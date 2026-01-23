"""
HIR -> LIR Lowering

Lowers high-level IR (with loops and branches) to low-level IR
(with basic blocks and jumps).
"""

from typing import Optional

from problem import SCRATCH_SIZE

from .hir import (
    SSAValue, Const, Operand, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from .lir import LIROpcode, LIRInst, Phi, BasicBlock, LIRFunction


class LoweringContext:
    """Context for lowering HIR to LIR."""

    def __init__(self):
        self.lir = LIRFunction(entry="entry")
        self.current_block: Optional[BasicBlock] = None
        self._block_counter = 0
        self._scratch_ptr = 0
        self._ssa_to_scratch: dict[int, int] = {}  # SSAValue.id -> scratch addr
        self._const_scratch: dict[int, int] = {}   # const value -> scratch addr
        self._pending_consts: list[tuple[int, int]] = []  # (scratch_addr, value) - deferred const loads

    def new_block(self, prefix: str = "bb") -> BasicBlock:
        """Create a new basic block."""
        name = f"{prefix}_{self._block_counter}"
        self._block_counter += 1
        block = BasicBlock(name=name)
        self.lir.blocks[name] = block
        return block

    def set_block(self, block: BasicBlock):
        """Set the current block for emission."""
        self.current_block = block

    def alloc_scratch(self, ssa: Optional[SSAValue] = None) -> int:
        """Allocate a scratch address, optionally binding to an SSA value."""
        addr = self._scratch_ptr
        self._scratch_ptr += 1
        # Note: Scratch overflow is checked in codegen, not here
        if ssa is not None:
            self._ssa_to_scratch[ssa.id] = addr
        return addr

    def get_scratch(self, ssa: SSAValue) -> int:
        """Get the scratch address for an SSA value."""
        if ssa.id not in self._ssa_to_scratch:
            # Allocate on demand
            self.alloc_scratch(ssa)
        return self._ssa_to_scratch[ssa.id]

    def get_operand(self, op: Operand) -> int:
        """Get scratch address for an operand (SSA value or const)."""
        if isinstance(op, SSAValue):
            return self.get_scratch(op)
        elif isinstance(op, Const):
            return self.get_const(op.value)
        else:
            raise ValueError(f"Unknown operand type: {op}")

    def get_const(self, value: int) -> int:
        """Get scratch address for a constant (with caching).

        Constants are deferred and emitted to the entry block later to ensure
        they dominate all uses (fixing the control-flow miscompilation bug).
        """
        if value not in self._const_scratch:
            addr = self.alloc_scratch()
            self._const_scratch[value] = addr
            # Defer const load - will be emitted to entry block later
            self._pending_consts.append((addr, value))
        return self._const_scratch[value]

    def emit_pending_consts(self):
        """Emit all pending constant loads to the entry block.

        This must be called after lowering is complete to ensure constants
        are materialized in the entry block where they dominate all uses.
        """
        if not self._pending_consts:
            return

        entry_block = self.lir.blocks[self.lir.entry]
        # Insert const loads at the beginning of the entry block
        const_insts = [
            LIRInst(LIROpcode.CONST, addr, [value], "load")
            for addr, value in self._pending_consts
        ]
        entry_block.instructions = const_insts + entry_block.instructions
        self._pending_consts = []

    def emit(self, inst: LIRInst):
        """Emit an instruction to the current block."""
        assert self.current_block is not None, "No current block"
        self.current_block.instructions.append(inst)

    def set_terminator(self, inst: LIRInst):
        """Set the terminator for the current block."""
        assert self.current_block is not None, "No current block"
        self.current_block.terminator = inst


def lower_to_lir(hir: HIRFunction) -> LIRFunction:
    """Lower HIR to LIR."""
    ctx = LoweringContext()

    # Create entry block
    entry = ctx.new_block("entry")
    ctx.lir.entry = entry.name  # Set the actual entry block name
    ctx.set_block(entry)

    # Lower all statements
    for stmt in hir.body:
        _lower_statement(stmt, ctx)

    # If the last block doesn't have a terminator, add halt
    if ctx.current_block and ctx.current_block.terminator is None:
        ctx.set_terminator(LIRInst(LIROpcode.HALT, None, [], "flow"))

    # Emit all pending constants to entry block (ensures they dominate all uses)
    ctx.emit_pending_consts()

    # Record max scratch used for phi elimination temp allocation
    ctx.lir.max_scratch_used = ctx._scratch_ptr - 1 if ctx._scratch_ptr > 0 else -1

    return ctx.lir


def _lower_statement(stmt: Statement, ctx: LoweringContext):
    """Lower a single HIR statement to LIR."""
    if isinstance(stmt, Op):
        _lower_op(stmt, ctx)
    elif isinstance(stmt, ForLoop):
        _lower_for_loop(stmt, ctx)
    elif isinstance(stmt, If):
        _lower_if(stmt, ctx)
    elif isinstance(stmt, Halt):
        ctx.set_terminator(LIRInst(LIROpcode.HALT, None, [], "flow"))
    elif isinstance(stmt, Pause):
        # Pause is a regular instruction, not a terminator
        # Execution continues after pause when run() is called again
        ctx.emit(LIRInst(LIROpcode.PAUSE, None, [], "flow"))
    else:
        raise ValueError(f"Unknown statement type: {stmt}")


def _lower_op(op: Op, ctx: LoweringContext):
    """Lower an Op to LIR instructions."""
    opcode_map = {
        "+": LIROpcode.ADD, "-": LIROpcode.SUB, "*": LIROpcode.MUL,
        "//": LIROpcode.DIV, "%": LIROpcode.MOD, "^": LIROpcode.XOR,
        "&": LIROpcode.AND, "|": LIROpcode.OR, "<<": LIROpcode.SHL,
        ">>": LIROpcode.SHR, "<": LIROpcode.LT, "==": LIROpcode.EQ,
        "load": LIROpcode.LOAD, "const": LIROpcode.CONST,
        "store": LIROpcode.STORE, "select": LIROpcode.SELECT,
    }

    lir_opcode = opcode_map.get(op.opcode)
    if lir_opcode is None:
        raise ValueError(f"Unknown opcode: {op.opcode}")

    # Get destination scratch address
    dest = None
    if op.result is not None:
        dest = ctx.get_scratch(op.result)

    # Get operand scratch addresses
    if op.opcode == "const":
        # Const has immediate value as operand
        operands = [op.operands[0].value]
    elif op.opcode == "store":
        # Store: (addr, value)
        operands = [ctx.get_operand(op.operands[0]), ctx.get_operand(op.operands[1])]
    else:
        operands = [ctx.get_operand(o) for o in op.operands]

    ctx.emit(LIRInst(lir_opcode, dest, operands, op.engine))


def _lower_for_loop(loop: ForLoop, ctx: LoweringContext):
    """Lower a ForLoop to LIR basic blocks with phis."""
    # Create blocks
    init_block = ctx.new_block("for_init")
    header_block = ctx.new_block("for_header")
    body_block = ctx.new_block("for_body")
    exit_block = ctx.new_block("for_exit")

    # Jump from current block to init
    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [init_block.name], "flow"))

    # === Init block ===
    ctx.set_block(init_block)

    # Load start value
    start_scratch = ctx.get_operand(loop.start)

    # Load end value
    end_scratch = ctx.get_operand(loop.end)

    # Allocate counter_init (copy of start for phi incoming from init)
    counter_init = ctx.alloc_scratch()
    zero_scratch = ctx.get_const(0)
    ctx.emit(LIRInst(LIROpcode.ADD, counter_init, [start_scratch, zero_scratch], "alu"))

    # Allocate iter_arg scratches (copy initial values)
    iter_arg_scratches = []
    for i, arg in enumerate(loop.iter_args):
        src = ctx.get_scratch(arg)
        dst = ctx.alloc_scratch()
        ctx.emit(LIRInst(LIROpcode.ADD, dst, [src, zero_scratch], "alu"))
        iter_arg_scratches.append(dst)

    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [header_block.name], "flow"))

    # === Header block ===
    ctx.set_block(header_block)

    # Allocate phi destinations
    counter_phi = ctx.alloc_scratch(loop.counter)
    param_scratches = [ctx.alloc_scratch(p) for p in loop.body_params]

    # We'll fill in phi incoming from body after lowering body
    # For now, set up phis with init incoming
    header_block.phis.append(Phi(counter_phi, {init_block.name: counter_init}))
    for i, param in enumerate(loop.body_params):
        header_block.phis.append(Phi(param_scratches[i], {init_block.name: iter_arg_scratches[i]}))

    # Condition: counter < end
    cond_scratch = ctx.alloc_scratch()
    ctx.emit(LIRInst(LIROpcode.LT, cond_scratch, [counter_phi, end_scratch], "alu"))
    ctx.set_terminator(LIRInst(LIROpcode.COND_JUMP, None, [cond_scratch, body_block.name, exit_block.name], "flow"))

    # === Body block ===
    ctx.set_block(body_block)

    # Lower body statements
    for stmt in loop.body:
        _lower_statement(stmt, ctx)

    # Get yield scratches
    yield_scratches = [ctx.get_scratch(y) for y in loop.yields]

    # Increment counter
    one_scratch = ctx.get_const(1)
    counter_next = ctx.alloc_scratch()
    ctx.emit(LIRInst(LIROpcode.ADD, counter_next, [counter_phi, one_scratch], "alu"))

    # Update phis with body incoming
    # Find the block that jumps back to header (could be current or a nested block)
    back_edge_block = ctx.current_block
    header_block.phis[0].incoming[back_edge_block.name] = counter_next
    for i in range(len(loop.body_params)):
        header_block.phis[i + 1].incoming[back_edge_block.name] = yield_scratches[i] if i < len(yield_scratches) else param_scratches[i]

    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [header_block.name], "flow"))

    # === Exit block ===
    ctx.set_block(exit_block)

    # Map results to param scratches (final values)
    for i, result in enumerate(loop.results):
        ctx._ssa_to_scratch[result.id] = param_scratches[i]


def _lower_if(if_stmt: If, ctx: LoweringContext):
    """Lower an If statement to LIR basic blocks with phis."""
    # Create blocks
    then_block = ctx.new_block("if_then")
    else_block = ctx.new_block("if_else")
    merge_block = ctx.new_block("if_merge")

    # Branch
    cond_scratch = ctx.get_scratch(if_stmt.cond)
    ctx.set_terminator(LIRInst(LIROpcode.COND_JUMP, None, [cond_scratch, then_block.name, else_block.name], "flow"))

    # === Then block ===
    ctx.set_block(then_block)
    for stmt in if_stmt.then_body:
        _lower_statement(stmt, ctx)
    then_yield_scratches = [ctx.get_scratch(y) for y in if_stmt.then_yields]
    then_exit_block = ctx.current_block
    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [merge_block.name], "flow"))

    # === Else block ===
    ctx.set_block(else_block)
    for stmt in if_stmt.else_body:
        _lower_statement(stmt, ctx)
    else_yield_scratches = [ctx.get_scratch(y) for y in if_stmt.else_yields]
    else_exit_block = ctx.current_block
    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [merge_block.name], "flow"))

    # === Merge block ===
    ctx.set_block(merge_block)

    # Set up phis
    for i, result in enumerate(if_stmt.results):
        result_scratch = ctx.alloc_scratch(result)
        merge_block.phis.append(Phi(
            dest=result_scratch,
            incoming={
                then_exit_block.name: then_yield_scratches[i],
                else_exit_block.name: else_yield_scratches[i],
            }
        ))
