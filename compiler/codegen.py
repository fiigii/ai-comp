"""
LIR -> VLIW Codegen

Generates VLIW bundles from LIR instructions.
"""

from typing import Optional

from problem import SCRATCH_SIZE

from .lir import LIROpcode, LIRInst, LIRFunction


def linearize(lir: LIRFunction) -> tuple[list[LIRInst], dict[str, int]]:
    """
    Linearize LIR into a sequence of instructions.

    Returns (instructions, label_map) where label_map maps block names to instruction indices.
    """
    instructions = []
    label_map = {}

    # Simple linearization: visit blocks in order starting from entry
    visited = set()
    worklist = [lir.entry]
    block_order = []

    while worklist:
        name = worklist.pop(0)
        if name in visited:
            continue
        visited.add(name)
        block = lir.blocks[name]
        block_order.append(block)

        # Add successors to worklist
        if block.terminator:
            if block.terminator.opcode == LIROpcode.JUMP:
                worklist.append(block.terminator.operands[0])
            elif block.terminator.opcode == LIROpcode.COND_JUMP:
                # operands: [cond, true_target, false_target]
                worklist.append(block.terminator.operands[1])
                worklist.append(block.terminator.operands[2])

    # Emit instructions in block order
    for i, block in enumerate(block_order):
        label_map[block.name] = len(instructions)
        instructions.extend(block.instructions)
        if block.terminator:
            # Omit unconditional jump to immediate next block (fallthrough)
            if (
                block.terminator.opcode == LIROpcode.JUMP
                and i + 1 < len(block_order)
                and block.terminator.operands[0] == block_order[i + 1].name
            ):
                continue
            instructions.append(block.terminator)

    return instructions, label_map


def resolve_labels(instructions: list[LIRInst], label_map: dict[str, int]) -> list[LIRInst]:
    """Replace label references with instruction addresses."""
    resolved = []
    for inst in instructions:
        new_operands = []
        for op in inst.operands:
            if isinstance(op, str) and op in label_map:
                new_operands.append(label_map[op])
            else:
                new_operands.append(op)
        resolved.append(LIRInst(inst.opcode, inst.dest, new_operands, inst.engine))
    return resolved


def codegen(instructions: list[LIRInst], zero_scratch: int) -> list[dict]:
    """
    Generate VLIW bundles from LIR instructions.

    Simple strategy: one instruction per bundle.
    """
    bundles = []

    for inst in instructions:
        slot = _inst_to_slot(inst, zero_scratch)
        if slot is not None:
            bundle = {inst.engine: [slot]}
            bundles.append(bundle)

    return bundles


def _inst_to_slot(inst: LIRInst, zero_scratch: int) -> Optional[tuple]:
    """Convert a LIR instruction to a machine slot tuple."""
    match inst.opcode:
        # ALU operations
        case LIROpcode.ADD | LIROpcode.SUB | LIROpcode.MUL | LIROpcode.DIV | \
             LIROpcode.MOD | LIROpcode.XOR | LIROpcode.AND | LIROpcode.OR | \
             LIROpcode.SHL | LIROpcode.SHR | LIROpcode.LT | LIROpcode.EQ:
            return (inst.opcode.value, inst.dest, inst.operands[0], inst.operands[1])

        # Load operations
        case LIROpcode.CONST:
            return ("const", inst.dest, inst.operands[0])
        case LIROpcode.LOAD:
            return ("load", inst.dest, inst.operands[0])

        # Store operations
        case LIROpcode.STORE:
            return ("store", inst.operands[0], inst.operands[1])

        # Flow operations
        case LIROpcode.SELECT:
            return ("select", inst.dest, inst.operands[0], inst.operands[1], inst.operands[2])
        case LIROpcode.JUMP:
            return ("jump", inst.operands[0])
        case LIROpcode.COND_JUMP:
            # cond_jump cond, true_target, false_target
            # Machine has: cond_jump cond, target (jumps if true)
            # We need to handle false_target as fallthrough or separate jump
            # For now, emit cond_jump for true, then jump for false
            # This is handled specially below
            return None  # Handled in codegen
        case LIROpcode.HALT:
            return ("halt",)
        case LIROpcode.PAUSE:
            return ("pause",)

        # Pseudo-ops
        case LIROpcode.COPY:
            # Implement copy as add with zero
            return ("+", inst.dest, inst.operands[0], zero_scratch)

        # Vector ALU operations
        # Machine uses base scratch address; VM knows to access VLEN consecutive locations
        case LIROpcode.VADD | LIROpcode.VSUB | LIROpcode.VMUL | LIROpcode.VDIV | \
             LIROpcode.VMOD | LIROpcode.VXOR | LIROpcode.VAND | LIROpcode.VOR | \
             LIROpcode.VSHL | LIROpcode.VSHR | LIROpcode.VLT | LIROpcode.VEQ:
            # Strip the "v" prefix to get the scalar op name
            scalar_op = inst.opcode.value[1:]  # "v+" -> "+"
            dest_base = inst.dest[0]
            a_base = inst.operands[0][0]
            b_base = inst.operands[1][0]
            return (scalar_op, dest_base, a_base, b_base)

        case LIROpcode.VBROADCAST:
            # vbroadcast: scalar -> vector
            dest_base = inst.dest[0]
            scalar = inst.operands[0]
            return ("vbroadcast", dest_base, scalar)

        case LIROpcode.MULTIPLY_ADD:
            # multiply_add: vec, vec, vec -> vec
            dest_base = inst.dest[0]
            a_base = inst.operands[0][0]
            b_base = inst.operands[1][0]
            c_base = inst.operands[2][0]
            return ("multiply_add", dest_base, a_base, b_base, c_base)

        case LIROpcode.VLOAD:
            # vload: addr (scalar) -> vector
            dest_base = inst.dest[0]
            addr = inst.operands[0]
            return ("vload", dest_base, addr)

        case LIROpcode.VSTORE:
            # vstore: addr (scalar), vec -> None
            addr = inst.operands[0]
            src_base = inst.operands[1][0]
            return ("vstore", addr, src_base)

        case LIROpcode.VSELECT:
            # vselect: cond (vec), a (vec), b (vec) -> vec
            dest_base = inst.dest[0]
            cond_base = inst.operands[0][0]
            a_base = inst.operands[1][0]
            b_base = inst.operands[2][0]
            return ("vselect", dest_base, cond_base, a_base, b_base)

        case _:
            raise NotImplementedError(f"Codegen for {inst.opcode}")


def compile_to_vliw(lir: LIRFunction) -> list[dict]:
    """Full compilation from LIR to VLIW bundles.

    Note: Phis must be eliminated before calling this function.
    """
    # Find zero constant for copy operations
    zero_scratch = None
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.opcode == LIROpcode.CONST and inst.operands[0] == 0:
                zero_scratch = inst.dest
                break
        if zero_scratch is not None:
            break

    # If no zero constant found, materialize one in the entry block
    if zero_scratch is None:
        # Find the maximum used scratch address to allocate a new one
        max_scratch = -1
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.dest is not None:
                    if isinstance(inst.dest, list):
                        # Vector dest - use the maximum address in the list
                        max_scratch = max(max_scratch, max(inst.dest))
                    else:
                        max_scratch = max(max_scratch, inst.dest)
                # Skip CONST operands - they are immediate values, not scratch indices
                if inst.opcode != LIROpcode.CONST:
                    for op in inst.operands:
                        if isinstance(op, list):
                            # Vector operand
                            max_scratch = max(max_scratch, max(op))
                        elif isinstance(op, int) and op >= 0:
                            max_scratch = max(max_scratch, op)

        zero_scratch = max_scratch + 1
        assert zero_scratch < SCRATCH_SIZE, "Out of scratch space for zero constant"

        # Add const 0 at the beginning of entry block
        entry_block = lir.blocks[lir.entry]
        zero_inst = LIRInst(LIROpcode.CONST, zero_scratch, [0], "load")
        entry_block.instructions.insert(0, zero_inst)

    # Linearize
    instructions, label_map = linearize(lir)

    # Handle COND_JUMP specially (expand to cond_jump + optional jump)
    expanded = []
    for i, inst in enumerate(instructions):
        if inst.opcode == LIROpcode.COND_JUMP:
            cond, true_target, false_target = inst.operands
            # Emit conditional jump to true target
            expanded.append(LIRInst(LIROpcode.COND_JUMP, None, [cond, true_target], "flow"))
            # Emit unconditional jump to false target only if it isn't fallthrough
            false_is_fallthrough = (
                isinstance(false_target, str)
                and false_target in label_map
                and label_map[false_target] == i + 1
            )
            if not false_is_fallthrough:
                expanded.append(LIRInst(LIROpcode.JUMP, None, [false_target], "flow"))
        else:
            expanded.append(inst)

    # Recompute label map after expansion
    # We need to adjust labels since we added instructions
    new_label_map = {}
    old_to_new_idx = {}
    new_idx = 0
    old_idx = 0
    for i, inst in enumerate(instructions):
        old_to_new_idx[old_idx] = new_idx
        if inst.opcode == LIROpcode.COND_JUMP:
            # cond_jump + optional jump
            cond, true_target, false_target = inst.operands
            false_is_fallthrough = (
                isinstance(false_target, str)
                and false_target in label_map
                and label_map[false_target] == i + 1
            )
            new_idx += 1
            if not false_is_fallthrough:
                new_idx += 1
        else:
            new_idx += 1
        old_idx += 1

    for name, old_idx in label_map.items():
        new_label_map[name] = old_to_new_idx.get(old_idx, old_idx)

    # Resolve labels
    resolved = resolve_labels(expanded, new_label_map)

    # Generate bundles
    bundles = []
    for inst in resolved:
        if inst.opcode == LIROpcode.COND_JUMP:
            # cond_jump cond, target
            slot = ("cond_jump", inst.operands[0], inst.operands[1])
            bundles.append({"flow": [slot]})
        else:
            slot = _inst_to_slot(inst, zero_scratch)
            if slot is not None:
                bundles.append({inst.engine: [slot]})

    return bundles
