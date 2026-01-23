"""
Common Subexpression Elimination (CSE) Pass

Eliminates redundant computations using value numbering.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..hir import (
    SSAValue, Const, Operand, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..ssa_context import SSARenumberContext


@dataclass
class CSEContext:
    """Context for CSE value numbering."""

    # Maps value_number (tuple) -> first SSA computing it
    expr_to_ssa: dict[tuple, SSAValue] = field(default_factory=dict)

    # Maps SSAValue.id -> its value number (tuple)
    ssa_to_value_number: dict[int, tuple] = field(default_factory=dict)

    # Memory epoch counter (increments on store operations)
    # Enables finer-grained load CSE: loads with same address and epoch can be merged
    mem_epoch: int = 0

    # Parent context for nested scopes
    parent: Optional['CSEContext'] = None

    def lookup(self, value_number: tuple) -> Optional[SSAValue]:
        """Look up a value number in this context or parent contexts."""
        if value_number in self.expr_to_ssa:
            return self.expr_to_ssa[value_number]
        if self.parent is not None:
            return self.parent.lookup(value_number)
        return None

    def get_value_number(self, ssa_id: int) -> Optional[tuple]:
        """Get the value number for an SSA value from this context or parents."""
        if ssa_id in self.ssa_to_value_number:
            return self.ssa_to_value_number[ssa_id]
        if self.parent is not None:
            return self.parent.get_value_number(ssa_id)
        return None

    def get_mem_epoch(self) -> int:
        """Get current memory epoch (including parent epochs)."""
        if self.parent is not None:
            return self.parent.get_mem_epoch() + self.mem_epoch
        return self.mem_epoch

    def increment_epoch(self):
        """Increment memory epoch (called on store)."""
        self.mem_epoch += 1

    def record(self, value_number: tuple, ssa: SSAValue):
        """Record a new expression -> SSA mapping."""
        self.expr_to_ssa[value_number] = ssa
        self.ssa_to_value_number[ssa.id] = value_number

    def child_context(self) -> 'CSEContext':
        """Create a child context for nested scopes."""
        return CSEContext(parent=self)


# Operations that are safe for CSE
CSE_SAFE_OPS = {
    # ALU ops
    "+", "-", "*", "//", "%", "^", "&", "|", "<<", ">>", "<", "==",
    # Load engine ops that are CSE-safe
    "const",
    # Flow engine ops
    "select",
}

# Load ops that can be CSE'd when memory isn't clobbered
LOAD_OPS = {"load"}

# Commutative operations (a op b == b op a)
COMMUTATIVE_OPS = {"+", "*", "^", "&", "|", "=="}


class CSEPass(Pass):
    """
    Common Subexpression Elimination using value numbering.

    Eliminates redundant computations by tracking expressions and their values:
    - Safe for CSE: ALU ops, const, select
    - Conditional CSE: load (only when memory hasn't been clobbered by a store)
    - Never CSE'd: store (has side effects)

    Memory operations are handled conservatively:
    - Any store operation invalidates ALL subsequent load CSE opportunities
    - This is safe but may miss some optimization opportunities
    """

    def __init__(self):
        super().__init__()
        self._expressions_analyzed = 0
        self._expressions_eliminated = 0
        self._consts_eliminated = 0
        self._loads_eliminated = 0

    @property
    def name(self) -> str:
        return "cse"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        # Initialize metrics
        self._init_metrics()
        self._expressions_analyzed = 0
        self._expressions_eliminated = 0
        self._consts_eliminated = 0
        self._loads_eliminated = 0

        # Check if pass is enabled
        if not config.enabled:
            return hir

        # Create SSA context for renumbering (in case we need new SSAs)
        ssa_ctx = SSARenumberContext(hir.num_ssa_values)

        # Create CSE context
        cse_ctx = CSEContext()

        # Transform body
        new_body = self._transform_statements(hir.body, cse_ctx, ssa_ctx)

        # Record custom metrics
        if self._metrics:
            self._metrics.custom = {
                "expressions_analyzed": self._expressions_analyzed,
                "expressions_eliminated": self._expressions_eliminated,
                "consts_eliminated": self._consts_eliminated,
                "loads_eliminated": self._loads_eliminated,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=ssa_ctx.next_id
        )

    def _transform_statements(
        self,
        stmts: list[Statement],
        ctx: CSEContext,
        ssa_ctx: SSARenumberContext
    ) -> list[Statement]:
        """Transform a list of statements, eliminating common subexpressions."""
        result = []

        for stmt in stmts:
            # Apply any SSA bindings from eliminated expressions
            stmt = self._apply_bindings(stmt, ssa_ctx.result_bindings)

            if isinstance(stmt, Op):
                transformed = self._transform_op(stmt, ctx, ssa_ctx)
                if transformed is not None:
                    result.append(transformed)
            elif isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt, ctx, ssa_ctx))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, ctx, ssa_ctx))
            else:
                # Halt, Pause - keep as is
                result.append(stmt)

        return result

    def _transform_op(
        self,
        op: Op,
        ctx: CSEContext,
        ssa_ctx: SSARenumberContext
    ) -> Optional[Op]:
        """
        Apply CSE to a single Op.

        Returns the Op to keep, or None if eliminated.
        """
        self._expressions_analyzed += 1

        # Store operations are never CSE'd and increment memory epoch
        if op.opcode == "store":
            ctx.increment_epoch()
            return op

        # Check if this is a CSE-able operation
        is_safe_op = op.opcode in CSE_SAFE_OPS
        is_load_op = op.opcode in LOAD_OPS

        if not is_safe_op and not is_load_op:
            # Unknown op, can't CSE but keep it
            return op

        # Compute value number for this expression (epoch included for loads)
        value_number = self._make_value_number_key(op, ctx)

        if value_number is None:
            # Couldn't compute value number (operand not in context)
            # Still keep the op and record its value number if possible
            if op.result is not None:
                # Create a fresh value number based on the result SSA
                fresh_vn = ("ssa", op.result.id)
                ctx.record(fresh_vn, op.result)
            return op

        # Check if we've seen this expression before
        existing_ssa = ctx.lookup(value_number)

        if existing_ssa is not None:
            # Found a match - eliminate this expression
            self._expressions_eliminated += 1

            if op.opcode == "const":
                self._consts_eliminated += 1
            elif op.opcode == "load":
                self._loads_eliminated += 1

            # Bind the result to the existing SSA
            if op.result is not None:
                ssa_ctx.bind_result(op.result.id, existing_ssa)

            return None  # Eliminate the op

        # New expression - record it
        if op.result is not None:
            ctx.record(value_number, op.result)

        return op

    def _transform_for_loop(
        self,
        loop: ForLoop,
        ctx: CSEContext,
        ssa_ctx: SSARenumberContext
    ) -> ForLoop:
        """Transform a ForLoop, handling nested scope for CSE."""
        # Create child context for loop body
        # Loop body params get fresh value numbers each iteration
        child_ctx = ctx.child_context()

        # Give body_params fresh value numbers (they're unique per iteration)
        for param in loop.body_params:
            fresh_vn = ("loop_param", param.id)
            child_ctx.record(fresh_vn, param)

        # Transform loop body
        new_body = self._transform_statements(loop.body, child_ctx, ssa_ctx)

        # Propagate memory epoch up to parent
        ctx.mem_epoch += child_ctx.mem_epoch

        # Resolve yields through bindings
        new_yields = [
            self._resolve_binding(y, ssa_ctx.result_bindings)
            for y in loop.yields
        ]

        return ForLoop(
            counter=loop.counter,
            start=loop.start,
            end=loop.end,
            iter_args=loop.iter_args,
            body_params=loop.body_params,
            body=new_body,
            yields=new_yields,
            results=loop.results
        )

    def _transform_if(
        self,
        if_stmt: If,
        ctx: CSEContext,
        ssa_ctx: SSARenumberContext
    ) -> If:
        """Transform an If statement, handling separate scopes for each branch."""
        # Create separate child contexts for each branch
        # Branches don't see each other's expressions
        then_ctx = ctx.child_context()
        else_ctx = ctx.child_context()

        # Transform each branch
        new_then_body = self._transform_statements(if_stmt.then_body, then_ctx, ssa_ctx)
        new_else_body = self._transform_statements(if_stmt.else_body, else_ctx, ssa_ctx)

        # Propagate memory epoch up (use max since either branch could execute)
        ctx.mem_epoch += max(then_ctx.mem_epoch, else_ctx.mem_epoch)

        # Resolve yields through bindings
        new_then_yields = [
            self._resolve_binding(y, ssa_ctx.result_bindings)
            for y in if_stmt.then_yields
        ]
        new_else_yields = [
            self._resolve_binding(y, ssa_ctx.result_bindings)
            for y in if_stmt.else_yields
        ]

        return If(
            cond=if_stmt.cond,
            then_body=new_then_body,
            then_yields=new_then_yields,
            else_body=new_else_body,
            else_yields=new_else_yields,
            results=if_stmt.results
        )

    def _make_value_number_key(self, op: Op, ctx: CSEContext) -> Optional[tuple]:
        """
        Compute the value number key for an operation.

        Returns None if we can't compute a value number (e.g., unknown operand).
        """
        if op.opcode == "const":
            # Constants are identified by their value
            return ("const", op.operands[0].value)

        # For other ops, use opcode + operand value numbers
        operand_vns = []
        for operand in op.operands:
            vn = self._get_operand_value_number(operand, ctx)
            if vn is None:
                return None
            operand_vns.append(vn)

        # Canonicalize commutative operations by sorting operands
        if op.opcode in COMMUTATIVE_OPS and len(operand_vns) == 2:
            operand_vns = sorted(operand_vns, key=lambda x: str(x))

        # Include memory epoch for load operations
        # This enables loads with same address but different epochs to have different value numbers
        if op.opcode in LOAD_OPS:
            return (op.opcode, tuple(operand_vns), ctx.get_mem_epoch())

        return (op.opcode, tuple(operand_vns))

    def _get_operand_value_number(
        self,
        operand: Operand,
        ctx: CSEContext
    ) -> Optional[tuple]:
        """Get the value number for an operand."""
        if isinstance(operand, Const):
            return ("const", operand.value)

        if isinstance(operand, SSAValue):
            # Look up in context
            vn = ctx.get_value_number(operand.id)
            if vn is not None:
                return vn
            # Unknown SSA - use its ID as value number
            return ("ssa", operand.id)

        return None

    def _resolve_binding(
        self,
        value: SSAValue,
        bindings: dict[int, SSAValue]
    ) -> SSAValue:
        """Resolve SSA bindings transitively."""
        while value.id in bindings:
            value = bindings[value.id]
        return value

    def _apply_bindings(
        self,
        stmt: Statement,
        bindings: dict[int, SSAValue]
    ) -> Statement:
        """Apply SSA value bindings to a statement's operands."""
        if not bindings:
            return stmt

        if isinstance(stmt, Op):
            new_operands = []
            for op in stmt.operands:
                if isinstance(op, SSAValue) and op.id in bindings:
                    new_operands.append(self._resolve_binding(op, bindings))
                else:
                    new_operands.append(op)
            return Op(stmt.opcode, stmt.result, new_operands, stmt.engine)

        elif isinstance(stmt, ForLoop):
            new_iter_args = []
            for arg in stmt.iter_args:
                if arg.id in bindings:
                    new_iter_args.append(self._resolve_binding(arg, bindings))
                else:
                    new_iter_args.append(arg)

            new_start = stmt.start
            if isinstance(stmt.start, SSAValue) and stmt.start.id in bindings:
                new_start = self._resolve_binding(stmt.start, bindings)

            new_end = stmt.end
            if isinstance(stmt.end, SSAValue) and stmt.end.id in bindings:
                new_end = self._resolve_binding(stmt.end, bindings)

            new_yields = [
                self._resolve_binding(y, bindings) if y.id in bindings else y
                for y in stmt.yields
            ]

            return ForLoop(
                counter=stmt.counter,
                start=new_start,
                end=new_end,
                iter_args=new_iter_args,
                body_params=stmt.body_params,
                body=stmt.body,
                yields=new_yields,
                results=stmt.results
            )

        elif isinstance(stmt, If):
            new_cond = stmt.cond
            if stmt.cond.id in bindings:
                new_cond = self._resolve_binding(stmt.cond, bindings)

            new_then_yields = [
                self._resolve_binding(y, bindings) if y.id in bindings else y
                for y in stmt.then_yields
            ]
            new_else_yields = [
                self._resolve_binding(y, bindings) if y.id in bindings else y
                for y in stmt.else_yields
            ]

            return If(
                cond=new_cond,
                then_body=stmt.then_body,
                then_yields=new_then_yields,
                else_body=stmt.else_body,
                else_yields=new_else_yields,
                results=stmt.results
            )

        return stmt
