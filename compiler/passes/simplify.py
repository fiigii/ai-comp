"""
Simplify Pass

Performs constant folding and algebraic identity simplifications on HIR.
"""

from typing import Optional

from ..hir import (
    SSAValue, Const, Operand, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..ssa_context import SSARenumberContext


# Operations that can be constant-folded (all binary arithmetic)
FOLDABLE_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "//": lambda a, b: a // b if b != 0 else None,
    "%": lambda a, b: a % b if b != 0 else None,
    "^": lambda a, b: a ^ b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "<<": lambda a, b: a << b,
    ">>": lambda a, b: a >> b,
    "<": lambda a, b: 1 if a < b else 0,
    "==": lambda a, b: 1 if a == b else 0,
}


class SimplifyPass(Pass):
    """
    Simplify pass that performs constant folding and algebraic identity simplifications.

    Transformations:
    - Constant folding: Const(a) op Const(b) -> Const(result)
    - Identity: x + 0 -> x, x * 1 -> x, x ^ 0 -> x, x | 0 -> x
    - Annihilation: x * 0 -> 0, x & 0 -> 0
    """

    def __init__(self):
        super().__init__()
        self._constants_folded = 0
        self._identities_simplified = 0
        # Maps SSA id -> known constant value
        self._known_constants: dict[int, int] = {}

    @property
    def name(self) -> str:
        return "simplify"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        # Initialize metrics
        self._init_metrics()
        self._constants_folded = 0
        self._identities_simplified = 0
        self._known_constants = {}

        # Check if pass is enabled
        if not config.enabled:
            return hir

        # Create SSA context for renumbering (in case we need new SSAs)
        ssa_ctx = SSARenumberContext(hir.num_ssa_values)

        # Transform body
        new_body = self._transform_statements(hir.body, ssa_ctx)

        # Record custom metrics
        if self._metrics:
            self._metrics.custom = {
                "constants_folded": self._constants_folded,
                "identities_simplified": self._identities_simplified,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=ssa_ctx.next_id
        )

    def _transform_statements(
        self,
        stmts: list[Statement],
        ssa_ctx: SSARenumberContext
    ) -> list[Statement]:
        """Transform a list of statements."""
        result = []

        for stmt in stmts:
            # Apply any SSA bindings from simplified expressions
            stmt = self._apply_bindings(stmt, ssa_ctx.result_bindings)

            if isinstance(stmt, Op):
                transformed = self._transform_op(stmt, ssa_ctx)
                result.append(transformed)
            elif isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt, ssa_ctx))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, ssa_ctx))
            else:
                # Halt, Pause - keep as is
                result.append(stmt)

        return result

    def _transform_op(self, op: Op, ssa_ctx: SSARenumberContext) -> Op:
        """Apply simplifications to a single Op."""
        # Track const ops for future constant folding
        if op.opcode == "const" and op.result is not None and len(op.operands) == 1:
            const_operand = op.operands[0]
            if isinstance(const_operand, Const):
                self._known_constants[op.result.id] = const_operand.value
            return op

        # Skip ops without results or with wrong operand count
        if op.result is None or len(op.operands) != 2:
            return op

        left, right = op.operands

        # Try constant folding
        folded = self._try_constant_fold(op.opcode, left, right)
        if folded is not None:
            self._constants_folded += 1
            # Track the result as a known constant
            self._known_constants[op.result.id] = folded
            # Create a const op instead
            return Op("const", op.result, [Const(folded)], "load")

        # Try algebraic identity simplifications
        simplified = self._try_simplify_identity(op.opcode, left, right, op.result, ssa_ctx)
        if simplified is not None:
            self._identities_simplified += 1
            return simplified

        return op

    def _get_const_value(self, operand: Operand) -> Optional[int]:
        """Get constant value if operand is a known constant."""
        if isinstance(operand, Const):
            return operand.value
        if isinstance(operand, SSAValue) and operand.id in self._known_constants:
            return self._known_constants[operand.id]
        return None

    def _try_constant_fold(self, opcode: str, left: Operand, right: Operand) -> Optional[int]:
        """Try to fold two constants. Returns result value or None."""
        if opcode not in FOLDABLE_OPS:
            return None

        left_val = self._get_const_value(left)
        right_val = self._get_const_value(right)

        if left_val is None or right_val is None:
            return None

        fold_fn = FOLDABLE_OPS[opcode]
        result = fold_fn(left_val, right_val)
        return result

    def _try_simplify_identity(
        self,
        opcode: str,
        left: Operand,
        right: Operand,
        result: SSAValue,
        ssa_ctx: SSARenumberContext
    ) -> Optional[Op]:
        """Try to simplify using algebraic identities. Returns replacement Op or None."""
        left_val = self._get_const_value(left)
        right_val = self._get_const_value(right)
        left_is_const = left_val is not None
        right_is_const = right_val is not None

        # x + 0 -> x, 0 + x -> x
        if opcode == "+":
            if right_is_const and right_val == 0:
                ssa_ctx.bind_result(result.id, left)
                # Return a no-op const that will get DCE'd, we bound the result
                return Op("const", result, [Const(0)], "load")
            if left_is_const and left_val == 0:
                ssa_ctx.bind_result(result.id, right)
                return Op("const", result, [Const(0)], "load")

        # x - 0 -> x
        if opcode == "-":
            if right_is_const and right_val == 0:
                ssa_ctx.bind_result(result.id, left)
                return Op("const", result, [Const(0)], "load")

        # x * 1 -> x, 1 * x -> x
        if opcode == "*":
            if right_is_const and right_val == 1:
                ssa_ctx.bind_result(result.id, left)
                return Op("const", result, [Const(0)], "load")
            if left_is_const and left_val == 1:
                ssa_ctx.bind_result(result.id, right)
                return Op("const", result, [Const(0)], "load")
            # x * 0 -> 0, 0 * x -> 0
            if right_is_const and right_val == 0:
                return Op("const", result, [Const(0)], "load")
            if left_is_const and left_val == 0:
                return Op("const", result, [Const(0)], "load")

        # x ^ 0 -> x, 0 ^ x -> x
        if opcode == "^":
            if right_is_const and right_val == 0:
                ssa_ctx.bind_result(result.id, left)
                return Op("const", result, [Const(0)], "load")
            if left_is_const and left_val == 0:
                ssa_ctx.bind_result(result.id, right)
                return Op("const", result, [Const(0)], "load")

        # x & 0 -> 0, 0 & x -> 0
        if opcode == "&":
            if (right_is_const and right_val == 0) or (left_is_const and left_val == 0):
                return Op("const", result, [Const(0)], "load")

        # x | 0 -> x, 0 | x -> x
        if opcode == "|":
            if right_is_const and right_val == 0:
                ssa_ctx.bind_result(result.id, left)
                return Op("const", result, [Const(0)], "load")
            if left_is_const and left_val == 0:
                ssa_ctx.bind_result(result.id, right)
                return Op("const", result, [Const(0)], "load")

        return None

    def _transform_for_loop(self, loop: ForLoop, ssa_ctx: SSARenumberContext) -> ForLoop:
        """Transform a ForLoop."""
        new_body = self._transform_statements(loop.body, ssa_ctx)

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

    def _transform_if(self, if_stmt: If, ssa_ctx: SSARenumberContext) -> If:
        """Transform an If statement."""
        new_then_body = self._transform_statements(if_stmt.then_body, ssa_ctx)
        new_else_body = self._transform_statements(if_stmt.else_body, ssa_ctx)

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
