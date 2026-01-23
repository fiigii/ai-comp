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
    - Strength reduction: % 2 -> & 1, * 2 -> << 1, * power_of_2 -> << log2(n)
    - Select optimization: select(cond, x, 0) -> *(x, cond) when cond is boolean
    - Parity pattern: ==(x & 1, 0) followed by select(cond, 1, 2) -> (x & 1) + 1
    """

    def __init__(self):
        super().__init__()
        self._constants_folded = 0
        self._identities_simplified = 0
        self._strength_reductions = 0
        self._select_optimizations = 0
        self._parity_patterns = 0
        # Maps SSA id -> known constant value
        self._known_constants: dict[int, int] = {}
        # SSA ids known to be boolean (0 or 1) - from comparisons or & 1
        self._boolean_values: set[int] = set()
        # Maps SSA id -> the SSA id it's negated from (for ==(x, 0) where x is boolean)
        # If _negated_boolean[a] = b, then a = (1 - b) = !b
        self._negated_boolean: dict[int, int] = {}
        # Feature options (set in run())
        self._opts: dict[str, bool] = {}

    @property
    def name(self) -> str:
        return "simplify"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        # Initialize metrics
        self._init_metrics()
        self._constants_folded = 0
        self._identities_simplified = 0
        self._strength_reductions = 0
        self._select_optimizations = 0
        self._parity_patterns = 0
        self._known_constants = {}
        self._boolean_values = set()
        self._negated_boolean = {}

        # Read feature options from config (all enabled by default)
        self._opts = {
            "constant_folding": config.options.get("constant_folding", True),
            "identities": config.options.get("identities", True),
            "strength_reduction": config.options.get("strength_reduction", True),
            "select_optimization": config.options.get("select_optimization", True),
            "parity_pattern": config.options.get("parity_pattern", True),
        }

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
                "strength_reductions": self._strength_reductions,
                "select_optimizations": self._select_optimizations,
                "parity_patterns": self._parity_patterns,
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
                # Constants 0 and 1 are boolean
                if const_operand.value in (0, 1):
                    self._boolean_values.add(op.result.id)
            return op

        # Handle select (3 operands)
        if op.opcode == "select" and op.result is not None and len(op.operands) == 3:
            simplified = self._try_simplify_select(op, ssa_ctx)
            if simplified is not None:
                # Counter incremented inside _try_simplify_select
                return simplified
            return op

        # Skip ops without results or with wrong operand count
        if op.result is None or len(op.operands) != 2:
            return op

        left, right = op.operands

        # Try constant folding
        if self._opts.get("constant_folding", True):
            folded = self._try_constant_fold(op.opcode, left, right)
            if folded is not None:
                self._constants_folded += 1
                # Track the result as a known constant
                self._known_constants[op.result.id] = folded
                # Create a const op instead
                return Op("const", op.result, [Const(folded)], "load")

        # Try algebraic identity simplifications (returns op, metric_type)
        simplified, metric_type = self._try_simplify_identity(op.opcode, left, right, op.result, ssa_ctx)
        if simplified is not None:
            # Increment appropriate counter
            if metric_type == "identity":
                self._identities_simplified += 1
            elif metric_type == "strength":
                self._strength_reductions += 1
            # Track boolean status if the simplified op produces a boolean
            if simplified.opcode in ("<", "=="):
                self._boolean_values.add(op.result.id)
            elif simplified.opcode == "&" and len(simplified.operands) == 2:
                # Check if this is & 1
                r_val = self._get_const_value(simplified.operands[1])
                if r_val == 1:
                    self._boolean_values.add(op.result.id)
            return simplified

        # Track boolean values from comparisons
        if op.opcode in ("<", "=="):
            self._boolean_values.add(op.result.id)
            # Track negated booleans: ==(x, 0) where x is boolean -> result is !x
            if op.opcode == "==":
                right_val = self._get_const_value(right)
                if right_val == 0 and self._is_boolean(left):
                    if isinstance(left, SSAValue):
                        self._negated_boolean[op.result.id] = left.id

        # Track & 1 as producing boolean
        if op.opcode == "&":
            right_val = self._get_const_value(right)
            if right_val == 1:
                self._boolean_values.add(op.result.id)

        return op

    def _get_const_value(self, operand: Operand) -> Optional[int]:
        """Get constant value if operand is a known constant."""
        if isinstance(operand, Const):
            return operand.value
        if isinstance(operand, SSAValue) and operand.id in self._known_constants:
            return self._known_constants[operand.id]
        return None

    def _is_boolean(self, operand: Operand) -> bool:
        """Check if operand is known to be boolean (0 or 1)."""
        if isinstance(operand, Const):
            return operand.value in (0, 1)
        if isinstance(operand, SSAValue):
            return operand.id in self._boolean_values
        return False

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
        # Apply 32-bit wrap semantics (VM uses mod 2**32)
        if result is not None:
            result = result & 0xFFFFFFFF
        return result

    def _try_simplify_identity(
        self,
        opcode: str,
        left: Operand,
        right: Operand,
        result: SSAValue,
        ssa_ctx: SSARenumberContext
    ) -> tuple[Optional[Op], Optional[str]]:
        """Try to simplify using algebraic identities.

        Returns tuple of (replacement Op or None, metric_type or None).
        metric_type is "identity" for algebraic identities, "strength" for strength reductions.
        """
        left_val = self._get_const_value(left)
        right_val = self._get_const_value(right)
        left_is_const = left_val is not None
        right_is_const = right_val is not None

        # Algebraic identities (only if enabled)
        if self._opts.get("identities", True):
            # x + 0 -> x, 0 + x -> x
            if opcode == "+":
                if right_is_const and right_val == 0:
                    ssa_ctx.bind_result(result.id, left)
                    # Return a no-op const that will get DCE'd, we bound the result
                    return Op("const", result, [Const(0)], "load"), "identity"
                if left_is_const and left_val == 0:
                    ssa_ctx.bind_result(result.id, right)
                    return Op("const", result, [Const(0)], "load"), "identity"

            # x - 0 -> x
            if opcode == "-":
                if right_is_const and right_val == 0:
                    ssa_ctx.bind_result(result.id, left)
                    return Op("const", result, [Const(0)], "load"), "identity"

            # x * 1 -> x, 1 * x -> x
            if opcode == "*":
                if right_is_const and right_val == 1:
                    ssa_ctx.bind_result(result.id, left)
                    return Op("const", result, [Const(0)], "load"), "identity"
                if left_is_const and left_val == 1:
                    ssa_ctx.bind_result(result.id, right)
                    return Op("const", result, [Const(0)], "load"), "identity"
                # x * 0 -> 0, 0 * x -> 0
                if right_is_const and right_val == 0:
                    return Op("const", result, [Const(0)], "load"), "identity"
                if left_is_const and left_val == 0:
                    return Op("const", result, [Const(0)], "load"), "identity"

            # x ^ 0 -> x, 0 ^ x -> x
            if opcode == "^":
                if right_is_const and right_val == 0:
                    ssa_ctx.bind_result(result.id, left)
                    return Op("const", result, [Const(0)], "load"), "identity"
                if left_is_const and left_val == 0:
                    ssa_ctx.bind_result(result.id, right)
                    return Op("const", result, [Const(0)], "load"), "identity"

            # x & 0 -> 0, 0 & x -> 0
            if opcode == "&":
                if (right_is_const and right_val == 0) or (left_is_const and left_val == 0):
                    return Op("const", result, [Const(0)], "load"), "identity"

            # x | 0 -> x, 0 | x -> x
            if opcode == "|":
                if right_is_const and right_val == 0:
                    ssa_ctx.bind_result(result.id, left)
                    return Op("const", result, [Const(0)], "load"), "identity"
                if left_is_const and left_val == 0:
                    ssa_ctx.bind_result(result.id, right)
                    return Op("const", result, [Const(0)], "load"), "identity"

        # Strength reductions (only if enabled)
        if self._opts.get("strength_reduction", True):
            # Strength reduction: % 2 -> & 1
            if opcode == "%" and right_is_const and right_val == 2:
                # The result of & 1 is boolean (0 or 1)
                self._boolean_values.add(result.id)
                return Op("&", result, [left, Const(1)], "alu"), "strength"

            # Strength reduction: * 2 -> << 1, * power_of_2 -> << log2(n)
            if opcode == "*":
                if right_is_const and right_val is not None and right_val > 0:
                    # Check if power of 2: n & (n-1) == 0 for powers of 2
                    if (right_val & (right_val - 1)) == 0:
                        shift_amount = right_val.bit_length() - 1
                        return Op("<<", result, [left, Const(shift_amount)], "alu"), "strength"

        return None, None

    def _try_simplify_select(self, op: Op, ssa_ctx: SSARenumberContext) -> Optional[Op]:
        """Try to simplify select operations."""
        cond, true_val, false_val = op.operands
        result = op.result

        true_const = self._get_const_value(true_val)
        false_const = self._get_const_value(false_val)

        # Parity pattern: select(is_zero, 1, 2) where is_zero = ==(lsb, 0)
        # -> lsb + 1 (since if lsb=0, we want 1; if lsb=1, we want 2)
        # Check parity pattern first as it's more specific
        if self._opts.get("parity_pattern", True):
            if true_const == 1 and false_const == 2:
                if isinstance(cond, SSAValue) and cond.id in self._negated_boolean:
                    # cond is ==(lsb, 0), so cond=1 when lsb=0, cond=0 when lsb=1
                    # select(cond, 1, 2) = 1 when lsb=0, 2 when lsb=1 = lsb + 1
                    lsb_id = self._negated_boolean[cond.id]
                    lsb = SSAValue(lsb_id)
                    self._parity_patterns += 1
                    return Op("+", result, [lsb, Const(1)], "alu")

        # select(cond, x, 0) -> *(x, cond) when cond is boolean (0/1)
        if self._opts.get("select_optimization", True):
            if false_const == 0 and self._is_boolean(cond):
                self._select_optimizations += 1
                return Op("*", result, [true_val, cond], "alu")

        # select(cond, 0, x) -> *(x, 1-cond) is more complex, skip for now

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
            results=loop.results,
            pragma_unroll=loop.pragma_unroll
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
                results=stmt.results,
                pragma_unroll=stmt.pragma_unroll
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
