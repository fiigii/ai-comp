"""
Simplify Pass

Performs constant folding and algebraic identity simplifications on HIR.
"""

from typing import Optional

from ..hir import (
    SSAValue, Const, Value, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..use_def import UseDefContext


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
    - Strength reduction: % 2 -> & 1, << n -> * 2^n
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
        # SSA values known to be boolean (0 or 1) - from comparisons or & 1
        self._boolean_values: set[SSAValue] = set()
        # Maps SSA value -> the SSA value it's negated from (for ==(x, 0) where x is boolean)
        # If _negated_boolean[a] = b, then a = (1 - b) = !b
        self._negated_boolean: dict[SSAValue, SSAValue] = {}
        # Feature options (set in run())
        self._opts: dict[str, bool] = {}
        # Use-def context for efficient replacements
        self._use_def_ctx: Optional[UseDefContext] = None

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

        # Create use-def context for efficient value replacement
        self._use_def_ctx = UseDefContext(hir)

        # Transform body
        new_body = self._transform_statements(hir.body)

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
            num_ssa_values=hir.num_ssa_values
        )

    def _transform_statements(self, stmts: list[Statement]) -> list[Statement]:
        """Transform a list of statements."""
        result = []

        for stmt in stmts:
            if isinstance(stmt, Op):
                transformed = self._transform_op(stmt)
                if transformed is not None:
                    result.append(transformed)
            elif isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt))
            else:
                # Halt, Pause - keep as is
                result.append(stmt)

        return result

    def _transform_op(self, op: Op) -> Op:
        """Apply simplifications to a single Op."""
        # Handle select (3 operands)
        if op.opcode == "select" and op.result is not None and len(op.operands) == 3:
            simplified = self._try_simplify_select(op)
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
                self._use_def_ctx.replace_all_uses(op.result, Const(folded))
                return None

        # Try algebraic identity simplifications (returns op, metric_type)
        simplified, metric_type = self._try_simplify_identity(op.opcode, left, right, op.result)
        if metric_type is not None:
            # Increment appropriate counter
            if metric_type == "identity":
                self._identities_simplified += 1
            elif metric_type == "strength":
                self._strength_reductions += 1
            if simplified is None:
                return None
            # Track boolean status if the simplified op produces a boolean
            if simplified.opcode in ("<", "=="):
                self._boolean_values.add(op.result)
            elif simplified.opcode == "&" and len(simplified.operands) == 2:
                # Check if this is & 1
                r_val = self._get_const_value(simplified.operands[1])
                if r_val == 1:
                    self._boolean_values.add(op.result)
            return simplified

        # Track boolean values from comparisons
        if op.opcode in ("<", "=="):
            self._boolean_values.add(op.result)
            # Track negated booleans: ==(x, 0) where x is boolean -> result is !x
            if op.opcode == "==":
                right_val = self._get_const_value(right)
                if right_val == 0 and self._is_boolean(left):
                    if isinstance(left, SSAValue):
                        self._negated_boolean[op.result] = left

        # Track & 1 as producing boolean
        if op.opcode == "&":
            right_val = self._get_const_value(right)
            if right_val == 1:
                self._boolean_values.add(op.result)

        return op

    def _get_const_value(self, operand: Value) -> Optional[int]:
        """Get constant value if operand is a known constant."""
        if isinstance(operand, Const):
            return operand.value
        return None

    def _is_boolean(self, operand: Value) -> bool:
        """Check if operand is known to be boolean (0 or 1)."""
        if isinstance(operand, Const):
            return operand.value in (0, 1)
        if isinstance(operand, SSAValue):
            return operand in self._boolean_values
        return False

    def _try_constant_fold(self, opcode: str, left: Value, right: Value) -> Optional[int]:
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
        left: Value,
        right: Value,
        result: SSAValue
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
                    self._use_def_ctx.replace_all_uses(result, left)
                    return None, "identity"
                if left_is_const and left_val == 0:
                    self._use_def_ctx.replace_all_uses(result, right)
                    return None, "identity"

            # x - 0 -> x
            if opcode == "-":
                if right_is_const and right_val == 0:
                    self._use_def_ctx.replace_all_uses(result, left)
                    return None, "identity"

            # x * 1 -> x, 1 * x -> x
            if opcode == "*":
                if right_is_const and right_val == 1:
                    self._use_def_ctx.replace_all_uses(result, left)
                    return None, "identity"
                if left_is_const and left_val == 1:
                    self._use_def_ctx.replace_all_uses(result, right)
                    return None, "identity"
                # x * 0 -> 0, 0 * x -> 0
                if right_is_const and right_val == 0:
                    self._use_def_ctx.replace_all_uses(result, Const(0))
                    return None, "identity"
                if left_is_const and left_val == 0:
                    self._use_def_ctx.replace_all_uses(result, Const(0))
                    return None, "identity"

            # x ^ 0 -> x, 0 ^ x -> x
            if opcode == "^":
                if right_is_const and right_val == 0:
                    self._use_def_ctx.replace_all_uses(result, left)
                    return None, "identity"
                if left_is_const and left_val == 0:
                    self._use_def_ctx.replace_all_uses(result, right)
                    return None, "identity"

            # x & 0 -> 0, 0 & x -> 0
            if opcode == "&":
                if (right_is_const and right_val == 0) or (left_is_const and left_val == 0):
                    self._use_def_ctx.replace_all_uses(result, Const(0))
                    return None, "identity"

            # x | 0 -> x, 0 | x -> x
            if opcode == "|":
                if right_is_const and right_val == 0:
                    self._use_def_ctx.replace_all_uses(result, left)
                    return None, "identity"
                if left_is_const and left_val == 0:
                    self._use_def_ctx.replace_all_uses(result, right)
                    return None, "identity"

        # Strength reductions (only if enabled)
        if self._opts.get("strength_reduction", True):
            # Strength reduction: % 2 -> & 1
            if opcode == "%" and right_is_const and right_val == 2:
                # The result of & 1 is boolean (0 or 1)
                self._boolean_values.add(result)
                return Op("&", result, [left, Const(1)], "alu"), "strength"

            # Strength reduction: << n -> * 2^n (multiplication can be faster on VLIW due to more ALU slots)
            if opcode == "<<":
                if right_is_const and right_val is not None and right_val >= 0 and right_val < 32:
                    mul_val = 1 << right_val
                    return Op("*", result, [left, Const(mul_val)], "alu"), "strength"

        return None, None

    def _try_simplify_select(self, op: Op) -> Optional[Op]:
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
                if isinstance(cond, SSAValue) and cond in self._negated_boolean:
                    # cond is ==(lsb, 0), so cond=1 when lsb=0, cond=0 when lsb=1
                    # select(cond, 1, 2) = 1 when lsb=0, 2 when lsb=1 = lsb + 1
                    lsb = self._negated_boolean[cond]
                    self._parity_patterns += 1
                    return Op("+", result, [lsb, Const(1)], "alu")

        # select(cond, x, 0) -> *(x, cond) when cond is boolean (0/1)
        if self._opts.get("select_optimization", True):
            if false_const == 0 and self._is_boolean(cond):
                self._select_optimizations += 1
                return Op("*", result, [true_val, cond], "alu")

        # select(cond, 0, x) -> *(x, 1-cond) is more complex, skip for now

        return None

    def _transform_for_loop(self, loop: ForLoop) -> ForLoop:
        """Transform a ForLoop."""
        new_body = self._transform_statements(loop.body)

        return ForLoop(
            counter=loop.counter,
            start=loop.start,
            end=loop.end,
            iter_args=loop.iter_args,
            body_params=loop.body_params,
            body=new_body,
            yields=loop.yields,
            results=loop.results,
            pragma_unroll=loop.pragma_unroll
        )

    def _transform_if(self, if_stmt: If) -> If:
        """Transform an If statement."""
        new_then_body = self._transform_statements(if_stmt.then_body)
        new_else_body = self._transform_statements(if_stmt.else_body)

        return If(
            cond=if_stmt.cond,
            then_body=new_then_body,
            then_yields=if_stmt.then_yields,
            else_body=new_else_body,
            else_yields=if_stmt.else_yields,
            results=if_stmt.results
        )
