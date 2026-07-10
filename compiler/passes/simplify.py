"""
Simplify Pass

Performs constant folding and algebraic identity simplifications on HIR.
"""

from typing import Optional

from ..hir import (
    SSAValue, Const, Value, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..range_analysis import RangeAnalysis
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
    - Canonicalization: % 2 -> & 1, << n -> * 2^n
    - Select-to-mul: select(cond, x, 0) -> *(x, cond) when cond is boolean
    - Select-to-ALU mux: select(cond, a, b) -> b + cond * (a - b) when cond is boolean
    - Parity pattern: ==(x & 1, 0) followed by select(cond, 1, 2) -> (x & 1) + 1
    - Range fold: structured interval analysis (loops and ifs included); ops
      whose result range is a single point become Const (folds provable wrap
      checks like lt(x, C), inside retained loops too)
    """

    def __init__(self):
        super().__init__()
        self._constants_folded = 0
        self._identities_simplified = 0
        self._canonicalizations = 0
        self._select_to_muls = 0
        self._select_to_alu_mux = 0
        self._parity_patterns = 0
        self._add_mul_folds = 0
        self._ranges_folded = 0
        self._selects_const_folded = 0
        self._assoc_folds = 0
        self._mul_dists = 0
        # Value ranges computed by RangeAnalysis (range_fold option)
        self._range: Optional[RangeAnalysis] = None
        # Defs emitted so far in this run (fresher than the use-def context,
        # which is built once at run start). Any def, stale or fresh, is
        # semantically valid to pattern-match against because rewrites
        # preserve value; the fresh map just exposes more patterns.
        self._local_defs: dict[SSAValue, Op] = {}
        # Per-run adjustments to use counts for uses removed by rewrites
        self._use_adjust: dict[SSAValue, int] = {}
        # SSA values known to be boolean (0 or 1) - from comparisons or & 1
        self._boolean_values: set[SSAValue] = set()
        # Maps SSA value -> the SSA value it's negated from (for ==(x, 0) where x is boolean)
        # If _negated_boolean[a] = b, then a = (1 - b) = !b
        self._negated_boolean: dict[SSAValue, SSAValue] = {}
        # Feature options (set in run())
        self._opts: dict[str, bool] = {}
        # Use-def context for efficient replacements
        self._use_def_ctx: Optional[UseDefContext] = None
        self._next_ssa_id = 0

    @property
    def name(self) -> str:
        return "simplify"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        # Initialize metrics
        self._init_metrics()
        self._constants_folded = 0
        self._identities_simplified = 0
        self._canonicalizations = 0
        self._select_to_muls = 0
        self._select_to_alu_mux = 0
        self._parity_patterns = 0
        self._add_mul_folds = 0
        self._ranges_folded = 0
        self._selects_const_folded = 0
        self._assoc_folds = 0
        self._mul_dists = 0
        self._range = None
        self._local_defs = {}
        self._use_adjust = {}
        self._boolean_values = set()
        self._negated_boolean = {}

        # Read feature options from config (all enabled by default)
        self._opts = {
            "constant_folding": config.options.get("constant_folding", True),
            "identities": config.options.get("identities", True),
            "canonicalization": config.options.get("canonicalization", True),
            "select_to_mul": config.options.get("select_to_mul", True),
            "select_to_alu_mux": config.options.get("select_to_alu_mux", False),
            "parity_pattern": config.options.get("parity_pattern", True),
            "add_mul_fold": config.options.get("add_mul_fold", True),
            "range_fold": config.options.get("range_fold", False),
            "assoc_fold": config.options.get("assoc_fold", False),
            "mul_dist": config.options.get("mul_dist", False),
        }

        # Check if pass is enabled
        if not config.enabled:
            return hir

        # Create use-def context for efficient value replacement
        self._use_def_ctx = UseDefContext(hir)
        self._next_ssa_id = hir.num_ssa_values

        # Structured interval analysis: control flow is handled inside
        # RangeAnalysis via joins, loop fixpoints, and widening
        if self._opts["range_fold"]:
            self._range = RangeAnalysis(hir)

        # Transform body
        new_body = self._transform_statements(hir.body)

        # Record custom metrics
        if self._metrics:
            self._metrics.custom = {
                "constants_folded": self._constants_folded,
                "identities_simplified": self._identities_simplified,
                "canonicalizations": self._canonicalizations,
                "select_to_muls": self._select_to_muls,
                "select_to_alu_mux": self._select_to_alu_mux,
                "parity_patterns": self._parity_patterns,
                "add_mul_folds": self._add_mul_folds,
                "ranges_folded": self._ranges_folded,
                "selects_const_folded": self._selects_const_folded,
                "assoc_folds": self._assoc_folds,
                "mul_dists": self._mul_dists,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=max(hir.num_ssa_values, self._next_ssa_id),
            num_vec_ssa_values=hir.num_vec_ssa_values,
        )

    def _transform_statements(self, stmts: list[Statement]) -> list[Statement]:
        """Transform a list of statements."""
        result = []

        for stmt in stmts:
            if isinstance(stmt, Op):
                transformed = self._transform_op(stmt)
                if transformed is None:
                    continue
                if isinstance(transformed, list):
                    result.extend(transformed)
                    for t in transformed:
                        if isinstance(t, Op) and t.result is not None:
                            self._local_defs[t.result] = t
                else:
                    result.append(transformed)
                    if transformed.result is not None:
                        self._local_defs[transformed.result] = transformed
            elif isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt))
            else:
                # Halt, Pause - keep as is
                result.append(stmt)

        return result

    def _transform_op(self, op: Op) -> Op | list[Op] | None:
        """Apply simplifications to a single Op."""
        # Range fold first: it applies to every op with a result (selects
        # included, e.g. select(c, 7, 7)), so it must run before the
        # select-specific early return below.
        if self._range is not None and op.result is not None and op.opcode != "load":
            point = self._range.try_const(op.result)
            if point is not None:
                self._ranges_folded += 1
                self._use_def_ctx.replace_all_uses(
                    op.result, Const(point), auto_invalidate=False
                )
                return None

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
                self._use_def_ctx.replace_all_uses(op.result, Const(folded), auto_invalidate=False)
                return None

        # Associative constant chain fold: op(op(x, C1), C2) -> op(x, C1?C2)
        if self._opts.get("assoc_fold", False):
            folded_op = self._try_assoc_fold(op)
            if folded_op is not None:
                self._assoc_folds += 1
                return folded_op

        # Multiply distribution over add-const: (x + C1) * K -> x*K + C1*K.
        # Also matches << by constant (shift = multiply by power of two).
        # This exposes multiply_add fusion across hash stages and shortens
        # the dependency chain.
        if self._opts.get("mul_dist", False):
            dist_ops = self._try_mul_dist(op)
            if dist_ops is not None:
                self._mul_dists += 1
                return dist_ops

        # Try algebraic identity simplifications (returns op, metric_type)
        simplified, metric_type = self._try_simplify_identity(op.opcode, left, right, op.result)
        if metric_type is not None:
            # Increment appropriate counter
            if metric_type == "identity":
                self._identities_simplified += 1
            elif metric_type == "canonicalization":
                self._canonicalizations += 1
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

        # Try algebraic add-mul fold: (a + C) + (a * K) -> a * (K+1) + C
        if op.opcode == "+" and self._opts.get("add_mul_fold", True):
            folded = self._try_fold_add_mul(op)
            if folded is not None:
                self._add_mul_folds += 1
                return folded

        return op

    def _get_const_value(self, operand: Value) -> Optional[int]:
        """Get constant value if operand is a known constant."""
        if isinstance(operand, Const):
            # The VM reduces every value mod 2**32 (const immediates
            # included), so folding must read constants the same way.
            return operand.value & 0xFFFFFFFF
        return None

    def _is_boolean(self, operand: Value) -> bool:
        """Check if operand is known to be boolean (0 or 1)."""
        if isinstance(operand, Const):
            return (operand.value & 0xFFFFFFFF) in (0, 1)
        if isinstance(operand, SSAValue):
            if operand in self._boolean_values:
                return True
            # Interval analysis proves booleans the syntactic tracker
            # misses (e.g. x >> 31, If results, loop-carried bits).
            return self._range is not None and self._range.is_boolean(operand)
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
        metric_type is "identity" for algebraic identities, "canonicalization" for canonicalizations.
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
                    self._use_def_ctx.replace_all_uses(result, left, auto_invalidate=False)
                    return None, "identity"
                if left_is_const and left_val == 0:
                    self._use_def_ctx.replace_all_uses(result, right, auto_invalidate=False)
                    return None, "identity"

            # x - 0 -> x
            if opcode == "-":
                if right_is_const and right_val == 0:
                    self._use_def_ctx.replace_all_uses(result, left, auto_invalidate=False)
                    return None, "identity"

            # x * 1 -> x, 1 * x -> x
            if opcode == "*":
                if right_is_const and right_val == 1:
                    self._use_def_ctx.replace_all_uses(result, left, auto_invalidate=False)
                    return None, "identity"
                if left_is_const and left_val == 1:
                    self._use_def_ctx.replace_all_uses(result, right, auto_invalidate=False)
                    return None, "identity"
                # x * 0 -> 0, 0 * x -> 0
                if right_is_const and right_val == 0:
                    self._use_def_ctx.replace_all_uses(result, Const(0), auto_invalidate=False)
                    return None, "identity"
                if left_is_const and left_val == 0:
                    self._use_def_ctx.replace_all_uses(result, Const(0), auto_invalidate=False)
                    return None, "identity"

            # x ^ 0 -> x, 0 ^ x -> x
            if opcode == "^":
                if right_is_const and right_val == 0:
                    self._use_def_ctx.replace_all_uses(result, left, auto_invalidate=False)
                    return None, "identity"
                if left_is_const and left_val == 0:
                    self._use_def_ctx.replace_all_uses(result, right, auto_invalidate=False)
                    return None, "identity"

            # x & 0 -> 0, 0 & x -> 0
            if opcode == "&":
                if (right_is_const and right_val == 0) or (left_is_const and left_val == 0):
                    self._use_def_ctx.replace_all_uses(result, Const(0), auto_invalidate=False)
                    return None, "identity"

            # x | 0 -> x, 0 | x -> x
            if opcode == "|":
                if right_is_const and right_val == 0:
                    self._use_def_ctx.replace_all_uses(result, left, auto_invalidate=False)
                    return None, "identity"
                if left_is_const and left_val == 0:
                    self._use_def_ctx.replace_all_uses(result, right, auto_invalidate=False)
                    return None, "identity"

        # Canonicalizations (only if enabled)
        if self._opts.get("canonicalization", True):
            # Canonicalization: % 2 -> & 1
            if opcode == "%" and right_is_const and right_val == 2:
                # The result of & 1 is boolean (0 or 1)
                self._boolean_values.add(result)
                return Op("&", result, [left, Const(1)], "alu"), "canonicalization"

            # Canonicalization: << n -> * 2^n (multiplication can be faster on VLIW due to more ALU slots)
            if opcode == "<<":
                if right_is_const and right_val is not None and right_val >= 0 and right_val < 32:
                    mul_val = 1 << right_val
                    return Op("*", result, [left, Const(mul_val)], "alu"), "canonicalization"

        return None, None

    _ASSOC_COMBINE = {
        "+": lambda a, b: (a + b) & 0xFFFFFFFF,
        "*": lambda a, b: (a * b) & 0xFFFFFFFF,
        "^": lambda a, b: a ^ b,
        "&": lambda a, b: a & b,
        "|": lambda a, b: a | b,
    }

    def _lookup_def(self, ssa: Value) -> Optional[Op]:
        """Find the defining op of an SSA value.

        Prefers defs (re)emitted earlier in this run; falls back to the
        use-def context built at run start. Stale defs are semantically
        equivalent (rewrites preserve value), so either source is sound.
        """
        if not isinstance(ssa, SSAValue):
            return None
        local = self._local_defs.get(ssa)
        if local is not None:
            return local
        def_loc = self._use_def_ctx.get_def(ssa)
        if def_loc is not None and isinstance(def_loc.statement, Op):
            return def_loc.statement
        return None

    def _effective_use_count(self, ssa: SSAValue) -> int:
        """Use count adjusted for uses removed by rewrites in this run.

        The use-def context is built once at run start; rewrites like
        assoc_fold rewire a user away from a value without updating it.
        """
        return self._use_def_ctx.use_count(ssa) + self._use_adjust.get(ssa, 0)

    def _note_use_removed(self, ssa: Value) -> None:
        if isinstance(ssa, SSAValue):
            self._use_adjust[ssa] = self._use_adjust.get(ssa, 0) - 1

    def _try_assoc_fold(self, op: Op) -> Optional[Op]:
        """op(op(x, C1), C2) -> op(x, C1 combined C2) for associative ops."""
        combine = self._ASSOC_COMBINE.get(op.opcode)
        if combine is None:
            return None
        var, c2 = self._extract_var_const(op)
        if var is None:
            return None
        inner = self._lookup_def(var)
        if inner is None or inner.opcode != op.opcode:
            return None
        x, c1 = self._extract_var_const(inner)
        if x is None:
            return None
        self._note_use_removed(var)
        return Op(op.opcode, op.result, [x, Const(combine(c1, c2))], op.engine)

    @staticmethod
    def _shift_var_const(op: Op) -> tuple[Optional[SSAValue], Optional[int]]:
        """Split a '<<' op into (value, shift amount). Shifts are NOT
        commutative: the constant must be the right operand."""
        if len(op.operands) != 2:
            return None, None
        val, amt = op.operands
        if (isinstance(val, SSAValue) and isinstance(amt, Const)
                and 0 <= (amt.value & 0xFFFFFFFF) < 32):
            return val, amt.value & 0xFFFFFFFF
        return None, None

    def _try_mul_dist(self, op: Op) -> Optional[list[Op]]:
        """(x + C1) * K -> x' * K' + (C1*K), looking through x = a * K2.

        Matches outer '*' by Const or '<<' by Const (as multiply by 2**s).
        Enables cross-stage multiply_add fusion, e.g. hash stages 2+3:
        ((a*33 + C2) << 9) becomes a*16896 + (C2<<9).

        Only fires when the inner add has a single (remaining) use, so the
        rewrite never duplicates the computation.
        """
        if op.opcode == "*":
            v, k = self._extract_var_const(op)
        elif op.opcode == "<<":
            v, s = self._shift_var_const(op)
            k = None if s is None else (1 << s)
        else:
            return None
        if v is None or k is None:
            return None
        inner = self._lookup_def(v)
        if inner is None or inner.opcode != "+":
            return None
        if self._effective_use_count(v) > 1:
            return None
        x, c1 = self._extract_var_const(inner)
        if x is None:
            return None

        # Look through x = a * K2 to fold the two multiplies directly
        mul_src, mul_k = x, k
        x_def = self._lookup_def(x)
        if x_def is not None and x_def.opcode == "*":
            a, k2 = self._extract_var_const(x_def)
            if a is not None and k2 is not None:
                mul_src = a
                mul_k = (k2 * k) & 0xFFFFFFFF
        elif x_def is not None and x_def.opcode == "<<":
            a, s2 = self._shift_var_const(x_def)
            if a is not None and s2 is not None:
                mul_src = a
                mul_k = ((1 << s2) * k) & 0xFFFFFFFF

        self._note_use_removed(v)
        temp = self._new_temp("mdist")
        return [
            Op("*", temp, [mul_src, Const(mul_k)], "alu"),
            Op("+", op.result, [temp, Const((c1 * k) & 0xFFFFFFFF)], "alu"),
        ]

    def _try_fold_add_mul(self, op: Op) -> Optional[list[Op]]:
        """Try to fold (a + C) + (a * K) -> a * (K+1) + C.

        This pattern arises after canonicalization converts << to *,
        producing val + C + val * K which can be a single multiply-add.
        """
        assert op.opcode == "+" and op.result is not None
        left, right = op.operands

        # Both operands must be SSA values with known definitions
        if not isinstance(left, SSAValue) or not isinstance(right, SSAValue):
            return None

        left_stmt = self._lookup_def(left)
        right_stmt = self._lookup_def(right)

        if left_stmt is None or right_stmt is None:
            return None

        # Try both orderings: (add_side, mul_side)
        for add_op, add_val, mul_op, mul_val in [
            (left_stmt, left, right_stmt, right),
            (right_stmt, right, left_stmt, left),
        ]:
            if add_op.opcode != "+" or mul_op.opcode != "*":
                continue
            if len(add_op.operands) != 2 or len(mul_op.operands) != 2:
                continue

            # Extract: add_op = a + C (or C + a), mul_op = a * K (or K * a)
            a_add, c_val = self._extract_var_const(add_op)
            a_mul, k_val = self._extract_var_const(mul_op)

            if a_add is None or a_mul is None:
                continue
            if a_add != a_mul:
                continue

            # Check that the mul intermediate has only one use (this combining ADD)
            if self._use_def_ctx.use_count(mul_val) != 1:
                continue

            # Fold: result = a * (K+1) + C
            new_k = ((k_val + 1) % (1 << 32))
            c_const = c_val % (1 << 32)
            temp = self._new_temp("amf")
            return [
                Op("*", temp, [a_add, Const(new_k)], "alu"),
                Op("+", op.result, [temp, Const(c_const)], "alu"),
            ]

        return None

    def _extract_var_const(self, op: Op) -> tuple[Optional[SSAValue], Optional[int]]:
        """Extract (variable, constant) from a binary op like a + C or C + a.

        Returns (None, None) if not in the expected form.
        """
        if len(op.operands) != 2:
            return None, None

        left, right = op.operands
        left_val = self._get_const_value(left)
        right_val = self._get_const_value(right)

        if right_val is not None and isinstance(left, SSAValue):
            return left, right_val
        if left_val is not None and isinstance(right, SSAValue):
            return right, left_val
        return None, None

    def _try_simplify_select(self, op: Op) -> Optional[Op | list[Op]]:
        """Try to simplify select operations."""
        cond, true_val, false_val = op.operands
        result = op.result

        # select(Const c, a, b) -> a if c != 0 else b
        if self._opts.get("constant_folding", True):
            cond_const = self._get_const_value(cond)
            if cond_const is not None:
                chosen = true_val if cond_const != 0 else false_val
                self._selects_const_folded += 1
                self._use_def_ctx.replace_all_uses(result, chosen, auto_invalidate=False)
                return []

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
        if self._opts.get("select_to_mul", True):
            if false_const == 0 and self._is_boolean(cond):
                self._select_to_muls += 1
                return Op("*", result, [true_val, cond], "alu")

        # Generic boolean select to ALU mux:
        # select(cond, a, b) = b + cond * (a - b), where cond is 0/1.
        if self._opts.get("select_to_alu_mux", False) and self._is_boolean(cond):
            self._select_to_alu_mux += 1
            delta = self._new_temp("sel_delta")
            scaled = self._new_temp("sel_scaled")
            return [
                Op("-", delta, [true_val, false_val], "alu"),
                Op("*", scaled, [cond, delta], "alu"),
                Op("+", result, [false_val, scaled], "alu"),
            ]

        # select(cond, 0, x) -> *(x, 1-cond) is more complex, skip for now

        return None

    def _new_temp(self, name: Optional[str] = None) -> SSAValue:
        ssa = SSAValue(self._next_ssa_id, name)
        self._next_ssa_id += 1
        return ssa

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
