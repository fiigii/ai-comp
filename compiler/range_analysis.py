"""
Range Analysis Module

Provides range analysis infrastructure for tracking value ranges of SSA values.
Used by level cache pass to determine when vgather can be replaced with preload + vselect.
"""

from dataclasses import dataclass
from typing import Optional

from .hir import SSAValue, VectorSSAValue, Variable, Const, Value, Op, ForLoop, If, Statement
from .use_def import UseDefContext


@dataclass(frozen=True)
class Range:
    """
    Represents a range of integer values [min_val, max_val].

    Used for range analysis to determine when idx has a small, predictable range
    that enables the level cache optimization.
    """
    min_val: int
    max_val: int

    @property
    def size(self) -> int:
        """Number of distinct values in the range."""
        return self.max_val - self.min_val + 1

    @staticmethod
    def point(val: int) -> "Range":
        """Create a point range containing a single value."""
        return Range(val, val)

    @staticmethod
    def unknown() -> "Range":
        """Create an unknown range (conservative default)."""
        return Range(0, 2**31 - 1)

    def union(self, other: "Range") -> "Range":
        """Return the smallest range containing both ranges."""
        return Range(min(self.min_val, other.min_val), max(self.max_val, other.max_val))

    def intersect(self, other: "Range") -> Optional["Range"]:
        """Return the intersection of two ranges, or None if disjoint."""
        new_min = max(self.min_val, other.min_val)
        new_max = min(self.max_val, other.max_val)
        if new_min <= new_max:
            return Range(new_min, new_max)
        return None

    def __add__(self, other: "Range") -> "Range":
        """Range of a + b."""
        return Range(self.min_val + other.min_val, self.max_val + other.max_val)

    def __sub__(self, other: "Range") -> "Range":
        """Range of a - b."""
        return Range(self.min_val - other.max_val, self.max_val - other.min_val)

    def __mul__(self, other: "Range") -> "Range":
        """Range of a * b (handles both positive and negative)."""
        products = [
            self.min_val * other.min_val,
            self.min_val * other.max_val,
            self.max_val * other.min_val,
            self.max_val * other.max_val,
        ]
        return Range(min(products), max(products))

    def shift_left(self, shift_amount: "Range") -> "Range":
        """Range of a << b."""
        # Conservative: if shift is not constant, return unknown
        if shift_amount.min_val != shift_amount.max_val:
            return Range.unknown()
        shift = shift_amount.min_val
        if shift < 0 or shift >= 32:
            return Range.unknown()
        return Range(self.min_val << shift, self.max_val << shift)

    def shift_right(self, shift_amount: "Range") -> "Range":
        """Range of a >> b (arithmetic shift)."""
        if shift_amount.min_val != shift_amount.max_val:
            return Range.unknown()
        shift = shift_amount.min_val
        if shift < 0 or shift >= 32:
            return Range.unknown()
        return Range(self.min_val >> shift, self.max_val >> shift)

    def bitwise_and(self, other: "Range") -> "Range":
        """Range of a & b."""
        # If masking with a constant, we can be more precise
        if other.min_val == other.max_val:
            mask = other.min_val
            if mask >= 0:
                # Result is in [0, mask] if self is non-negative
                if self.min_val >= 0:
                    return Range(0, min(self.max_val, mask))
                return Range(0, mask)
        # Conservative fallback
        if self.min_val >= 0 and other.min_val >= 0:
            return Range(0, min(self.max_val, other.max_val))
        return Range.unknown()

    def contains(self, val: int) -> bool:
        """Check if a value is in the range."""
        return self.min_val <= val <= self.max_val

    def __repr__(self) -> str:
        if self.min_val == self.max_val:
            return f"Range({self.min_val})"
        if self.max_val >= 2**30:
            return "Range(unknown)"
        return f"Range({self.min_val}, {self.max_val})"


class RangeAnalysis:
    """
    Tracks value ranges for SSA values.

    Uses use-def chains to compute ranges by walking the definition chain.
    Results are cached for efficiency.
    """

    # Maximum depth for recursive range computation
    MAX_DEPTH = 10

    def __init__(self, use_def: UseDefContext, max_depth: int = MAX_DEPTH):
        self._use_def = use_def
        self._cache: dict[Variable, Range] = {}
        self._max_depth = max_depth
        self._in_progress: set[Variable] = set()  # Track cycles

    def get_range(self, val: Value, depth: int = 0) -> Range:
        """
        Get range for a value (Const, SSAValue, etc.)

        Args:
            val: The value to analyze
            depth: Current recursion depth

        Returns:
            Range containing all possible values
        """
        # Early termination on depth limit
        if depth > self._max_depth:
            return Range.unknown()

        if isinstance(val, Const):
            return Range.point(val.value)
        elif isinstance(val, SSAValue):
            return self._compute_range(val, depth)
        elif isinstance(val, VectorSSAValue):
            # Vector values need per-lane analysis, return unknown for now
            return Range.unknown()
        else:
            return Range.unknown()

    def _compute_range(self, ssa: SSAValue, depth: int = 0) -> Range:
        """
        Walk def chain to compute range.

        Args:
            ssa: The SSA value to analyze
            depth: Current recursion depth

        Returns:
            Range containing all possible values of the SSA value
        """
        # Check cache first
        if ssa in self._cache:
            return self._cache[ssa]

        # Early termination on depth limit
        if depth > self._max_depth:
            return Range.unknown()

        # Cycle detection
        if ssa in self._in_progress:
            return Range.unknown()

        # Mark as in-progress
        self._in_progress.add(ssa)

        try:
            # Find the defining operation
            def_loc = self._use_def.get_def(ssa)
            if def_loc is None:
                # No definition found (external input) - return unknown
                result = Range.unknown()
            else:
                result = self._compute_range_from_def(ssa, def_loc, depth)

            self._cache[ssa] = result
            return result
        finally:
            self._in_progress.discard(ssa)

    def _compute_range_from_def(self, ssa: SSAValue, def_loc, depth: int) -> Range:
        """Compute range based on the defining statement."""
        stmt = def_loc.statement

        if isinstance(stmt, Op):
            return self._compute_range_from_op(stmt, depth + 1)
        elif isinstance(stmt, ForLoop):
            return self._compute_range_from_loop(ssa, stmt, def_loc.def_kind, depth + 1)
        elif isinstance(stmt, If):
            return self._compute_range_from_if(ssa, stmt, def_loc.def_kind, depth + 1)
        else:
            return Range.unknown()

    def _compute_range_from_op(self, op: Op, depth: int) -> Range:
        """Compute range from an Op definition."""
        opcode = op.opcode

        if opcode == "const":
            # const(value) -> point range
            if len(op.operands) == 1 and isinstance(op.operands[0], Const):
                return Range.point(op.operands[0].value)
            return Range.unknown()

        if opcode == "load":
            # Load from memory - unknown value
            return Range.unknown()

        if len(op.operands) >= 2:
            left_range = self.get_range(op.operands[0], depth)
            # Early termination: if left is unknown, many ops will be unknown
            if left_range.max_val >= 2**30:
                if opcode in ("^", "load"):
                    return Range.unknown()

            right_range = self.get_range(op.operands[1], depth)

            if opcode == "+":
                return left_range + right_range
            elif opcode == "-":
                return left_range - right_range
            elif opcode == "*":
                return left_range * right_range
            elif opcode == "<<":
                return left_range.shift_left(right_range)
            elif opcode == ">>":
                return left_range.shift_right(right_range)
            elif opcode == "&":
                return left_range.bitwise_and(right_range)
            elif opcode == "|":
                # Bitwise OR is harder to analyze precisely
                if left_range.min_val >= 0 and right_range.min_val >= 0:
                    return Range(0, max(left_range.max_val, right_range.max_val) * 2)
                return Range.unknown()
            elif opcode == "^":
                # XOR is hard to analyze precisely
                return Range.unknown()
            elif opcode == "//":
                # Integer division
                if right_range.min_val > 0:
                    return Range(
                        left_range.min_val // right_range.max_val,
                        left_range.max_val // right_range.min_val
                    )
                return Range.unknown()
            elif opcode == "%":
                # Modulo
                if right_range.min_val > 0:
                    return Range(0, right_range.max_val - 1)
                return Range.unknown()
            elif opcode == "<":
                # Comparison returns 0 or 1
                return Range(0, 1)
            elif opcode == "==":
                # Equality returns 0 or 1
                return Range(0, 1)

        if opcode == "select" and len(op.operands) == 3:
            # select(cond, true_val, false_val) -> union of true/false ranges
            true_range = self.get_range(op.operands[1], depth)
            false_range = self.get_range(op.operands[2], depth)
            return true_range.union(false_range)

        return Range.unknown()

    def _compute_range_from_loop(self, ssa: SSAValue, loop: ForLoop, def_kind: str, depth: int) -> Range:
        """Compute range from a ForLoop definition."""
        if def_kind == "counter":
            # Loop counter has range [start, end-1]
            start_range = self.get_range(loop.start, depth)
            end_range = self.get_range(loop.end, depth)
            if start_range.min_val == start_range.max_val and end_range.min_val == end_range.max_val:
                # Both are constant
                return Range(start_range.min_val, end_range.max_val - 1)
            return Range.unknown()

        elif def_kind == "body_param":
            # Body param - need to analyze the corresponding iter_arg and yield
            # This can be expensive, so limit depth more aggressively
            if depth > self._max_depth // 2:
                return Range.unknown()
            idx = None
            for i, param in enumerate(loop.body_params):
                if param == ssa:
                    idx = i
                    break
            if idx is not None:
                # Range is union of iter_arg and yield
                if idx < len(loop.iter_args):
                    iter_range = self.get_range(loop.iter_args[idx], depth)
                else:
                    iter_range = Range.unknown()
                if idx < len(loop.yields):
                    yield_range = self.get_range(loop.yields[idx], depth)
                else:
                    yield_range = Range.unknown()
                return iter_range.union(yield_range)
            return Range.unknown()

        elif def_kind == "loop_result":
            # Loop result - comes from the final yield
            idx = None
            for i, result in enumerate(loop.results):
                if result == ssa:
                    idx = i
                    break
            if idx is not None and idx < len(loop.yields):
                return self.get_range(loop.yields[idx], depth)
            return Range.unknown()

        return Range.unknown()

    def _compute_range_from_if(self, ssa: SSAValue, if_stmt: If, def_kind: str, depth: int) -> Range:
        """Compute range from an If definition."""
        if def_kind == "if_result":
            # Result is phi of then and else yields
            idx = None
            for i, result in enumerate(if_stmt.results):
                if result == ssa:
                    idx = i
                    break
            if idx is not None:
                then_range = Range.unknown()
                else_range = Range.unknown()
                if idx < len(if_stmt.then_yields):
                    then_range = self.get_range(if_stmt.then_yields[idx], depth)
                if idx < len(if_stmt.else_yields):
                    else_range = self.get_range(if_stmt.else_yields[idx], depth)
                return then_range.union(else_range)
            return Range.unknown()

        return Range.unknown()

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._cache.clear()
