"""Unsigned 32-bit value-range analysis over structured HIR.

Computes a sound interval (lo, hi) for every scalar SSA value. Straight-line
code is a single forward pass; structured control flow is handled by
structural induction:

- If: both branches are analyzed under branch refinements derived from the
  condition (e.g. inside the then-branch of ``x < C``, x is narrowed to
  [lo, C-1]); the If results join the two sides' yields. Values defined
  inside a branch keep their refined ranges globally, which is sound because
  SSA dominance confines their uses to that branch.
- ForLoop: loop-carried values (iter_args -> body_params -> yields) are
  iterated to a fixpoint. The interval lattice is too tall for naive
  iteration, so after a few exact rounds unstable bounds are widened along
  a 2**k - 1 threshold ladder, which preserves the mask/wrap-shaped bounds
  that matter on this VM while guaranteeing termination. The loop counter
  ranges over [lo(start), hi(end) - 1] (the lowered loop is top-tested);
  loop results account for the zero-trip case unless start < end is
  provable.

Any operation that may wrap modulo 2**32 widens to the full range, which
keeps the intervals sound for the VM's arithmetic.

Mirrors the AliasAnalysis conventions: constructed by a pass over the
current HIR, queried through a small API, never reused across passes.
"""

from __future__ import annotations

from typing import Optional

from .hir import (
    Const,
    ForLoop,
    HIRFunction,
    If,
    Op,
    SSAValue,
    Statement,
    Value,
    Variable,
)

WORD_MASK = 0xFFFFFFFF
FULL_RANGE = (0, WORD_MASK)

_Range = tuple  # (lo, hi), inclusive, always non-empty

# Exact join rounds before widening kicks in: the common wrap-select
# recurrence (idx' = select(2*idx + d < 2**k - 1, ..., 0)) stabilizes in two.
_EXACT_ITERATIONS = 2
# Widening ladder: 1, 3, 7, ..., 2**32 - 1. Bounded climb per carried value.
_WIDEN_THRESHOLDS = tuple((1 << k) - 1 for k in range(1, 33))
# Hard stop far above the ladder bound (~33 widening rounds per value).
_MAX_ITERATIONS = 200
# Descending rounds recovering precision lost to widening overshoot.
_NARROW_ITERATIONS = 2
# Statement-visit budget. Nested dynamic loops re-solve inner fixpoints per
# outer round, which is multiplicative in depth; past this budget the
# analysis degrades to all-FULL (sound) instead of stalling compilation.
_ANALYSIS_BUDGET = 2_000_000


class _BudgetExhausted(Exception):
    pass


def _hull(a: _Range, b: _Range) -> _Range:
    return (min(a[0], b[0]), max(a[1], b[1]))


def _contains(outer: _Range, inner: _Range) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _widen(old: _Range, new: _Range) -> _Range:
    lo = new[0] if new[0] >= old[0] else 0
    hi = new[1]
    if hi > old[1]:
        for threshold in _WIDEN_THRESHOLDS:
            if threshold >= hi:
                hi = threshold
                break
    return (lo, hi)


class RangeAnalysis:
    """Interval analysis over a structured HIR body."""

    def __init__(self, hir: HIRFunction):
        # Only non-full entries are stored; absent means FULL_RANGE.
        self._ranges: dict[SSAValue, _Range] = {}
        self._defs: dict[SSAValue, Op] = {}
        # Warm-start seeds: id(loop) -> the loop's last fixpoint params.
        # Never trusted as-is; every solve re-verifies them (see _visit_for).
        self._loop_memo: dict[int, tuple] = {}
        self.queries = 0
        self.budget_exhausted = False
        self._visits = 0
        try:
            self._walk(hir.body, {})
        except _BudgetExhausted:
            # Degrade soundly: every query answers FULL_RANGE.
            self._ranges = {}
            self._defs = {}
            self.budget_exhausted = True

    # === Queries ===

    def range_of(self, value: Value) -> _Range:
        """Sound unsigned 32-bit interval for value at its program point."""
        self.queries += 1
        if isinstance(value, Const):
            c = value.value & WORD_MASK
            return (c, c)
        if isinstance(value, SSAValue):
            return self._ranges.get(value, FULL_RANGE)
        return FULL_RANGE

    def try_const(self, value: Value) -> Optional[int]:
        """The single point of a degenerate interval, or None."""
        lo, hi = self.range_of(value)
        return lo if lo == hi else None

    def try_compare_lt(self, left: Value, right: Value) -> Optional[bool]:
        """Provable result of unsigned left < right, or None."""
        l1, h1 = self.range_of(left)
        l2, h2 = self.range_of(right)
        if h1 < l2:
            return True
        if l1 >= h2:
            return False
        return None

    def try_compare_eq(self, left: Value, right: Value) -> Optional[bool]:
        """Provable result of left == right, or None."""
        l1, h1 = self.range_of(left)
        l2, h2 = self.range_of(right)
        if l1 == h1 == l2 == h2:
            return True
        if h1 < l2 or h2 < l1:
            return False
        return None

    def is_boolean(self, value: Value) -> bool:
        """Whether value provably lies in {0, 1}."""
        _, hi = self.range_of(value)
        return hi <= 1

    # === Analysis ===

    def _rng(self, value: Value, overlay: dict) -> _Range:
        if isinstance(value, Const):
            c = value.value & WORD_MASK
            return (c, c)
        if isinstance(value, SSAValue):
            refined = overlay.get(value)
            if refined is not None:
                return refined
            return self._ranges.get(value, FULL_RANGE)
        return FULL_RANGE

    def _set(self, value: Variable, r: _Range) -> None:
        # Iterative re-walks may grow an entry back to full; drop it then so
        # absence stays the canonical encoding of FULL_RANGE.
        if r == FULL_RANGE:
            self._ranges.pop(value, None)
        else:
            self._ranges[value] = r

    def _walk(self, body: list[Statement], overlay: dict) -> None:
        self._visits += len(body)
        if self._visits > _ANALYSIS_BUDGET:
            raise _BudgetExhausted()
        for stmt in body:
            if isinstance(stmt, Op):
                if stmt.result is not None:
                    if isinstance(stmt.result, SSAValue):
                        self._defs[stmt.result] = stmt
                    self._set(stmt.result, self._transfer(stmt, overlay))
            elif isinstance(stmt, ForLoop):
                self._visit_for(stmt, overlay)
            elif isinstance(stmt, If):
                self._visit_if(stmt, overlay)

    def _visit_if(self, stmt: If, overlay: dict) -> None:
        cond_range = self._rng(stmt.cond, overlay)
        then_overlay = dict(overlay)
        else_overlay = dict(overlay)
        self._refine(stmt.cond, then_overlay, else_overlay, overlay)

        self._walk(stmt.then_body, then_overlay)
        then_yields = [self._rng(y, then_overlay) for y in stmt.then_yields]
        self._walk(stmt.else_body, else_overlay)
        else_yields = [self._rng(y, else_overlay) for y in stmt.else_yields]

        for i, result in enumerate(stmt.results):
            if cond_range == (0, 0):
                joined = else_yields[i]
            elif cond_range[0] >= 1:
                joined = then_yields[i]
            else:
                joined = _hull(then_yields[i], else_yields[i])
            if isinstance(result, SSAValue):
                self._set(result, joined)

    def _refine(
        self,
        cond: Value,
        then_overlay: dict,
        else_overlay: dict,
        overlay: dict,
    ) -> None:
        """Narrow the compared values inside each branch.

        Refinements are optional precision: a refinement that would produce
        an empty interval (a statically dead branch) is skipped instead of
        modeled, keeping every stored interval non-empty.
        """
        if not isinstance(cond, SSAValue):
            return
        definition = self._defs.get(cond)
        if definition is None or len(definition.operands) != 2:
            return
        left, right = definition.operands
        left_range = self._rng(left, overlay)
        right_range = self._rng(right, overlay)

        def narrow(target: dict, value: Value, lo: int, hi: int) -> None:
            if isinstance(value, SSAValue) and lo <= hi:
                target[value] = (lo, hi)

        if definition.opcode == "<":
            # then: left < right
            narrow(then_overlay, left,
                   left_range[0], min(left_range[1], right_range[1] - 1))
            narrow(then_overlay, right,
                   max(right_range[0], left_range[0] + 1), right_range[1])
            # else: left >= right
            narrow(else_overlay, left,
                   max(left_range[0], right_range[0]), left_range[1])
            narrow(else_overlay, right,
                   right_range[0], min(right_range[1], left_range[1]))
        elif definition.opcode == "==":
            intersect_lo = max(left_range[0], right_range[0])
            intersect_hi = min(left_range[1], right_range[1])
            narrow(then_overlay, left, intersect_lo, intersect_hi)
            narrow(then_overlay, right, intersect_lo, intersect_hi)
            # else: trim a point endpoint off the other side
            for point, other, other_range in (
                (left_range, right, right_range),
                (right_range, left, left_range),
            ):
                if point[0] != point[1]:
                    continue
                p = point[0]
                if p == other_range[0]:
                    narrow(else_overlay, other, p + 1, other_range[1])
                elif p == other_range[1]:
                    narrow(else_overlay, other, other_range[0], p - 1)

    def _visit_for(self, loop: ForLoop, overlay: dict) -> None:
        start_range = self._rng(loop.start, overlay)
        end_range = self._rng(loop.end, overlay)
        init_ranges = [self._rng(arg, overlay) for arg in loop.iter_args]

        # The lowered loop is top-tested (while counter < end), so it may
        # run zero times; results then carry the iter_args through.
        may_run = start_range[0] < end_range[1]
        must_run = start_range[1] < end_range[0]
        if not may_run:
            for result, r in zip(loop.results, init_ranges):
                if isinstance(result, SSAValue):
                    self._set(result, r)
            return

        if isinstance(loop.counter, SSAValue):
            self._set(loop.counter, (start_range[0], end_range[1] - 1))

        def walk_with(params: list) -> list:
            for param, r in zip(loop.body_params, params):
                if isinstance(param, SSAValue):
                    self._set(param, r)
            self._walk(loop.body, overlay)
            return [self._rng(y, overlay) for y in loop.yields]

        # Warm start: an enclosing loop's fixpoint re-solves this loop once
        # per outer round. Seeding from the previous fixpoint makes the
        # first ascending walk a VERIFICATION: if it is still a
        # post-fixpoint under the current environment (captured outer SSAs
        # included -- nothing is assumed cacheable), the solve costs one
        # walk; otherwise the ascent simply continues from there. The
        # descending phase below re-narrows when the seed was too wide.
        params = list(init_ranges)
        memo = self._loop_memo.get(id(loop))
        if memo is not None and len(memo) == len(params):
            params = [_hull(m, i) for m, i in zip(memo, params)]

        # Ascending phase: exact joins, then threshold widening.
        for iteration in range(_MAX_ITERATIONS):
            yield_ranges = walk_with(params)
            joined = [_hull(p, y) for p, y in zip(params, yield_ranges)]
            if joined == params:
                break
            if iteration + 1 >= _EXACT_ITERATIONS:
                joined = [_widen(p, j) for p, j in zip(params, joined)]
            params = joined
        else:
            # Unreachable given the finite widening ladder; stay sound.
            params = [FULL_RANGE] * len(params)
            yield_ranges = walk_with(params)

        # Descending phase: params is a post-fixpoint, so re-evaluating the
        # body may shrink it back toward hull(init, yields) and recover the
        # precision a widening overshoot (or a stale-wide warm start)
        # discarded (e.g. [0, 7] -> [0, 6] for a wrap bounded by 7).
        for _ in range(_NARROW_ITERATIONS):
            candidate = [
                _hull(i, y) if _contains(p, _hull(i, y)) else p
                for p, i, y in zip(params, init_ranges, yield_ranges)
            ]
            if candidate == params:
                break
            params = candidate
            yield_ranges = walk_with(params)
        self._loop_memo[id(loop)] = tuple(params)

        for i, result in enumerate(loop.results):
            r = yield_ranges[i]
            if not must_run:
                r = _hull(init_ranges[i], r)
            if isinstance(result, SSAValue):
                self._set(result, r)

    # === Transfer functions ===

    def _transfer(self, stmt: Op, overlay: dict) -> _Range:
        M = WORD_MASK
        FULL = FULL_RANGE
        opcode = stmt.opcode
        r = FULL

        if opcode == "select" and len(stmt.operands) == 3:
            c = self._rng(stmt.operands[0], overlay)
            # Each arm is only taken when the condition holds, so evaluate it
            # under the matching refinement. This is what lets the wrap
            # recurrence select(x < n, x, 0) stabilize at [0, n - 1].
            then_overlay = dict(overlay)
            else_overlay = dict(overlay)
            self._refine(stmt.operands[0], then_overlay, else_overlay, overlay)
            a = self._rng(stmt.operands[1], then_overlay)
            b = self._rng(stmt.operands[2], else_overlay)
            if c == (0, 0):
                r = b
            elif c[0] >= 1:
                r = a
            else:
                r = _hull(a, b)
        elif len(stmt.operands) == 2:
            l1, h1 = self._rng(stmt.operands[0], overlay)
            l2, h2 = self._rng(stmt.operands[1], overlay)
            if opcode == "+":
                r = (l1 + l2, h1 + h2)
                if r[1] > M:
                    r = FULL
            elif opcode == "-":
                r = (l1 - h2, h1 - l2) if l1 >= h2 else FULL
            elif opcode == "*":
                r = (l1 * l2, h1 * h2)
                if r[1] > M:
                    r = FULL
            elif opcode == "<<":
                if h2 < 32 and (h1 << h2) <= M:
                    r = (l1 << l2, h1 << h2)
            elif opcode == ">>":
                if h2 < 32:
                    r = (l1 >> h2, h1 >> l2)
            elif opcode == "&":
                r = (0, min(h1, h2))
            elif opcode == "|":
                r = (max(l1, l2),
                     (1 << max(h1.bit_length(), h2.bit_length())) - 1)
            elif opcode == "^":
                r = (0, (1 << max(h1.bit_length(), h2.bit_length())) - 1)
            elif opcode == "%":
                if l2 == h2 and l2 > 0:
                    r = (l1, h1) if h1 < l2 else (0, l2 - 1)
                elif l2 > 0:
                    r = (0, h2 - 1)
            elif opcode == "//":
                if l2 > 0:
                    r = (l1 // h2, h1 // l2)
            elif opcode == "<":
                if h1 < l2:
                    r = (1, 1)
                elif l1 >= h2:
                    r = (0, 0)
                else:
                    r = (0, 1)
            elif opcode == "==":
                if l1 == h1 and l2 == h2 and l1 == l2:
                    r = (1, 1)
                elif h1 < l2 or h2 < l1:
                    r = (0, 0)
                else:
                    r = (0, 1)
        return r
