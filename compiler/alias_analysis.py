"""
Alias analysis utilities for HIR.

Provides a simple base+constant-offset normalization and cached alias queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .hir import SSAValue, VectorSSAValue, Const, VectorConst, Value, Op, WORD_MASK
from .range_analysis import RangeAnalysis
from .use_def import UseDefContext


@dataclass(frozen=True)
class AddrKey:
    """Normalized address: base + constant offset."""
    base: object
    offset: int


class AliasResult(Enum):
    MUST_ALIAS = 1
    NO_ALIAS = 2
    MAY_ALIAS = 3


_CONST_BASE = object()
_ADDRESS_SPACE = 1 << 32


def base_roots(base: object) -> frozenset:
    """Root components of a (possibly composite) base.

    A composite base ("add", a, b) contains the roots of both components.
    Root-set disjointness is what restrict_ptr acts on: table+i and table+j
    share the table root, so restrict_ptr alone never proves them disjoint.
    For bounded i, j the value-range proof in _alias_shared_component can
    still establish NO_ALIAS.
    """
    if isinstance(base, tuple) and len(base) == 3 and base[0] == "add":
        return base_roots(base[1]) | base_roots(base[2])
    if isinstance(base, tuple) and len(base) == 3 and base[0] == "rooted_load":
        return base_roots(base[2])
    if isinstance(base, tuple) and len(base) == 3 and base[0] == "derived":
        return base[2]
    if isinstance(base, tuple) and len(base) == 4 and base[0] == "union":
        return base_roots(base[2]) | base_roots(base[3])
    return frozenset([base])


class AliasAnalysis:
    """
    Simple alias analysis based on base + constant offset normalization.

    - If base matches and ranges overlap:
        - exact same range => MUST_ALIAS
        - overlapping ranges => MAY_ALIAS
    - If bases differ but share a symbolic root:
        - bounded dynamic components may still prove NO_ALIAS by value range
        - otherwise => MAY_ALIAS
    - If roots are disjoint and restrict_ptr is set => NO_ALIAS
    - Without restrict_ptr, different symbolic roots remain MAY_ALIAS
    """

    def __init__(self, use_def: UseDefContext, restrict_ptr: bool = False):
        self._use_def = use_def
        self._restrict_ptr = restrict_ptr
        self._norm_cache: dict[SSAValue, Optional[AddrKey]] = {}
        self._alias_cache: dict[tuple[Optional[AddrKey], int, Optional[AddrKey], int], AliasResult] = {}
        self.alias_queries = 0
        self.alias_cache_hits = 0
        # Value-range analysis, built lazily on the first query that can
        # use it (shared-root bases with rangeable dynamic components).
        self._ranges: Optional[RangeAnalysis] = None
        self._ranges_built = False

    def _range_analysis(self) -> Optional[RangeAnalysis]:
        if not self._ranges_built:
            self._ranges_built = True
            hir = getattr(self._use_def, "hir", None)
            if hir is not None:
                self._ranges = RangeAnalysis(hir)
        return self._ranges

    def normalize(self, val: Value) -> Optional[AddrKey]:
        """Normalize an address expression to (base, offset)."""
        if isinstance(val, Const):
            return AddrKey(_CONST_BASE, val.value & WORD_MASK)
        if isinstance(val, SSAValue):
            return self._normalize_ssa(val)
        # Vector addresses and vector consts are treated as unknown
        if isinstance(val, (VectorSSAValue, VectorConst)):
            return None
        return None

    def _normalize_ssa(self, ssa: SSAValue) -> Optional[AddrKey]:
        if ssa in self._norm_cache:
            return self._norm_cache[ssa]

        # Default: treat this SSA as a base pointer
        base = self._canonical_base(ssa)
        key = AddrKey(base, 0)
        self._norm_cache[ssa] = key

        # If defined as add with constant, fold into base+offset
        def_loc = self._use_def.get_def(ssa)
        if def_loc is None:
            return key
        stmt = def_loc.statement
        if isinstance(stmt, Op) and stmt.opcode == "+":
            const_val, other = self._extract_const_add(stmt)
            if const_val is not None and other is not None:
                other_key = self.normalize(other)
                if other_key is not None:
                    key = AddrKey(
                        other_key.base,
                        (other_key.offset + const_val) & WORD_MASK,
                    )
                    self._norm_cache[ssa] = key
                    return key
            # Handle ssa + ssa: normalize both sides and create composite base.
            # This is needed for partial loop unrolling where addresses are
            # ptr + (scaled + j) — both operands are SSA values but one
            # carries a constant offset that SLP needs to see.
            a, b = stmt.operands
            if isinstance(a, SSAValue) and isinstance(b, SSAValue):
                a_key = self._normalize_ssa(a)
                b_key = self._normalize_ssa(b)
                if a_key is not None and b_key is not None:
                    # Sort bases for commutativity (a+b == b+a)
                    bases = (a_key.base, b_key.base)
                    if repr(bases[0]) > repr(bases[1]):
                        bases = (bases[1], bases[0])
                    key = AddrKey(
                        ("add", bases[0], bases[1]),
                        (a_key.offset + b_key.offset) & WORD_MASK,
                    )
                    self._norm_cache[ssa] = key
                    return key
        elif (isinstance(stmt, Op) and stmt.opcode == "-"
              and len(stmt.operands) == 2
              and isinstance(stmt.operands[1], Const)):
            left, right = stmt.operands
            left_key = self.normalize(left)
            if left_key is not None:
                key = AddrKey(
                    left_key.base,
                    (left_key.offset - right.value) & WORD_MASK,
                )
                self._norm_cache[ssa] = key
                return key
        elif (isinstance(stmt, Op) and stmt.opcode == "select"
              and len(stmt.operands) == 3):
            _, true_value, false_value = stmt.operands
            true_key = self.normalize(true_value)
            false_key = self.normalize(false_value)
            if true_key is None or false_key is None:
                self._norm_cache[ssa] = None
                return None
            # Keep each select's base unique so two independently selected
            # pointers are not mistaken for MUST_ALIAS. base_roots exposes
            # both alternatives for conservative cross-base queries.
            key = AddrKey(
                ("union", ssa, true_key.base, false_key.base),
                0,
            )
            self._norm_cache[ssa] = key
            return key
        elif isinstance(stmt, Op) and stmt.opcode != "load":
            operand_roots = frozenset()
            for operand in stmt.operands:
                if isinstance(operand, Const):
                    continue
                if not isinstance(operand, SSAValue):
                    self._norm_cache[ssa] = None
                    return None
                operand_key = self.normalize(operand)
                if operand_key is None:
                    self._norm_cache[ssa] = None
                    return None
                operand_roots |= base_roots(operand_key.base)
            if operand_roots:
                # Keep the result identity unique: an unsupported operation
                # is not precise enough to establish MUST_ALIAS. Retaining
                # every operand root still prevents restrict_ptr from proving
                # it disjoint from pointers it may have been derived from.
                key = AddrKey(("derived", ssa, operand_roots), 0)
                self._norm_cache[ssa] = key
                return key
            self._norm_cache[ssa] = None
            return None
        return key

    def _canonical_base(self, ssa: SSAValue) -> object:
        """Give constant-slot loads unique bases with shared provenance roots."""
        def_loc = self._use_def.get_def(ssa)
        if def_loc is None:
            return ssa
        stmt = def_loc.statement
        if isinstance(stmt, Op) and stmt.opcode == "load":
            address_key = self.normalize(stmt.operands[0])
            if address_key is not None and address_key.base is _CONST_BASE:
                root = ("memslot", address_key.offset)
                return ("rooted_load", ssa, root)
        return ssa

    @staticmethod
    def _extract_const_add(op: Op) -> tuple[Optional[int], Optional[Value]]:
        """Return (const_val, other_operand) for add with constant, else (None, None)."""
        if len(op.operands) != 2:
            return None, None
        a, b = op.operands
        if isinstance(a, Const):
            return a.value, b
        if isinstance(b, Const):
            return b.value, a
        return None, None

    def alias_keys(self, a_key: Optional[AddrKey], a_width: int,
                   b_key: Optional[AddrKey], b_width: int) -> AliasResult:
        """Alias query on normalized keys with widths."""
        self.alias_queries += 1
        cache_key = (a_key, a_width, b_key, b_width)
        cached = self._alias_cache.get(cache_key)
        if cached is not None:
            self.alias_cache_hits += 1
            return cached

        if a_key is None or b_key is None:
            res = AliasResult.MAY_ALIAS
        elif a_key.base == b_key.base:
            res = self._alias_same_base(a_key, a_width, b_key, b_width)
        else:
            # Different bases can only be disjoint when they share no root:
            # composite bases like table+i vs table+j (distinct dynamic
            # index SSAs) share the table root and may alias for any i, j --
            # UNLESS value-range analysis proves the dynamic components
            # disjoint around one identical shared component.
            a_roots = base_roots(a_key.base)
            b_roots = base_roots(b_key.base)
            if a_roots & b_roots:
                res = (self._alias_shared_component(a_key, a_width,
                                                    b_key, b_width)
                       or AliasResult.MAY_ALIAS)
            elif self._restrict_ptr:
                res = AliasResult.NO_ALIAS
            else:
                res = AliasResult.MAY_ALIAS

        # Cache symmetric results
        self._alias_cache[cache_key] = res
        self._alias_cache[(b_key, b_width, a_key, a_width)] = res
        return res

    @staticmethod
    def _component_ssa(base: object) -> Optional[SSAValue]:
        """The SSA whose runtime value a non-composite base contributes.

        Every non-composite base form is anchored to one SSA: the address
        value of a key (base, offset) is value(anchor) + offset mod 2**32.
        """
        if isinstance(base, SSAValue):
            return base
        if (isinstance(base, tuple) and len(base) >= 2
                and base[0] in ("rooted_load", "derived", "union")
                and isinstance(base[1], SSAValue)):
            return base[1]
        return None

    @classmethod
    def _decompositions(cls, key: AddrKey) -> list:
        """(shared_component, dynamic_part) views of an address key.

        dynamic_part is ("ssa", value) or ("zero",); the address equals
        value(shared_component's anchor) + value(dynamic_part) + offset.
        Composite ("add", x, y) bases yield both orientations.
        """
        base = key.base
        views = []
        if isinstance(base, tuple) and len(base) == 3 and base[0] == "add":
            for common, dynamic in ((base[1], base[2]), (base[2], base[1])):
                dyn_ssa = cls._component_ssa(dynamic)
                if dyn_ssa is not None:
                    views.append((common, ("ssa", dyn_ssa)))
        else:
            views.append((base, ("zero",)))
        return views

    def _alias_shared_component(
        self, a_key: AddrKey, a_width: int, b_key: AddrKey, b_width: int
    ) -> Optional[AliasResult]:
        """Range-based NO_ALIAS proof for bases sharing one component.

        When both addresses decompose as value(X) + dyn + offset around the
        SAME component X, the shared value(X) rotates both footprints by the
        same amount mod 2**32, so it cancels: the accesses are disjoint iff
        the circular arcs [dyn_lo + offset, dyn_hi + offset + width) are.
        Returns NO_ALIAS on proof, None when no proof is possible.
        """
        pairs = [
            (a_dyn, b_dyn)
            for a_common, a_dyn in self._decompositions(a_key)
            for b_common, b_dyn in self._decompositions(b_key)
            if a_common == b_common
        ]
        if not pairs:
            return None
        ranges = self._range_analysis()
        if ranges is None:
            return None

        def arc(dyn, offset: int, width: int):
            if dyn[0] == "zero":
                lo, hi = 0, 0
            else:
                lo, hi = ranges.range_of(dyn[1])
            length = (hi - lo) + width
            if length >= _ADDRESS_SPACE:
                return None
            start = (lo + offset) & WORD_MASK
            end = start + length
            if end <= _ADDRESS_SPACE:
                return ((start, end),)
            return ((start, _ADDRESS_SPACE), (0, end - _ADDRESS_SPACE))

        for a_dyn, b_dyn in pairs:
            a_arcs = arc(a_dyn, a_key.offset, a_width)
            b_arcs = arc(b_dyn, b_key.offset, b_width)
            if a_arcs is None or b_arcs is None:
                continue
            if not any(max(a_lo, b_lo) < min(a_hi, b_hi)
                       for a_lo, a_hi in a_arcs
                       for b_lo, b_hi in b_arcs):
                return AliasResult.NO_ALIAS
        return None

    @staticmethod
    def _alias_same_base(a_key: AddrKey, a_width: int,
                         b_key: AddrKey, b_width: int) -> AliasResult:
        if a_width <= 0 or b_width <= 0:
            raise ValueError("memory access widths must be positive")

        a_start = a_key.offset & WORD_MASK
        b_start = b_key.offset & WORD_MASK
        if a_start == b_start and a_width == b_width:
            return AliasResult.MUST_ALIAS

        def intervals(start: int, width: int) -> tuple[tuple[int, int], ...]:
            if width >= _ADDRESS_SPACE:
                return ((0, _ADDRESS_SPACE),)
            end = start + width
            if end <= _ADDRESS_SPACE:
                return ((start, end),)
            return ((start, _ADDRESS_SPACE), (0, end - _ADDRESS_SPACE))

        a_intervals = intervals(a_start, a_width)
        b_intervals = intervals(b_start, b_width)
        for a_lo, a_hi in a_intervals:
            for b_lo, b_hi in b_intervals:
                if max(a_lo, b_lo) < min(a_hi, b_hi):
                    return AliasResult.MAY_ALIAS
        return AliasResult.NO_ALIAS
