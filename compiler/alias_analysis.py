"""
Alias analysis utilities for HIR.

Provides a simple base+constant-offset normalization and cached alias queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .hir import SSAValue, VectorSSAValue, Const, VectorConst, Value, Op
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


def _is_memslot_base(base: object) -> bool:
    return isinstance(base, tuple) and len(base) == 2 and base[0] == "memslot"


class AliasAnalysis:
    """
    Simple alias analysis based on base + constant offset normalization.

    - If base matches and ranges overlap:
        - exact same range => MUST_ALIAS
        - overlapping ranges => MAY_ALIAS
    - If base differs and restrict_ptr is set => NO_ALIAS (ignores offsets)
    - If base differs and both are distinct memslot roots => NO_ALIAS
    - Otherwise => MAY_ALIAS
    """

    def __init__(self, use_def: UseDefContext, restrict_ptr: bool = False):
        self._use_def = use_def
        self._restrict_ptr = restrict_ptr
        self._norm_cache: dict[SSAValue, Optional[AddrKey]] = {}
        self._alias_cache: dict[tuple[Optional[AddrKey], int, Optional[AddrKey], int], AliasResult] = {}
        self.alias_queries = 0
        self.alias_cache_hits = 0

    def normalize(self, val: Value) -> Optional[AddrKey]:
        """Normalize an address expression to (base, offset)."""
        if isinstance(val, Const):
            return AddrKey(_CONST_BASE, val.value)
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
                    key = AddrKey(other_key.base, other_key.offset + const_val)
                    self._norm_cache[ssa] = key
                    return key
        return key

    def _canonical_base(self, ssa: SSAValue) -> object:
        """If SSA is loaded from a constant slot, use that slot as a root base."""
        def_loc = self._use_def.get_def(ssa)
        if def_loc is None:
            return ssa
        stmt = def_loc.statement
        if isinstance(stmt, Op) and stmt.opcode == "load":
            addr = stmt.operands[0]
            if isinstance(addr, Const):
                return ("memslot", addr.value)
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
            if self._restrict_ptr:
                res = AliasResult.NO_ALIAS
            elif _is_memslot_base(a_key.base) and _is_memslot_base(b_key.base):
                res = AliasResult.NO_ALIAS
            else:
                res = AliasResult.MAY_ALIAS

        # Cache symmetric results
        self._alias_cache[cache_key] = res
        self._alias_cache[(b_key, b_width, a_key, a_width)] = res
        return res

    @staticmethod
    def _alias_same_base(a_key: AddrKey, a_width: int,
                         b_key: AddrKey, b_width: int) -> AliasResult:
        a_start = a_key.offset
        a_end = a_key.offset + a_width - 1
        b_start = b_key.offset
        b_end = b_key.offset + b_width - 1

        if a_end < b_start or b_end < a_start:
            return AliasResult.NO_ALIAS

        if a_start == b_start and a_width == b_width:
            return AliasResult.MUST_ALIAS

        return AliasResult.MAY_ALIAS
