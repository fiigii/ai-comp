"""Exact-base pointer provenance for HIR addresses.

Given one designated base SSA value, classify how any other value derives
from it: at a static word offset, at a dynamic (optionally range-bounded)
offset, or not derived from it at all.

This is the directed complement of :mod:`compiler.alias_analysis`: alias
analysis answers the symmetric may-alias question over canonicalized
roots, while provenance answers "where is this address relative to THIS
base".  It carries no local-memory semantics; consumers include region
promotion (:mod:`compiler.passes.sroa`) and, historically, read-only
table caching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .hir import (
    Const,
    HIRFunction,
    Op,
    SSAValue,
    Value,
    Variable,
    VectorSSAValue,
    WORD_MASK,
)
from .use_def import UseDefContext


_MEMORY_OPS = frozenset(("load", "store", "vload", "vstore", "vgather"))


@dataclass(frozen=True)
class AddressRelation:
    """How an address value is derived from the designated exact base."""

    kind: str  # "unrelated", "static", or "dynamic"
    offset: Optional[int] = None
    # For "dynamic": the non-wrapping interval of base-relative offsets the
    # address can take, when value-range analysis can bound it. Lets a
    # consumer prove a dynamic access lies entirely outside a region.
    offset_range: Optional[tuple] = None


class PointerProvenance:
    """Classify exact-base-derived addresses without assuming pointer slots.

    A load from the memory slot that originally held ``base`` is deliberately
    unrelated: the slot may have been overwritten since ``base`` was loaded.

    ``ranges_provider`` (a zero-argument callable returning a RangeAnalysis)
    is consulted lazily to bound dynamic offsets from the base.
    """

    def __init__(
        self,
        hir: HIRFunction,
        base: SSAValue,
        use_def: Optional[UseDefContext] = None,
        ranges_provider=None,
    ) -> None:
        self._use_def = use_def or UseDefContext(hir)
        self._ranges_provider = ranges_provider
        self._cache: dict[Variable, AddressRelation] = {
            base: AddressRelation("static", 0)
        }

    def classify(self, value: Value) -> AddressRelation:
        """Return the exact-base provenance and static word offset of value."""

        if not isinstance(value, (SSAValue, VectorSSAValue)):
            return AddressRelation("unrelated")
        cached = self._cache.get(value)
        if cached is not None:
            return cached

        # Explicit DFS avoids Python recursion limits on long straight-line
        # address chains. The expanded flag means all operand facts are ready.
        stack: list[tuple[Variable, bool]] = [(value, False)]
        active: set[Variable] = set()
        while stack:
            current, expanded = stack.pop()
            if current in self._cache:
                continue
            if expanded:
                active.discard(current)
                def_loc = self._use_def.get_def(current)
                assert def_loc is not None and isinstance(def_loc.statement, Op)
                self._cache[current] = self._classify_op(def_loc.statement)
                continue
            if current in active:
                self._cache[current] = AddressRelation("dynamic")
                continue

            def_loc = self._use_def.get_def(current)
            if (def_loc is None or not isinstance(def_loc.statement, Op)
                    or def_loc.statement.opcode in _MEMORY_OPS):
                self._cache[current] = AddressRelation("unrelated")
                continue

            active.add(current)
            stack.append((current, True))
            for operand in reversed(def_loc.statement.operands):
                if (isinstance(operand, (SSAValue, VectorSSAValue))
                        and operand not in self._cache):
                    stack.append((operand, False))

        return self._cache[value]

    def _classify_op(self, op: Op) -> AddressRelation:
        def relation(value: Value) -> AddressRelation:
            if not isinstance(value, (SSAValue, VectorSSAValue)):
                return AddressRelation("unrelated")
            return self._cache[value]

        if op.opcode == "+" and len(op.operands) == 2:
            left, right = op.operands
            if isinstance(left, Const):
                right_relation = relation(right)
                if right_relation.kind == "static":
                    assert right_relation.offset is not None
                    return AddressRelation(
                        "static",
                        (right_relation.offset + left.value) & WORD_MASK,
                    )
            if isinstance(right, Const):
                left_relation = relation(left)
                if left_relation.kind == "static":
                    assert left_relation.offset is not None
                    return AddressRelation(
                        "static",
                        (left_relation.offset + right.value) & WORD_MASK,
                    )
        elif op.opcode == "-" and len(op.operands) == 2:
            left, right = op.operands
            if isinstance(right, Const):
                left_relation = relation(left)
                if left_relation.kind == "static":
                    assert left_relation.offset is not None
                    return AddressRelation(
                        "static",
                        (left_relation.offset - right.value) & WORD_MASK,
                    )

        if any(relation(operand).kind != "unrelated" for operand in op.operands):
            bounded = self._bounded_dynamic_offset(op, relation)
            if bounded is not None:
                return AddressRelation("dynamic", offset_range=bounded)
            return AddressRelation("dynamic")
        return AddressRelation("unrelated")

    def _bounded_dynamic_offset(self, op: Op, relation) -> Optional[tuple]:
        """Non-wrapping offset interval of static-base +/- ranged value.

        Only the exact shapes ``static + unrelated`` and ``static -
        unrelated`` are bounded; intervals that could wrap mod 2**32 are
        rejected so a consumer may compare the result against a region
        extent directly.
        """
        if self._ranges_provider is None or len(op.operands) != 2:
            return None
        left, right = op.operands

        def static_offset(value: Value) -> Optional[int]:
            rel = relation(value)
            return rel.offset if rel.kind == "static" else None

        def unrelated_ssa(value: Value) -> Optional[SSAValue]:
            if (isinstance(value, SSAValue)
                    and relation(value).kind == "unrelated"):
                return value
            return None

        if op.opcode == "+":
            for static_side, dyn_side in ((left, right), (right, left)):
                offset = static_offset(static_side)
                dyn = unrelated_ssa(dyn_side)
                if offset is None or dyn is None:
                    continue
                lo, hi = self._ranges_provider().range_of(dyn)
                if offset + hi <= WORD_MASK:
                    return (offset + lo, offset + hi)
        elif op.opcode == "-":
            offset = static_offset(left)
            dyn = unrelated_ssa(right)
            if offset is not None and dyn is not None:
                lo, hi = self._ranges_provider().range_of(dyn)
                if offset >= hi:
                    return (offset - hi, offset - lo)
        return None
