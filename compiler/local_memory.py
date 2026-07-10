"""Language-level local-memory contracts for HIR.

``assume_local_memory(base, length)`` is a marker, not an allocation.  From
the marker onward the program promises that the indicated word-addressed
region is zero initialized, private, non-escaping, and unobservable at pause
boundaries and when the function returns. Optimization passes may ignore the
promise, but programs that violate it have undefined behavior.

This module only parses and collects the contract.  Promotion is implemented
by :mod:`compiler.passes.local_mem2reg`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

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
    VectorSSAValue,
)
from .use_def import UseDefContext


LOCAL_MEMORY_OPCODE = "assume_local_memory"
WORD_MASK = (1 << 32) - 1
_MEMORY_OPS = frozenset(("load", "store", "vload", "vstore", "vgather"))


@dataclass(frozen=True)
class LocalMemoryMarker:
    """An occurrence of an ``assume_local_memory`` marker in a function."""

    op: Op
    top_level_index: int
    control_flow_depth: int


@dataclass(frozen=True)
class StaticLocalMemoryRegion:
    """A syntactically valid, statically sized local-memory contract."""

    marker: LocalMemoryMarker
    base: SSAValue
    length: int


@dataclass(frozen=True)
class LocalAddressRelation:
    """How an address value is derived from a local region's exact base."""

    kind: str  # "unrelated", "static", or "dynamic"
    offset: Optional[int] = None
    # For "dynamic": the non-wrapping interval of base-relative offsets the
    # address can take, when value-range analysis can bound it. Lets a
    # consumer prove a dynamic access lies entirely outside the region.
    offset_range: Optional[tuple] = None


class LocalPointerProvenance:
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
        self._cache: dict[Variable, LocalAddressRelation] = {
            base: LocalAddressRelation("static", 0)
        }

    def classify(self, value: Value) -> LocalAddressRelation:
        """Return the exact-base provenance and static word offset of value."""

        if not isinstance(value, (SSAValue, VectorSSAValue)):
            return LocalAddressRelation("unrelated")
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
                self._cache[current] = LocalAddressRelation("dynamic")
                continue

            def_loc = self._use_def.get_def(current)
            if (def_loc is None or not isinstance(def_loc.statement, Op)
                    or def_loc.statement.opcode in _MEMORY_OPS):
                self._cache[current] = LocalAddressRelation("unrelated")
                continue

            active.add(current)
            stack.append((current, True))
            for operand in reversed(def_loc.statement.operands):
                if (isinstance(operand, (SSAValue, VectorSSAValue))
                        and operand not in self._cache):
                    stack.append((operand, False))

        return self._cache[value]

    def _classify_op(self, op: Op) -> LocalAddressRelation:
        def relation(value: Value) -> LocalAddressRelation:
            if not isinstance(value, (SSAValue, VectorSSAValue)):
                return LocalAddressRelation("unrelated")
            return self._cache[value]

        if op.opcode == "+" and len(op.operands) == 2:
            left, right = op.operands
            if isinstance(left, Const):
                right_relation = relation(right)
                if right_relation.kind == "static":
                    assert right_relation.offset is not None
                    return LocalAddressRelation(
                        "static",
                        (right_relation.offset + left.value) & WORD_MASK,
                    )
            if isinstance(right, Const):
                left_relation = relation(left)
                if left_relation.kind == "static":
                    assert left_relation.offset is not None
                    return LocalAddressRelation(
                        "static",
                        (left_relation.offset + right.value) & WORD_MASK,
                    )
        elif op.opcode == "-" and len(op.operands) == 2:
            left, right = op.operands
            if isinstance(right, Const):
                left_relation = relation(left)
                if left_relation.kind == "static":
                    assert left_relation.offset is not None
                    return LocalAddressRelation(
                        "static",
                        (left_relation.offset - right.value) & WORD_MASK,
                    )

        if any(relation(operand).kind != "unrelated" for operand in op.operands):
            bounded = self._bounded_dynamic_offset(op, relation)
            if bounded is not None:
                return LocalAddressRelation("dynamic", offset_range=bounded)
            return LocalAddressRelation("dynamic")
        return LocalAddressRelation("unrelated")

    def _bounded_dynamic_offset(self, op: Op, relation) -> Optional[tuple]:
        """Non-wrapping offset interval of static-base +/- ranged value.

        Only the exact shapes ``static + unrelated`` and ``static -
        unrelated`` are bounded; intervals that could wrap mod 2**32 are
        rejected so a consumer may compare the result against the region
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


def is_local_memory_marker(stmt: Statement) -> bool:
    """Return whether *stmt* is a local-memory contract marker."""

    return isinstance(stmt, Op) and stmt.opcode == LOCAL_MEMORY_OPCODE


def parse_static_local_memory_marker(
    marker: LocalMemoryMarker,
) -> Optional[StaticLocalMemoryRegion]:
    """Parse a positive, constant-length scalar region marker.

    Invalid markers are deliberately returned as ``None`` rather than raising:
    the optimization can conservatively leave their memory accesses intact.
    Front-end validation may diagnose malformed contracts separately.
    """

    op = marker.op
    if op.result is not None or len(op.operands) != 2:
        return None
    base, length = op.operands
    if not isinstance(base, SSAValue):
        return None
    if not isinstance(length, Const) or length.value <= 0:
        return None
    return StaticLocalMemoryRegion(marker=marker, base=base, length=length.value)


def _walk_markers(
    body: list[Statement],
    top_level_index: int,
    depth: int,
) -> Iterator[LocalMemoryMarker]:
    for stmt in body:
        if is_local_memory_marker(stmt):
            assert isinstance(stmt, Op)
            yield LocalMemoryMarker(stmt, top_level_index, depth)
        elif isinstance(stmt, ForLoop):
            yield from _walk_markers(stmt.body, top_level_index, depth + 1)
        elif isinstance(stmt, If):
            yield from _walk_markers(
                stmt.then_body, top_level_index, depth + 1
            )
            yield from _walk_markers(
                stmt.else_body, top_level_index, depth + 1
            )


def collect_local_memory_markers(hir: HIRFunction) -> list[LocalMemoryMarker]:
    """Collect all local-memory markers, including markers in control flow."""

    result: list[LocalMemoryMarker] = []
    for index, stmt in enumerate(hir.body):
        if is_local_memory_marker(stmt):
            assert isinstance(stmt, Op)
            result.append(LocalMemoryMarker(stmt, index, 0))
        elif isinstance(stmt, ForLoop):
            result.extend(_walk_markers(stmt.body, index, 1))
        elif isinstance(stmt, If):
            result.extend(_walk_markers(stmt.then_body, index, 1))
            result.extend(_walk_markers(stmt.else_body, index, 1))
    return result


def collect_static_local_memory_regions(
    hir: HIRFunction,
) -> list[StaticLocalMemoryRegion]:
    """Collect syntactically valid local regions from *hir*.

    Regions nested in retained control flow are included in the result so a
    consumer can diagnose or count them.  The first promotion implementation
    rejects them atomically because it does not construct phis.
    """

    regions: list[StaticLocalMemoryRegion] = []
    for marker in collect_local_memory_markers(hir):
        region = parse_static_local_memory_marker(marker)
        if region is not None:
            regions.append(region)
    return regions
