"""Language-level local-memory contracts for HIR.

``assume_local_memory(base, length)`` is a marker, not an allocation.  From
the marker onward the program promises that the indicated word-addressed
region is zero initialized, private, non-escaping, and unobservable at pause
boundaries and when the function returns. Optimization passes may ignore the
promise, but programs that violate it have undefined behavior.

This module only parses and collects the contract.  Promotion is implemented
by :mod:`compiler.passes.sroa`.
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
)


LOCAL_MEMORY_OPCODE = "assume_local_memory"


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
