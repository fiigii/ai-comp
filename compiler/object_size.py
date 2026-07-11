"""Object-size analysis for statically bounded HIR memory views."""

from __future__ import annotations

from dataclasses import dataclass

from .hir import Const, HIRFunction, Op, SSAValue, Value, WORD_MASK
from .pointer_provenance import PointerProvenance
from .use_def import UseDefContext


OBJECT_EXTENT_OPCODE = "object_extent"


@dataclass(frozen=True)
class ObjectExtent:
    """Trusted frontend/ABI description of one non-wrapping memory object."""

    base: Value
    length: int


def collect_object_extents(hir: HIRFunction) -> list[ObjectExtent]:
    """Collect valid top-level, whole-invocation object descriptors."""
    result: list[ObjectExtent] = []
    for stmt in hir.body:
        if (not isinstance(stmt, Op)
                or stmt.opcode != OBJECT_EXTENT_OPCODE
                or stmt.engine != "meta"
                or stmt.result is not None
                or len(stmt.operands) != 2):
            continue
        base, length = stmt.operands
        if (not isinstance(base, (SSAValue, Const))
                or not isinstance(length, Const)
                or isinstance(length.value, bool)
                or not isinstance(length.value, int)
                or length.value <= 0
                or length.value > WORD_MASK + 1):
            continue
        if isinstance(base, Const):
            if (isinstance(base.value, bool)
                    or not isinstance(base.value, int)
                    or (base.value & WORD_MASK) + length.value
                        > WORD_MASK + 1):
                continue
        result.append(ObjectExtent(base, length.value))
    return result


class ObjectSizeAnalysis:
    """Prove that base-relative access windows stay inside memory objects."""

    def __init__(
        self,
        hir: HIRFunction,
        use_def: UseDefContext | None = None,
    ) -> None:
        self._use_def = use_def or UseDefContext(hir)
        self._provenance = [
            (
                extent,
                PointerProvenance(hir, extent.base, self._use_def)
                if isinstance(extent.base, SSAValue) else None,
            )
            for extent in collect_object_extents(hir)
        ]

    def contains_window(
        self,
        base: Value,
        offset: int,
        width: int,
    ) -> bool:
        """Whether every word in ``base + [offset, offset + width)`` is valid.

        The proof intentionally rejects modulo-wrapping derivations. They may
        land back inside an object, but retaining only non-wrapping facts keeps
        object bounds independent of the VM's 32-bit address arithmetic.
        """
        if offset < 0 or width <= 0:
            return False
        for extent, provenance in self._provenance:
            if provenance is not None:
                relation = provenance.classify(base)
                if relation.kind != "static" or relation.offset is None:
                    continue
                base_offset = relation.offset
            else:
                if not isinstance(base, Const):
                    continue
                assert isinstance(extent.base, Const)
                if (isinstance(base.value, bool)
                        or not isinstance(base.value, int)):
                    continue
                object_base = extent.base.value & WORD_MASK
                candidate_base = base.value & WORD_MASK
                if candidate_base < object_base:
                    continue
                base_offset = candidate_base - object_base
            start = base_offset + offset
            end = start + width
            if (start <= WORD_MASK
                    and end <= WORD_MASK + 1
                    and end <= extent.length):
                return True
        return False
