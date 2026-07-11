"""Tests for trusted HIR memory-object extents."""

import pytest

from compiler import Const, HIRBuilder, Op, PassConfig
from compiler.lowering import lower_to_lir
from compiler.object_size import (
    OBJECT_EXTENT_OPCODE,
    ObjectSizeAnalysis,
    collect_object_extents,
)
from compiler.passes import CSEPass, DCEPass, LoopUnrollPass, SimplifyPass


def test_object_size_proves_exact_and_derived_windows():
    b = HIRBuilder()
    base = b.load(b.const(4), "base")
    b.memory_view(base, 8)
    derived = b.add(base, b.const(3), "derived")
    hir = b.build()

    sizes = ObjectSizeAnalysis(hir)
    assert sizes.contains_window(base, 0, 8)
    assert sizes.contains_window(derived, 2, 3)
    assert not sizes.contains_window(derived, 2, 4)


def test_object_size_rejects_wrapping_derivation():
    b = HIRBuilder()
    base = b.load(b.const(4), "base")
    b.memory_view(base, 8)
    before = b.sub(base, b.const(1), "before")
    hir = b.build()

    assert not ObjectSizeAnalysis(hir).contains_window(before, 1, 1)


def test_constant_object_extent_survives_simplify():
    b = HIRBuilder()
    folded_base = b.add(b.const(100), b.const(0), "folded_base")
    b.memory_view(folded_base, 4)

    hir = SimplifyPass().run(
        b.build(), PassConfig(name="simplify", enabled=True, options={})
    )
    extent = collect_object_extents(hir)[0]
    assert extent.base == Const(100)
    assert ObjectSizeAnalysis(hir).contains_window(Const(102), 0, 2)


def test_malformed_object_extent_is_not_trusted():
    b = HIRBuilder()
    base = b.load(b.const(4), "base")
    b._emit(Op(
        OBJECT_EXTENT_OPCODE, None, [base, Const(True)], "meta"
    ))
    b._emit(Op(
        OBJECT_EXTENT_OPCODE, None, [base, Const(4)], "alu"
    ))

    assert collect_object_extents(b.build()) == []


def test_object_extent_survives_pre_sroa_passes():
    b = HIRBuilder()
    root = b.load(b.const(4), "root")
    base = b.add(root, b.const(0), "base")
    b.memory_view(base, 16)
    b.store(b.const(20), b.load(base, "value"))
    hir = b.build()

    passes = [DCEPass(), LoopUnrollPass(), SimplifyPass(), DCEPass(), CSEPass()]
    for compiler_pass in passes:
        hir = compiler_pass.run(
            hir,
            PassConfig(
                name=compiler_pass.name,
                enabled=True,
                options={},
            ),
        )
        extents = collect_object_extents(hir)
        assert len(extents) == 1
        assert extents[0].length == 16
    assert collect_object_extents(hir)[0].base == root


def test_object_extent_keeps_metadata_only_base_live():
    b = HIRBuilder()
    base = b.load(b.const(4), "base")
    b.memory_view(base, 4)

    hir = DCEPass().run(
        b.build(), PassConfig(name="dce", enabled=True, options={})
    )
    extents = collect_object_extents(hir)
    assert len(extents) == 1
    assert extents[0].base == base
    assert any(getattr(stmt, "result", None) == base for stmt in hir.body)


def test_object_extent_operand_follows_cse_replacement():
    b = HIRBuilder()
    first = b.load(b.const(4), "first")
    duplicate = b.load(b.const(4), "duplicate")
    b.memory_view(duplicate, 4)

    hir = CSEPass().run(
        b.build(), PassConfig(name="cse", enabled=True, options={})
    )
    assert collect_object_extents(hir)[0].base == first


def test_object_extent_is_renumbered_by_full_unroll():
    b = HIRBuilder()
    root = b.load(b.const(4), "root")

    def body(counter, _params):
        base = b.add(root, counter, "base")
        b.memory_view(base, 1)
        return []

    b.for_loop(b.const(0), b.const(2), [], body, pragma_unroll=0)
    hir = LoopUnrollPass().run(
        b.build(),
        PassConfig(name="loop-unroll", enabled=True, options={}),
    )
    extents = collect_object_extents(hir)
    assert len(extents) == 2
    assert extents[0].base != extents[1].base


def test_lowering_defensively_erases_object_extent_metadata():
    b = HIRBuilder()
    base = b.load(b.const(4), "base")
    b.memory_view(base, 2)
    b.store(b.const(20), b.load(base, "value"))

    lir = lower_to_lir(b.build())
    assert lir.blocks


@pytest.mark.parametrize("length", [0, -1, (1 << 32) + 1])
def test_memory_view_rejects_invalid_length(length):
    b = HIRBuilder()
    base = b.load(b.const(4), "base")
    with pytest.raises(ValueError):
        b.memory_view(base, length)
