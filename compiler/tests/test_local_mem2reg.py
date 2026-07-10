"""Tests for generic local-memory SROA and mem2reg promotion."""

from __future__ import annotations

import unittest

from compiler.hir import Const, ForLoop, If, Op, SSAValue, Statement
from compiler.compile import compile_hir_to_vliw
from compiler.local_memory import collect_static_local_memory_regions
from compiler.lowering import lower_to_lir
from compiler.pass_manager import PassConfig
from compiler.passes.local_mem2reg import LocalMem2RegPass
from compiler.tests.conftest import DebugInfo, HIRBuilder, Machine, N_CORES


def _mark_local(b: HIRBuilder, base, length) -> None:
    """Emit the marker directly so this test is independent of builder sugar."""

    length_value = length if isinstance(length, (Const, SSAValue)) else b.const(length)
    b._emit(Op("assume_local_memory", None, [base, length_value], "meta"))


def _walk(body: list[Statement]):
    for stmt in body:
        yield stmt
        if isinstance(stmt, ForLoop):
            yield from _walk(stmt.body)
        elif isinstance(stmt, If):
            yield from _walk(stmt.then_body)
            yield from _walk(stmt.else_body)


def _ops(hir, opcode=None):
    return [
        stmt for stmt in _walk(hir.body)
        if isinstance(stmt, Op) and (opcode is None or stmt.opcode == opcode)
    ]


def _named_op(hir, name):
    for op in _ops(hir):
        if op.result is not None and op.result.name == name:
            return op
    return None


def _run_pass(hir):
    promotion = LocalMem2RegPass()
    result = promotion.run(
        hir, PassConfig(name="local-mem2reg", enabled=True, options={})
    )
    return result, promotion.get_metrics()


def _execute_instructions(instructions, memory):
    machine = Machine(
        memory,
        instructions,
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine


def _execute(hir, memory):
    return _execute_instructions(compile_hir_to_vliw(hir), memory)


class TestLocalMem2RegPass(unittest.TestCase):
    def test_default_pipeline_executes_promoted_region(self):
        b = HIRBuilder()
        base = b.load(b.const(10), "base")
        b.assume_local_memory(base, b.const(2))
        initial = b.load(base, "initial")
        b.store(b.add(base, b.const(1)), b.add(initial, b.const(17)))
        result = b.load(b.add(base, b.const(1)), "result")
        b.store(b.const(0), result)

        memory = [0] * 64
        memory[10] = 32
        machine = _execute(b.build(), memory)

        self.assertEqual(machine.mem[0], 17)

    def test_default_pipeline_preserves_dynamic_address_fallback(self):
        b = HIRBuilder()
        base = b.load(b.const(10), "base")
        store_index = b.load(b.const(1), "store_index")
        load_index = b.load(b.const(2), "load_index")
        b.assume_local_memory(base, b.const(8))
        store_address = b.add(base, store_index, "store_address")
        b.store(store_address, b.const(42))
        loaded = b.load(b.add(base, load_index), "loaded")
        b.store(b.const(0), loaded)
        b.store(store_address, b.const(99))

        memory = [0] * 128
        memory[1] = 3
        memory[2] = 3
        memory[10] = 64
        machine = _execute(b.build(), memory)

        self.assertEqual(machine.mem[0], 42)

    def test_default_pipeline_preserves_retained_if_fallback(self):
        b = HIRBuilder()
        condition = b.load(b.const(1), "condition")
        base = b.load(b.const(10), "base")
        b.assume_local_memory(base, b.const(1))

        def then_body():
            b.store(base, b.const(5))
            return []

        def else_body():
            b.store(base, b.const(6))
            return []

        b.if_stmt(condition, then_body, else_body)
        b.store(b.const(0), b.load(base, "result"))

        instructions = compile_hir_to_vliw(b.build())
        for condition_value, expected in ((0, 6), (1, 5)):
            memory = [0] * 64
            memory[1] = condition_value
            memory[10] = 32
            machine = _execute_instructions(instructions, memory)
            self.assertEqual(machine.mem[0], expected)

    def test_promotes_zero_initialized_scalar_region(self):
        b = HIRBuilder()
        workspace = b.load(b.const(13), "arbitrary_workspace_base")
        _mark_local(b, workspace, 4)

        initial = b.load(b.add(workspace, b.const(0), "addr0"), "initial")
        incremented = b.add(initial, b.const(5), "incremented")
        b.store(b.add(workspace, b.const(1), "addr1_store"), incremented)
        reloaded = b.load(b.add(workspace, b.const(1), "addr1_load"), "reloaded")
        b.store(b.const(30), reloaded)
        hir = b.build()

        transformed, metrics = _run_pass(hir)

        self.assertFalse(_ops(transformed, "assume_local_memory"))
        self.assertIsNone(_named_op(transformed, "initial"))
        self.assertIsNone(_named_op(transformed, "reloaded"))

        incremented_op = _named_op(transformed, "incremented")
        self.assertIsNotNone(incremented_op)
        self.assertEqual(incremented_op.operands, [Const(0), Const(5)])
        output_store = next(
            op for op in _ops(transformed, "store") if op.operands[0] == Const(30)
        )
        self.assertEqual(output_store.operands[1], incremented)

        self.assertEqual(metrics.custom["regions_promoted"], 1)
        self.assertEqual(metrics.custom["loads_promoted"], 2)
        self.assertEqual(metrics.custom["stores_removed"], 1)

    def test_absent_annotation_is_noop(self):
        b = HIRBuilder()
        base = b.load(b.const(4), "base")
        value = b.load(b.add(base, b.const(0)), "value")
        b.store(b.add(base, b.const(1)), value)
        hir = b.build()

        transformed, metrics = _run_pass(hir)

        self.assertIs(transformed, hir)
        self.assertEqual(transformed.body, hir.body)
        self.assertEqual(metrics.custom["regions_seen"], 0)
        self.assertEqual(metrics.custom["regions_promoted"], 0)

    def test_promotes_multiple_independent_regions(self):
        b = HIRBuilder()
        left = b.load(b.const(17), "left")
        right = b.load(b.const(23), "right")
        _mark_local(b, left, 3)
        _mark_local(b, right, 5)

        left_zero = b.load(b.add(left, b.const(2)), "left_zero")
        right_zero = b.load(b.add(right, b.const(4)), "right_zero")
        b.store(b.add(left, b.const(0)), b.const(11))
        b.store(b.add(right, b.const(1)), b.const(22))
        left_value = b.load(b.add(left, b.const(0)), "left_value")
        right_value = b.load(b.add(right, b.const(1)), "right_value")
        b.store(b.const(40), b.add(left_zero, left_value))
        b.store(b.const(41), b.add(right_zero, right_value))

        transformed, metrics = _run_pass(b.build())

        for name in ("left_zero", "right_zero", "left_value", "right_value"):
            self.assertIsNone(_named_op(transformed, name))
        self.assertEqual(metrics.custom["regions_promoted"], 2)
        self.assertEqual(metrics.custom["loads_promoted"], 4)
        self.assertEqual(metrics.custom["stores_removed"], 2)

    def test_same_base_markers_reject_shared_access_plan_atomically(self):
        b = HIRBuilder()
        base = b.load(b.const(10), "base")
        b.assume_local_memory(base, b.const(1))
        b.assume_local_memory(base, b.const(1))
        b.store(base, b.const(41))
        observed = b.load(base, "observed")
        b.store(b.const(0), observed)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "observed"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(metrics.custom["regions_rejected"], 2)
        self.assertEqual(
            metrics.custom["rejection_reasons"],
            {"overlapping_regions": 2},
        )

    def test_partially_overlapping_regions_preserve_memory_semantics(self):
        b = HIRBuilder()
        base = b.load(b.const(10), "base")
        overlapping_base = b.add(base, b.const(1), "overlapping_base")
        b.assume_local_memory(base, b.const(2))
        b.assume_local_memory(overlapping_base, b.const(2))
        b.store(overlapping_base, b.const(73))
        observed = b.load(overlapping_base, "observed")
        b.store(b.const(0), observed)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "observed"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(metrics.custom["regions_rejected"], 2)
        self.assertEqual(
            metrics.custom["rejection_reasons"],
            {"overlapping_regions": 2},
        )

        memory = [0] * 64
        memory[10] = 32
        machine = _execute(transformed, memory)
        self.assertEqual(machine.mem[0], 73)

    def test_canonical_address_forms_share_one_slot(self):
        b = HIRBuilder()
        base = b.load(b.const(17), "base")
        b.assume_local_memory(base, b.const(4))

        b.store(b.add(base, b.const(1)), b.const(37))
        commuted = b.load(b.add(b.const(1), base), "commuted")
        nested_base = b.add(base, b.const(0), "nested_base")
        nested = b.load(b.add(nested_base, b.const(1)), "nested")
        subtracted = b.load(
            b.sub(base, b.const(0xFFFFFFFF)), "subtracted"
        )
        total = b.add(commuted, nested, "partial_total")
        total = b.add(total, subtracted, "total")
        b.store(b.const(40), total)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNone(_named_op(transformed, "commuted"))
        self.assertIsNone(_named_op(transformed, "nested"))
        self.assertIsNone(_named_op(transformed, "subtracted"))
        self.assertEqual(_named_op(transformed, "partial_total").operands,
                         [Const(37), Const(37)])
        self.assertEqual(metrics.custom["loads_promoted"], 3)
        self.assertEqual(metrics.custom["stores_removed"], 1)

    def test_deep_address_chain_does_not_recurse(self):
        b = HIRBuilder()
        base = b.load(b.const(17), "base")
        b.assume_local_memory(base, b.const(1))
        address = base
        for i in range(1200):
            address = b.add(address, b.const(0), f"address_{i}")
        value = b.load(address, "value")
        b.store(b.const(40), value)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNone(_named_op(transformed, "value"))
        self.assertEqual(metrics.custom["loads_promoted"], 1)

    def test_dynamic_same_root_access_rejects_whole_region(self):
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 8)
        static_value = b.load(b.add(base, b.const(0)), "static_value")
        dynamic_offset = b.load(b.const(6), "dynamic_offset")
        dynamic_value = b.load(b.add(base, dynamic_offset), "dynamic_value")
        b.store(b.add(base, b.const(1)), b.const(99))
        b.store(b.const(30), b.add(static_value, dynamic_value))

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "static_value"))
        self.assertIsNotNone(_named_op(transformed, "dynamic_value"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(metrics.custom["loads_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"dynamic_address": 1}
        )

    def test_static_out_of_range_accesses_are_preserved(self):
        b = HIRBuilder()
        base = b.load(b.const(7), "base")
        _mark_local(b, base, 2)
        in_range = b.load(b.add(base, b.const(1)), "in_range")
        high = b.load(b.add(base, b.const(2)), "high")
        low_addr = b.add(base, b.const(-1), "low_addr")
        b.store(low_addr, b.const(77))
        b.store(b.const(31), b.add(in_range, high))

        transformed, metrics = _run_pass(b.build())

        self.assertIsNone(_named_op(transformed, "in_range"))
        self.assertIsNotNone(_named_op(transformed, "high"))
        self.assertTrue(
            any(op.operands[0] == low_addr for op in _ops(transformed, "store"))
        )
        self.assertEqual(metrics.custom["regions_promoted"], 1)
        self.assertEqual(metrics.custom["loads_promoted"], 1)
        self.assertEqual(metrics.custom["stores_removed"], 0)

    def test_32bit_wrapped_offset_aliases_local_slot(self):
        b = HIRBuilder()
        base = b.load(b.const(7), "base")
        b.assume_local_memory(base, b.const(2))
        wrapped = b.add(base, b.const(1 << 32), "wrapped")
        b.store(wrapped, b.const(91))
        value = b.load(base, "value")
        b.store(b.const(31), value)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNone(_named_op(transformed, "value"))
        output = next(
            op for op in _ops(transformed, "store")
            if op.operands[0] == Const(31)
        )
        self.assertEqual(output.operands[1], Const(91))
        self.assertEqual(metrics.custom["loads_promoted"], 1)
        self.assertEqual(metrics.custom["stores_removed"], 1)

    def test_promoted_constant_value_uses_32bit_word_semantics(self):
        b = HIRBuilder()
        base = b.load(b.const(7), "base")
        b.assume_local_memory(base, b.const(1))
        b.store(base, b.const(1 << 32))
        value = b.load(base, "value")
        below_one = b.lt(value, b.const(1), "below_one")
        b.store(b.const(31), below_one)

        transformed, _ = _run_pass(b.build())

        self.assertIsNone(_named_op(transformed, "value"))
        self.assertEqual(
            _named_op(transformed, "below_one").operands,
            [Const(0), Const(1)],
        )

    def test_wrapped_vector_footprint_rejects_region(self):
        b = HIRBuilder()
        base = b.load(b.const(15), "base")
        b.assume_local_memory(base, b.const(4))
        scalar = b.load(base, "scalar")
        before_base = b.add(base, b.const(0xFFFFFFFF), "before_base")
        vector = b.vload(before_base, "vector")
        b.store(b.const(50), b.add(scalar, b.vextract(vector, 1)))

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "scalar"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"vector_access": 1}
        )

    def test_access_in_retained_for_loop_rejects_region(self):
        b = HIRBuilder()
        base = b.load(b.const(9), "base")
        _mark_local(b, base, 4)
        outside = b.load(b.add(base, b.const(0)), "outside")

        def body(_counter, _params):
            inside = b.load(b.add(base, b.const(1)), "inside")
            b.store(b.const(35), inside)
            return []

        b.for_loop(b.const(0), b.const(2), [], body, pragma_unroll=1)
        b.store(b.const(36), outside)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "outside"))
        self.assertIsNotNone(_named_op(transformed, "inside"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"access_in_control_flow": 1}
        )

    def test_access_in_retained_if_rejects_region(self):
        b = HIRBuilder()
        base = b.load(b.const(9), "base")
        cond = b.load(b.const(10), "cond")
        _mark_local(b, base, 4)

        def then_body():
            return [b.load(b.add(base, b.const(0)), "then_value")]

        def else_body():
            return [b.const(7)]

        merged = b.if_stmt(cond, then_body, else_body)[0]
        b.store(b.const(36), merged)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "then_value"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"access_in_control_flow": 1}
        )

    def test_pointer_through_pre_marker_if_result_is_tainted(self):
        b = HIRBuilder()
        condition = b.load(b.const(1), "condition")
        base = b.load(b.const(10), "base")
        merged_pointer = b.if_stmt(
            condition,
            lambda: [base],
            lambda: [base],
        )[0]
        b.assume_local_memory(base, b.const(1))
        loaded = b.load(merged_pointer, "loaded_through_if")
        b.store(b.const(0), loaded)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "loaded_through_if"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"dynamic_address": 1}
        )

    def test_pointer_through_pre_marker_loop_counter_is_tainted(self):
        b = HIRBuilder()
        base = b.load(b.const(8), "base")
        end = b.add(base, b.const(1), "end")

        def body(counter, _params):
            return [counter]

        loop_result = b.for_loop(
            base,
            end,
            [b.const(0)],
            body,
            pragma_unroll=1,
        )[0]
        b.assume_local_memory(base, b.const(4))
        b.store(loop_result, b.const(42))
        loaded = b.load(base, "loaded_after_loop")
        b.store(b.const(9), loaded)

        hir = b.build()
        transformed, metrics = _run_pass(hir)

        self.assertIsNotNone(_named_op(transformed, "loaded_after_loop"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"dynamic_address": 1}
        )

        memory = [0] * 128
        memory[8] = 100
        machine = _execute(hir, memory)
        self.assertEqual(machine.mem[9], 42)

    def test_unknown_unrelated_memory_does_not_block_promotion(self):
        b = HIRBuilder()
        local_base = b.load(b.const(11), "local_base")
        other_base = b.load(b.const(12), "other_base")
        dynamic_offset = b.load(b.const(13), "dynamic_offset")
        _mark_local(b, local_base, 4)

        local_zero = b.load(b.add(local_base, b.const(0)), "local_zero")
        unknown = b.load(b.add(other_base, dynamic_offset), "unknown")
        b.store(b.add(local_base, b.const(1)), unknown)
        local_copy = b.load(b.add(local_base, b.const(1)), "local_copy")
        b.store(b.const(42), b.add(local_zero, local_copy))

        transformed, metrics = _run_pass(b.build())

        self.assertIsNone(_named_op(transformed, "local_zero"))
        self.assertIsNone(_named_op(transformed, "local_copy"))
        self.assertIsNotNone(_named_op(transformed, "unknown"))
        self.assertEqual(metrics.custom["regions_promoted"], 1)

    def test_reloaded_pointer_slot_is_not_the_original_base(self):
        b = HIRBuilder()
        base = b.load(b.const(10), "base")
        b.assume_local_memory(base, b.const(1))
        other = b.load(b.const(20), "other")
        b.store(b.const(10), other)
        reloaded = b.load(b.const(10), "reloaded")
        b.store(reloaded, b.const(7))
        local_value = b.load(base, "local_value")
        b.store(b.const(30), local_value)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNone(_named_op(transformed, "local_value"))
        unrelated_store = next(
            op for op in _ops(transformed, "store")
            if op.operands[0] == reloaded
        )
        self.assertEqual(unrelated_store.operands[1], Const(7))
        output = next(
            op for op in _ops(transformed, "store")
            if op.operands[0] == Const(30)
        )
        self.assertEqual(output.operands[1], Const(0))
        self.assertEqual(metrics.custom["loads_promoted"], 1)
        self.assertEqual(metrics.custom["stores_removed"], 0)

    def test_vector_overlap_rejects_region_atomically(self):
        b = HIRBuilder()
        base = b.load(b.const(15), "base")
        _mark_local(b, base, 16)
        scalar = b.load(b.add(base, b.const(0)), "scalar")
        vector = b.vload(b.add(base, b.const(4)), "vector")
        b.store(b.const(50), b.add(scalar, b.vextract(vector, 0)))

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "scalar"))
        self.assertIsNotNone(_named_op(transformed, "vector"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"vector_access": 1}
        )

    def test_tainted_select_rejects_region(self):
        b = HIRBuilder()
        base = b.load(b.const(17), "base")
        other = b.load(b.const(18), "other")
        cond = b.load(b.const(19), "cond")
        _mark_local(b, base, 4)
        selected = b.select(cond, base, other, "selected")
        value = b.load(selected, "selected_load")
        b.store(b.const(52), value)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "selected_load"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"tainted_select": 1}
        )

    def test_derived_pointer_store_value_rejects_region_atomically(self):
        b = HIRBuilder()
        base = b.load(b.const(17), "base")
        b.assume_local_memory(base, b.const(2))
        local_value = b.load(base, "local_value")
        derived_pointer = b.add(base, b.const(1), "derived_pointer")
        b.store(b.const(30), derived_pointer)
        b.store(b.const(31), local_value)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "local_value"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(metrics.custom["loads_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"tainted_escape": 1}
        )

    def test_marker_only_scopes_following_accesses(self):
        b = HIRBuilder()
        base = b.load(b.const(20), "base")
        before = b.load(b.add(base, b.const(0)), "before")
        _mark_local(b, base, 2)
        after = b.load(b.add(base, b.const(0)), "after")
        b.store(b.const(60), before)
        b.store(b.const(61), after)

        transformed, metrics = _run_pass(b.build())

        self.assertIsNotNone(_named_op(transformed, "before"))
        self.assertIsNone(_named_op(transformed, "after"))
        output = next(
            op for op in _ops(transformed, "store") if op.operands[0] == Const(61)
        )
        self.assertEqual(output.operands[1], Const(0))
        self.assertEqual(metrics.custom["loads_promoted"], 1)

    def test_invalid_dynamic_length_marker_is_removed_without_promotion(self):
        b = HIRBuilder()
        base = b.load(b.const(21), "base")
        length = b.load(b.const(22), "length")
        _mark_local(b, base, length)
        value = b.load(b.add(base, b.const(0)), "value")
        b.store(b.const(62), value)

        hir = b.build()
        self.assertEqual(collect_static_local_memory_regions(hir), [])
        transformed, metrics = _run_pass(hir)

        self.assertFalse(_ops(transformed, "assume_local_memory"))
        self.assertIsNotNone(_named_op(transformed, "value"))
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(metrics.custom["regions_rejected"], 1)

    def test_marker_is_ignored_when_lowering_without_promotion(self):
        b = HIRBuilder()
        base = b.load(b.const(21), "base")
        b.assume_local_memory(base, b.const(2))
        value = b.load(base, "value")
        b.store(b.const(62), value)

        lir = lower_to_lir(b.build())

        self.assertTrue(lir.blocks)



class TestRangeBoundedDynamicAccesses(unittest.TestCase):
    """Dynamic derived addresses provably outside the region do not force a
    rejection: they are ordinary accesses to other memory."""

    def test_out_of_region_dynamic_access_keeps_promotion(self):
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        b.store(b.add(base, b.const(0)), b.const(41))
        loaded = b.load(b.add(base, b.const(0)), "loaded")
        # offset in [4, 7]: entirely outside the 4-word region
        t = b.add(b.and_(b.load(b.const(6), "x"), b.const(3), "masked"),
                  b.const(4), "t")
        outside = b.load(b.add(base, t, "outside_addr"), "outside")
        b.store(b.const(30), b.add(loaded, outside))

        transformed, metrics = _run_pass(b.build())

        self.assertEqual(metrics.custom["regions_promoted"], 1)
        self.assertEqual(metrics.custom["loads_promoted"], 1)
        self.assertEqual(metrics.custom["stores_removed"], 1)
        # The out-of-region access is preserved untouched.
        self.assertIsNotNone(_named_op(transformed, "outside"))

    def test_overlapping_dynamic_range_still_rejects(self):
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        _ = b.load(b.add(base, b.const(0)), "in_range")
        # offset in [3, 6]: overlaps the region
        t = b.add(b.and_(b.load(b.const(6), "x"), b.const(3), "masked"),
                  b.const(3), "t")
        _ = b.load(b.add(base, t, "overlap_addr"), "overlap")
        b.store(b.const(30), b.const(1))

        transformed, metrics = _run_pass(b.build())

        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"dynamic_address": 1})


class TestRangeBoundedDynamicAccessShapes(unittest.TestCase):
    """Subtraction, vector footprints, and wrap edges of the out-of-region
    proof, plus an execution-level check of the promoted program."""

    def test_subtraction_out_of_region_keeps_promotion(self):
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        b.store(b.add(base, b.const(0)), b.const(41))
        loaded = b.load(b.add(base, b.const(0)), "loaded")
        # offsets 10 - [0, 3] = [7, 10]: outside the 4-word region
        t = b.and_(b.load(b.const(6), "x"), b.const(3), "masked")
        outside = b.load(
            b.alu("-", b.add(base, b.const(10), "rel"), t, "outside_addr"),
            "outside")
        b.store(b.const(30), b.add(loaded, outside))

        transformed, metrics = _run_pass(b.build())
        self.assertEqual(metrics.custom["regions_promoted"], 1)
        self.assertIsNotNone(_named_op(transformed, "outside"))

    def test_vector_footprint_outside_region_keeps_promotion(self):
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        b.store(b.add(base, b.const(0)), b.const(41))
        loaded = b.load(b.add(base, b.const(0)), "loaded")
        b.store(b.const(30), loaded)
        # vload lanes cover [4, 7+7]: entirely past the region
        t = b.add(b.and_(b.load(b.const(6), "x"), b.const(3), "masked"),
                  b.const(4), "t")
        vec = b.vload(b.add(base, t, "vaddr"), "vec")
        b.vstore(b.const(40), vec)

        transformed, metrics = _run_pass(b.build())
        self.assertEqual(metrics.custom["regions_promoted"], 1)

    def test_wrapping_dynamic_offset_still_rejects(self):
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        _ = b.load(b.add(base, b.const(0)), "in_range")
        # static offset 2**32 - 2 plus [0, 3] wraps mod 2**32: unprovable
        t = b.and_(b.load(b.const(6), "x"), b.const(3), "masked")
        rel = b.add(base, b.const((1 << 32) - 2), "rel")
        _ = b.load(b.add(rel, t, "wrap_addr"), "wrapping")
        b.store(b.const(30), b.const(1))

        transformed, metrics = _run_pass(b.build())
        self.assertEqual(metrics.custom["regions_promoted"], 0)
        self.assertEqual(
            metrics.custom["rejection_reasons"], {"dynamic_address": 1})

    def test_out_of_region_promotion_executes_correctly(self):
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        b.store(b.add(base, b.const(1)), b.const(41))
        v = b.load(b.add(base, b.const(1)), "v")
        t = b.add(b.and_(b.load(b.const(6), "x"), b.const(3), "m"),
                  b.const(4), "t")
        outside = b.load(b.add(base, t, "oaddr"), "outside")
        b.store(b.const(30), b.add(v, outside, "sum"))
        hir = b.build()

        mem = [0] * 64
        mem[5] = 16          # region [16, 20), zero-initialized
        mem[6] = 2           # t = 6 -> outside reads mem[22]
        mem[22] = 100
        instrs = compile_hir_to_vliw(hir)
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}),
                          n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        self.assertEqual(machine.mem[30], 141)
        # The promoted store never reaches memory (contract: unobservable).
        self.assertEqual(machine.mem[16:20], [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
