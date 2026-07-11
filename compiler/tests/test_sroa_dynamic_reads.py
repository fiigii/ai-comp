"""Tests for the SROA pass's unified promotion (local x dynamic reads)."""

import unittest

from compiler.hir import Const, Op
from compiler.pass_manager import PassConfig
from compiler.passes.sroa import SROAPass
from compiler.tests.conftest import (
    DebugInfo,
    HIRBuilder,
    Machine,
    N_CORES,
    compile_hir_to_vliw,
)


def _run_sroa(hir, **extra):
    options = {"table_promotion": True, "restrict_ptr": True,
               "max_window": 8, "share_window": 4, "repreload_gap": 10000}
    options.update(extra)
    sroa = SROAPass()
    transformed = sroa.run(hir, PassConfig(name="sroa", enabled=True,
                                           options=options))
    return transformed, sroa.get_metrics().custom


def _execute(hir, mem):
    machine = Machine(list(mem), compile_hir_to_vliw(hir),
                      DebugInfo(scratch_map={}), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine


def _mark_local(b, base, length):
    b.assume_local_memory(base, b.const(length))


class TestLocalDynamicReads(unittest.TestCase):
    """The dynamic-read quadrant: a bounded dynamic load from a LOCAL
    region selects over the tracked state snapshot -- no memory access,
    no preloads, and the region (stores included) still promotes."""

    def _build(self):
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        # Fill slots 0..3 with distinct values derived from an input.
        seed = b.load(b.const(6), "seed")
        for j in range(4):
            b.store(b.add(base, b.const(j)),
                    b.add(seed, b.const(10 * j), "v%d" % j))
        # Dynamic read: index in [0, 3], inside the region.
        raw = b.load(b.const(7), "raw")
        bit0 = b.and_(raw, b.const(1), "bit0")
        bit1 = b.and_(b.shr(raw, b.const(1), "sh"), b.const(1), "bit1")
        index = b.add(bit0, b.mul(bit1, b.const(2), "b2"), "index")
        picked = b.load(b.add(base, index, "daddr"), "picked")
        b.store(b.const(30), picked)
        return b.build()

    def test_promotes_region_and_selects_over_state(self):
        transformed, stats = _run_sroa(self._build())
        self.assertEqual(stats["regions_promoted"], 1)
        self.assertEqual(stats["stores_removed"], 4)
        self.assertEqual(stats["dynamic_loads_promoted"], 1)
        # No loads or stores against the region survive; the dynamic read
        # became selects over state values.
        selects = sum(1 for s in transformed.body
                      if isinstance(s, Op) and s.opcode == "select")
        self.assertEqual(selects, 3)

    def test_executes_correctly_for_every_index(self):
        for index_value in range(4):
            mem = [0] * 64
            mem[5] = 40          # region base: must stay untouched
            mem[6] = 700         # seed
            mem[7] = index_value
            machine = _execute(self._build(), mem)
            self.assertEqual(machine.mem[30], 700 + 10 * index_value,
                             f"index {index_value}")
            # Promoted stores never reach memory.
            self.assertEqual(machine.mem[40:44], [0, 0, 0, 0])

    def test_read_of_partially_unwritten_state_is_zero(self):
        # Slots 2..3 are never stored: zero-initialized by contract.
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        b.store(b.add(base, b.const(0)), b.const(111))
        raw = b.load(b.const(7), "raw")
        index = b.and_(raw, b.const(3), "index")
        picked = b.load(b.add(base, index, "daddr"), "picked")
        b.store(b.const(30), picked)
        hir = b.build()

        _, stats = _run_sroa(hir)
        self.assertEqual(stats["regions_promoted"], 1)
        self.assertEqual(stats["dynamic_loads_promoted"], 1)

        mem = [0] * 64
        mem[5] = 40
        mem[7] = 2               # unwritten slot -> contract zero
        machine = _execute(hir, mem)
        self.assertEqual(machine.mem[30], 0)
        mem[7] = 0               # written slot
        machine = _execute(hir, mem)
        self.assertEqual(machine.mem[30], 111)

    def test_dynamic_read_sees_state_at_its_program_point(self):
        # The select leaves are the state AT THE READ, not the final state.
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 2)
        b.store(b.add(base, b.const(0)), b.const(1))
        b.store(b.add(base, b.const(1)), b.const(2))
        raw = b.load(b.const(7), "raw")
        index = b.and_(raw, b.const(1), "index")
        early = b.load(b.add(base, index, "daddr"), "early")
        b.store(b.add(base, b.const(0)), b.const(99))   # later overwrite
        late = b.load(b.add(base, index, "daddr2"), "late")
        b.store(b.const(30), early)
        b.store(b.const(31), late)
        hir = b.build()

        _, stats = _run_sroa(hir)
        self.assertEqual(stats["regions_promoted"], 1)
        self.assertEqual(stats["dynamic_loads_promoted"], 2)

        mem = [0] * 64
        mem[5] = 40
        mem[7] = 0
        machine = _execute(hir, mem)
        self.assertEqual(machine.mem[30], 1)     # before the overwrite
        self.assertEqual(machine.mem[31], 99)    # after the overwrite

    def test_wide_dynamic_read_still_rejects_region(self):
        # Window wider than max_window: no select explosion; the region
        # conservatively stays in memory (same behavior as before).
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 32)
        b.store(b.add(base, b.const(0)), b.const(1))
        raw = b.load(b.const(7), "raw")
        index = b.and_(raw, b.const(31), "index")   # [0, 31] > max_window
        picked = b.load(b.add(base, index, "daddr"), "picked")
        b.store(b.const(30), picked)

        _, stats = _run_sroa(b.build())
        self.assertEqual(stats["regions_promoted"], 0)
        self.assertEqual(
            stats["rejection_reasons"], {"dynamic_address": 1})

    def test_dynamic_store_still_rejects_region(self):
        # Select scatter is not implemented: dynamic stores disqualify.
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        _mark_local(b, base, 4)
        raw = b.load(b.const(7), "raw")
        index = b.and_(raw, b.const(3), "index")
        b.store(b.add(base, index, "daddr"), b.const(1))
        _ = b.load(b.add(base, b.const(0)), "read")
        b.store(b.const(30), b.const(0))

        _, stats = _run_sroa(b.build())
        self.assertEqual(stats["regions_promoted"], 0)
        self.assertEqual(
            stats["rejection_reasons"], {"dynamic_address": 1})


if __name__ == "__main__":
    unittest.main()
