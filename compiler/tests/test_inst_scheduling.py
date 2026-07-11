"""Tests for instruction scheduling (LIR -> MIR)."""

import json
import os
import tempfile
import unittest

from compiler.hir import Const
from compiler.lir import LIRFunction, BasicBlock, LIRInst, LIROpcode
from compiler.mir import MachineInst
from compiler.passes import InstSchedulingPass
from compiler.passes.inst_scheduling import _parse_auto_stagger_options
from compiler.pass_manager import PassConfig
from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    HIRBuilder,
    compile_hir_to_vliw,
)


def _cfg(name, **opts):
    return PassConfig(name=name, enabled=True, options=opts)


def _schedule_single_block(instructions, **opts):
    entry = BasicBlock(
        name="entry",
        instructions=instructions,
        terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
    )
    lir = LIRFunction(entry="entry", blocks={"entry": entry})
    mir = InstSchedulingPass().run(lir, _cfg("inst-scheduling", **opts))
    return mir.blocks["entry"].bundles


def _find_bundle_index(bundles, opcode):
    for i, bundle in enumerate(bundles):
        for inst in bundle.instructions:
            if inst.opcode == opcode:
                return i
    return None


def _bundle_signature(bundles):
    sig = []
    for bundle in bundles:
        insts = []
        for inst in bundle.instructions:
            dest = tuple(inst.dest) if isinstance(inst.dest, list) else inst.dest
            ops = []
            for op in inst.operands:
                if isinstance(op, list):
                    ops.append(tuple(op))
                else:
                    ops.append(op)
            insts.append((inst.opcode.value, dest, tuple(ops), inst.engine))
        sig.append(tuple(insts))
    return tuple(sig)


class TestInstructionScheduling(unittest.TestCase):
    """Scheduling correctness and determinism tests."""

    def test_no_same_bundle_raw(self):
        const = LIRInst(LIROpcode.CONST, 0, [1], "load")
        add = LIRInst(LIROpcode.ADD, 1, [0, 0], "alu")
        bundles = _schedule_single_block([const, add])

        const_idx = _find_bundle_index(bundles, LIROpcode.CONST)
        add_idx = _find_bundle_index(bundles, LIROpcode.ADD)

        self.assertIsNotNone(const_idx)
        self.assertIsNotNone(add_idx)
        self.assertNotEqual(const_idx, add_idx, "RAW must not co-issue in same bundle")
        self.assertLess(const_idx, add_idx, "RAW consumer must be scheduled after producer")

    def test_store_then_load_separated(self):
        const_addr = LIRInst(LIROpcode.CONST, 0, [10], "load")
        const_val = LIRInst(LIROpcode.CONST, 1, [7], "load")
        store = LIRInst(LIROpcode.STORE, None, [0, 1], "store")
        load = LIRInst(LIROpcode.LOAD, 2, [0], "load")
        bundles = _schedule_single_block([const_addr, const_val, store, load])

        store_idx = _find_bundle_index(bundles, LIROpcode.STORE)
        load_idx = _find_bundle_index(bundles, LIROpcode.LOAD)

        self.assertIsNotNone(store_idx)
        self.assertIsNotNone(load_idx)
        self.assertNotEqual(store_idx, load_idx, "Store->load must not co-issue")
        self.assertLess(store_idx, load_idx, "Load must be scheduled after prior store")

    def test_load_then_store_can_coissue(self):
        const_addr = LIRInst(LIROpcode.CONST, 0, [10], "load")
        const_val = LIRInst(LIROpcode.CONST, 1, [5], "load")
        load = LIRInst(LIROpcode.LOAD, 2, [0], "load")
        store = LIRInst(LIROpcode.STORE, None, [0, 1], "store")
        bundles = _schedule_single_block([const_addr, const_val, load, store])

        load_idx = _find_bundle_index(bundles, LIROpcode.LOAD)
        store_idx = _find_bundle_index(bundles, LIROpcode.STORE)

        self.assertIsNotNone(load_idx)
        self.assertIsNotNone(store_idx)
        self.assertEqual(load_idx, store_idx, "Load->store should be able to co-issue")

    def test_wrapped_vstore_precedes_overlapping_scalar_load(self):
        const_header = LIRInst(LIROpcode.CONST, 0, [4], "load")
        forest_base = LIRInst(LIROpcode.LOAD, 1, [0], "load")
        const_one = LIRInst(LIROpcode.CONST, 2, [1], "load")
        before_base = LIRInst(LIROpcode.SUB, 3, [1, 2], "alu")
        vstore = LIRInst(
            LIROpcode.VSTORE, None, [3, list(range(100, 108))], "store"
        )
        root_load = LIRInst(LIROpcode.LOAD, 4, [1], "load")
        bundles = _schedule_single_block([
            const_header, forest_base, const_one, before_base,
            vstore, root_load,
        ])

        vstore_idx = _find_bundle_index(bundles, LIROpcode.VSTORE)
        root_load_idx = next(
            i for i, bundle in enumerate(bundles)
            if any(inst.opcode == LIROpcode.LOAD and inst.dest == 4
                   for inst in bundle.instructions)
        )
        self.assertIsNotNone(vstore_idx)
        self.assertLess(
            vstore_idx, root_load_idx,
            "vstore(base-1, width=8) overlaps load(base) and must precede it",
        )

    def test_distinct_header_roots_may_alias_without_restrict(self):
        instructions = [
            LIRInst(LIROpcode.CONST, 0, [4], "load"),
            LIRInst(LIROpcode.CONST, 1, [5], "load"),
            LIRInst(LIROpcode.LOAD, 2, [0], "load"),
            LIRInst(LIROpcode.LOAD, 3, [1], "load"),
            LIRInst(LIROpcode.CONST, 4, [11], "load"),
            LIRInst(LIROpcode.STORE, None, [2, 4], "store"),
            LIRInst(LIROpcode.LOAD, 5, [3], "load"),
        ]

        bundles = _schedule_single_block(instructions, restrict_ptr=False)
        store_index = _find_bundle_index(bundles, LIROpcode.STORE)
        load_index = next(
            i for i, bundle in enumerate(bundles)
            if any(inst.opcode == LIROpcode.LOAD and inst.dest == 5
                   for inst in bundle.instructions)
        )

        self.assertLess(
            store_index,
            load_index,
            "distinct symbolic roots may be equal at runtime without restrict",
        )

    def test_distinct_header_roots_disambiguate_with_restrict(self):
        instructions = [
            LIRInst(LIROpcode.CONST, 0, [4], "load"),
            LIRInst(LIROpcode.CONST, 1, [5], "load"),
            LIRInst(LIROpcode.LOAD, 2, [0], "load"),
            LIRInst(LIROpcode.LOAD, 3, [1], "load"),
            LIRInst(LIROpcode.CONST, 4, [11], "load"),
            LIRInst(LIROpcode.STORE, None, [2, 4], "store"),
            LIRInst(LIROpcode.LOAD, 5, [3], "load"),
        ]

        bundles = _schedule_single_block(instructions, restrict_ptr=True)
        store_index = _find_bundle_index(bundles, LIROpcode.STORE)
        load_index = next(
            i for i, bundle in enumerate(bundles)
            if any(inst.opcode == LIROpcode.LOAD and inst.dest == 5
                   for inst in bundle.instructions)
        )

        self.assertLess(
            load_index,
            store_index,
            "an explicit restrict contract should allow independent roots to reorder",
        )

    def test_deterministic_bundles(self):
        const0 = LIRInst(LIROpcode.CONST, 0, [1], "load")
        const1 = LIRInst(LIROpcode.CONST, 1, [2], "load")
        add = LIRInst(LIROpcode.ADD, 2, [0, 1], "alu")
        mul = LIRInst(LIROpcode.MUL, 3, [0, 1], "alu")

        bundles_a = _schedule_single_block([const0, const1, add, mul])
        bundles_b = _schedule_single_block([const0, const1, add, mul])

        self.assertEqual(
            _bundle_signature(bundles_a),
            _bundle_signature(bundles_b),
            "Scheduling must be deterministic",
        )

    def test_invalid_stream_stagger_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stream_stagger_threshold"):
            _schedule_single_block([], stream_stagger_threshold=0)

    def test_devectorize_valu_to_alu_when_valu_saturated(self):
        def vec(base):
            return [base + i for i in range(8)]

        instructions = [
            LIRInst(LIROpcode.VADD, vec(100 + i * 8), [vec(1000), vec(2000)], "valu")
            for i in range(7)
        ]

        bundles_no_dev = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=False,
        )
        bundles_dev = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=True,
        )

        non_term_no_dev = bundles_no_dev[:-1]
        non_term_dev = bundles_dev[:-1]
        self.assertEqual(len(non_term_no_dev), 2)
        self.assertEqual(len(non_term_dev), 1)

        first_bundle = non_term_dev[0]
        valu_count = sum(1 for inst in first_bundle.instructions if inst.engine == "valu")
        alu_count = sum(1 for inst in first_bundle.instructions if inst.engine == "alu")
        self.assertEqual(valu_count, 6)
        self.assertEqual(alu_count, 8)
        self.assertTrue(
            any(inst.opcode == LIROpcode.ADD and inst.engine == "alu" for inst in first_bundle.instructions),
            "Expected devectorized scalar ALU instructions",
        )

    def test_devectorize_can_spill_second_vector_across_bundles(self):
        def vec(base):
            return [base + i for i in range(8)]

        instructions = [
            LIRInst(LIROpcode.VADD, vec(100 + i * 8), [vec(1000), vec(2000)], "valu")
            for i in range(8)
        ]

        bundles_without_spill = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=True,
        )
        bundles_with_spill = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=True,
            devectorize_partial_alu_fill=True,
        )

        non_term_without = bundles_without_spill[:-1]
        non_term_with = bundles_with_spill[:-1]
        self.assertEqual(len(non_term_without), 2)
        self.assertEqual(len(non_term_with), 2)

        # Without partial spill, only one vector gets devectorized in first bundle.
        first_without = non_term_without[0]
        second_without = non_term_without[1]
        self.assertEqual(sum(1 for inst in first_without.instructions if inst.engine == "valu"), 6)
        self.assertEqual(sum(1 for inst in first_without.instructions if inst.engine == "alu"), 8)
        self.assertEqual(sum(1 for inst in second_without.instructions if inst.engine == "valu"), 1)

        # With partial spill, second vector starts devectorizing in bundle 1 and
        # remaining lanes spill to bundle 2.
        first_bundle = non_term_with[0]
        second_bundle = non_term_with[1]
        first_valu = sum(1 for inst in first_bundle.instructions if inst.engine == "valu")
        first_alu = sum(1 for inst in first_bundle.instructions if inst.engine == "alu")
        second_valu = sum(1 for inst in second_bundle.instructions if inst.engine == "valu")
        second_alu = sum(1 for inst in second_bundle.instructions if inst.engine == "alu")

        self.assertEqual(first_valu, 6)
        self.assertEqual(first_alu, 12, "Expected ALU bundle to be fully filled")
        self.assertEqual(second_valu, 0, "Second vector op should be continued as scalar lanes")
        self.assertEqual(second_alu, 4, "Expected remaining 4 scalar lanes in next bundle")

    def test_devectorize_valu_to_alu_skips_multiply_add(self):
        def vec(base):
            return [base + i for i in range(8)]

        instructions = [
            LIRInst(LIROpcode.VADD, vec(100 + i * 8), [vec(1000), vec(2000)], "valu")
            for i in range(6)
        ]
        instructions.append(
            LIRInst(LIROpcode.MULTIPLY_ADD, vec(200), [vec(300), vec(400), vec(500)], "valu")
        )

        bundles = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=True,
        )

        non_term = bundles[:-1]
        self.assertEqual(len(non_term), 2)
        self.assertEqual(
            sum(1 for inst in non_term[0].instructions if inst.engine == "alu"),
            0,
            "multiply_add must not be devectorized",
        )
        self.assertTrue(
            any(inst.opcode == LIROpcode.MULTIPLY_ADD for inst in non_term[1].instructions),
            "multiply_add should remain a valu instruction",
        )

    def test_devectorize_vbroadcast_knob(self):
        def vec(base):
            return [base + i for i in range(8)]

        instructions = [
            LIRInst(LIROpcode.VBROADCAST, vec(100 + i * 8), [42 + i], "valu")
            for i in range(7)
        ]

        bundles_no_vbroadcast_dev = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=True,
            devectorize_vbroadcast_to_alu=False,
        )
        bundles_with_vbroadcast_dev = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=True,
            devectorize_vbroadcast_to_alu=True,
        )

        non_term_no = bundles_no_vbroadcast_dev[:-1]
        non_term_yes = bundles_with_vbroadcast_dev[:-1]
        self.assertEqual(len(non_term_no), 2)
        self.assertEqual(len(non_term_yes), 1)
        self.assertEqual(
            sum(1 for inst in non_term_no[0].instructions if inst.engine == "alu"),
            0,
        )
        self.assertGreater(
            sum(1 for inst in non_term_yes[0].instructions if inst.engine == "alu"),
            0,
        )


class TestAutoStaggerOptions(unittest.TestCase):
    def test_parse_defaults(self):
        options = _parse_auto_stagger_options({})

        self.assertEqual(options.min_gap_pct, 25)
        self.assertEqual(options.pressure_headroom, 64)
        self.assertEqual(options.candidate_start, 1)
        self.assertEqual(options.candidate_multiplier, 2)
        self.assertIsNone(options.candidate_max)
        self.assertEqual(options.direction, "auto")

    def test_parse_all_options(self):
        options = _parse_auto_stagger_options({
            "stream_stagger_auto_min_gap_pct": 40,
            "stream_stagger_auto_pressure_headroom": 96,
            "stream_stagger_auto_candidate_start": 3,
            "stream_stagger_auto_candidate_multiplier": 3,
            "stream_stagger_auto_candidate_max": 81,
            "stream_stagger_auto_direction": "both",
        })

        self.assertEqual(options.min_gap_pct, 40)
        self.assertEqual(options.pressure_headroom, 96)
        self.assertEqual(options.candidate_start, 3)
        self.assertEqual(options.candidate_multiplier, 3)
        self.assertEqual(options.candidate_max, 81)
        self.assertEqual(options.direction, "both")

    def test_invalid_options_are_rejected(self):
        cases = [
            {"stream_stagger_auto_min_gap_pct": -1},
            {"stream_stagger_auto_pressure_headroom": 1536},
            {"stream_stagger_auto_candidate_start": 0},
            {"stream_stagger_auto_candidate_multiplier": 1},
            {
                "stream_stagger_auto_candidate_start": 4,
                "stream_stagger_auto_candidate_max": 2,
            },
            {"stream_stagger_auto_direction": "inside-out"},
        ]
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    _parse_auto_stagger_options(options)

    def test_candidate_range_controls_search_count(self):
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [1], "load"),
                LIRInst(LIROpcode.CONST, 1, [2], "load"),
                LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        scheduler = InstSchedulingPass()
        scheduler.run(
            LIRFunction(entry="entry", blocks={"entry": entry}),
            _cfg(
                "inst-scheduling",
                stream_stagger="auto",
                stream_stagger_auto_pressure_headroom=1535,
                stream_stagger_auto_candidate_start=1,
                stream_stagger_auto_candidate_multiplier=2,
                stream_stagger_auto_candidate_max=8,
                stream_stagger_auto_direction="unidirectional",
            ),
        )

        self.assertEqual(
            scheduler.get_metrics().custom["auto_stagger_candidates"],
            5,
        )


class TestSchedulerConfigKnobs(unittest.TestCase):
    """End-to-end tests for the new scheduler config knobs.

    Each test compiles a small program with pass_config overrides (tempfile
    pattern from test_regressions.TestRegisterAllocatorVectorLanes), runs it
    on the VM, and asserts both correct results and that the knob actually
    changed the emitted VLIW code.
    """

    N = 32
    BASE_SRC = 0
    BASE_DST = 64

    def _compile_with_options(self, hir, **sched_opts):
        """Compile with inst-scheduling option overrides on the default config."""
        config_path = os.path.join(os.path.dirname(__file__), "..",
                                   "pass_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["passes"]["inst-scheduling"]["options"].update(sched_opts)
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as tf:
            json.dump(cfg, tf)
            tmp_path = tf.name
        try:
            return compile_hir_to_vliw(hir, pass_config=tmp_path)
        finally:
            os.unlink(tmp_path)

    def _run_program(self, instrs, mem):
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def _build_array_hir(self):
        """32-element array computation with lots of parallelism.

        out[i] = (in[i] ^ 12345) * 5 + 999, fully unrolled so slp vectorizes
        it and the scheduler sees many independent streams.

        NOTE: passes mutate HIR in place, so callers must build a fresh HIR
        (call this again) for every compile.
        """
        b = HIRBuilder()
        bs = b.const(self.BASE_SRC)
        bd = b.const(self.BASE_DST)
        key = b.const(12345)
        mfac = b.const(5)
        addend = b.const(999)

        def body(i, params):
            sa = b.add(bs, i, "sa")
            da = b.add(bd, i, "da")
            v = b.load(sa, "v")
            t = b.xor(v, key, "t")
            t2 = b.mul(t, mfac, "t2")
            t3 = b.add(t2, addend, "t3")
            b.store(da, t3)
            return []

        b.for_loop(start=Const(0), end=Const(self.N), iter_args=[],
                   body_fn=body, pragma_unroll=0)
        return b.build()

    def _array_src_and_expected(self):
        src = [i * 17 + 3 for i in range(self.N)]
        exp = [((v ^ 12345) * 5 + 999) & 0xFFFFFFFF for v in src]
        return src, exp

    def _array_mem(self):
        src, _ = self._array_src_and_expected()
        mem = [0] * 128
        mem[self.BASE_SRC:self.BASE_SRC + self.N] = src
        return mem

    def _check_array_result(self, machine):
        _, exp = self._array_src_and_expected()
        self.assertEqual(
            list(machine.mem[self.BASE_DST:self.BASE_DST + self.N]), exp)

    @staticmethod
    def _count_engine_slots(instrs, engine):
        return sum(len(bundle.get(engine, [])) for bundle in instrs)

    @staticmethod
    def _count_opcode_slots(instrs, engine, opcode):
        return sum(1 for bundle in instrs
                   for slot in bundle.get(engine, []) if slot[0] == opcode)

    def test_stream_stagger_deterministic(self):
        """stream_stagger=6 must be deterministic and actually change the
        schedule relative to stream_stagger=0 (correctness of staggered
        schedules is covered by the regalloc regression test)."""
        instrs_a = self._compile_with_options(
            self._build_array_hir(),
            stream_stagger=6, stream_stagger_bidirectional=False)
        instrs_b = self._compile_with_options(
            self._build_array_hir(),
            stream_stagger=6, stream_stagger_bidirectional=False)
        self.assertEqual(instrs_a, instrs_b,
                         "Compiling twice with stream_stagger=6 must yield "
                         "identical instructions")

        instrs_zero = self._compile_with_options(
            self._build_array_hir(),
            stream_stagger=0, stream_stagger_bidirectional=False)
        self.assertNotEqual(instrs_a, instrs_zero,
                            "stream_stagger=6 should produce a different "
                            "schedule than stream_stagger=0")

        machine = self._run_program(instrs_a, self._array_mem())
        self._check_array_result(machine)

    def test_stream_stagger_bidirectional_correctness(self):
        """stream_stagger_bidirectional=True must produce a distinct, correct
        schedule."""
        instrs_bi = self._compile_with_options(
            self._build_array_hir(),
            stream_stagger=6, stream_stagger_bidirectional=True)
        instrs_uni = self._compile_with_options(
            self._build_array_hir(),
            stream_stagger=6, stream_stagger_bidirectional=False)
        self.assertNotEqual(instrs_bi, instrs_uni,
                            "Bidirectional stagger should change the schedule")

        machine = self._run_program(instrs_bi, self._array_mem())
        self._check_array_result(machine)

    def test_auto_stagger_keeps_efficient_natural_schedule(self):
        """Auto mode must not impose staggering on a block whose natural
        schedule is already close to its resource lower bound."""
        instrs_auto = self._compile_with_options(
            self._build_array_hir(), stream_stagger="auto")
        instrs_zero = self._compile_with_options(
            self._build_array_hir(), stream_stagger=0,
            stream_stagger_bidirectional=False)

        self.assertEqual(
            instrs_auto,
            instrs_zero,
            "auto staggering should retain the shorter natural schedule",
        )
        machine = self._run_program(instrs_auto, self._array_mem())
        self._check_array_result(machine)

    def test_static_devectorize_pct_shifts_valu_work_to_alu(self):
        """static_devectorize_pct=50 pre-expands half the eligible valu ops
        to scalar alu ops: more alu slots, fewer valu slots than pct=0, and
        the program still computes correct results."""
        instrs_pct0 = self._compile_with_options(
            self._build_array_hir(), static_devectorize_pct=0)
        instrs_pct50 = self._compile_with_options(
            self._build_array_hir(), static_devectorize_pct=50)

        valu_pct0 = self._count_engine_slots(instrs_pct0, "valu")
        self.assertGreater(valu_pct0, 0,
                           "Test program should contain slp-vectorized valu "
                           "ops at pct=0")

        alu_pct50 = self._count_engine_slots(instrs_pct50, "alu")
        alu_pct0 = self._count_engine_slots(instrs_pct0, "alu")
        valu_pct50 = self._count_engine_slots(instrs_pct50, "valu")
        self.assertGreater(alu_pct50, alu_pct0,
                           "pct=50 should emit more scalar alu slots")
        self.assertLess(valu_pct50, valu_pct0,
                        "pct=50 should emit fewer valu slots")

        machine = self._run_program(instrs_pct50, self._array_mem())
        self._check_array_result(machine)

    def test_const_via_flow_materializes_consts_on_flow_engine(self):
        """const_via_flow=True (skip=0) converts non-anchor consts into
        add_imm ops on the flow engine, reducing load-engine const slots,
        without changing program results."""

        def build_const_hir():
            # Several distinct constants; stride-3 addresses avoid vstore
            # vectorization so the consts stay scalar.
            b = HIRBuilder()
            for i in range(10):
                addr = b.const(i * 3)
                val = b.const(i * 7 + 100)
                b.store(addr, val)
            return b.build()

        instrs_flow = self._compile_with_options(
            build_const_hir(), const_via_flow=True, const_via_flow_skip=0)
        instrs_load = self._compile_with_options(
            build_const_hir(), const_via_flow=False)

        n_add_imm = self._count_opcode_slots(instrs_flow, "flow", "add_imm")
        self.assertGreater(n_add_imm, 0,
                           "const_via_flow=True should emit ('add_imm', ...) "
                           "slots on the flow engine")
        self.assertEqual(
            self._count_opcode_slots(instrs_load, "flow", "add_imm"), 0,
            "const_via_flow=False should not emit add_imm slots")

        n_const_flow = self._count_opcode_slots(instrs_flow, "load", "const")
        n_const_load = self._count_opcode_slots(instrs_load, "load", "const")
        self.assertLess(n_const_flow, n_const_load,
                        "const_via_flow=True should emit fewer load-engine "
                        "const slots")

        machine = self._run_program(instrs_flow, [0] * 64)
        for i in range(10):
            self.assertEqual(machine.mem[i * 3], i * 7 + 100,
                             f"mem[{i * 3}] should hold const {i * 7 + 100}")


class TestAddImmDefUse(unittest.TestCase):
    """Unit tests for the LIROpcode.ADD_IMM def/use handling."""

    def test_add_imm_immediate_not_treated_as_register(self):
        """ADD_IMM's second operand is an immediate: get_uses() must only
        return the scratch source, and get_defs() the destination."""
        inst = MachineInst(opcode=LIROpcode.ADD_IMM, dest=5,
                           operands=[3, 1234], engine="flow")
        self.assertEqual(inst.get_uses(), {3},
                         "ADD_IMM immediate must not be treated as a register")
        self.assertEqual(inst.get_defs(), {5})

    def test_add_imm_lir_inst_uses_match(self):
        """LIRInst shares the def/use mixin: same semantics for ADD_IMM."""
        inst = LIRInst(LIROpcode.ADD_IMM, 7, [2, 99999], "flow")
        self.assertEqual(inst.get_uses(), {2})
        self.assertEqual(inst.get_defs(), {7})


def _execute_default(hir, mem):
    instrs = compile_hir_to_vliw(hir)
    machine = Machine(list(mem), instrs, DebugInfo(scratch_map={}),
                      n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine


class TestSchedulerAliasRegressions(unittest.TestCase):
    """Two confirmed miscompiles in the scheduler's value tracking."""

    def test_loop_carried_offset_not_assumed_invariant(self):
        """Regression: single-pass RPO ignored back edges, so a loop-carried
        offset inherited the preheader constant in EVERY iteration; the
        in-loop load was then reordered/co-issued across the second
        iteration's conflicting store (read 3 instead of 2)."""
        b = HIRBuilder()
        ptr = b.load(b.const(4), "ptr")

        def body(r, params):
            off, acc = params
            b.store(b.add(ptr, off), b.const(2))
            v = b.load(b.add(ptr, b.const(1)), "v")
            return [b.add(off, b.const(1)), b.add(acc, v)]

        res = b.for_loop(b.const(0), b.load(b.const(0), "n"),
                         [b.const(0), b.const(0)], body)
        b.store(b.const(30), res[1])

        mem = [0] * 64
        mem[0] = 2
        mem[4] = 16
        mem[16] = 3
        mem[17] = 3
        machine = _execute_default(b.build(), mem)
        # iter 1: store mem[16], load mem[17] -> 3
        # iter 2: store mem[17]=2, THEN load mem[17] -> must be 2
        self.assertEqual(machine.mem[30], 5)

    def test_reloaded_pointer_slot_gets_fresh_identity(self):
        """Regression: every load of one memory slot shared one address
        base, so offsets from two DIFFERENT pointer values (the slot was
        overwritten in between) were compared as if from one pointer and
        a true store->load conflict was scheduled out of order."""
        b = HIRBuilder()
        p = b.load(b.const(4), "p")
        w = b.load(b.const(21), "w")     # runtime 4: overwrites the slot
        b.store(w, b.const(9))           # unknown address: defeats forwarding
        q = b.load(b.const(4), "q")      # genuine reload -> 9
        val = b.load(b.const(20), "seed")
        for k in range(10):              # delay the store's value operand
            val = b.alu("^", b.mul(val, b.const(5)), b.const(1 + k))
        b.store(p, val)                  # mem[10] = val
        x = b.load(b.add(q, b.const(1)), "x")   # 9 + 1 = 10: must see val
        b.store(b.const(30), x)

        mem = [0] * 64
        mem[4] = 10
        mem[10] = 5
        mem[20] = 7
        mem[21] = 4
        machine = _execute_default(b.build(), mem)
        expected = 7
        for k in range(10):
            expected = ((expected * 5) ^ (1 + k)) & 0xFFFFFFFF
        self.assertEqual(machine.mem[30], expected)


class TestStreamStaggerEffect(unittest.TestCase):
    """stream_stagger on its canonical shape: independent lanes with a
    compute-only era (cached rounds) followed by a load-bound era (gather
    rounds). In lockstep the load engine idles through era one and the alu
    drains through era two; staggering overlaps the eras.

    The shape encodes two structural preconditions discovered the hard way:
    lanes must be emitted round-major (final stores would otherwise chain
    conservative may-alias edges lane-to-lane and glue the components), and
    lane addresses must be rooted (loads from constant slots)."""

    LANES = 24

    def _build(self):
        b = HIRBuilder()
        base = b.load(b.const(4), "base")
        out = b.load(b.const(6), "out")
        t = [b.load(b.add(base, b.const(100 + i)), "seed%d" % i)
             for i in range(self.LANES)]
        for k in range(12):              # era 1: no loads at all
            for i in range(self.LANES):
                t[i] = b.alu("^", b.mul(t[i], b.const(5)),
                             b.const(0x9E37 + k))
        for r in range(4):               # era 2: load-bound, thin compute
            for i in range(self.LANES):
                idx = b.and_(t[i], b.const(63))
                v1 = b.load(b.add(base, idx))
                v2 = b.load(b.add(b.add(base, b.const(64)), idx))
                t[i] = b.alu("^", t[i], b.alu("^", v1, v2))
        for i in range(self.LANES):
            b.store(b.add(out, b.const(i)), t[i])
        return b.build()

    def _run(self, stagger, bidirectional=False, **scheduler_options):
        default_path = os.path.join(os.path.dirname(__file__), "..",
                                    "pass_config.json")
        with open(default_path) as f:
            config = json.load(f)
        config["passes"]["slp-vectorization"] = {"enabled": False, "options": {}}
        options = config["passes"]["inst-scheduling"]["options"]
        options["stream_stagger"] = stagger
        options["stream_stagger_bidirectional"] = bidirectional
        options.update(scheduler_options)
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as f:
            json.dump(config, f)
            config_path = f.name
        try:
            instrs = compile_hir_to_vliw(self._build(),
                                         pass_config=config_path)
        finally:
            os.unlink(config_path)
        mem = [0] * 512
        mem[4] = 16
        mem[6] = 300
        for j in range(128):
            mem[16 + j] = 7919 * j % 65536
        for i in range(self.LANES):
            mem[116 + i] = 12345 + i
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}),
                          n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine.cycle, list(machine.mem[300:300 + self.LANES])

    def test_stagger_overlaps_eras_and_preserves_semantics(self):
        base_cycles, base_out = self._run(stagger=0)
        stag_cycles, stag_out = self._run(stagger=2)
        self.assertEqual(base_out, stag_out,
                         "staggering must not change program results")
        self.assertLess(stag_cycles, base_cycles,
                        "staggering the lanes must overlap the load-idle "
                        "compute era with the load-bound gather era")

    def test_auto_stagger_finds_profitable_wavefront(self):
        base_cycles, base_out = self._run(stagger=0)
        auto_cycles, auto_out = self._run(stagger="auto")
        tuned_cycles, tuned_out = self._run(stagger=2, bidirectional=True)

        self.assertEqual(auto_out, base_out)
        self.assertEqual(auto_out, tuned_out)
        self.assertLess(auto_cycles, base_cycles)
        self.assertEqual(
            auto_cycles,
            tuned_cycles,
            "auto mode should find the profitable stagger strength",
        )

    def test_auto_stagger_search_controls_are_effective(self):
        base_cycles, base_out = self._run(stagger=0)
        limited_cycles, limited_out = self._run(
            stagger="auto",
            stream_stagger_auto_candidate_max=1,
        )
        disabled_cycles, disabled_out = self._run(
            stagger="auto",
            stream_stagger_auto_min_gap_pct=1000,
            stream_stagger_auto_pressure_headroom=0,
        )

        self.assertEqual(limited_out, base_out)
        self.assertEqual(disabled_out, base_out)
        self.assertGreater(limited_cycles, 135)
        self.assertEqual(disabled_cycles, base_cycles)


if __name__ == "__main__":
    unittest.main()
