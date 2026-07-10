"""Tests for the straight-line strength reduction pass (slsr.py).

The pass rewrites per-round index-update chains

    off  = bit + 1
    idx' = idx*2 + off
    addr = base + idx'

to track j = idx + 1 instead of idx, rewriting address adds to
(base - 1) + j. These tests cover:

1. The end-to-end feature on a mini tree traversal (match_link, parity /
   mul-distributed roots, address rewrite) plus the links_rewritten
   metric when the pass is run standalone after the pipeline prefix.
2. Regression: the hoisted slsr_comp = base - 1 op must be inserted AFTER
   the base pointer's definition when the base is defined after the chain
   link (use-before-def bug: the program stored 0 instead of the loaded
   value).
3. Regression: s = y1 + y2 where both operands are chain-link results.
   The first link rewrite replaces y1 in s; the second rewrite must
   re-check the CURRENT operands of s (stale use records) or s is
   corrupted (off by one).
"""

import json
import unittest

from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    HIRBuilder,
    compile_hir_to_vliw,
)
from compiler.compile import PASS_REGISTRY
from compiler.pass_manager import PassConfig
from compiler.hir import Op, Const


def _default_config():
    """Load the default pass_config.json used by compile_hir_to_vliw."""
    import os
    import compiler
    path = os.path.join(os.path.dirname(compiler.__file__), "pass_config.json")
    with open(path) as f:
        return json.load(f)


class TestSLSRPass(unittest.TestCase):
    """Tests for the slsr j = idx + 1 strength reduction pass."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def _run_pipeline_to_slsr(self, hir, slsr_options=None):
        """Run the default pipeline prefix up to slsr, then run the
        SLSRPass standalone so its metrics can be inspected.

        Single-lane toy programs have no cross-lane amortization of the
        hoisted compensation ops, so the default cost model correctly
        declines them; tests that assert the REWRITE behavior pass
        slsr_options={"min_savings": 0}.

        Returns (post_pass_hir, slsr_pass_instance).
        """
        config_data = _default_config()
        passes_cfg = config_data.get("passes", {})
        pipeline_order = config_data["pipeline"]
        self.assertIn("slsr", pipeline_order)
        prefix = pipeline_order[:pipeline_order.index("slsr")]
        for name in prefix:
            cfg_data = passes_cfg.get(name, {})
            cfg = PassConfig(
                name=name,
                enabled=cfg_data.get("enabled", True),
                options=cfg_data.get("options", {}),
            )
            hir = PASS_REGISTRY[name]().run(hir, cfg)
        ac = PASS_REGISTRY["slsr"]()
        cfg_data = passes_cfg.get("slsr", {})
        options = dict(cfg_data.get("options", {}))
        if slsr_options:
            options.update(slsr_options)
        cfg = PassConfig(
            name="slsr",
            enabled=cfg_data.get("enabled", True),
            options=options,
        )
        hir = ac.run(hir, cfg)
        return hir, ac

    # ------------------------------------------------------------------
    # 1. Feature end-to-end: mini traversal
    # ------------------------------------------------------------------

    def _build_traversal(self):
        """4-round unrolled traversal from idx = 0.

        Round r: bit = load(r) & 1; off = bit + 1; idx' = idx*2 + off;
                 v = load(base + idx'); store(8 + r, v).

        Round 0 constant-folds to the parity root idx1 = bit0 + 1, round 1
        picks up the mul-distributed root +(*(bit0, 2), 2) from simplify's
        mul_dist, and rounds 2-3 chain through j_of.
        """
        b = HIRBuilder()
        base = b.load(b.const(7), "base")
        idx = b.const(0)
        for r in range(4):
            raw = b.load(b.const(r), "raw%d" % r)
            bit = b.and_(raw, b.const(1), "bit%d" % r)
            off = b.add(bit, b.const(1), "off%d" % r)
            dbl = b.mul(idx, b.const(2), "dbl%d" % r)
            idx = b.add(dbl, off, "idx%d" % r)
            addr = b.add(base, idx, "addr%d" % r)
            v = b.load(addr, "v%d" % r)
            b.store(b.const(8 + r), v)
        b.halt()
        return b.build()

    def _traversal_mem(self):
        mem = [0] * 64
        mem[0], mem[1], mem[2], mem[3] = 5, 2, 7, 4  # bits 1, 0, 1, 0
        mem[7] = 16  # base pointer to the node table
        for i in range(32):
            mem[16 + i] = 100 + 3 * i
        return mem

    def _build_two_lane_traversal(self):
        """Two parallel 4-round lanes sharing the base pointer: the hoisted
        compensation ops (base, k) are shared across lanes, so the default
        cost model approves the rewrite (unlike the single-lane toy)."""
        b = HIRBuilder()
        base = b.load(b.const(7), "base")
        for lane in range(2):
            idx = b.const(0)
            for r in range(4):
                raw = b.load(b.const(lane * 4 + r), "raw%d_%d" % (lane, r))
                bit = b.and_(raw, b.const(1), "bit%d_%d" % (lane, r))
                off = b.add(bit, b.const(1), "off%d_%d" % (lane, r))
                dbl = b.mul(idx, b.const(2), "dbl%d_%d" % (lane, r))
                idx = b.add(dbl, off, "idx%d_%d" % (lane, r))
                addr = b.add(base, idx, "addr%d_%d" % (lane, r))
                v = b.load(addr, "v%d_%d" % (lane, r))
                b.store(b.const(40 + lane * 4 + r), v)
        b.halt()
        return b.build()

    def test_j_chain_traversal_end_to_end(self):
        """Traversal values must match a Python model under default config."""
        mem = self._traversal_mem()

        # Python model of the same traversal
        expected = []
        idx = 0
        for r in range(4):
            bit = mem[r] & 1
            idx = 2 * idx + bit + 1
            expected.append(mem[mem[7] + idx])

        instrs = compile_hir_to_vliw(self._build_traversal())
        machine = self._run_program(instrs, list(mem))
        got = [machine.mem[8 + r] for r in range(4)]
        self.assertEqual(got, expected,
                         "traversal loads must match the Python model")

    def test_j_chain_traversal_rewrites_links(self):
        """With two lanes sharing the base pointer the default cost model
        approves the chains; links must be rewritten and slsr_val chain ops
        must appear in the body."""
        hir, ac = self._run_pipeline_to_slsr(self._build_two_lane_traversal())

        metrics = ac.get_metrics()
        self.assertIsNotNone(metrics)
        rewritten = metrics.custom.get("links_rewritten", 0)
        self.assertGreater(rewritten, 0,
                           "slsr did not rewrite any links")
        self.assertGreaterEqual(metrics.custom.get("chains", 0), 2,
                                "both lanes should form approved chains")

        names = set()
        for stmt in hir.body:
            if isinstance(stmt, Op) and stmt.result is not None and stmt.result.name:
                names.add(stmt.result.name)
        self.assertTrue(any(n.startswith("slsr_") for n in names),
                        "rewritten body must contain slsr chain ops")

    # ------------------------------------------------------------------
    # 2. Regression: base pointer defined AFTER the chain link
    # ------------------------------------------------------------------

    def _build_late_base(self):
        """Chain computed first, base pointer loaded afterwards.

        idx1 = (load(0) & 1) + 1 folds to the parity root; the round-1 link
        becomes +(+(*(bit0, 2), 2), off1) via mul_dist. Only THEN is the
        base pointer p = load(2) defined, followed by
        addr = p + idx2; v = load(addr); store(3, v).
        """
        b = HIRBuilder()
        idx = b.const(0)
        for r in range(2):
            raw = b.load(b.const(r), "raw%d" % r)
            bit = b.and_(raw, b.const(1), "bit%d" % r)
            off = b.add(bit, b.const(1), "off%d" % r)
            dbl = b.mul(idx, b.const(2), "dbl%d" % r)
            idx = b.add(dbl, off, "idx%d" % r)
        p = b.load(b.const(2), "base_p")
        addr = b.add(p, idx, "addr")
        v = b.load(addr, "v")
        b.store(b.const(3), v)
        b.halt()
        return b.build()

    def test_base_defined_after_chain_link_value(self):
        """slsr_comp hoisting must not read the base before it is loaded.

        bits (1, 0) -> idx2 = 2*(2*0 + 1 + 1) + 0 + 1 = 5, so the load hits
        mem[20 + 5] = 777. Before the fix, slsr_comp = p - 1 was inserted at
        the link position (before p's def), read an unwritten register, and
        the program stored 0.
        """
        mem = [0] * 64
        mem[0] = 1   # bit 1 -> idx1 = 2
        mem[1] = 0   # bit 0 -> idx2 = 5
        mem[2] = 20  # base pointer, defined after the chain in the HIR
        mem[25] = 777

        instrs = compile_hir_to_vliw(self._build_late_base())
        machine = self._run_program(instrs, list(mem))
        self.assertEqual(machine.mem[3], 777,
                         "store must see the value loaded through base + idx"
                         " (0 means slsr_comp was emitted before the base def)")

    def test_base_defined_after_chain_link_insertion_position(self):
        """No op inserted by SLSR may read a value before its definition.

        The original bug hoisted a compensation op above the (late) base
        pointer load. The general invariant: in the post-pass body every
        SSA operand of every op must be defined at an earlier position.
        """
        hir, ac = self._run_pipeline_to_slsr(self._build_late_base(),
                                             slsr_options={'min_savings': 0})
        metrics = ac.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.custom.get("links_rewritten", 0), 0,
                           "slsr did not fire on the late-base shape")

        def_pos = {}
        for i, stmt in enumerate(hir.body):
            if isinstance(stmt, Op) and stmt.result is not None:
                def_pos[stmt.result] = i
        for i, stmt in enumerate(hir.body):
            if not isinstance(stmt, Op):
                continue
            for o in stmt.operands:
                if o in def_pos:
                    self.assertLess(def_pos[o], i,
                                    "use-before-def introduced by slsr: "
                                    "%s at %d uses %s defined at %d"
                                    % (stmt, i, o, def_pos[o]))

    # ------------------------------------------------------------------
    # 3. Regression: add of two chain-link results
    # ------------------------------------------------------------------

    def _build_two_link_add(self):
        """s = y1 + y2 where y1, y2 are independent 1-round chain links.

        Each chain: root idxc1 = bit + 1 (folded from idx0 = 0), then
        yc = idxc1*2 + (bit' + 1). Both links treat the other operand of s
        as an "address base", so both get rewritten into s.
        """
        b = HIRBuilder()
        ys = []
        for c in range(2):
            idx = b.const(0)
            for r in range(2):
                slot = 2 * c + r
                raw = b.load(b.const(slot), "raw_%d_%d" % (c, r))
                bit = b.and_(raw, b.const(1), "bit_%d_%d" % (c, r))
                off = b.add(bit, b.const(1), "off_%d_%d" % (c, r))
                dbl = b.mul(idx, b.const(2), "dbl_%d_%d" % (c, r))
                idx = b.add(dbl, off, "idx_%d_%d" % (c, r))
            ys.append(idx)
        s = b.add(ys[0], ys[1], "s")
        b.store(b.const(4), s)
        b.halt()
        return b.build()

    def test_two_link_add_value(self):
        """Rewriting both operands of s = y1 + y2 must keep s correct.

        bits (1, 0) -> y1 = 5; bits (1, 1) -> y2 = 6; s = 11. Before the
        stale-use re-check fix, the second link rewrite hit the already
        rewritten add and produced s = y1 + y2 + 1 = 12.
        """
        mem = [0] * 64
        mem[0], mem[1] = 1, 0  # chain A: idx1 = 2, y1 = 5
        mem[2], mem[3] = 1, 1  # chain B: idx1 = 2, y2 = 6

        instrs = compile_hir_to_vliw(self._build_two_link_add())
        machine = self._run_program(instrs, list(mem))
        self.assertEqual(machine.mem[4], 11,
                         "s = y1 + y2 corrupted by the second link rewrite")

    def _build_two_link_addr_uses(self):
        """Two independent chains whose values feed plain address adds.

        Unlike _build_two_link_add, the compensable use's other operand is
        an unrelated base value, so the chains do not compensate into each
        other. (Chains that compensate into EACH OTHER's values may be
        legitimately declined by the dead-op cost model: keeping the other
        chain's value alive for the compensation op collapses the dying
        set. The value test above stays correct either way.)"""
        b = HIRBuilder()
        base = b.load(b.const(5), "base")
        for c in range(2):
            idx = b.const(0)
            for r in range(2):
                slot = 2 * c + r
                raw = b.load(b.const(slot), "raw_%d_%d" % (c, r))
                bit = b.and_(raw, b.const(1), "bit_%d_%d" % (c, r))
                off = b.add(bit, b.const(1), "off_%d_%d" % (c, r))
                dbl = b.mul(idx, b.const(2), "dbl_%d_%d" % (c, r))
                idx = b.add(dbl, off, "idx_%d_%d" % (c, r))
            addr = b.add(base, idx, "addr_%d" % c)
            b.store(addr, b.const(7 + c))
        b.halt()
        return b.build()

    def test_two_link_add_pass_fires(self):
        """Both chains must be recognized and rewritten when their uses are
        plain address adds."""
        hir, ac = self._run_pipeline_to_slsr(self._build_two_link_addr_uses(), slsr_options={'min_savings': 0})
        metrics = ac.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.custom.get("links_rewritten", 0), 2,
                         "both chain links should be rewritten")




class TestSLSRCostModel(unittest.TestCase):
    """The Level-2 cost model must decline unprofitable rewrites."""

    def test_single_lane_chain_rejected_at_default_threshold(self):
        """A single-lane chain where every link needs its own hoisted
        compensation op has zero net savings; the default min_savings=1
        must reject it (links_found > 0 but links_rewritten == 0)."""
        base = TestSLSRPass()
        hir, ac = base._run_pipeline_to_slsr(base._build_traversal())
        metrics = ac.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.custom.get("links_found", 0), 0,
                           "chain links should still be discovered")
        self.assertEqual(metrics.custom.get("links_rewritten", 0), 0,
                         "unprofitable single-lane chain must not be rewritten")




class TestSLSRGeneralizedChains(unittest.TestCase):
    """The generalized matcher handles multipliers other than 2 and negative
    step signs, and the rewrites must ACTUALLY fire (asserted on metrics)."""

    def _run(self, hir, mem):
        instrs = compile_hir_to_vliw(hir)
        m = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        m.enable_pause = False
        m.enable_debug = False
        m.run()
        return m

    def _metrics_for(self, hir):
        base = TestSLSRPass()
        _, ac = base._run_pipeline_to_slsr(hir)
        return ac.get_metrics().custom

    def _build_addressed(self, a, c, lanes=2, rounds=3):
        """lanes x rounds chains x' = a*x + (v + c) with DIRECT base + x
        addressing (compensable '+' uses), values bounded so addresses stay
        inside the table."""
        b = HIRBuilder()
        base = b.load(b.const(7), "base")
        for lane in range(lanes):
            x = b.load(b.const(8 + lane), "x0_%d" % lane)
            for r in range(rounds):
                v = b.alu("&", b.load(b.const(10 + lane * rounds + r),
                                      "ld%d_%d" % (lane, r)), Const(1),
                          "v%d_%d" % (lane, r))
                off = b.add(v, Const(c), "off%d_%d" % (lane, r))
                m = b.mul(x, Const(a), "m%d_%d" % (lane, r))
                x = b.add(m, off, "x%d_%d" % (lane, r + 1))
                addr = b.add(base, x, "a%d_%d" % (lane, r))
                val = b.load(addr, "n%d_%d" % (lane, r))
                b.store(b.const(40 + lane * rounds + r), val)
        b.halt()
        return b.build()

    def _check_addressed(self, a, c, lanes=2, rounds=3):
        mem = [0] * 768
        mem[7] = 256                        # table base
        mem[8], mem[9] = 1, 2               # lane starting values
        for i in range(8):
            mem[10 + i] = (i * 5 + 3) & 0xFF
        for i in range(400):
            mem[256 + i] = 900 + i
        expected = {}
        for lane in range(lanes):
            x = mem[8 + lane]
            for r in range(rounds):
                v = mem[10 + lane * rounds + r] & 1
                x = (a * x + v + c) & 0xFFFFFFFF
                self.assertLess(x, 400, "test construction: address in table")
                expected[40 + lane * rounds + r] = mem[256 + x]

        metrics = self._metrics_for(self._build_addressed(a, c, lanes, rounds))
        self.assertGreater(metrics.get("links_rewritten", 0), 0,
                           "SLSR must fire on a=%d chains" % a)

        m = self._run(self._build_addressed(a, c, lanes, rounds), mem)
        for slot, val in expected.items():
            self.assertEqual(m.mem[slot], val, "a=%d slot %d" % (a, slot))

    def test_a3_chain_rewritten_and_correct(self):
        self._check_addressed(a=3, c=2)

    def test_a5_chain_rewritten_and_correct(self):
        self._check_addressed(a=5, c=4)

    def test_negative_step_chain_rewritten_and_correct(self):
        """x' = 2*x - (v + 1): '-' link with '-' compensable uses."""
        rounds = 3

        def build():
            b = HIRBuilder()
            for lane in range(2):
                x = b.load(b.const(8 + lane), "x0_%d" % lane)
                for r in range(rounds):
                    v = b.alu("&", b.load(b.const(10 + lane * rounds + r),
                                          "ld%d_%d" % (lane, r)), Const(1),
                              "v%d_%d" % (lane, r))
                    off = b.add(v, Const(1), "off%d_%d" % (lane, r))
                    m = b.mul(x, Const(2), "m%d_%d" % (lane, r))
                    x = b.sub(m, off, "x%d_%d" % (lane, r + 1))
                    u = b.sub(Const(1 << 20), x, "u%d_%d" % (lane, r))
                    b.store(b.const(40 + lane * rounds + r), u)
            b.halt()
            return b.build()

        mem = [0] * 64
        mem[8], mem[9] = 1000, 2000
        for i in range(8):
            mem[10 + i] = (i * 7 + 1) & 0xFF
        expected = {}
        for lane in range(2):
            x = mem[8 + lane]
            for r in range(rounds):
                v = mem[10 + lane * rounds + r] & 1
                x = (2 * x - (v + 1)) & 0xFFFFFFFF
                expected[40 + lane * rounds + r] = ((1 << 20) - x) & 0xFFFFFFFF

        metrics = self._metrics_for(build())
        self.assertGreater(metrics.get("links_rewritten", 0), 0,
                           "SLSR must fire on negative-step chains")

        m = self._run(build(), mem)
        for slot, val in expected.items():
            self.assertEqual(m.mem[slot], val, "slot %d" % slot)


class TestSLSRCostModelRegressions(unittest.TestCase):
    """Regressions for the approval fixpoint and root-offset policies."""

    def _pipeline(self, hir, slsr_options=None):
        base = TestSLSRPass()
        return base._run_pipeline_to_slsr(hir, slsr_options=slsr_options)

    def test_partial_chain_survivor_declined(self):
        """A member kept alive by a NON-approved link loses its saving.

        y = 2*x + 1 is compensably used (9 - y) but also feeds z = 2*y + 1
        whose direct store makes z unrewritable: y must stay alive for z,
        so rewriting y is a net op increase and must be declined. (Before
        the approved-closure fix this was approved and grew the IR.)
        """
        def build():
            b = HIRBuilder()
            x = b.load(b.const(0), "x")
            y = b.add(b.mul(x, Const(2), "m1"), Const(1), "y")
            b.store(b.const(2), b.sub(Const(9), y, "u"))
            z = b.add(b.mul(y, Const(2), "m2"), Const(1), "z")
            b.store(b.const(3), z)
            return b.build()

        hir, ac = self._pipeline(build())
        metrics = ac.get_metrics()
        self.assertEqual(metrics.custom.get("links_rewritten", 0), 0,
                         "negative-benefit rewrite must be declined")

    def test_uniform_single_lane_chain_approved_via_fixpoint(self):
        """A uniform single-lane chain is unprofitable with k_root = 0
        (per-link ks need distinct compensation ops) but profitable with
        the fixpoint root offset (one shared compensation + one root add);
        the policy comparison must find and apply it."""
        def build():
            b = HIRBuilder()
            base = b.load(b.const(7), "base")
            x = b.load(b.const(0), "x0")
            for r in range(4):
                v = b.alu("&", b.load(b.const(1 + r), "ld%d" % r), Const(1),
                          "v%d" % r)
                off = b.add(v, Const(1), "off%d" % r)
                m = b.mul(x, Const(2), "m%d" % r)
                x = b.add(m, off, "x%d" % (r + 1))
                addr = b.add(base, x, "a%d" % r)
                val = b.load(addr, "n%d" % r)
                b.store(b.const(40 + r), val)
            b.halt()
            return b.build()

        hir, ac = self._pipeline(build())
        metrics = ac.get_metrics()
        self.assertGreater(metrics.custom.get("links_rewritten", 0), 0,
                           "fixpoint policy should approve the chain")
        self.assertEqual(metrics.custom.get("policy"), "fixpoint")

        # And it must execute correctly end-to-end under the default config
        mem = [0] * 256
        mem[7] = 64
        mem[0] = 1
        for r in range(4):
            mem[1 + r] = (r * 3 + 1) & 0xFF
        for i in range(64):
            mem[64 + i] = 500 + i
        expected = {}
        x = mem[0]
        for r in range(4):
            v = mem[1 + r] & 1
            x = (2 * x + v + 1) & 0xFFFFFFFF
            expected[40 + r] = mem[64 + x]

        instrs = compile_hir_to_vliw(build())
        m = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        m.enable_pause = False
        m.enable_debug = False
        m.run()
        for slot, val in expected.items():
            self.assertEqual(m.mem[slot], val)


class TestRecurrenceMergedLinks(unittest.TestCase):
    """Association-order independence: y = (A*x + v) + C is discovered as
    one full link even when the inner add was greedily frozen first."""

    def test_const_on_top_of_zero_c_link_merges(self):
        from compiler.use_def import UseDefContext
        from compiler.recurrence import find_chain_links
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        v = b.load(b.const(1), "v")
        t = b.add(b.mul(x, Const(3), "m"), v, "t")
        y = b.add(t, Const(5), "y")
        b.store(b.const(2), y)
        hir = b.build()
        links = find_chain_links(hir.body, UseDefContext(hir))
        self.assertIn(y, links)
        self.assertEqual((links[y].a, links[y].c), (3, 5))
        self.assertEqual(links[y].x, x)


class TestSLSRReReview(unittest.TestCase):
    """Regressions from the SLSR re-review round: shared roots with
    different fixpoint offsets, dead-op-liveness cost accounting, partial
    link retirement, and the CRT fixpoint solver."""

    def _pipeline(self, hir, slsr_options=None):
        base = TestSLSRPass()
        return base._run_pipeline_to_slsr(hir, slsr_options=slsr_options)

    def _execute(self, hir, mem):
        instrs = compile_hir_to_vliw(hir)
        m = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        m.enable_pause = False
        m.enable_debug = False
        m.run()
        return m

    def test_shared_root_distinct_fixpoint_offsets(self):
        """Two chains rooted at the SAME value with different fixpoint
        k_root (C=1 -> k=1; C=3 -> k=3) must not share a root
        materialization (was a confirmed miscompile: the second chain
        reused the first chain's x+1)."""
        def build():
            b = HIRBuilder()
            base = b.load(b.const(7), "base")
            x = b.load(b.const(0), "x")
            y = x
            z = x
            for r in range(4):
                v = b.and_(b.load(b.const(1 + r), "lv%d" % r), b.const(1),
                           "v%d" % r)
                y = b.add(b.mul(y, Const(2), "my%d" % r),
                          b.add(v, Const(1), "ov%d" % r), "y%d" % (r + 1))
                b.store(b.const(40 + r),
                        b.load(b.add(base, y, "ay%d" % r), "ny%d" % r))
            for r in range(4):
                w = b.and_(b.load(b.const(1 + r), "lw%d" % r), b.const(1),
                           "w%d" % r)
                z = b.add(b.mul(z, Const(2), "mz%d" % r),
                          b.add(w, Const(3), "ow%d" % r), "z%d" % (r + 1))
                b.store(b.const(50 + r),
                        b.load(b.add(base, z, "az%d" % r), "nz%d" % r))
            b.halt()
            return b.build()

        mem = [0] * 640
        mem[7] = 128
        mem[0] = 1
        for r in range(4):
            mem[1 + r] = (r * 3 + 1) & 0xFF
        for i in range(300):
            mem[128 + i] = 100 + i
        expected = {}
        y = z = mem[0]
        for r in range(4):
            v = mem[1 + r] & 1
            y = (2 * y + v + 1) & 0xFFFFFFFF
            expected[40 + r] = mem[128 + y]
        for r in range(4):
            w = mem[1 + r] & 1
            z = (2 * z + w + 3) & 0xFFFFFFFF
            expected[50 + r] = mem[128 + z]

        m = self._execute(build(), mem)
        for slot, val in expected.items():
            self.assertEqual(m.mem[slot], val, "slot %d" % slot)

    def _build_shared_interior(self):
        """4-layer chain whose x2 also feeds an unrewritable link q; after
        CSE the mul *(x2, 2) is SHARED between the approved x3 and the
        non-approved q. Rewriting must not grow the IR (the dead-op
        liveness model sees the shared op stays alive)."""
        b = HIRBuilder()
        base = b.load(b.const(7), "base")
        x = b.load(b.const(0), "x0")
        xs = []
        for r in range(4):
            v = b.and_(b.load(b.const(1 + r), "ld%d" % r), b.const(1),
                       "v%d" % r)
            x = b.add(b.mul(x, Const(2), "m%d" % r),
                      b.add(v, Const(1), "off%d" % r), "x%d" % (r + 1))
            xs.append(x)
            b.store(b.const(40 + r),
                    b.load(b.add(base, x, "a%d" % r), "n%d" % r))
        q = b.add(b.mul(xs[1], Const(2), "mq"), Const(1), "q")
        b.store(b.const(60), q)
        b.halt()
        return b.build()

    def _op_count_after_dce(self, hir_after_slsr):
        from compiler.compile import PASS_REGISTRY
        from compiler.pass_manager import PassConfig
        p = PASS_REGISTRY["dce"]()
        out = p.run(hir_after_slsr,
                    PassConfig(name="dce", enabled=True, options={}))
        return sum(1 for s in out.body if isinstance(s, Op))

    def test_rejected_descendant_does_not_grow_ir(self):
        hir_on, ac = self._pipeline(self._build_shared_interior())
        n_on = self._op_count_after_dce(hir_on)
        hir_off, _ = self._pipeline(self._build_shared_interior(),
                                    slsr_options={"min_savings": 10**9})
        n_off = self._op_count_after_dce(hir_off)
        self.assertLessEqual(n_on, n_off,
                             "rewrite must not grow the IR")
        metrics = ac.get_metrics().custom
        if metrics.get("links_rewritten", 0):
            self.assertGreaterEqual(n_off - n_on,
                                    metrics.get("net_savings", 0),
                                    "metrics must not overstate savings")

    def test_partial_link_retired_after_merge(self):
        """t = 2*x + v; x' = t + 1 chains: the covered zero-C partials are
        retired so the full-C chain solves a fixpoint and gets rewritten."""
        def build():
            b = HIRBuilder()
            base = b.load(b.const(7), "base")
            x = b.load(b.const(0), "x0")
            for r in range(4):
                v = b.and_(b.load(b.const(1 + r), "ld%d" % r), b.const(1),
                           "v%d" % r)
                t = b.add(b.mul(x, Const(2), "m%d" % r), v, "t%d" % r)
                x = b.add(t, Const(1), "x%d" % (r + 1))
                b.store(b.const(40 + r),
                        b.load(b.add(base, x, "a%d" % r), "n%d" % r))
            b.halt()
            return b.build()

        hir, ac = self._pipeline(build())
        self.assertGreater(ac.get_metrics().custom.get("links_rewritten", 0),
                           0, "merged chain should be approved")

        mem = [0] * 640
        mem[7] = 128
        mem[0] = 1
        for r in range(4):
            mem[1 + r] = (r * 3 + 1) & 0xFF
        for i in range(300):
            mem[128 + i] = 700 + i
        expected = {}
        x = mem[0]
        for r in range(4):
            v = mem[1 + r] & 1
            x = (2 * x + v + 1) & 0xFFFFFFFF
            expected[40 + r] = mem[128 + x]
        m = self._execute(build(), mem)
        for slot, val in expected.items():
            self.assertEqual(m.mem[slot], val)

    def test_fixpoint_solver_crt_intersection(self):
        from compiler.passes.slsr import _solve_fixpoint_k
        from compiler.recurrence import ChainLink
        from compiler.hir import SSAValue

        def link(a, c):
            return ChainLink(y=SSAValue(0), x=SSAValue(1), a=a, v=None,
                             sign=1, c=c, pos=0, interior_op_ids=set())

        k = _solve_fixpoint_k([link(3, 2), link(5, 4)])
        self.assertIsNotNone(k)
        self.assertEqual((2 * k - 2) % (1 << 32), 0)
        self.assertEqual((4 * k - 4) % (1 << 32), 0)
        self.assertIsNone(_solve_fixpoint_k([link(5, 8), link(3, 2)]))


if __name__ == "__main__":
    unittest.main()
