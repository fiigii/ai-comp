"""Tests for alias analysis, especially composite base normalization."""

import unittest

from compiler import HIRBuilder, Const, PassManager
from compiler.hir import SSAValue, Op
from compiler.use_def import UseDefContext
from compiler.alias_analysis import AliasAnalysis, AddrKey, AliasResult


class TestCompositeBaseNormalization(unittest.TestCase):
    """Test that AliasAnalysis normalizes ssa+ssa addresses into composite bases."""

    def test_ssa_plus_const_offset(self):
        """Basic: ptr + Const(j) should give base=ptr, offset=j."""
        b = HIRBuilder()
        ptr_val = b.load(b.const(100), "ptr")
        addr = b.add(ptr_val, b.const(5), "addr")
        b.store(addr, b.const(42))
        hir = b.build()

        ud = UseDefContext(hir)
        aa = AliasAnalysis(ud)

        key = aa.normalize(addr)
        self.assertIsNotNone(key)
        self.assertEqual(key.offset, 5)

    def test_composite_base_from_ssa_plus_ssa(self):
        """ptr + scaled should produce a composite base ("add", ptr_base, scaled_base)."""
        b = HIRBuilder()
        ptr = b.load(b.const(100), "ptr")
        scaled = b.load(b.const(200), "scaled")
        addr = b.add(ptr, scaled, "addr")
        b.store(addr, b.const(42))
        hir = b.build()

        ud = UseDefContext(hir)
        aa = AliasAnalysis(ud)

        key = aa.normalize(addr)
        self.assertIsNotNone(key)
        # Should be a composite ("add", ..., ...) base
        self.assertIsInstance(key.base, tuple)
        self.assertEqual(key.base[0], "add")
        self.assertEqual(key.offset, 0)

    def test_composite_base_extracts_offset(self):
        """ptr + (scaled + Const(j)) should give composite base with offset=j."""
        b = HIRBuilder()
        ptr = b.load(b.const(100), "ptr")
        scaled = b.load(b.const(200), "scaled")
        inner = b.add(scaled, b.const(3), "inner")
        addr = b.add(ptr, inner, "addr")
        b.store(addr, b.const(42))
        hir = b.build()

        ud = UseDefContext(hir)
        aa = AliasAnalysis(ud)

        key = aa.normalize(addr)
        self.assertIsNotNone(key)
        self.assertIsInstance(key.base, tuple)
        self.assertEqual(key.base[0], "add")
        self.assertEqual(key.offset, 3)

    def test_consecutive_offsets_share_composite_base(self):
        """ptr + (scaled + 0..7) should all share the same composite base with offsets 0..7."""
        b = HIRBuilder()
        ptr = b.load(b.const(100), "ptr")
        scaled = b.load(b.const(200), "scaled")

        addrs = []
        for j in range(8):
            inner = b.add(scaled, b.const(j), f"inner_{j}")
            addr = b.add(ptr, inner, f"addr_{j}")
            b.store(addr, b.const(j * 10))
            addrs.append(addr)

        hir = b.build()
        ud = UseDefContext(hir)
        aa = AliasAnalysis(ud)

        keys = [aa.normalize(a) for a in addrs]
        self.assertTrue(all(k is not None for k in keys))

        # All should share the same composite base
        bases = {k.base for k in keys}
        self.assertEqual(len(bases), 1, f"Expected 1 shared base, got {len(bases)}: {bases}")

        # Offsets should be 0..7
        offsets = sorted(k.offset for k in keys)
        self.assertEqual(offsets, list(range(8)))

    def test_commutativity(self):
        """a + b and b + a should produce the same composite base."""
        b = HIRBuilder()
        x = b.load(b.const(100), "x")
        y = b.load(b.const(200), "y")

        addr1 = b.add(x, y, "addr1")
        b.store(addr1, b.const(1))
        addr2 = b.add(y, x, "addr2")
        b.store(addr2, b.const(2))

        hir = b.build()
        ud = UseDefContext(hir)
        aa = AliasAnalysis(ud)

        key1 = aa.normalize(addr1)
        key2 = aa.normalize(addr2)
        self.assertIsNotNone(key1)
        self.assertIsNotNone(key2)
        self.assertEqual(key1.base, key2.base, "Commutative adds should have same base")
        self.assertEqual(key1.offset, key2.offset)

    def test_alias_no_alias_composite_base(self):
        """Two addresses with same composite base but different offsets -> NO_ALIAS."""
        b = HIRBuilder()
        ptr = b.load(b.const(100), "ptr")
        scaled = b.load(b.const(200), "scaled")

        inner0 = b.add(scaled, b.const(0), "inner0")
        addr0 = b.add(ptr, inner0, "addr0")
        b.store(addr0, b.const(10))

        inner1 = b.add(scaled, b.const(1), "inner1")
        addr1 = b.add(ptr, inner1, "addr1")
        b.store(addr1, b.const(20))

        hir = b.build()
        ud = UseDefContext(hir)
        aa = AliasAnalysis(ud)

        key0 = aa.normalize(addr0)
        key1 = aa.normalize(addr1)
        self.assertIsNotNone(key0)
        self.assertIsNotNone(key1)

        result = aa.alias_keys(key0, 1, key1, 1)
        self.assertEqual(result, AliasResult.NO_ALIAS)

    def test_alias_must_alias_composite_base(self):
        """Two addresses with same composite base and same offset -> MUST_ALIAS."""
        b = HIRBuilder()
        ptr = b.load(b.const(100), "ptr")
        scaled = b.load(b.const(200), "scaled")

        inner_a = b.add(scaled, b.const(5), "inner_a")
        addr_a = b.add(ptr, inner_a, "addr_a")
        b.store(addr_a, b.const(10))

        inner_b = b.add(scaled, b.const(5), "inner_b")
        addr_b = b.add(ptr, inner_b, "addr_b")
        b.store(addr_b, b.const(20))

        hir = b.build()
        ud = UseDefContext(hir)
        aa = AliasAnalysis(ud)

        key_a = aa.normalize(addr_a)
        key_b = aa.normalize(addr_b)
        self.assertIsNotNone(key_a)
        self.assertIsNotNone(key_b)

        result = aa.alias_keys(key_a, 1, key_b, 1)
        self.assertEqual(result, AliasResult.MUST_ALIAS)

    def test_partial_unroll_pattern_slp_seeds(self):
        """End-to-end: a partially-unrolled loop with ptr+(scaled+j) stores gets SLP seeds."""
        from compiler.passes import SLPVectorizationPass, DCEPass
        from compiler.passes.slp import VLEN

        b = HIRBuilder()
        out_p = b.load(b.const(0), "out_p")

        # Simulate partial unroll: for i in 0..1 (1 iteration, body has 8 stores)
        # Each store goes to out_p + (i*8 + j) for j=0..7
        def body(i, params):
            scaled = b.mul(i, b.const(8), "scaled")
            for j in range(VLEN):
                inner = b.add(scaled, b.const(j), f"inner_{j}")
                addr = b.add(out_p, inner, f"addr_{j}")
                b.store(addr, b.const(j + 1))
            return []

        b.for_loop(
            start=Const(0),
            end=Const(1),
            iter_args=[],
            body_fn=body,
            pragma_unroll=1,
        )
        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        slp = SLPVectorizationPass()
        pm.add_pass(slp)
        pm.run(hir)

        metrics = slp.get_metrics()
        seeds = metrics.custom.get("seeds_found", 0)
        self.assertGreaterEqual(seeds, 1,
                                f"SLP should find seeds from composite-base stores, got {seeds}")


if __name__ == "__main__":
    unittest.main()
