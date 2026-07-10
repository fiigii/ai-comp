"""Tests for alias analysis, especially composite base normalization."""

import unittest

from compiler import HIRBuilder, Const, PassManager
from compiler.hir import SSAValue, Op
from compiler.use_def import UseDefContext
from compiler.alias_analysis import AliasAnalysis, AddrKey, AliasResult, base_roots


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


class TestBaseRoots(unittest.TestCase):
    """Unit tests for base_roots on plain and composite bases."""

    def test_plain_base_is_its_own_root(self):
        base = ("memslot", 7)
        self.assertEqual(base_roots(base), frozenset([base]))

    def test_composite_add_base_has_both_roots(self):
        a = ("memslot", 1)
        b = ("memslot", 2)
        self.assertEqual(base_roots(("add", a, b)), frozenset([a, b]))

    def test_nested_composite_base_flattens_all_roots(self):
        a = ("memslot", 1)
        b = ("memslot", 2)
        c = ("memslot", 3)
        self.assertEqual(base_roots(("add", ("add", a, b), c)),
                         frozenset([a, b, c]))

    def test_union_base_has_both_arm_roots(self):
        a = ("memslot", 1)
        b = ("memslot", 2)
        identity = SSAValue(99, "selected")
        self.assertEqual(
            base_roots(("union", identity, a, b)),
            frozenset([a, b]),
        )

    def test_rooted_load_exposes_its_memslot_root(self):
        root = ("memslot", 7)
        identity = SSAValue(99, "loaded")
        self.assertEqual(
            base_roots(("rooted_load", identity, root)),
            frozenset([root]),
        )

    def test_generic_derived_base_exposes_all_operand_roots(self):
        a = ("memslot", 1)
        b = ("memslot", 2)
        identity = SSAValue(99, "derived")
        self.assertEqual(
            base_roots(("derived", identity, frozenset([a, b]))),
            frozenset([a, b]),
        )


class TestCompositeBaseRootsRestrictPtr(unittest.TestCase):
    """Regression tests for the composite-base root fix.

    Under restrict_ptr, two composite bases that share a root (e.g.
    table+i vs table+j with dynamic i, j) must be MAY_ALIAS; before
    the fix any base mismatch was treated as NO_ALIAS.
    """

    def _build(self):
        """Flat HIR with composite addresses over shared/disjoint roots."""
        b = HIRBuilder()
        p = b.load(b.const(0), "p")
        i = b.load(b.const(1), "i")
        j = b.load(b.const(2), "j")
        q = b.load(b.const(3), "q")
        j2 = b.load(b.const(4), "j2")
        p2 = b.load(b.const(5), "p2")

        a1 = b.add(p, i, "a1")
        b.store(a1, b.const(10))
        a2 = b.add(p, j, "a2")
        b.store(a2, b.const(20))
        a3 = b.add(q, j2, "a3")
        b.store(a3, b.const(30))
        a4 = b.add(p2, i, "a4")
        b.store(a4, b.const(40))

        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)
        return aa, a1, a2, a3, a4

    def test_same_pointer_root_may_alias_under_restrict_ptr(self):
        """p+i vs p+j share root p: MAY_ALIAS (was NO_ALIAS before fix)."""
        aa, a1, a2, _, _ = self._build()
        k1 = aa.normalize(a1)
        k2 = aa.normalize(a2)
        self.assertIsNotNone(k1)
        self.assertIsNotNone(k2)
        # Composite bases differ (different dynamic index roots) ...
        self.assertNotEqual(k1.base, k2.base)
        # ... but they share the pointer root, so restrict_ptr must not
        # prove them disjoint.
        self.assertEqual(aa.alias_keys(k1, 1, k2, 1), AliasResult.MAY_ALIAS)

    def test_disjoint_roots_no_alias_under_restrict_ptr(self):
        """p+i vs q+j2 share no root: still NO_ALIAS under restrict_ptr."""
        aa, a1, _, a3, _ = self._build()
        k1 = aa.normalize(a1)
        k3 = aa.normalize(a3)
        self.assertIsNotNone(k1)
        self.assertIsNotNone(k3)
        self.assertNotEqual(k1.base, k3.base)
        self.assertEqual(aa.alias_keys(k1, 1, k3, 1), AliasResult.NO_ALIAS)

    def test_shared_dynamic_index_root_may_alias_under_restrict_ptr(self):
        """p+i vs p2+i share the index root i: MAY_ALIAS under restrict_ptr."""
        aa, a1, _, _, a4 = self._build()
        k1 = aa.normalize(a1)
        k4 = aa.normalize(a4)
        self.assertIsNotNone(k1)
        self.assertIsNotNone(k4)
        self.assertNotEqual(k1.base, k4.base)
        self.assertEqual(aa.alias_keys(k1, 1, k4, 1), AliasResult.MAY_ALIAS)


class TestCompositeBaseRootsWithoutRestrict(unittest.TestCase):
    """Different symbolic roots remain MAY_ALIAS without a noalias contract."""

    def test_disjoint_composite_roots_may_equal_at_runtime(self):
        b = HIRBuilder()
        p = b.load(b.const(0), "p")
        i = b.load(b.const(1), "i")
        q = b.load(b.const(2), "q")
        j = b.load(b.const(3), "j")
        p_plus_i = b.add(p, i, "p_plus_i")
        q_plus_j = b.add(q, j, "q_plus_j")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=False)

        left = aa.normalize(p_plus_i)
        right = aa.normalize(q_plus_j)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertTrue(base_roots(left.base).isdisjoint(base_roots(right.base)))
        # For example p=64, i=3, q=65, j=2 makes both addresses 67.
        self.assertEqual(aa.alias_keys(left, 1, right, 1), AliasResult.MAY_ALIAS)

    def test_distinct_plain_pointer_roots_are_also_may_alias(self):
        b = HIRBuilder()
        p = b.load(b.const(0), "p")
        q = b.load(b.const(1), "q")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=False)

        self.assertEqual(
            aa.alias_keys(aa.normalize(p), 1, aa.normalize(q), 1),
            AliasResult.MAY_ALIAS,
        )


class Test32BitAddressWrapping(unittest.TestCase):
    def _analysis(self):
        b = HIRBuilder()
        base = b.load(b.const(100), "base")
        plus_space = b.add(base, b.const(1 << 32), "plus_space")
        minus_one = b.add(base, b.const(-1), "minus_one")
        nested_wrap = b.add(minus_one, b.const(1), "nested_wrap")
        hir = b.build()
        return AliasAnalysis(UseDefContext(hir)), base, plus_space, minus_one, nested_wrap

    def test_base_plus_address_space_is_same_address(self):
        aa, base, plus_space, _, nested_wrap = self._analysis()
        base_key = aa.normalize(base)
        plus_space_key = aa.normalize(plus_space)
        nested_wrap_key = aa.normalize(nested_wrap)

        self.assertEqual(plus_space_key, base_key)
        self.assertEqual(nested_wrap_key, base_key)
        self.assertEqual(
            aa.alias_keys(base_key, 1, plus_space_key, 1),
            AliasResult.MUST_ALIAS,
        )

    def test_constant_addresses_and_memslot_roots_wrap(self):
        b = HIRBuilder()
        low = b.load(b.const(4), "low")
        wrapped = b.load(b.const((1 << 32) + 4), "wrapped")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir))

        self.assertEqual(aa.normalize(b.const(-1)).offset, (1 << 32) - 1)
        low_key = aa.normalize(low)
        wrapped_key = aa.normalize(wrapped)
        self.assertNotEqual(low_key.base, wrapped_key.base)
        self.assertEqual(base_roots(low_key.base), base_roots(wrapped_key.base))
        self.assertEqual(
            aa.alias_keys(low_key, 1, wrapped_key, 1),
            AliasResult.MAY_ALIAS,
        )

    def test_same_loaded_ssa_still_must_alias_itself(self):
        b = HIRBuilder()
        loaded = b.load(b.const(4), "loaded")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)

        key = aa.normalize(loaded)
        self.assertEqual(
            aa.alias_keys(key, 1, key, 1),
            AliasResult.MUST_ALIAS,
        )

    def test_equivalent_constant_slot_expressions_share_provenance(self):
        b = HIRBuilder()
        literal = b.load(b.const(4), "literal")
        computed_slot = b.add(b.const(2), b.const(2), "computed_slot")
        computed = b.load(computed_slot, "computed")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)

        literal_key = aa.normalize(literal)
        computed_key = aa.normalize(computed)
        self.assertNotEqual(literal_key.base, computed_key.base)
        self.assertEqual(
            base_roots(literal_key.base),
            base_roots(computed_key.base),
        )
        self.assertEqual(
            aa.alias_keys(literal_key, 1, computed_key, 1),
            AliasResult.MAY_ALIAS,
        )

    def test_vector_range_crossing_zero_overlaps_low_addresses(self):
        aa, base, _, minus_one, _ = self._analysis()
        base_key = aa.normalize(base)
        minus_one_key = aa.normalize(minus_one)

        self.assertEqual(minus_one_key.offset, (1 << 32) - 1)
        self.assertEqual(
            aa.alias_keys(minus_one_key, 8, base_key, 1),
            AliasResult.MAY_ALIAS,
        )
        self.assertEqual(
            aa.alias_keys(minus_one_key, 1, base_key, 1),
            AliasResult.NO_ALIAS,
        )

    def test_equal_wrapped_vector_ranges_must_alias(self):
        aa, _, _, minus_one, _ = self._analysis()
        minus_one_key = aa.normalize(minus_one)
        equivalent = AddrKey(minus_one_key.base, -1)

        self.assertEqual(
            aa.alias_keys(minus_one_key, 8, equivalent, 8),
            AliasResult.MUST_ALIAS,
        )

    def test_subtract_constant_wraps_and_vector_range_overlaps(self):
        b = HIRBuilder()
        base = b.load(b.const(100), "base")
        before_base = b.sub(base, b.const(1), "before_base")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)

        base_key = aa.normalize(base)
        before_key = aa.normalize(before_base)
        self.assertEqual(before_key.base, base_key.base)
        self.assertEqual(before_key.offset, (1 << 32) - 1)
        self.assertEqual(
            aa.alias_keys(before_key, 8, base_key, 1),
            AliasResult.MAY_ALIAS,
        )


class TestSelectPointerProvenance(unittest.TestCase):
    def test_selected_pointer_may_alias_either_arm_under_restrict(self):
        b = HIRBuilder()
        p = b.load(b.const(0), "p")
        q = b.load(b.const(1), "q")
        cond = b.load(b.const(2), "cond")
        selected = b.select(cond, p, q, "selected")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)

        p_key = aa.normalize(p)
        q_key = aa.normalize(q)
        selected_key = aa.normalize(selected)
        self.assertEqual(
            aa.alias_keys(selected_key, 1, p_key, 1),
            AliasResult.MAY_ALIAS,
        )
        self.assertEqual(
            aa.alias_keys(selected_key, 1, q_key, 1),
            AliasResult.MAY_ALIAS,
        )

    def test_distinct_selects_are_not_must_alias(self):
        b = HIRBuilder()
        p = b.load(b.const(0), "p")
        q = b.load(b.const(1), "q")
        cond_a = b.load(b.const(2), "cond_a")
        cond_b = b.load(b.const(3), "cond_b")
        selected_a = b.select(cond_a, p, q, "selected_a")
        selected_b = b.select(cond_b, p, q, "selected_b")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)

        self.assertEqual(
            aa.alias_keys(
                aa.normalize(selected_a), 1,
                aa.normalize(selected_b), 1,
            ),
            AliasResult.MAY_ALIAS,
        )


class TestGenericDerivedPointerProvenance(unittest.TestCase):
    def test_dynamic_subtraction_retains_both_operand_roots(self):
        b = HIRBuilder()
        forest = b.load(b.const(0), "forest")
        dynamic = b.load(b.const(1), "dynamic")
        address = b.sub(forest, dynamic, "address")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)

        forest_key = aa.normalize(forest)
        dynamic_key = aa.normalize(dynamic)
        address_key = aa.normalize(address)
        address_roots = base_roots(address_key.base)
        self.assertTrue(address_roots & base_roots(forest_key.base))
        self.assertTrue(address_roots & base_roots(dynamic_key.base))
        self.assertEqual(
            aa.alias_keys(address_key, 1, forest_key, 1),
            AliasResult.MAY_ALIAS,
        )

    def test_other_unsupported_scalar_op_retains_operand_roots(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        mask = b.load(b.const(1), "mask")
        address = b.xor(base, mask, "address")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)

        base_key = aa.normalize(base)
        address_key = aa.normalize(address)
        self.assertTrue(
            base_roots(address_key.base) & base_roots(base_key.base)
        )
        self.assertEqual(
            aa.alias_keys(address_key, 1, base_key, 1),
            AliasResult.MAY_ALIAS,
        )

    def test_constants_do_not_become_generic_provenance_roots(self):
        b = HIRBuilder()
        index = b.load(b.const(0), "index")
        scaled = b.mul(index, b.const(8), "scaled")
        constant_only = b.mul(b.const(2), b.const(8), "constant_only")
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir), restrict_ptr=True)

        index_key = aa.normalize(index)
        scaled_key = aa.normalize(scaled)
        self.assertEqual(
            base_roots(scaled_key.base),
            base_roots(index_key.base),
        )
        self.assertIsNone(aa.normalize(constant_only))



class TestSharedComponentRangeDisjointness(unittest.TestCase):
    """Value ranges prove p+i vs p+j disjoint without any restrict contract."""

    def _build(self):
        from compiler.hir_builder import HIRBuilder as _HB
        b = _HB()
        p = b.load(b.const(100), "p")
        x = b.load(b.const(101), "x")
        y = b.load(b.const(102), "y")
        i = b.and_(x, b.const(7), "i")                        # [0, 7]
        j = b.add(b.and_(y, b.const(7), "j0"), b.const(8), "j")  # [8, 15]
        k = b.and_(y, b.const(15), "k")                       # [0, 15]
        addrs = {
            "low": b.add(p, i, "a_low"),
            "high": b.add(p, j, "a_high"),
            "wide": b.add(p, k, "a_wide"),
            "const9": b.add(p, b.const(9), "a_const9"),
            "const7": b.add(p, b.const(7), "a_const7"),
        }
        for name, addr in addrs.items():
            b.store(addr, b.const(1))
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir))  # NO restrict_ptr
        return aa, addrs

    def _alias(self, aa, addrs, a, aw, b, bw):
        return aa.alias_keys(aa.normalize(addrs[a]), aw,
                             aa.normalize(addrs[b]), bw)

    def test_disjoint_dynamic_indices_no_alias(self):
        aa, addrs = self._build()
        self.assertEqual(self._alias(aa, addrs, "low", 1, "high", 1),
                         AliasResult.NO_ALIAS)

    def test_overlapping_dynamic_indices_may_alias(self):
        aa, addrs = self._build()
        self.assertEqual(self._alias(aa, addrs, "low", 1, "wide", 1),
                         AliasResult.MAY_ALIAS)

    def test_const_offset_vs_disjoint_dynamic_no_alias(self):
        aa, addrs = self._build()
        self.assertEqual(self._alias(aa, addrs, "const9", 1, "low", 1),
                         AliasResult.NO_ALIAS)

    def test_const_offset_inside_dynamic_range_may_alias(self):
        aa, addrs = self._build()
        self.assertEqual(self._alias(aa, addrs, "const7", 1, "low", 1),
                         AliasResult.MAY_ALIAS)

    def test_width_extends_footprint_into_overlap(self):
        aa, addrs = self._build()
        # [0,7] with width 8 covers up to 14: overlaps [8,15].
        self.assertEqual(self._alias(aa, addrs, "low", 8, "high", 1),
                         AliasResult.MAY_ALIAS)

    def test_unbounded_dynamic_component_may_alias(self):
        from compiler.hir_builder import HIRBuilder as _HB
        b = _HB()
        p = b.load(b.const(100), "p")
        u = b.load(b.const(101), "u")            # FULL range
        v = b.and_(b.load(b.const(102), "v0"), b.const(3), "v")
        a1 = b.add(p, u, "a1")
        a2 = b.add(p, v, "a2")
        b.store(a1, b.const(1))
        b.store(a2, b.const(1))
        hir = b.build()
        aa = AliasAnalysis(UseDefContext(hir))
        self.assertEqual(
            aa.alias_keys(aa.normalize(a1), 1, aa.normalize(a2), 1),
            AliasResult.MAY_ALIAS)


if __name__ == "__main__":
    unittest.main()
