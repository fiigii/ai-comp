"""Tests for the recurrence analysis (compiler/recurrence.py)."""

import unittest

from compiler.tests.conftest import HIRBuilder
from compiler.hir import Const, WORD_MASK
from compiler.use_def import UseDefContext
from compiler.recurrence import RecurrenceAnalysis, find_chain_links


class TestAffineEvaluation(unittest.TestCase):
    """Symbolic affine evaluation over flat SSA bodies."""

    def _analysis(self, hir):
        return RecurrenceAnalysis(hir.body, UseDefContext(hir))

    def test_const_and_atoms(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        hir = b.build()
        ra = self._analysis(hir)

        e = ra.affine_of(Const(41))
        self.assertEqual((e.terms, e.const), ({}, 41))
        e = ra.affine_of(x)
        self.assertEqual((e.terms, e.const), ({x: 1}, 0))

    def test_add_sub_mul_shift(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        y = b.load(b.const(1), "y")
        t1 = b.add(x, Const(5), "t1")           # x + 5
        t2 = b.mul(t1, Const(3), "t2")          # 3x + 15
        t3 = b.alu("<<", t2, Const(2), "t3")    # 12x + 60
        t4 = b.sub(t3, y, "t4")                 # 12x - y + 60
        hir = b.build()
        ra = self._analysis(hir)

        e = ra.affine_of(t4)
        self.assertEqual(e.const, 60)
        self.assertEqual(e.terms[x], 12)
        self.assertEqual(e.terms[y], (-1) & WORD_MASK)

    def test_shift_amount_must_be_const_right_operand(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        z = b.alu("<<", Const(2), x, "z")   # 2 << x: NOT affine in x
        hir = b.build()
        ra = self._analysis(hir)

        e = ra.affine_of(z)
        # z stays atomic (no misread as x << 2)
        self.assertEqual((e.terms, e.const), ({z: 1}, 0))

    def test_term_cancellation(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        t = b.sub(b.mul(x, Const(4), "m"), b.mul(x, Const(4), "m2"), "t")
        hir = b.build()
        ra = self._analysis(hir)
        e = ra.affine_of(t)
        self.assertEqual((e.terms, e.const), ({}, 0))

    def test_stop_set_keeps_value_atomic(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        t = b.add(x, Const(5), "t")
        u = b.add(t, Const(2), "u")
        hir = b.build()
        ra = self._analysis(hir)

        e = ra.affine_of(u, stop={t})
        self.assertEqual((e.terms, e.const), ({t: 1}, 2))

    def test_term_budget_degrades_to_atoms(self):
        """Expansion beyond the term budget degrades gracefully: the
        overflowing subexpression stays atomic instead of failing."""
        b = HIRBuilder()
        atoms = [b.load(b.const(i), "a%d" % i) for i in range(8)]
        acc = atoms[0]
        for a in atoms[1:]:
            acc = b.add(acc, a, "acc")
        hir = b.build()
        ra = RecurrenceAnalysis(hir.body, UseDefContext(hir), max_terms=4)
        e = ra.affine_of(acc)
        self.assertIsNotNone(e)
        self.assertLessEqual(len(e.terms), 4)


class TestChainDiscovery(unittest.TestCase):
    """find_chain_links over straight-line chains."""

    def _links(self, hir):
        return find_chain_links(hir.body, UseDefContext(hir))

    def _build_chain(self, a, c, rounds=3, sign=1):
        """x_{r+1} = a*x_r + sign*(v_r + c): the constant rides on the step
        operand (like tree-hash's off = bit + 1), so the intermediate add
        has no coeff >= 2 atom and cannot preempt the real chain value."""
        b = HIRBuilder()
        x = b.load(b.const(0), "x0")
        chain = [x]
        for r in range(rounds):
            v = b.load(b.const(1 + r), "v%d" % r)
            off = b.add(v, Const(c), "off%d" % r)
            m = b.mul(x, Const(a), "m%d" % r)
            x = b.add(m, off, "x%d" % (r + 1)) if sign > 0 \
                else b.sub(m, off, "x%d" % (r + 1))
            chain.append(x)
            b.store(b.const(20 + r), x)
        b.halt()
        return b.build(), chain

    def test_a3_chain_links(self):
        hir, chain = self._build_chain(a=3, c=7)
        links = self._links(hir)
        for y in chain[1:]:
            self.assertIn(y, links)
            self.assertEqual(links[y].a, 3)
            self.assertEqual(links[y].c, 7)
            self.assertEqual(links[y].sign, 1)
            self.assertIsNotNone(links[y].v)
        # chained: x of each later link is the previous chain value
        self.assertEqual(links[chain[2]].x, chain[1])
        self.assertEqual(links[chain[3]].x, chain[2])

    def test_negative_step_sign(self):
        hir, chain = self._build_chain(a=2, c=1, sign=-1)
        links = self._links(hir)
        self.assertIn(chain[1], links)
        self.assertEqual(links[chain[1]].sign, -1)

    def test_v_absent_link(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        y = b.add(b.mul(x, Const(4), "m"), Const(5), "y")   # y = 4x + 5
        b.store(b.const(1), y)
        hir = b.build()
        links = self._links(hir)
        self.assertIn(y, links)
        link = links[y]
        self.assertEqual((link.a, link.v, link.c), (4, None, 5))

    def test_ambiguous_step_rejected(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        u = b.load(b.const(1), "u")
        w = b.load(b.const(2), "w")
        m = b.mul(x, Const(2), "m")
        y = b.add(b.add(m, u, "t"), w, "y")   # 2x + u + w: two step candidates
        b.store(b.const(3), y)
        hir = b.build()
        self.assertNotIn(y, self._links(hir))

    def test_pure_scaling_not_a_link(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        m = b.mul(x, Const(8), "m")   # '*' top: interior, never a link
        b.store(b.const(1), m)
        hir = b.build()
        self.assertNotIn(m, self._links(hir))


if __name__ == "__main__":
    unittest.main()
