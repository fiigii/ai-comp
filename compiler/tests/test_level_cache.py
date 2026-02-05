"""Tests for Level Cache Pass and Range Analysis."""

import unittest

from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    HIRBuilder,
    compile_hir_to_vliw,
)
from compiler import (
    PassManager,
    PassConfig,
    Const,
    count_statements,
)
from compiler.hir import Op, ForLoop, SSAValue, VectorSSAValue, VectorConst, HIRFunction
from compiler.use_def import UseDefContext
from compiler.range_analysis import Range, RangeAnalysis
from compiler.passes.level_cache import LevelCachePass


class TestRange(unittest.TestCase):
    """Test Range class operations."""

    def test_point_range(self):
        """Test point range creation."""
        r = Range.point(5)
        self.assertEqual(r.min_val, 5)
        self.assertEqual(r.max_val, 5)
        self.assertEqual(r.size, 1)

    def test_range_size(self):
        """Test range size calculation."""
        r = Range(3, 6)
        self.assertEqual(r.size, 4)  # 3, 4, 5, 6

    def test_range_add(self):
        """Test range addition."""
        r1 = Range(1, 3)
        r2 = Range(10, 20)
        result = r1 + r2
        self.assertEqual(result.min_val, 11)
        self.assertEqual(result.max_val, 23)

    def test_range_sub(self):
        """Test range subtraction."""
        r1 = Range(10, 20)
        r2 = Range(1, 3)
        result = r1 - r2
        self.assertEqual(result.min_val, 7)   # 10 - 3
        self.assertEqual(result.max_val, 19)  # 20 - 1

    def test_range_mul(self):
        """Test range multiplication."""
        r1 = Range(2, 4)
        r2 = Range(3, 5)
        result = r1 * r2
        self.assertEqual(result.min_val, 6)   # 2 * 3
        self.assertEqual(result.max_val, 20)  # 4 * 5

    def test_range_bitwise_and_const(self):
        """Test range bitwise AND with constant."""
        r = Range(0, 15)
        mask = Range.point(7)
        result = r.bitwise_and(mask)
        self.assertEqual(result.min_val, 0)
        self.assertEqual(result.max_val, 7)

    def test_range_union(self):
        """Test range union."""
        r1 = Range(1, 5)
        r2 = Range(3, 10)
        result = r1.union(r2)
        self.assertEqual(result.min_val, 1)
        self.assertEqual(result.max_val, 10)

    def test_range_intersect(self):
        """Test range intersection."""
        r1 = Range(1, 5)
        r2 = Range(3, 10)
        result = r1.intersect(r2)
        self.assertIsNotNone(result)
        self.assertEqual(result.min_val, 3)
        self.assertEqual(result.max_val, 5)

    def test_range_intersect_disjoint(self):
        """Test range intersection of disjoint ranges."""
        r1 = Range(1, 5)
        r2 = Range(10, 20)
        result = r1.intersect(r2)
        self.assertIsNone(result)

    def test_range_contains(self):
        """Test range contains."""
        r = Range(3, 7)
        self.assertTrue(r.contains(5))
        self.assertTrue(r.contains(3))
        self.assertTrue(r.contains(7))
        self.assertFalse(r.contains(2))
        self.assertFalse(r.contains(8))


class TestRangeAnalysis(unittest.TestCase):
    """Test RangeAnalysis class."""

    def test_const_range(self):
        """Test range of constant values."""
        b = HIRBuilder()
        c = b.const(42)
        hir = b.build()
        use_def = UseDefContext(hir)
        analysis = RangeAnalysis(use_def)

        r = analysis.get_range(Const(42))
        self.assertEqual(r.min_val, 42)
        self.assertEqual(r.max_val, 42)

    def test_add_range(self):
        """Test range of addition."""
        b = HIRBuilder()
        a = b.const(10)
        c = b.const(5)
        result = b.add(a, c, "sum")
        hir = b.build()

        use_def = UseDefContext(hir)
        analysis = RangeAnalysis(use_def)

        r = analysis.get_range(result)
        self.assertEqual(r.min_val, 15)
        self.assertEqual(r.max_val, 15)

    def test_select_range(self):
        """Test range of select operation."""
        b = HIRBuilder()
        cond = b.const(1)
        true_val = b.const(10)
        false_val = b.const(20)
        result = b.select(cond, true_val, false_val, "sel")
        hir = b.build()

        use_def = UseDefContext(hir)
        analysis = RangeAnalysis(use_def)

        r = analysis.get_range(result)
        self.assertEqual(r.min_val, 10)
        self.assertEqual(r.max_val, 20)


class TestLevelCachePass(unittest.TestCase):
    """Test Level Cache Pass."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_pass_disabled(self):
        """Test that pass can be disabled."""
        b = HIRBuilder()
        addr = b.const(0)
        val = b.const(42)
        b.store(addr, val)
        hir = b.build()

        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(LevelCachePass())
        pm.config["level-cache"] = PassConfig(name="level-cache", enabled=False)
        result = pm.run(hir)

        self.assertEqual(count_statements(result), original_count)

    def test_pass_no_vgather(self):
        """Test pass with no vgather operations."""
        b = HIRBuilder()
        addr = b.const(0)
        val = b.const(42)
        b.store(addr, val)
        hir = b.build()

        pm = PassManager()
        pm.add_pass(LevelCachePass())
        result = pm.run(hir)

        # Should not change anything
        instrs = compile_hir_to_vliw(result)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 42)

    def test_pass_metrics(self):
        """Test that pass reports metrics."""
        b = HIRBuilder()
        addr = b.const(0)
        val = b.const(42)
        b.store(addr, val)
        hir = b.build()

        level_cache = LevelCachePass()
        pm = PassManager()
        pm.add_pass(level_cache)
        pm.run(hir)

        metrics = level_cache.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("vgathers_found", metrics.custom)


class TestLevelCacheIntegration(unittest.TestCase):
    """Integration tests for level cache with the full compiler."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_full_pipeline_correctness(self):
        """Test that level cache doesn't break correctness."""
        # Build a simple program that stores values
        b = HIRBuilder()
        for i in range(8):
            addr = b.const(i)
            val = b.const(i * 10)
            b.store(addr, val)
        hir = b.build()

        instrs = compile_hir_to_vliw(hir)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        for i in range(8):
            self.assertEqual(machine.mem[i], i * 10)


if __name__ == "__main__":
    unittest.main()
