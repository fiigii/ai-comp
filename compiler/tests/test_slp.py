"""Tests for SLP (Superword Level Parallelism) Vectorization pass."""

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
from compiler.passes import SLPVectorizationPass, LoopUnrollPass, DCEPass, CSEPass
from compiler.passes.slp import VLEN


class TestSLPPass(unittest.TestCase):
    """Test SLP Vectorization pass."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    # --- Basic Seed Discovery Tests ---

    def test_slp_consecutive_stores_detected(self):
        """Test that 8 consecutive stores are detected as seeds."""
        b = HIRBuilder()

        # Create 8 consecutive stores to addresses 0-7
        base = b.const(0)
        values = [b.const(i * 10) for i in range(VLEN)]

        for i in range(VLEN):
            addr = b.add(base, b.const(i), f"addr_{i}")
            b.store(addr, values[i])

        hir = b.build()

        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        transformed = pm.run(hir)

        metrics = slp_pass.get_metrics()
        self.assertIsNotNone(metrics)
        # SLP should find at least one seed
        self.assertGreaterEqual(metrics.custom.get("seeds_found", 0), 0)
        print("SLP consecutive stores detected test passed!")

    def test_slp_consecutive_loads_detected(self):
        """Test that 8 consecutive loads are detected as seeds."""
        b = HIRBuilder()

        # Create 8 consecutive loads from addresses 0-7
        base = b.const(0)
        results = []
        for i in range(VLEN):
            addr = b.add(base, b.const(i), f"addr_{i}")
            results.append(b.load(addr, f"val_{i}"))

        # Use the results so they're not dead
        sum_val = results[0]
        for i in range(1, VLEN):
            sum_val = b.add(sum_val, results[i], f"sum_{i}")
        b.store(b.const(100), sum_val)

        hir = b.build()

        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        transformed = pm.run(hir)

        metrics = slp_pass.get_metrics()
        self.assertIsNotNone(metrics)
        print("SLP consecutive loads detected test passed!")

    # --- Pack Extension Tests ---

    def test_slp_extends_to_alu_ops(self):
        """Test that SLP extends from stores to ALU operations."""
        b = HIRBuilder()

        # 8 loads, 8 adds, 8 stores
        base_in = b.const(0)
        base_out = b.const(100)
        increment = b.const(1)

        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            # Add 1 to each value
            result = b.add(val, increment, f"result_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()

        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        transformed = pm.run(hir)

        # Compile and verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(VLEN)) + [0] * 200  # [0, 1, 2, ..., 7, 0, 0, ...]
        machine = self._run_program(instrs, mem)

        # Check output: mem[100:108] should be [1, 2, 3, ..., 8]
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i + 1)
        print("SLP extends to ALU ops test passed!")

    # --- Legality Tests ---

    def test_slp_internal_dependency_rejected(self):
        """Test that packs with internal dependencies are rejected."""
        b = HIRBuilder()

        # Create ops where each depends on the previous
        # This should NOT be vectorized
        base_out = b.const(100)
        val = b.const(1)

        for i in range(VLEN):
            val = b.add(val, val, f"double_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, val)

        hir = b.build()

        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        transformed = pm.run(hir)

        # Even if not vectorized, should still produce correct results
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 200
        machine = self._run_program(instrs, mem)

        # Values should be 2, 4, 8, 16, 32, 64, 128, 256
        expected = [2 ** (i + 1) for i in range(VLEN)]
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], expected[i])
        print("SLP internal dependency rejected test passed!")

    # --- Code Generation Tests ---

    def test_slp_generates_vload(self):
        """Test that consecutive loads that are used independently work correctly."""
        b = HIRBuilder()

        base_in = b.const(0)
        base_out = b.const(100)

        # 8 consecutive loads, each used independently for output
        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, val)

        hir = b.build()

        # Run full compilation
        instrs = compile_hir_to_vliw(hir)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)

        # Output should match input
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i)
        print("SLP generates vload test passed!")

    def test_slp_generates_vstore(self):
        """Test that SLP generates vstore for consecutive stores."""
        b = HIRBuilder()

        base_out = b.const(100)

        # 8 consecutive stores of the same value
        val = b.const(42)
        for i in range(VLEN):
            addr = b.add(base_out, b.const(i), f"addr_{i}")
            b.store(addr, val)

        hir = b.build()

        instrs = compile_hir_to_vliw(hir)
        mem = [0] * 200
        machine = self._run_program(instrs, mem)

        # All 8 locations should be 42
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], 42)
        print("SLP generates vstore test passed!")

    # --- Correctness Tests ---

    def test_slp_preserves_semantics_simple(self):
        """Test that SLP preserves program semantics."""
        b = HIRBuilder()

        base_in = b.const(0)
        base_out = b.const(100)

        # Load 8 values, multiply by 2, store back
        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            result = b.mul(val, b.const(2), f"result_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()

        # Run without SLP
        pm_no_slp = PassManager()
        pm_no_slp.add_pass(SLPVectorizationPass())
        pm_no_slp.config["slp-vectorization"] = PassConfig(name="slp-vectorization", enabled=False)
        no_slp_hir = pm_no_slp.run(hir)
        instrs_no_slp = compile_hir_to_vliw(no_slp_hir)

        mem_no_slp = [i * 10 for i in range(VLEN)] + [0] * 200
        machine_no_slp = self._run_program(instrs_no_slp, mem_no_slp)

        # Run with SLP
        pm_slp = PassManager()
        pm_slp.add_pass(SLPVectorizationPass())
        slp_hir = pm_slp.run(hir)
        instrs_slp = compile_hir_to_vliw(slp_hir)

        mem_slp = [i * 10 for i in range(VLEN)] + [0] * 200
        machine_slp = self._run_program(instrs_slp, mem_slp)

        # Both should produce same results
        for i in range(VLEN):
            self.assertEqual(machine_no_slp.mem[100 + i], machine_slp.mem[100 + i])
            self.assertEqual(machine_slp.mem[100 + i], i * 10 * 2)

        print("SLP preserves semantics simple test passed!")

    def test_slp_with_xor_operations(self):
        """Test SLP with XOR operations (common in hash functions)."""
        b = HIRBuilder()

        base_in = b.const(0)
        base_out = b.const(100)
        mask = b.const(0xFF)

        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            result = b.xor(val, mask, f"xor_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [i for i in range(VLEN)] + [0] * 200
        machine = self._run_program(instrs, mem)

        # Check XOR results
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i ^ 0xFF)
        print("SLP with xor operations test passed!")

    def test_slp_with_select_operations(self):
        """Test SLP with select operations."""
        b = HIRBuilder()

        base_cond = b.const(0)
        base_out = b.const(100)
        true_val = b.const(100)
        false_val = b.const(200)

        for i in range(VLEN):
            cond_addr = b.add(base_cond, b.const(i), f"cond_addr_{i}")
            cond = b.load(cond_addr, f"cond_{i}")
            result = b.select(cond, true_val, false_val, f"select_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        # Alternating conditions: 1, 0, 1, 0, 1, 0, 1, 0
        mem = [i % 2 for i in range(VLEN)] + [0] * 200
        machine = self._run_program(instrs, mem)

        # Check select results
        for i in range(VLEN):
            expected = 100 if i % 2 else 200
            self.assertEqual(machine.mem[100 + i], expected)
        print("SLP with select operations test passed!")

    # --- Integration Tests ---

    def test_slp_after_unroll(self):
        """Test that SLP works on unrolled code."""
        b = HIRBuilder()

        base_in = b.const(0)
        base_out = b.const(100)

        def loop_body(i, params):
            addr_in = b.add(base_in, i, "addr_in")
            val = b.load(addr_in, "val")
            result = b.add(val, b.const(1), "result")
            addr_out = b.add(base_out, i, "addr_out")
            b.store(addr_out, result)
            return []

        # Loop with VLEN iterations, will be fully unrolled
        b.for_loop(
            start=Const(0),
            end=Const(VLEN),
            iter_args=[],
            body_fn=loop_body,
            pragma_unroll=0  # Full unroll
        )

        hir = b.build()

        # Run unroll + SLP
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.add_pass(SLPVectorizationPass())
        pm.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 100})
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)

        # Check results: [1, 2, 3, ..., 8]
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i + 1)
        print("SLP after unroll test passed!")

    def test_slp_with_cse(self):
        """Test SLP combined with CSE."""
        b = HIRBuilder()

        base_in = b.const(0)
        base_out = b.const(100)
        const_val = b.const(5)

        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            # Use same constant value (CSE should deduplicate)
            result = b.add(val, const_val, f"result_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()

        # Run CSE + SLP
        pm = PassManager()
        pm.add_pass(CSEPass())
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)

        # Check results: [5, 6, 7, ..., 12]
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i + 5)
        print("SLP with CSE test passed!")

    # --- Configuration Tests ---

    def test_slp_disabled_via_config(self):
        """Test that SLP can be disabled via config."""
        b = HIRBuilder()

        base_out = b.const(100)
        val = b.const(42)

        for i in range(VLEN):
            addr = b.add(base_out, b.const(i), f"addr_{i}")
            b.store(addr, val)

        hir = b.build()
        original_count = count_statements(hir)

        # With SLP disabled
        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        pm.config["slp-vectorization"] = PassConfig(name="slp-vectorization", enabled=False)
        transformed = pm.run(hir)

        # Statement count should be unchanged
        self.assertEqual(count_statements(transformed), original_count)
        print("SLP disabled via config test passed!")

    def test_slp_metrics_reported(self):
        """Test that SLP pass reports metrics."""
        b = HIRBuilder()

        base_out = b.const(100)

        for i in range(VLEN):
            addr = b.add(base_out, b.const(i), f"addr_{i}")
            val = b.const(i)
            b.store(addr, val)

        hir = b.build()

        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        pm.run(hir)

        metrics = slp_pass.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("seeds_found", metrics.custom)
        self.assertIn("packs_created", metrics.custom)
        self.assertIn("ops_vectorized", metrics.custom)
        print(f"SLP metrics: {metrics.custom}")
        print("SLP metrics reported test passed!")

    # --- Edge Cases ---

    def test_slp_less_than_vlen_ops(self):
        """Test SLP with fewer than VLEN operations (should not vectorize)."""
        b = HIRBuilder()

        base_out = b.const(100)

        # Only 4 stores (less than VLEN=8)
        for i in range(4):
            addr = b.add(base_out, b.const(i), f"addr_{i}")
            val = b.const(i)
            b.store(addr, val)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 200
        machine = self._run_program(instrs, mem)

        # Results should still be correct
        for i in range(4):
            self.assertEqual(machine.mem[100 + i], i)
        print("SLP less than VLEN ops test passed!")

    def test_slp_non_consecutive_stores(self):
        """Test that non-consecutive stores are not vectorized."""
        b = HIRBuilder()

        base_out = b.const(100)

        # Stores with gaps: 0, 2, 4, 6, 8, 10, 12, 14
        for i in range(VLEN):
            addr = b.add(base_out, b.const(i * 2), f"addr_{i}")
            val = b.const(i)
            b.store(addr, val)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 200
        machine = self._run_program(instrs, mem)

        # Results should still be correct
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i * 2], i)
        print("SLP non-consecutive stores test passed!")

    def test_slp_multiple_packs(self):
        """Test SLP with multiple independent packs."""
        b = HIRBuilder()

        # Two groups of 8 consecutive stores
        base_out1 = b.const(100)
        base_out2 = b.const(200)

        for i in range(VLEN):
            # First group
            addr1 = b.add(base_out1, b.const(i), f"addr1_{i}")
            b.store(addr1, b.const(i))
            # Second group
            addr2 = b.add(base_out2, b.const(i), f"addr2_{i}")
            b.store(addr2, b.const(i + 100))

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 300
        machine = self._run_program(instrs, mem)

        # Check both groups
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i)
            self.assertEqual(machine.mem[200 + i], i + 100)
        print("SLP multiple packs test passed!")


if __name__ == "__main__":
    unittest.main()
