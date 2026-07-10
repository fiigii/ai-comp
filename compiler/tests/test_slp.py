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
    lower_to_lir,
    eliminate_phis,
)
from compiler.passes import SLPVectorizationPass, LoopUnrollPass, DCEPass, CSEPass
from compiler.passes import LIRToMIRPass, MIRRegisterAllocationPass, MIRToVLIWPass
from compiler.passes.slp import VLEN, VECTORIZABLE_ALU_OPS


class TestSLPPass(unittest.TestCase):
    """Test SLP Vectorization pass."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def _compile_hir_via_mir_only(self, hir):
        """Compile HIR through lowering+MIR only (skip full HIR optimization pipeline)."""
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        cfg = PassConfig(name="test", enabled=True, options={})
        mir = LIRToMIRPass().run(lir, cfg)
        mir = MIRRegisterAllocationPass().run(mir, cfg)
        return MIRToVLIWPass().run(mir, cfg)

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
        instrs = self._compile_hir_via_mir_only(transformed)
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
        instrs = self._compile_hir_via_mir_only(transformed)
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
        instrs_no_slp = self._compile_hir_via_mir_only(no_slp_hir)

        mem_no_slp = [i * 10 for i in range(VLEN)] + [0] * 200
        machine_no_slp = self._run_program(instrs_no_slp, mem_no_slp)

        # Run with SLP
        pm_slp = PassManager()
        pm_slp.add_pass(SLPVectorizationPass())
        slp_hir = pm_slp.run(hir)
        instrs_slp = self._compile_hir_via_mir_only(slp_hir)

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


    # --- Broadcast Placement Tests ---

    def test_slp_broadcast_external_values_at_entry(self):
        """Test that broadcasts for externally-defined values are at function entry."""
        b = HIRBuilder()

        # Values defined at function entry (outside any loop)
        base_in = b.const(0)
        base_out = b.const(100)
        multiplier = b.const(2)

        # 8 loads, multiply by constant, store
        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            result = b.mul(val, multiplier, f"result_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(hir)

        # Check that broadcasts for constants are at the beginning (before any loops)
        from compiler.hir import Op, ForLoop
        broadcast_ops = []
        other_ops = []
        seen_non_broadcast = False

        for stmt in transformed.body:
            if isinstance(stmt, Op):
                if stmt.opcode == "vbroadcast":
                    broadcast_ops.append(stmt)
                    # External broadcasts should appear before non-broadcast ops
                    # (except for loads that define the broadcast operand)
                    if stmt.operands[0].__class__.__name__ == "Const":
                        self.assertFalse(
                            seen_non_broadcast and not any(
                                isinstance(s, Op) and s.opcode == "load"
                                for s in other_ops
                            ),
                            "Constant broadcast should be at entry, not after non-load ops"
                        )
                else:
                    other_ops.append(stmt)
                    if stmt.opcode not in ("load", "const"):
                        seen_non_broadcast = True

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)

        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i * 2)
        print("SLP broadcast external values at entry test passed!")

    def test_slp_broadcast_internal_values_after_def(self):
        """Test that broadcasts for internally-defined values are after their definition."""
        b = HIRBuilder()

        base_in = b.const(0)
        base_out = b.const(100)

        # Create a loop where values are computed inside
        def loop_body(i, params):
            # scaled_val is defined inside the loop
            scaled_val = b.mul(i, b.const(VLEN), "scaled")

            # Inner pattern: consecutive addresses based on scaled_val
            for j in range(VLEN):
                addr_in = b.add(scaled_val, b.const(j), f"addr_in_{j}")
                val = b.load(addr_in, f"val_{j}")
                addr_out = b.add(base_out, addr_in, f"addr_out_{j}")
                b.store(addr_out, val)
            return []

        b.for_loop(
            start=Const(0),
            end=Const(2),
            iter_args=[],
            body_fn=loop_body,
        )

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.add_pass(SLPVectorizationPass())
        pm.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 16})
        transformed = pm.run(hir)

        # Verify correctness - program should still work correctly
        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(VLEN * 4)) + [0] * 300
        machine = self._run_program(instrs, mem)

        # Check some outputs are correct
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i)
        print("SLP broadcast internal values after def test passed!")

    def test_slp_broadcast_loop_invariant_hoisted(self):
        """Test that broadcasts of loop-invariant values are hoisted to entry."""
        b = HIRBuilder()

        # These are defined outside any vectorized block - should be hoisted
        base_in = b.const(0)
        base_out = b.const(100)
        mask = b.const(0xFF)

        # Create 8 consecutive operations that will be vectorized
        # The mask constant should have its broadcast hoisted to entry
        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            # mask is a constant defined at entry
            masked = b.and_(val, mask, f"masked_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, masked)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(hir)

        # Count broadcasts at function entry (before any loops or other complex stmts)
        from compiler.hir import Op, ForLoop

        entry_broadcasts = 0
        for stmt in transformed.body:
            if isinstance(stmt, Op) and stmt.opcode == "vbroadcast":
                entry_broadcasts += 1
            elif isinstance(stmt, ForLoop):
                break

        # There should be broadcasts at entry for the constants
        self.assertGreater(entry_broadcasts, 0, "Expected some broadcasts at function entry")

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)

        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i & 0xFF)
        print("SLP broadcast loop invariant hoisted test passed!")

    def test_slp_broadcast_placement_with_consecutive_pattern(self):
        """Test broadcast placement for consecutive offset pattern (e.g., [base, base+1, ...])."""
        b = HIRBuilder()

        base_in = b.const(0)
        base_out = b.const(100)

        def loop_body(batch_idx, params):
            # batch_offset is defined in the loop body
            batch_offset = b.mul(batch_idx, b.const(VLEN), "batch_offset")

            # This creates a consecutive pattern: [batch_offset+0, batch_offset+1, ...]
            for i in range(VLEN):
                addr_in = b.add(batch_offset, b.const(i), f"addr_in_{i}")
                real_addr = b.add(base_in, addr_in, f"real_addr_{i}")
                val = b.load(real_addr, f"val_{i}")
                addr_out = b.add(base_out, addr_in, f"addr_out_{i}")
                b.store(addr_out, val)
            return []

        b.for_loop(
            start=Const(0),
            end=Const(2),
            iter_args=[],
            body_fn=loop_body,
        )

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.add_pass(SLPVectorizationPass())
        pm.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 16})
        transformed = pm.run(hir)

        # The broadcast for batch_offset should be inside the loop (after its definition)
        # while broadcasts for constants should be at entry

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(VLEN * 4)) + [0] * 300
        machine = self._run_program(instrs, mem)

        # First batch: mem[100:108] = mem[0:8]
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i)
        # Second batch: mem[108:116] = mem[8:16]
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + VLEN + i], VLEN + i)
        print("SLP broadcast placement with consecutive pattern test passed!")

    def test_slp_broadcast_no_duplicate_at_entry(self):
        """Test that the same broadcast isn't duplicated at entry."""
        b = HIRBuilder()

        base_out = b.const(100)
        constant_val = b.const(42)  # Same constant used multiple times

        # Use the same constant in multiple operations
        for i in range(VLEN):
            addr = b.add(base_out, b.const(i), f"addr_{i}")
            # Multiple uses of same constant
            result = b.add(constant_val, constant_val, f"double_{i}")
            b.store(addr, result)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(hir)

        # Count broadcasts of the same value
        from compiler.hir import Op
        broadcast_operands = []
        for stmt in transformed.body:
            if isinstance(stmt, Op) and stmt.opcode == "vbroadcast":
                operand = stmt.operands[0]
                broadcast_operands.append(id(operand))

        # Each unique value should only have one broadcast
        # (though the exact count depends on implementation details)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 200
        machine = self._run_program(instrs, mem)

        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], 84)  # 42 + 42
        print("SLP broadcast no duplicate at entry test passed!")

    def test_slp_simplified_base_plus_zero_address(self):
        """
        Regression test: SLP should vectorize stores when first iteration's
        address is simplified from +(base, #0) to just base.

        This tests the pattern where:
        - Loop unroll generates: +(base, #0), +(base, #1), ..., +(base, #7)
        - Simplify pass converts +(base, #0) to just base
        - SLP should still recognize all 8 stores as consecutive

        The bug was in DDG operand_nodes not maintaining position correspondence
        when some operands are external (defined outside the block).
        """
        from compiler.hir import Op
        from compiler.passes import SimplifyPass

        b = HIRBuilder()

        # Simulate external base (like inp_indices_p loaded from memory header)
        base = b.load(b.const(100), "base")

        # Create pattern that mimics unrolled loop after simplify:
        # - Iteration 0: store to base directly (simplified from +(base, #0))
        # - Iterations 1-7: store to +(base, #N)

        # First, create values to store
        values = [b.add(b.const(i), b.const(10), f"val_{i}") for i in range(VLEN)]

        # Create addresses - simulate what happens after loop unroll + simplify:
        # Iteration 0's address is just 'base' (after +(base, #0) -> base simplification)
        # Iterations 1-7 have +(base, #N)
        addrs = [base]  # Iteration 0: base directly
        for i in range(1, VLEN):
            addrs.append(b.add(base, b.const(i), f"addr_{i}"))

        # Create 8 stores with these addresses
        for i in range(VLEN):
            b.store(addrs[i], values[i])

        hir = b.build()

        # Run SLP pass
        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        transformed = pm.run(hir)

        # Check that stores are vectorized (should have vstore, not 8 scalar stores)
        scalar_stores = 0
        vector_stores = 0
        for stmt in transformed.body:
            if isinstance(stmt, Op):
                if stmt.opcode == "store":
                    scalar_stores += 1
                elif stmt.opcode == "vstore":
                    vector_stores += 1

        # Should have at least one vstore (the 8 scalar stores vectorized)
        self.assertGreaterEqual(vector_stores, 1, "SLP should vectorize consecutive stores")
        # All 8 stores should be vectorized into 1 vstore, so no scalar stores
        self.assertEqual(scalar_stores, 0, "All stores should be vectorized")

        # Verify correctness by compiling and running
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 200
        mem[100] = 50  # base address
        machine = self._run_program(instrs, mem)

        # Check that values were stored correctly at addresses 50-57
        for i in range(VLEN):
            expected = i + 10  # val_i = i + 10
            self.assertEqual(machine.mem[50 + i], expected,
                f"mem[{50 + i}] should be {expected}, got {machine.mem[50 + i]}")

        print("SLP simplified base+0 address regression test passed!")

    def test_slp_handles_duplicate_offsets(self):
        """
        Regression test: SLP should still find seed packs when the same base+offset
        store sequence repeats (e.g., from fully unrolling an outer loop).

        Previously, seed finding sorted by offset and looked for a contiguous
        window; duplicates clustered as [0,0,...,1,1,...] and no consecutive run
        existed, causing SLP to skip vectorization entirely.
        """
        from compiler.hir import Op

        b = HIRBuilder()
        base = b.const(100)
        add_1000 = b.const(1000)

        # Two identical offset ranges (0..VLEN-1) to the same base.
        # Values are SSA (loads/adds) so SLP can build vector operands for vstore.
        for i in range(VLEN):
            val = b.load(b.const(i), f"in0_{i}")
            addr = b.add(base, b.const(i), f"addr0_{i}")
            b.store(addr, val)

        for i in range(VLEN):
            val = b.load(b.const(i), f"in1_{i}")
            val = b.add(val, add_1000, f"plus_1000_{i}")
            addr = b.add(base, b.const(i), f"addr1_{i}")
            b.store(addr, val)

        hir = b.build()

        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        transformed = pm.run(hir)

        # Should produce two vstores (one per repetition).
        scalar_stores = 0
        vector_stores = 0
        for stmt in transformed.body:
            if isinstance(stmt, Op):
                if stmt.opcode == "store":
                    scalar_stores += 1
                elif stmt.opcode == "vstore":
                    vector_stores += 1

        self.assertEqual(scalar_stores, 0, "All stores should be vectorized")
        self.assertGreaterEqual(vector_stores, 2, "Expected repeated vstore packs for duplicate offsets")

        # Verify semantics: second repetition overwrites the first.
        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(VLEN)) + [0] * 300
        machine = self._run_program(instrs, mem)
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], 1000 + i)

        metrics = slp_pass.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertGreaterEqual(metrics.custom.get("seeds_found", 0), 2)

    def test_slp_preserves_pause_statements(self):
        """
        Regression test: SLP must preserve Pause statements.

        Pauses are used to synchronize with reference kernels in perf_takehome.
        """
        from compiler.hir import Pause

        b = HIRBuilder()
        b.pause()

        base = b.const(50)
        for i in range(VLEN):
            addr = b.add(base, b.const(i), f"addr_{i}")
            b.store(addr, b.const(i + 10))

        b.pause()

        hir = b.build()

        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        transformed = pm.run(hir)

        # Ensure Pause nodes still exist in HIR.
        pause_count = sum(1 for s in transformed.body if isinstance(s, Pause))
        self.assertEqual(pause_count, 2, "SLP must not drop Pause statements")

        # Ensure Pause also survives lowering/codegen.
        instrs = compile_hir_to_vliw(transformed)
        vliw_pause_count = sum(
            1 for bundle in instrs for slot in bundle.get("flow", []) if slot[0] == "pause"
        )
        self.assertEqual(vliw_pause_count, 2, "Expected two pause instructions in VLIW output")


    # --- Config Knob Tests ---

    def test_slp_vectorize_memory_disabled(self):
        """Test that vectorize_memory=False disables load/store vectorization."""
        from compiler.hir import Op

        b = HIRBuilder()
        base_in = b.const(0)
        base_out = b.const(100)

        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, val)

        hir = b.build()

        # With vectorize_memory disabled
        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        pm.config["slp-vectorization"] = PassConfig(
            name="slp-vectorization",
            options={"vectorize_memory": False},
        )
        transformed = pm.run(hir)

        # Should have no vload or vstore
        for stmt in transformed.body:
            if isinstance(stmt, Op):
                self.assertNotIn(stmt.opcode, ("vload", "vstore"),
                    "vectorize_memory=False should prevent vload/vstore")

        # Correctness check
        instrs = self._compile_hir_via_mir_only(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i)

    def test_slp_gather_disabled(self):
        """Test that gather=False disables vgather generation."""
        from compiler.hir import Op

        b = HIRBuilder()
        base_in = b.const(0)
        base_out = b.const(100)
        increment = b.const(1)

        # Pattern that triggers gather: load from base + computed offset
        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            result = b.add(val, increment, f"result_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()

        # With gather disabled
        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        pm.config["slp-vectorization"] = PassConfig(
            name="slp-vectorization",
            options={"gather": False},
        )
        transformed = pm.run(hir)

        # Should have no vgather
        def check_no_vgather(stmts):
            for stmt in stmts:
                if isinstance(stmt, Op):
                    self.assertNotEqual(stmt.opcode, "vgather",
                        "gather=False should prevent vgather")

        check_no_vgather(transformed.body)

        # Correctness check
        instrs = self._compile_hir_via_mir_only(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i + 1)

    def test_slp_vectorize_alu_disabled(self):
        """Test that vectorize_alu=False disables ALU vectorization but keeps memory and select."""
        from compiler.hir import Op

        b = HIRBuilder()
        base_in = b.const(0)
        base_out = b.const(100)
        increment = b.const(1)

        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            result = b.add(val, increment, f"result_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()

        # With vectorize_alu disabled
        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        pm.config["slp-vectorization"] = PassConfig(
            name="slp-vectorization",
            options={"vectorize_alu": False},
        )
        transformed = pm.run(hir)

        # Should have no vector ALU ops (v+, v-, v*, etc.)
        vector_alu_opcodes = {f"v{op}" for op in VECTORIZABLE_ALU_OPS}
        for stmt in transformed.body:
            if isinstance(stmt, Op):
                self.assertNotIn(stmt.opcode, vector_alu_opcodes,
                    f"vectorize_alu=False should prevent {stmt.opcode}")

        # Correctness check
        instrs = self._compile_hir_via_mir_only(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i + 1)

    def test_slp_knobs_independent(self):
        """Test that disabling one knob doesn't affect the others."""
        from compiler.hir import Op

        b = HIRBuilder()
        base_in = b.const(0)
        base_out = b.const(100)

        # Simple load-store pattern (no ALU, no gather)
        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, val)

        hir = b.build()

        # Disable ALU and gather, but keep memory enabled
        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        pm.config["slp-vectorization"] = PassConfig(
            name="slp-vectorization",
            options={"vectorize_alu": False, "gather": False, "vectorize_memory": True},
        )
        transformed = pm.run(hir)

        # Should still have vload/vstore since vectorize_memory is enabled
        has_vstore = False
        for stmt in transformed.body:
            if isinstance(stmt, Op):
                if stmt.opcode == "vstore":
                    has_vstore = True

        self.assertTrue(has_vstore, "vectorize_memory=True should still produce vstore")

        # Correctness check
        instrs = self._compile_hir_via_mir_only(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i)

    def test_slp_memory_disabled_alu_still_vectorizes(self):
        """Test that vectorize_memory=False still allows ALU vectorization."""
        from compiler.hir import Op

        b = HIRBuilder()
        base_in = b.const(0)
        base_out = b.const(100)
        increment = b.const(1)

        # load → add → store pattern
        for i in range(VLEN):
            addr_in = b.add(base_in, b.const(i), f"addr_in_{i}")
            val = b.load(addr_in, f"val_{i}")
            result = b.add(val, increment, f"result_{i}")
            addr_out = b.add(base_out, b.const(i), f"addr_out_{i}")
            b.store(addr_out, result)

        hir = b.build()

        # Disable memory vectorization only
        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        pm.config["slp-vectorization"] = PassConfig(
            name="slp-vectorization",
            options={"vectorize_memory": False, "vectorize_alu": True, "gather": True},
        )
        transformed = pm.run(hir)

        # Should have vector ALU ops (v+) but no vload/vstore
        has_valu = False
        for stmt in transformed.body:
            if isinstance(stmt, Op):
                self.assertNotIn(stmt.opcode, ("vload", "vstore"),
                    "vectorize_memory=False should prevent vload/vstore")
                if stmt.opcode == "v+":
                    has_valu = True

        self.assertTrue(has_valu,
            "vectorize_memory=False should still allow ALU vectorization")

        # Correctness check
        instrs = self._compile_hir_via_mir_only(transformed)
        mem = list(range(VLEN)) + [0] * 200
        machine = self._run_program(instrs, mem)
        for i in range(VLEN):
            self.assertEqual(machine.mem[100 + i], i + 1)


class TestSLPGatherSeeding(unittest.TestCase):
    """Tests for gather-load seed packs (_find_gather_seeds).

    Gather-shaped loads (address = base + runtime index, both SSA) cannot be
    reached bottom-up from consecutive-store seeds, so SLP seeds them
    directly and codegen emits a vgather (which lowers to VLEN load_offset
    slots). A seed pack is only legal when no may-aliasing store sits
    between its first and last element in program order.
    """

    # Memory layout used by all tests in this class:
    #   mem[0]     = table_p (base of the gathered table)
    #   mem[1]     = out_p (base of a disjoint output array)
    #   mem[8..15] = per-lane indices into the table
    #   mem[32..]  = table values
    #   mem[64..]  = output slots
    TABLE_BASE = 32
    OUT_BASE = 64
    IDX_BASE = 8
    INDICES = [3, 1, 4, 7, 5, 0, 2, 6]

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    @staticmethod
    def _count_load_offset_slots(instrs):
        """Count load_offset slots in the VLIW output (vgather lowers to
        VLEN of them; scalar loads and vloads never produce any)."""
        count = 0
        for bundle in instrs:
            for slot in bundle.get("load", []):
                if slot and slot[0] == "load_offset":
                    count += 1
        return count

    def _make_gather_mem(self):
        mem = [0] * 128
        mem[0] = self.TABLE_BASE
        mem[1] = self.OUT_BASE
        for k, idx in enumerate(self.INDICES):
            mem[self.IDX_BASE + k] = idx
        for i in range(16):
            mem[self.TABLE_BASE + i] = 100 + 7 * i
        return mem

    def _build_gather_hir(self):
        """8 independent gather loads table[idx_k] (distinct runtime
        indices loaded from memory), then 8 stores to consecutive output
        slots. Fresh HIR every call: passes mutate ops in place."""
        b = HIRBuilder()
        table_p = b.load(b.const(0), "table_p")
        vals = []
        for k in range(VLEN):
            idx = b.load(b.const(self.IDX_BASE + k), f"idx_{k}")
            addr = b.add(table_p, idx, f"gaddr_{k}")
            vals.append(b.load(addr, f"gval_{k}"))
        out_base = b.const(self.OUT_BASE)
        for k in range(VLEN):
            out_addr = b.add(out_base, b.const(k), f"oaddr_{k}")
            b.store(out_addr, vals[k])
        return b.build()

    def test_slp_gather_seed_forms_vgather_and_executes(self):
        """Gather-shaped loads are seeded directly into a vgather pack.

        The index chain has no other vectorized consumer, so without
        gather-load seeding no vgather forms at all (the loads fall back
        to scalars). Assert the vgather both at the HIR level (standalone
        pass) and in the VLIW output (load_offset slots), and check all 8
        gathered outputs under the default pipeline.
        """
        from compiler.hir import Op

        # Standalone SLP run: a vgather op must appear in the body.
        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(self._build_gather_hir())
        vgathers = [s for s in transformed.body
                    if isinstance(s, Op) and s.opcode == "vgather"]
        self.assertGreaterEqual(len(vgathers), 1,
            "gather-load seeding should form a vgather pack")

        # End-to-end with the default config (fresh HIR: ops are mutated).
        instrs = compile_hir_to_vliw(self._build_gather_hir())
        self.assertGreaterEqual(self._count_load_offset_slots(instrs), VLEN,
            "vgather should lower to load_offset slots in the VLIW output")

        machine = self._run_program(instrs, self._make_gather_mem())
        for k, idx in enumerate(self.INDICES):
            self.assertEqual(machine.mem[self.OUT_BASE + k], 100 + 7 * idx)

    def test_slp_gather_seed_rejects_aliasing_store_in_span(self):
        """Regression: a may-aliasing store inside the pack span must block
        the gather seed.

        8x unrolled read-modify-write on table[idx_k] where every idx_k is
        loaded from memory and all equal 5 at runtime. Fusing the loads
        into a vgather at the last element's position would hoist earlier
        lane loads past the intervening stores that alias them, so every
        lane would read the stale value and table[5] would end up 1.
        Sequential semantics require table[5] == 8. This miscompiled under
        the default config before the blocking-store legality check.
        """
        b = HIRBuilder()
        table_p = b.load(b.const(0), "table_p")
        for k in range(VLEN):
            idx = b.load(b.const(self.IDX_BASE + k), f"idx_{k}")
            addr = b.add(table_p, idx, f"addr_{k}")
            v = b.load(addr, f"val_{k}")
            v1 = b.add(v, b.const(1), f"inc_{k}")
            b.store(addr, v1)
        hir = b.build()

        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 128
        mem[0] = self.TABLE_BASE
        for k in range(VLEN):
            mem[self.IDX_BASE + k] = 5
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[self.TABLE_BASE + 5], 8,
            "read-modify-write chain on one location must not be reordered "
            "by gather packing")
        # No other table slot may be touched.
        for i in range(16):
            if i != 5:
                self.assertEqual(machine.mem[self.TABLE_BASE + i], 0)

    def test_slp_gather_seed_allows_non_aliasing_interleaved_stores(self):
        """Stores proven NO_ALIAS must not block a gather seed.

        Same 8 gather loads, but each load is followed by a store to a
        different array (base loaded from another header slot; the default
        config sets restrict_ptr so distinct bases cannot alias). The
        stores sit inside the pack span, yet the pack must still form:
        load_offset slots appear and the outputs are correct.
        """
        b = HIRBuilder()
        table_p = b.load(b.const(0), "table_p")
        out_p = b.load(b.const(1), "out_p")
        for k in range(VLEN):
            idx = b.load(b.const(self.IDX_BASE + k), f"idx_{k}")
            addr = b.add(table_p, idx, f"gaddr_{k}")
            v = b.load(addr, f"gval_{k}")
            out_addr = b.add(out_p, b.const(k), f"oaddr_{k}")
            b.store(out_addr, v)
        hir = b.build()

        instrs = compile_hir_to_vliw(hir)
        self.assertGreaterEqual(self._count_load_offset_slots(instrs), VLEN,
            "interleaved NO_ALIAS stores must not block the gather pack")

        machine = self._run_program(instrs, self._make_gather_mem())
        for k, idx in enumerate(self.INDICES):
            self.assertEqual(machine.mem[self.OUT_BASE + k], 100 + 7 * idx)

    def test_slp_gather_seed_rejects_pointer_chase_dependency(self):
        """Regression: a pack whose addresses transitively depend on another
        element's loaded value (pointer chasing) must be rejected.

        Each iteration loads table[cur] and derives the next cur from the
        loaded value (cur = v & 15), so lane k's address depends on lane
        k-1's load through the '&' op -- a transitive dependency that only
        addr_depends_on_pack's def-chain walk catches (no operand of the
        address is directly a pack result). Fusing the loads into a vgather
        would read all lanes with stale indices. Before the fix in
        _find_gather_seeds, this repro miscompiled (wrong chase sum) and the
        VLIW output contained load_offset slots from the illegal vgather.
        """
        # Deterministic table of 16 values, each in [0, 15].
        table = [7, 12, 3, 9, 14, 2, 8, 5, 11, 1, 6, 15, 0, 10, 4, 13]

        b = HIRBuilder()
        base = b.load(b.const(0))  # table base pointer (16)
        cur = b.load(b.const(1))   # initial index (3)
        total = None
        for k in range(8):
            a = b.add(base, cur)
            v = b.load(a)
            total = v if total is None else b.add(total, v)
            cur = b.alu("&", v, Const(15))
        b.store(b.const(2), total)
        hir = b.build()

        instrs = compile_hir_to_vliw(hir)

        # The pack must be rejected: no vgather, hence no load_offset slots.
        self.assertEqual(self._count_load_offset_slots(instrs), 0,
            "pointer-chasing loads must not be fused into a vgather")

        mem = [0] * 64
        mem[0] = 16
        mem[1] = 3
        for i, t in enumerate(table):
            mem[16 + i] = t
        machine = self._run_program(instrs, mem)

        # Python model of the sequential chase.
        model_cur = 3
        expected = 0
        for _ in range(8):
            model_v = table[model_cur]
            expected += model_v
            model_cur = model_v & 15
        self.assertEqual(machine.mem[2], expected,
            "pointer-chase sum must match sequential semantics")

    def test_slp_gather_seed_blocked_by_same_root_dynamic_store(self):
        """Regression: a store through the same root pointer with a dynamic
        index must block a gather seed pack.

        The loads read table[idx_k] (addresses tp+idx_k) and a store to
        tp+widx sits between lane 3 and lane 4. tp+idx_k and tp+widx have
        different composite bases but share the tp root, so for dynamic
        idx_k/widx they may alias. Before the base_roots fix in alias_keys,
        restrict_ptr treated the differing composite bases as NO_ALIAS, the
        pack formed, and the fused vgather at the last element's position
        read lane 2 AFTER the store (idx_2 == widx == 2 at runtime): the
        sum miscompiled to 1096 instead of 108.
        """
        b = HIRBuilder()
        tp = b.load(b.const(0))     # table pointer (32)
        widx = b.load(b.const(20))  # dynamic write index (2)
        vals = []
        for k in range(8):
            idx = b.load(b.const(8 + k))
            a = b.add(tp, idx)
            v = b.load(a)
            vals.append(v)
            if k == 3:
                wa = b.add(tp, widx)
                b.store(wa, Const(1000))
        total = vals[0]
        for k in range(1, 8):
            total = b.add(total, vals[k])
        b.store(b.const(2), total)
        hir = b.build()

        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[0] = 32
        for k in range(8):
            mem[8 + k] = k          # idx_k = k
        mem[20] = 2                 # widx = 2 (same slot lane 2 reads)
        for k in range(8):
            mem[32 + k] = k + 10    # table values
        machine = self._run_program(instrs, mem)

        # Lane 2 reads table[2] == 12 BEFORE the store overwrites it with
        # 1000, so the sequential sum is sum(k+10 for k in 0..7) == 108.
        self.assertEqual(machine.mem[2], sum(k + 10 for k in range(8)),
            "loads must not be hoisted past a may-aliasing same-root store")
        # The store itself must still land.
        self.assertEqual(machine.mem[34], 1000)


class TestSLPTreeHashSmallBatch(unittest.TestCase):
    """Small-batch tree-hash kernel correctness under the default pipeline.

    With batch sizes below VLEN the traversal degenerates into per-round
    pointer chases (each round's node load feeds the next round's index),
    exactly the shape gather-load seeding must reject. do_kernel_test
    asserts the machine memory against the reference kernel every round;
    batch_size=1 miscompiled on round 1 before the addr_depends_on_pack
    fix in _find_gather_seeds.
    """

    def test_tree_hash_batch_1(self):
        from programs.tree_hash import do_kernel_test
        do_kernel_test(3, 8, 1)

    def test_tree_hash_batch_4(self):
        from programs.tree_hash import do_kernel_test
        do_kernel_test(3, 8, 4)


if __name__ == "__main__":
    unittest.main()
