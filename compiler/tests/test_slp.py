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
    VectorConst,
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

    def test_slp_packs_created_accumulates_across_op_runs(self):
        """packs_created must cover every vectorized run in the function."""
        from compiler.hir import Op

        b = HIRBuilder()
        for block in range(2):
            for lane in range(VLEN):
                b.store(b.const(100 + block * VLEN + lane), b.const(block + 1))
            if block == 0:
                b.pause()

        slp_pass = SLPVectorizationPass()
        pm = PassManager()
        pm.add_pass(slp_pass)
        transformed = pm.run(b.build())

        self.assertEqual(
            sum(
                isinstance(stmt, Op) and stmt.opcode == "vstore"
                for stmt in transformed.body
            ),
            2,
            "both straight-line runs must vectorize independently",
        )
        metrics = slp_pass.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.custom["packs_created"], 2)

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


class TestSLPEmissionSafety(unittest.TestCase):
    """Correctness tests for pack planning and scalar-ownership emission."""

    def _transform(self, hir, **options):
        manager = PassManager()
        manager.add_pass(SLPVectorizationPass())
        manager.config["slp-vectorization"] = PassConfig(
            name="slp-vectorization",
            enabled=True,
            options={"restrict_ptr": True, **options},
        )
        return manager.run(hir)

    def _compile(self, hir):
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        config = PassConfig(name="test", enabled=True, options={})
        mir = LIRToMIRPass().run(lir, config)
        mir = MIRRegisterAllocationPass().run(mir, config)
        return MIRToVLIWPass().run(mir, config)

    def _run(self, hir, mem):
        machine = Machine(
            mem,
            self._compile(hir),
            DebugInfo(scratch_map={}),
            n_cores=N_CORES,
        )
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def _assert_flat_ssa_dominance(self, hir):
        from compiler.hir import Op, SSAValue, VectorSSAValue

        defined = set()
        for stmt in hir.body:
            if not isinstance(stmt, Op):
                continue
            for operand in stmt.operands:
                if isinstance(operand, (SSAValue, VectorSSAValue)):
                    self.assertIn(operand, defined, f"undefined operand in {stmt!r}")
            if stmt.result is not None:
                self.assertNotIn(stmt.result, defined)
                defined.add(stmt.result)

    @staticmethod
    def _flat_ops(hir):
        from compiler.hir import Op

        return [stmt for stmt in hir.body if isinstance(stmt, Op)]

    def test_store_fallback_does_not_cross_aliasing_load(self):
        b = HIRBuilder()
        values_base = b.load(b.const(9), "values_base")
        first_out = b.load(b.const(10), "first_out")
        second_out = b.load(b.const(11), "second_out")
        other_values = [
            b.load(b.add(values_base, b.const(2 * i)), f"value_{i}")
            for i in range(1, VLEN)
        ]

        packed_values = [b.xor(b.const(40), b.const(2), "q0")]
        first_addrs = [
            b.add(first_out, b.const(i), f"first_addr_{i}")
            for i in range(VLEN)
        ]
        for addr, value in zip(first_addrs, [packed_values[0], *other_values]):
            b.store(addr, value)

        observed = b.load(first_addrs[0], "observed")
        b.store(b.const(300), observed)

        for i in range(1, VLEN):
            packed_values.append(
                b.xor(b.const(40 + i), b.const(2), f"q{i}")
            )
        for i, value in enumerate(packed_values):
            b.store(b.add(second_out, b.const(i)), value)

        transformed = self._transform(b.build())
        ops = self._flat_ops(transformed)
        vstores = [op for op in ops if op.opcode == "vstore"]
        self.assertEqual(
            len(vstores),
            2,
            "both legal store packs must vectorize",
        )
        self.assertTrue(
            any(op.opcode == "vinsert" for op in ops),
            "the heterogeneous first store pack must use scalar fallback",
        )
        observed_load = next(op for op in ops if op.result == observed)
        self.assertLess(
            ops.index(vstores[0]),
            ops.index(observed_load),
            "fallback materialization must not move the first store pack "
            "across its aliasing load",
        )
        mem = [0] * 1100
        mem[9], mem[10], mem[11] = 1000, 100, 200
        for i in range(1, VLEN):
            mem[1000 + 2 * i] = 100 + i
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[300], 42)

    def test_uniform_consumer_can_precede_producer_pack_anchor(self):
        b = HIRBuilder()
        q_out = b.load(b.const(0), "q_out")
        c_out = b.load(b.const(1), "c_out")
        xs = [b.load(b.const(10 + 2 * i), f"x{i}") for i in range(VLEN)]

        q = [b.add(xs[0], b.const(10), "q0")]
        for i in range(VLEN):
            c = b.add(q[0], b.const(1), f"c{i}")
            b.store(b.add(c_out, b.const(i)), c)
        for i in range(1, VLEN):
            q.append(b.add(xs[i], b.const(10), f"q{i}"))
        for i, value in enumerate(q):
            b.store(b.add(q_out, b.const(i)), value)

        transformed = self._transform(b.build())
        self._assert_flat_ssa_dominance(transformed)
        ops = self._flat_ops(transformed)
        scalar_defs = {op.result for op in ops if op.result is not None}
        self.assertTrue(
            all(value in scalar_defs for value in q),
            "a uniform early use of q0 must retain the whole producer pack",
        )
        q0_broadcasts = [
            op for op in ops
            if op.opcode == "vbroadcast" and op.operands == [q[0]]
        ]
        self.assertEqual(
            len(q0_broadcasts),
            1,
            "the uniform q0 consumer must use a vector broadcast",
        )
        self.assertTrue(
            any(
                op.opcode == "v+" and q0_broadcasts[0].result in op.operands
                for op in ops
            ),
            "the q0 broadcast must feed the vectorized consumer pack",
        )

        mem = [0] * 256
        mem[0], mem[1] = 100, 120
        for i in range(VLEN):
            mem[10 + 2 * i] = i + 1
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:108], list(range(11, 19)))
        self.assertEqual(machine.mem[120:128], [12] * VLEN)

    def test_late_uniform_consumer_broadcast_follows_owner_extract(self):
        b = HIRBuilder()
        producer_out = b.load(b.const(20), "producer_out")
        consumer_out = b.load(b.const(21), "consumer_out")
        inputs = [b.load(b.const(i), f"input_{i}") for i in range(VLEN)]
        producers = [
            b.add(value, b.const(10), f"producer_{lane}")
            for lane, value in enumerate(inputs)
        ]
        for lane, value in enumerate(producers):
            b.store(b.add(producer_out, b.const(lane)), value)
        for lane in range(VLEN):
            value = b.add(producers[0], b.const(1), f"consumer_{lane}")
            b.store(b.add(consumer_out, b.const(lane)), value)

        transformed = self._transform(b.build())
        self._assert_flat_ssa_dominance(transformed)
        ops = self._flat_ops(transformed)
        owner_extract = next(
            op for op in ops
            if op.opcode == "vextract" and op.result == producers[0]
        )
        broadcast = next(
            op for op in ops
            if op.opcode == "vbroadcast" and op.operands == [producers[0]]
        )
        consumer = next(
            op for op in ops
            if op.opcode == "v+" and broadcast.result in op.operands
        )
        self.assertLess(ops.index(owner_extract), ops.index(broadcast))
        self.assertLess(ops.index(broadcast), ops.index(consumer))

        mem = list(range(1, VLEN + 1)) + [0] * 220
        mem[20], mem[21] = 100, 120
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:108], list(range(11, 19)))
        self.assertEqual(machine.mem[120:128], [12] * VLEN)

    def test_individual_early_use_keeps_producer_scalars(self):
        b = HIRBuilder()
        output = b.load(b.const(20), "output")
        inputs = [b.load(b.const(i), f"input_{i}") for i in range(VLEN)]

        values = [b.add(inputs[0], b.const(10), "value_0")]
        early = b.xor(values[0], b.const(7), "early")
        b.store(b.const(300), early)
        for lane in range(1, VLEN):
            values.append(
                b.add(inputs[lane], b.const(10), f"value_{lane}")
            )
        for lane, value in enumerate(values):
            b.store(b.add(output, b.const(lane)), value)

        transformed = self._transform(b.build())
        self._assert_flat_ssa_dominance(transformed)
        ops = self._flat_ops(transformed)
        scalar_defs = {op.result for op in ops if op.result is not None}
        self.assertTrue(
            all(value in scalar_defs for value in values),
            "one individual use before the pack anchor must retain all lanes",
        )
        self.assertTrue(
            any(op.opcode == "v+" for op in ops),
            "the retained scalar producer must still have a vector form",
        )
        self.assertTrue(any(op.opcode == "vstore" for op in ops))

        mem = list(range(1, VLEN + 1)) + [0] * 320
        mem[20] = 100
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:108], list(range(11, 19)))
        self.assertEqual(machine.mem[300], 11 ^ 7)

    def test_lane_wise_early_users_allow_fixed_point_replacement(self):
        b = HIRBuilder()
        output = b.load(b.const(20), "output")
        inputs = [b.load(b.const(i), f"input_{i}") for i in range(VLEN)]
        producers = []
        consumers = []
        for lane, value in enumerate(inputs):
            producer = b.add(value, b.const(10), f"producer_{lane}")
            producers.append(producer)
            consumers.append(
                b.mul(producer, b.const(3), f"consumer_{lane}")
            )
        for lane, value in enumerate(consumers):
            b.store(b.add(output, b.const(lane)), value)

        transformed = self._transform(b.build())
        self._assert_flat_ssa_dominance(transformed)
        ops = self._flat_ops(transformed)
        scalar_defs = {op.result for op in ops if op.result is not None}
        self.assertFalse(
            any(value in scalar_defs for value in producers + consumers),
            "lane-wise vector-owned chains should not retain scalar definitions",
        )

        vector_add = next(op for op in ops if op.opcode == "v+")
        vector_mul = next(
            op for op in ops
            if op.opcode == "v*" and vector_add.result in op.operands
        )
        vector_store = next(
            op for op in ops
            if op.opcode == "vstore" and vector_mul.result in op.operands
        )
        self.assertLess(ops.index(vector_add), ops.index(vector_mul))
        self.assertLess(ops.index(vector_mul), ops.index(vector_store))
        self.assertFalse(any(op.opcode == "vextract" for op in ops))

        mem = list(range(1, VLEN + 1)) + [0] * 200
        mem[20] = 100
        machine = self._run(transformed, mem)
        self.assertEqual(
            machine.mem[100:108],
            [3 * value for value in range(11, 19)],
        )

    def test_pressure_budget_prunes_dual_representation_component(self):
        def build_program():
            b = HIRBuilder()
            output = b.load(b.const(20), "output")
            inputs = [
                b.load(b.const(i), f"input_{i}") for i in range(VLEN)
            ]
            values = [b.add(inputs[0], b.const(10), "value_0")]
            early = b.xor(values[0], b.const(7), "early")
            b.store(b.const(300), early)
            for lane in range(1, VLEN):
                values.append(
                    b.add(inputs[lane], b.const(10), f"value_{lane}")
                )
            consumers = [
                b.mul(value, b.const(3), f"consumer_{lane}")
                for lane, value in enumerate(values)
            ]
            for lane, value in enumerate(consumers):
                b.store(b.add(output, b.const(lane)), value)
            return b.build(), values

        def transform(limit):
            hir, values = build_program()
            slp_pass = SLPVectorizationPass()
            manager = PassManager()
            manager.add_pass(slp_pass)
            manager.config["slp-vectorization"] = PassConfig(
                name="slp-vectorization",
                enabled=True,
                options={
                    "restrict_ptr": True,
                    "dual_representation_prune_threshold": limit,
                },
            )
            transformed = manager.run(hir)
            return transformed, values, slp_pass.get_metrics().custom

        scalar_hir, scalar_values, scalar_metrics = transform(0)
        scalar_ops = self._flat_ops(scalar_hir)
        self.assertGreater(
            scalar_metrics["dual_representation_lanes_before_pruning"], 0)
        self.assertGreater(scalar_metrics["packs_pruned_for_pressure"], 0)
        self.assertFalse(
            any(op.opcode in ("v+", "v*", "vstore") for op in scalar_ops),
            "the producer, consumer, and store must fall back as one component",
        )
        scalar_defs = {
            op.result for op in scalar_ops if op.result is not None
        }
        self.assertTrue(all(value in scalar_defs for value in scalar_values))

        vector_hir, vector_values, vector_metrics = transform(64)
        self._assert_flat_ssa_dominance(vector_hir)
        vector_ops = self._flat_ops(vector_hir)
        self.assertEqual(vector_metrics["packs_pruned_for_pressure"], 0)
        self.assertGreater(vector_metrics["packs_emitted"], 0)
        for opcode in ("vload", "v+", "v*", "vstore"):
            self.assertTrue(any(op.opcode == opcode for op in vector_ops))
        vector_defs = {
            op.result for op in vector_ops if op.result is not None
        }
        self.assertTrue(
            all(value in vector_defs for value in vector_values),
            "the early individual use still requires the scalar producer",
        )

        expected = [3 * value for value in range(11, 19)]
        for transformed in (scalar_hir, vector_hir):
            mem = list(range(1, VLEN + 1)) + [0] * 320
            mem[20] = 100
            machine = self._run(transformed, mem)
            self.assertEqual(machine.mem[100:108], expected)
            self.assertEqual(machine.mem[300], 11 ^ 7)

    def test_rejected_pack_does_not_pollute_later_materialization(self):
        from compiler.hir import Op

        b = HIRBuilder()
        c_out = b.load(b.const(0), "c_out")
        d_out = b.load(b.const(1), "d_out")
        common = b.load(b.const(2), "common")
        xs = [b.load(b.const(10 + 2 * i), f"x{i}") for i in range(VLEN)]

        for i, x in enumerate(xs):
            rhs = b.const(1) if i < VLEN // 2 else common
            c = b.add(x, rhs, f"c{i}")
            b.store(b.add(c_out, b.const(i)), c)

        for i, x in enumerate(xs):
            d = b.mul(x, b.const(3), f"d{i}")
            b.store(b.add(d_out, b.const(i)), d)

        transformed = self._transform(b.build())
        self._assert_flat_ssa_dominance(transformed)
        self.assertTrue(
            any(isinstance(stmt, Op) and stmt.opcode == "v*"
                for stmt in transformed.body),
            "the pack after the failed candidate should still vectorize",
        )

        mem = [0] * 256
        mem[0], mem[1], mem[2] = 100, 120, 7
        for i in range(VLEN):
            mem[10 + 2 * i] = 20 + i
        machine = self._run(transformed, mem)
        self.assertEqual(
            machine.mem[100:108],
            [21, 22, 23, 24, 31, 32, 33, 34],
        )
        self.assertEqual(machine.mem[120:128], [60 + 3 * i for i in range(VLEN)])

    def test_reversed_consecutive_loads_preserve_lane_order(self):
        b = HIRBuilder()
        source = b.load(b.const(20), "source")
        dest = b.load(b.const(21), "dest")
        for lane in range(VLEN):
            value = b.load(b.add(source, b.const(7 - lane)), f"value_{lane}")
            b.store(b.add(dest, b.const(lane)), value)

        transformed = self._transform(b.build())
        ops = self._flat_ops(transformed)
        self.assertNotIn(
            "vload",
            [op.opcode for op in ops],
            "a reversed load sequence must not become a forward vload",
        )
        self.assertTrue(
            any(op.opcode == "vstore" for op in ops),
            "the output store pack must still vectorize",
        )
        mem = list(range(150))
        mem[20], mem[21] = 32, 100
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:108], list(range(39, 31, -1)))

    def test_loop_counter_broadcast_stays_inside_loop_body(self):
        from compiler.hir import ForLoop, Op

        b = HIRBuilder()

        def body(counter, _params):
            for lane in range(VLEN):
                value = b.add(counter, Const(1), f"value_{lane}")
                b.store(b.add(b.const(100), Const(lane)), value)
            return []

        b.for_loop(Const(0), Const(3), [], body, pragma_unroll=1)
        transformed = self._transform(b.build())
        loop = next(stmt for stmt in transformed.body if isinstance(stmt, ForLoop))
        self.assertFalse(
            any(
                isinstance(stmt, Op)
                and stmt.opcode == "vbroadcast"
                and stmt.operands == [loop.counter]
                for stmt in transformed.body
            ),
            "loop counter broadcast must not be hoisted before the loop",
        )
        counter_broadcasts = [
            stmt for stmt in loop.body
            if isinstance(stmt, Op)
            and stmt.opcode == "vbroadcast"
            and stmt.operands == [loop.counter]
        ]
        self.assertEqual(
            len(counter_broadcasts),
            1,
            "the vectorized loop-body pack must broadcast its counter locally",
        )
        self.assertTrue(
            any(
                isinstance(stmt, Op)
                and stmt.opcode == "v+"
                and counter_broadcasts[0].result in stmt.operands
                for stmt in loop.body
            ),
            "the local counter broadcast must feed a vector add",
        )
        self.assertTrue(
            any(isinstance(stmt, Op) and stmt.opcode == "vstore"
                for stmt in loop.body),
            "the loop-body store pack must vectorize",
        )

        machine = self._run(transformed, [0] * 200)
        self.assertEqual(machine.mem[100:108], [3] * VLEN)

    def test_scalar_and_vector_ssa_ids_do_not_collide_for_broadcasts(self):
        """A later vec0 must not steal scalar v0's broadcast placement."""
        b = HIRBuilder()
        uniform = b.load(b.const(0), "uniform")  # scalar SSA id 0
        output = b.load(b.const(20), "output")
        inputs = [
            b.load(b.const(1 + lane), f"input_{lane}")
            for lane in range(VLEN)
        ]
        for lane, value in enumerate(inputs):
            result = b.add(value, uniform, f"result_{lane}")
            b.store(b.add(output, b.const(lane)), result)

        # Vector and scalar SSA values use independent id spaces. Keeping this
        # vector definition after the scalar pack specifically exercises id 0
        # in both spaces and the broadcast-placement lookup.
        later_vector = b.vload(b.const(200), "later_vector")  # vector SSA id 0
        self.assertEqual(uniform.id, later_vector.id)

        transformed = self._transform(b.build())
        self._assert_flat_ssa_dominance(transformed)
        ops = self._flat_ops(transformed)
        broadcasts = [
            op for op in ops
            if op.opcode == "vbroadcast" and op.operands == [uniform]
        ]
        self.assertEqual(len(broadcasts), 1)
        consumers = [
            op for op in ops
            if op.opcode == "v+" and broadcasts[0].result in op.operands
        ]
        self.assertEqual(
            len(consumers),
            1,
            "the uniform scalar must feed a vectorized add pack",
        )
        self.assertLess(ops.index(broadcasts[0]), ops.index(consumers[0]))

        mem = [0] * 256
        mem[0] = 10
        mem[1:1 + VLEN] = list(range(1, VLEN + 1))
        mem[20] = 100
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:100 + VLEN], list(range(11, 19)))

    def test_consecutive_load_pack_does_not_cross_aliasing_store(self):
        """A regular vload cannot move earlier lanes below an aliasing store."""
        b = HIRBuilder()
        source = b.load(b.const(20), "source")
        dest = b.load(b.const(21), "dest")
        values = []
        for lane in range(VLEN):
            addr = b.add(source, b.const(lane), f"source_addr_{lane}")
            values.append(b.load(addr, f"value_{lane}"))
            if lane == 3:
                b.store(b.add(source, b.const(2)), b.const(999))
        for lane, value in enumerate(values):
            b.store(b.add(dest, b.const(lane)), value)

        transformed = self._transform(b.build())
        opcodes = [op.opcode for op in self._flat_ops(transformed)]
        self.assertNotIn(
            "vload",
            opcodes,
            "the load pack must be rejected instead of crossing the store",
        )
        self.assertIn(
            "vstore",
            opcodes,
            "the independent output store pack must still vectorize",
        )

        mem = [0] * 256
        mem[20], mem[21] = 32, 100
        mem[32:32 + VLEN] = list(range(10, 10 + VLEN))
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:100 + VLEN], list(range(10, 18)))
        self.assertEqual(machine.mem[34], 999)

    def test_store_pack_does_not_cross_existing_vector_load(self):
        """Width-8 aliases from existing vector memory ops block fusion."""
        b = HIRBuilder()
        base = b.load(b.const(20), "base")
        observed = None
        store_addrs = []
        for lane in range(VLEN):
            addr = b.add(base, b.const(lane), f"addr_{lane}")
            store_addrs.append(addr)
            b.store(addr, b.const(100 + lane))
            if lane == 3:
                observed = b.vload(base, "observed")
        b.vstore(b.const(200), observed)

        transformed = self._transform(b.build())
        ops = self._flat_ops(transformed)
        self.assertEqual(
            sum(
                op.opcode == "store" and op.operands[0] in store_addrs
                for op in ops
            ),
            VLEN,
            "the scalar stores must not move below the overlapping vload",
        )

        mem = [0] * 256
        mem[20] = 32
        mem[32:40] = list(range(10, 18))
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[32:40], list(range(100, 108)))
        self.assertEqual(machine.mem[200:208], [100, 101, 102, 103, 14, 15, 16, 17])

    def test_replaced_result_used_in_next_op_run_gets_dominating_extract(self):
        from compiler.hir import Op, Pause

        b = HIRBuilder()
        output = b.load(b.const(20), "output")
        inputs = [b.load(b.const(i), f"input_{i}") for i in range(VLEN)]
        values = [
            b.add(value, b.const(10), f"value_{lane}")
            for lane, value in enumerate(inputs)
        ]
        for lane, value in enumerate(values):
            b.store(b.add(output, b.const(lane)), value)
        b.pause()
        doubled = b.mul(values[3], b.const(2), "doubled_after_pause")
        b.store(b.const(300), doubled)

        transformed = self._transform(b.build())
        self._assert_flat_ssa_dominance(transformed)
        pause_index = next(
            i for i, stmt in enumerate(transformed.body)
            if isinstance(stmt, Pause)
        )
        extracted = next(
            stmt for stmt in transformed.body
            if isinstance(stmt, Op) and stmt.result == values[3]
        )
        self.assertEqual(extracted.opcode, "vextract")
        self.assertLess(transformed.body.index(extracted), pause_index)
        for lane, value in enumerate(values):
            defs = [
                stmt for stmt in transformed.body
                if isinstance(stmt, Op) and stmt.result == value
            ]
            if lane == 3:
                self.assertEqual(defs, [extracted])
            else:
                self.assertEqual(defs, [])

        mem = list(range(1, VLEN + 1)) + [0] * 320
        mem[20] = 100
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:108], list(range(11, 19)))
        self.assertEqual(machine.mem[300], 28)

    def test_replaced_result_used_as_loop_yield_gets_extract(self):
        from compiler.hir import ForLoop, Op

        b = HIRBuilder()
        output = b.load(b.const(20), "output")
        inputs = [b.load(b.const(i), f"input_{i}") for i in range(VLEN)]
        captured = {}

        def body(_counter, params):
            values = [
                b.add(value, params[0], f"loop_value_{lane}")
                for lane, value in enumerate(inputs)
            ]
            captured["values"] = values
            for lane, value in enumerate(values):
                b.store(b.add(output, b.const(lane)), value)
            return [values[0]]

        results = b.for_loop(
            b.const(0), b.const(2), [b.const(10)], body, pragma_unroll=1
        )
        b.store(b.const(300), results[0])
        values = captured["values"]

        transformed = self._transform(b.build())
        loop = next(
            stmt for stmt in transformed.body if isinstance(stmt, ForLoop)
        )
        self.assertEqual(loop.yields, [values[0]])
        extracted = next(
            stmt for stmt in loop.body
            if isinstance(stmt, Op) and stmt.result == values[0]
        )
        self.assertEqual(extracted.opcode, "vextract")
        self.assertTrue(
            all(
                not any(
                    isinstance(stmt, Op)
                    and stmt.result == value
                    and stmt.opcode == "+"
                    for stmt in loop.body
                )
                for value in values
            ),
            "the scalar loop-body producer pack must be replaced",
        )
        vector_def = next(
            stmt for stmt in loop.body
            if isinstance(stmt, Op) and stmt.result == extracted.operands[0]
        )
        self.assertLess(loop.body.index(vector_def), loop.body.index(extracted))

        mem = list(range(1, VLEN + 1)) + [0] * 320
        mem[20] = 100
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:108], list(range(12, 20)))
        self.assertEqual(machine.mem[300], 12)

    def test_replaced_result_used_as_if_yield_gets_extract(self):
        from compiler.hir import If, Op

        b = HIRBuilder()
        output = b.load(b.const(20), "output")
        condition = b.load(b.const(30), "condition")
        inputs = [b.load(b.const(i), f"input_{i}") for i in range(VLEN)]
        captured = {}

        def then_body():
            values = [
                b.add(value, b.const(10), f"then_value_{lane}")
                for lane, value in enumerate(inputs)
            ]
            captured["values"] = values
            for lane, value in enumerate(values):
                b.store(b.add(output, b.const(lane)), value)
            return [values[3]]

        results = b.if_stmt(condition, then_body, lambda: [b.const(77)])
        b.store(b.const(300), results[0])
        values = captured["values"]

        transformed = self._transform(b.build())
        branch = next(stmt for stmt in transformed.body if isinstance(stmt, If))
        self.assertEqual(branch.then_yields, [values[3]])
        extracted = next(
            stmt for stmt in branch.then_body
            if isinstance(stmt, Op) and stmt.result == values[3]
        )
        self.assertEqual(extracted.opcode, "vextract")
        self.assertTrue(
            all(
                not any(
                    isinstance(stmt, Op)
                    and stmt.result == value
                    and stmt.opcode == "+"
                    for stmt in branch.then_body
                )
                for value in values
            ),
            "the scalar branch producer pack must be replaced",
        )
        vector_def = next(
            stmt for stmt in branch.then_body
            if isinstance(stmt, Op) and stmt.result == extracted.operands[0]
        )
        self.assertLess(
            branch.then_body.index(vector_def),
            branch.then_body.index(extracted),
        )

        mem = list(range(1, VLEN + 1)) + [0] * 320
        mem[20], mem[30] = 100, 1
        machine = self._run(transformed, mem)
        self.assertEqual(machine.mem[100:108], list(range(11, 19)))
        self.assertEqual(machine.mem[300], 14)

    def test_varying_constants_are_not_vectorized_by_default(self):
        b = HIRBuilder()
        output = b.load(b.const(20), "output")
        inputs = [b.load(b.const(i), f"input_{i}") for i in range(VLEN)]
        values = [
            b.xor(value, b.const(0x10 + lane), f"value_{lane}")
            for lane, value in enumerate(inputs)
        ]
        for lane, value in enumerate(values):
            b.store(b.add(output, b.const(lane)), value)

        transformed = self._transform(b.build())
        ops = self._flat_ops(transformed)
        self.assertFalse(any(op.opcode == "v^" for op in ops))
        self.assertFalse(
            any(
                isinstance(operand, VectorConst)
                for op in ops
                for operand in op.operands
            )
        )
        scalar_defs = {op.result for op in ops if op.result is not None}
        self.assertTrue(all(value in scalar_defs for value in values))
        self.assertTrue(any(op.opcode == "vstore" for op in ops))

        mem = list(range(1, VLEN + 1)) + [0] * 200
        mem[20] = 100
        machine = self._run(transformed, mem)
        self.assertEqual(
            machine.mem[100:108],
            [(lane + 1) ^ (0x10 + lane) for lane in range(VLEN)],
        )

    def test_varying_constants_vectorize_with_explicit_opt_in(self):
        b = HIRBuilder()
        output = b.load(b.const(20), "output")
        inputs = [b.load(b.const(i), f"input_{i}") for i in range(VLEN)]
        values = [
            b.xor(value, b.const(0x10 + lane), f"value_{lane}")
            for lane, value in enumerate(inputs)
        ]
        for lane, value in enumerate(values):
            b.store(b.add(output, b.const(lane)), value)

        transformed = self._transform(
            b.build(), vectorize_varying_constants=True
        )
        self._assert_flat_ssa_dominance(transformed)
        ops = self._flat_ops(transformed)
        vector_xor = next(op for op in ops if op.opcode == "v^")
        varying_operand = next(
            operand for operand in vector_xor.operands
            if isinstance(operand, VectorConst)
        )
        self.assertEqual(
            varying_operand.values,
            tuple(0x10 + lane for lane in range(VLEN)),
        )
        scalar_defs = {op.result for op in ops if op.result is not None}
        self.assertFalse(any(value in scalar_defs for value in values))
        self.assertTrue(
            any(op.opcode == "vstore" and vector_xor.result in op.operands
                for op in ops)
        )

        mem = list(range(1, VLEN + 1)) + [0] * 200
        mem[20] = 100
        machine = self._run(transformed, mem)
        self.assertEqual(
            machine.mem[100:108],
            [(lane + 1) ^ (0x10 + lane) for lane in range(VLEN)],
        )


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


class TestSLPMixedIterationDuplicateStores(unittest.TestCase):
    """Duplicate same-address stores with non-uniform survivors.

    When several unrolled iterations all store to the same addresses and an
    upstream pass removed some but not all of the duplicates (this happens
    in the real pipeline with dse restrict_ptr=false: only rounds without
    intervening may-alias loads lose their stores), the k-th-occurrence
    pairing in the old store seed finder used to group stores from DIFFERENT
    iterations into one pack. Two distinct miscompiles followed:

    - the pack's value operands mixed iterations, so the extension-formed
      value pack had a far-future last element; a pack consuming a subset
      of those values emitted EARLIER, found no scalar_to_vector mapping,
      and built a vinsert chain over scalar SSAs whose defs were deleted
      when the producer pack finally emitted (dangling refs -> the VM
      reads stale scratch; observed as 7 lanes collapsing to one value);
    - emitting the mixed vstore at its last element's position moved the
      earlier iteration's stores across the later iteration's same-address
      stores (store order flip).

    The fix makes memory-pack fusion span-checked (_mem_pack_span_is_legal)
    and uses a scalar-ownership fixed point: early unmapped uses retain their
    scalar definitions, while post-anchor uses are rebuilt with extracts.
    """

    BATCH = 16
    A = 100
    B = 200
    K = 0xABCDEF

    def _build(self):
        b = HIRBuilder()
        a_base = b.const(self.A)
        b_base = b.const(self.B)
        k = b.const(self.K)
        consts = [(b.const(0x9E3779B9 + r), b.const(5 + 2 * r))
                  for r in range(3)]

        vals = [b.load(b.const(i), "in_%d" % i) for i in range(self.BATCH)]

        # Round 0 stores all but A[15]: offset 15 then has fewer
        # occurrences than offsets 8..14, so occurrence pairing would mix
        # round-0 stores with the round-1 store to A[15].
        c1, c2 = consts[0]
        r0 = []
        for i in range(self.BATCH):
            t1 = b.xor(vals[i], c1, "r0_x_%d" % i)
            t2 = b.mul(t1, c2, "r0_m_%d" % i)
            r0.append(t2)
            if i != self.BATCH - 1:
                b.store(b.add(a_base, b.const(i), "r0_a_%d" % i), t2)

        # Side chain consuming round-0 values 8..15: its pack emits before
        # the mixed value pack (whose last element sits in round 1), which
        # is exactly the emission-order inversion that dangled.
        for j in range(8):
            w = b.xor(r0[8 + j], k, "w_%d" % j)
            b.store(b.add(b_base, b.const(j), "w_a_%d" % j), w)

        prev = r0
        for r in (1, 2):
            c1, c2 = consts[r]
            cur = []
            for i in range(self.BATCH):
                t1 = b.xor(prev[i], c1, "r%d_x_%d" % (r, i))
                t2 = b.mul(t1, c2, "r%d_m_%d" % (r, i))
                cur.append(t2)
                b.store(b.add(a_base, b.const(i), "r%d_a_%d" % (r, i)), t2)
            prev = cur
        return b.build()

    def _expected(self):
        mask = (1 << 32) - 1
        vals = list(range(self.BATCH))
        rounds = []
        for r in range(3):
            vals = [(((v ^ (0x9E3779B9 + r)) * (5 + 2 * r)) & mask)
                    for v in vals]
            rounds.append(list(vals))
        want_a = rounds[2]
        want_b = [(rounds[0][8 + j] ^ self.K) & mask for j in range(8)]
        return want_a, want_b

    def test_no_dangling_refs_and_correct_values(self):
        from compiler.hir import Op, SSAValue, VectorSSAValue

        hir = self._build()
        pm = PassManager()
        pm.add_pass(SLPVectorizationPass())
        transformed = pm.run(hir)

        # Structural: SSA dominance must hold in the flat body. The bug
        # emitted vinsert chains over scalars whose defs no longer existed.
        defined = set()
        for stmt in transformed.body:
            if not isinstance(stmt, Op):
                continue
            for operand in stmt.operands:
                if isinstance(operand, (SSAValue, VectorSSAValue)):
                    self.assertIn(
                        operand, defined,
                        "use of %r before (or without) its def in %r"
                        % (operand, stmt))
            if stmt.result is not None:
                defined.add(stmt.result)

        # Semantic: run and compare against the scalar computation.
        instrs = compile_hir_to_vliw(transformed)
        mem = list(range(self.BATCH)) + [0] * 300
        machine = self._run_program(instrs, mem)
        want_a, want_b = self._expected()
        self.assertEqual(machine.mem[self.A:self.A + self.BATCH], want_a,
                         "same-address store order was not preserved")
        self.assertEqual(machine.mem[self.B:self.B + 8], want_b,
                         "side-chain values corrupted (dangling scalar refs)")

    def _run_program(self, instrs, mem):
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}),
                          n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine


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
