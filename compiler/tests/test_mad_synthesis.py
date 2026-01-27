"""Tests for Multiply-Add (MAD) Synthesis pass."""

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
    count_statements,
)
from compiler.passes import MADSynthesisPass, SLPVectorizationPass, LoopUnrollPass, CSEPass
from compiler.hir import Const, Op


def count_ops_by_opcode(hir, opcode: str) -> int:
    """Count operations with a specific opcode in HIR."""
    from compiler.hir import ForLoop, If, Statement

    def count_in_body(body: list[Statement]) -> int:
        total = 0
        for stmt in body:
            if isinstance(stmt, Op) and stmt.opcode == opcode:
                total += 1
            if isinstance(stmt, ForLoop):
                total += count_in_body(stmt.body)
            elif isinstance(stmt, If):
                total += count_in_body(stmt.then_body)
                total += count_in_body(stmt.else_body)
        return total

    return count_in_body(hir.body)


class TestMADSynthesisPass(unittest.TestCase):
    """Test Multiply-Add Synthesis pass."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    # --- Basic Fusion Tests ---

    def test_basic_mad_fusion(self):
        """Test that v+(v*(a,b), c) becomes multiply_add."""
        b = HIRBuilder()
        addr0 = b.const(0)

        # Create vector values
        va = b.vbroadcast(b.const(2), "va")
        vb = b.vbroadcast(b.const(3), "vb")
        vc = b.vbroadcast(b.const(10), "vc")

        # v* followed by v+
        vmul_result = b.vmul(va, vb, "vmul")
        vadd_result = b.vadd(vmul_result, vc, "vadd")

        # Store first element to verify
        result = b.vextract(vadd_result, 0, "result")
        b.store(addr0, result)

        hir = b.build()

        # Verify we have v* and v+ before transformation
        self.assertEqual(count_ops_by_opcode(hir, "v*"), 1)
        self.assertEqual(count_ops_by_opcode(hir, "v+"), 1)

        pm = PassManager()
        pm.add_pass(MADSynthesisPass())
        transformed = pm.run(hir)

        # After fusion: v* should be gone, v+ should be gone, multiply_add should exist
        self.assertEqual(count_ops_by_opcode(transformed, "v*"), 0)
        self.assertEqual(count_ops_by_opcode(transformed, "v+"), 0)
        self.assertEqual(count_ops_by_opcode(transformed, "multiply_add"), 1)

        # Verify correctness: 2 * 3 + 10 = 16
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 16)

        print("Basic MAD fusion test passed!")

    def test_mad_commutative(self):
        """Test that v+(c, v*(a,b)) also fuses."""
        b = HIRBuilder()
        addr0 = b.const(0)

        # Create vector values
        va = b.vbroadcast(b.const(4), "va")
        vb = b.vbroadcast(b.const(5), "vb")
        vc = b.vbroadcast(b.const(7), "vc")

        # v+ with v* as second operand: v+(c, v*(a, b))
        vmul_result = b.vmul(va, vb, "vmul")
        vadd_result = b.vadd(vc, vmul_result, "vadd")

        # Store first element to verify
        result = b.vextract(vadd_result, 0, "result")
        b.store(addr0, result)

        hir = b.build()

        # Verify we have v* and v+ before transformation
        self.assertEqual(count_ops_by_opcode(hir, "v*"), 1)
        self.assertEqual(count_ops_by_opcode(hir, "v+"), 1)

        pm = PassManager()
        pm.add_pass(MADSynthesisPass())
        transformed = pm.run(hir)

        # After fusion
        self.assertEqual(count_ops_by_opcode(transformed, "v*"), 0)
        self.assertEqual(count_ops_by_opcode(transformed, "v+"), 0)
        self.assertEqual(count_ops_by_opcode(transformed, "multiply_add"), 1)

        # Verify correctness: 4 * 5 + 7 = 27
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 27)

        print("MAD commutative test passed!")

    def test_no_fusion_multiple_uses(self):
        """Test that v* with multiple users is NOT fused."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)

        # Create vector values
        va = b.vbroadcast(b.const(2), "va")
        vb = b.vbroadcast(b.const(3), "vb")
        vc = b.vbroadcast(b.const(10), "vc")

        # v* result is used by both v+ and another operation
        vmul_result = b.vmul(va, vb, "vmul")

        # First use: v+ (would be candidate for fusion)
        vadd_result = b.vadd(vmul_result, vc, "vadd")

        # Second use: another v+ (prevents fusion)
        vadd_result2 = b.vadd(vmul_result, vc, "vadd2")

        # Store both results
        result1 = b.vextract(vadd_result, 0, "result1")
        result2 = b.vextract(vadd_result2, 0, "result2")
        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        # Verify structure before
        self.assertEqual(count_ops_by_opcode(hir, "v*"), 1)
        self.assertEqual(count_ops_by_opcode(hir, "v+"), 2)

        pm = PassManager()
        pm.add_pass(MADSynthesisPass())
        transformed = pm.run(hir)

        # Should NOT fuse because v* has 2 uses
        self.assertEqual(count_ops_by_opcode(transformed, "v*"), 1)
        self.assertEqual(count_ops_by_opcode(transformed, "v+"), 2)
        self.assertEqual(count_ops_by_opcode(transformed, "multiply_add"), 0)

        # Verify correctness anyway
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        # 2 * 3 + 10 = 16
        self.assertEqual(machine.mem[0], 16)
        self.assertEqual(machine.mem[1], 16)

        print("No fusion multiple uses test passed!")

    def test_correctness_with_loaded_values(self):
        """Test MAD fusion correctness with values loaded from memory."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        addr2 = b.const(2)
        addr_result = b.const(10)

        # Load values from memory
        val_a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")
        val_c = b.load(addr2, "c")

        # Broadcast to vectors
        va = b.vbroadcast(val_a, "va")
        vb = b.vbroadcast(val_b, "vb")
        vc = b.vbroadcast(val_c, "vc")

        # MAD pattern
        vmul_result = b.vmul(va, vb, "vmul")
        vadd_result = b.vadd(vmul_result, vc, "vadd")

        # Extract and store
        result = b.vextract(vadd_result, 0, "result")
        b.store(addr_result, result)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(MADSynthesisPass())
        transformed = pm.run(hir)

        # Verify fusion happened
        self.assertEqual(count_ops_by_opcode(transformed, "multiply_add"), 1)

        # Compile and run with specific values
        instrs = compile_hir_to_vliw(transformed)
        mem = [5, 7, 11] + [0] * 97  # a=5, b=7, c=11
        machine = self._run_program(instrs, mem)
        # 5 * 7 + 11 = 46
        self.assertEqual(machine.mem[10], 46)

        print("MAD correctness with loaded values test passed!")

    def test_no_fusion_with_scalar_ops(self):
        """Test that scalar * and + are NOT fused (only vector ops)."""
        b = HIRBuilder()
        addr0 = b.const(0)

        # Scalar operations
        a = b.const(2)
        val_b = b.const(3)
        c = b.const(10)

        # Scalar mul and add (not vector)
        mul_result = b.mul(a, val_b, "mul")
        add_result = b.add(mul_result, c, "add")

        b.store(addr0, add_result)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(MADSynthesisPass())
        transformed = pm.run(hir)

        # Should NOT create multiply_add (that's for vectors)
        self.assertEqual(count_ops_by_opcode(transformed, "multiply_add"), 0)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 16)

        print("No fusion with scalar ops test passed!")

    def test_multiple_mad_patterns(self):
        """Test that multiple MAD patterns in the same block are all fused."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)

        # First MAD pattern
        va1 = b.vbroadcast(b.const(2), "va1")
        vb1 = b.vbroadcast(b.const(3), "vb1")
        vc1 = b.vbroadcast(b.const(10), "vc1")
        vmul1 = b.vmul(va1, vb1, "vmul1")
        vadd1 = b.vadd(vmul1, vc1, "vadd1")

        # Second MAD pattern
        va2 = b.vbroadcast(b.const(4), "va2")
        vb2 = b.vbroadcast(b.const(5), "vb2")
        vc2 = b.vbroadcast(b.const(20), "vc2")
        vmul2 = b.vmul(va2, vb2, "vmul2")
        vadd2 = b.vadd(vmul2, vc2, "vadd2")

        # Store results
        result1 = b.vextract(vadd1, 0, "result1")
        result2 = b.vextract(vadd2, 0, "result2")
        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        self.assertEqual(count_ops_by_opcode(hir, "v*"), 2)
        self.assertEqual(count_ops_by_opcode(hir, "v+"), 2)

        pm = PassManager()
        pm.add_pass(MADSynthesisPass())
        transformed = pm.run(hir)

        # Both should be fused
        self.assertEqual(count_ops_by_opcode(transformed, "v*"), 0)
        self.assertEqual(count_ops_by_opcode(transformed, "v+"), 0)
        self.assertEqual(count_ops_by_opcode(transformed, "multiply_add"), 2)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        # 2 * 3 + 10 = 16
        self.assertEqual(machine.mem[0], 16)
        # 4 * 5 + 20 = 40
        self.assertEqual(machine.mem[1], 40)

        print("Multiple MAD patterns test passed!")

    def test_mad_in_loop(self):
        """Test MAD fusion works inside loop bodies."""
        b = HIRBuilder()
        addr0 = b.const(0)
        init_sum = b.const(0)

        def loop_body(i, params):
            s = params[0]

            # MAD pattern inside loop
            va = b.vbroadcast(i, "va")
            vb = b.vbroadcast(b.const(2), "vb")
            vc = b.vbroadcast(b.const(1), "vc")

            vmul = b.vmul(va, vb, "vmul")
            vadd = b.vadd(vmul, vc, "vadd")

            result = b.vextract(vadd, 0, "result")
            new_s = b.add(s, result, "new_s")
            return [new_s]

        results = b.for_loop(start=Const(0), end=Const(3), iter_args=[init_sum], body_fn=loop_body)
        b.store(addr0, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(MADSynthesisPass())
        transformed = pm.run(hir)

        # Verify fusion happened in loop
        self.assertEqual(count_ops_by_opcode(transformed, "multiply_add"), 1)

        # Compile and run (need to unroll first for execution)
        pm_full = PassManager()
        pm_full.add_pass(LoopUnrollPass())
        pm_full.add_pass(MADSynthesisPass())
        pm_full.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 100})
        final = pm_full.run(hir)

        instrs = compile_hir_to_vliw(final)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        # Sum of (i * 2 + 1) for i in 0..3: (0*2+1) + (1*2+1) + (2*2+1) = 1 + 3 + 5 = 9
        self.assertEqual(machine.mem[0], 9)

        print("MAD in loop test passed!")

    def test_mad_preserves_semantics(self):
        """Test that MAD fusion preserves program semantics."""
        b = HIRBuilder()
        addr0 = b.const(0)

        # Create test values that would produce different results
        # if the operation was incorrect
        va = b.vbroadcast(b.const(100), "va")
        vb = b.vbroadcast(b.const(200), "vb")
        vc = b.vbroadcast(b.const(50), "vc")

        vmul = b.vmul(va, vb, "vmul")
        vadd = b.vadd(vmul, vc, "vadd")

        result = b.vextract(vadd, 0, "result")
        b.store(addr0, result)

        hir = b.build()

        # Run without MAD synthesis
        instrs_no_mad = compile_hir_to_vliw(hir)
        mem_no_mad = [0] * 100
        machine_no_mad = self._run_program(instrs_no_mad, mem_no_mad)
        result_no_mad = machine_no_mad.mem[0]

        # Run with MAD synthesis
        pm = PassManager()
        pm.add_pass(MADSynthesisPass())
        transformed = pm.run(hir)
        instrs_with_mad = compile_hir_to_vliw(transformed)
        mem_with_mad = [0] * 100
        machine_with_mad = self._run_program(instrs_with_mad, mem_with_mad)
        result_with_mad = machine_with_mad.mem[0]

        # Both should produce same result: 100 * 200 + 50 = 20050
        self.assertEqual(result_no_mad, 20050)
        self.assertEqual(result_with_mad, 20050)

        print("MAD preserves semantics test passed!")

    def test_mad_metrics_reported(self):
        """Test that MAD pass reports correct metrics."""
        b = HIRBuilder()
        addr0 = b.const(0)

        va = b.vbroadcast(b.const(2), "va")
        vb = b.vbroadcast(b.const(3), "vb")
        vc = b.vbroadcast(b.const(10), "vc")

        vmul = b.vmul(va, vb, "vmul")
        vadd = b.vadd(vmul, vc, "vadd")

        result = b.vextract(vadd, 0, "result")
        b.store(addr0, result)

        hir = b.build()

        mad_pass = MADSynthesisPass()
        pm = PassManager()
        pm.add_pass(mad_pass)
        pm.run(hir)

        metrics = mad_pass.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("patterns_matched", metrics.custom)
        self.assertIn("ops_fused", metrics.custom)
        self.assertEqual(metrics.custom["patterns_matched"], 1)
        self.assertEqual(metrics.custom["ops_fused"], 1)

        print(f"MAD metrics: {metrics.custom}")
        print("MAD metrics reported test passed!")

    def test_mad_config_disabled(self):
        """Test that MAD pass can be disabled via config."""
        b = HIRBuilder()
        addr0 = b.const(0)

        va = b.vbroadcast(b.const(2), "va")
        vb = b.vbroadcast(b.const(3), "vb")
        vc = b.vbroadcast(b.const(10), "vc")

        vmul = b.vmul(va, vb, "vmul")
        vadd = b.vadd(vmul, vc, "vadd")

        result = b.vextract(vadd, 0, "result")
        b.store(addr0, result)

        hir = b.build()

        # With MAD enabled (default)
        pm_enabled = PassManager()
        pm_enabled.add_pass(MADSynthesisPass())
        result_enabled = pm_enabled.run(hir)

        # With MAD disabled
        pm_disabled = PassManager()
        pm_disabled.add_pass(MADSynthesisPass())
        pm_disabled.config["mad-synthesis"] = PassConfig(name="mad-synthesis", enabled=False)
        result_disabled = pm_disabled.run(hir)

        # Disabled should not fuse
        self.assertEqual(count_ops_by_opcode(result_disabled, "multiply_add"), 0)
        self.assertEqual(count_ops_by_opcode(result_disabled, "v*"), 1)

        # Enabled should fuse
        self.assertEqual(count_ops_by_opcode(result_enabled, "multiply_add"), 1)
        self.assertEqual(count_ops_by_opcode(result_enabled, "v*"), 0)

        print("MAD config disabled test passed!")


if __name__ == "__main__":
    unittest.main()
