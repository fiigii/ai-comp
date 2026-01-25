"""Tests for Dead Code Elimination pass."""

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
    DCEPass,
    Const,
    count_statements,
)
from compiler.hir import ForLoop, If, Halt, Pause


def const_ssa(builder: HIRBuilder, value: int, name=None):
    """Materialize a constant as an SSA value using a simple add."""
    return builder.add(Const(value), Const(0), name)


class TestDCEPass(unittest.TestCase):
    """Test Dead Code Elimination pass."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    # --- Basic DCE Tests ---

    def test_dce_unused_const(self):
        """Test that unused const is eliminated."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val42 = const_ssa(b, 42, "val42")
        unused = const_ssa(b, 999, "unused")  # Never used
        b.store(addr0, val42)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # DCE should eliminate the unused const
        self.assertLess(count_statements(transformed), original_count)
        print("DCE unused const test passed!")

    def test_dce_unused_load(self):
        """Test that unused load is eliminated."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        addr1 = const_ssa(b, 1, "addr1")
        val = const_ssa(b, 100, "val")
        unused_load = b.load(addr1, "unused_load")  # Never used
        b.store(addr0, val)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # DCE should eliminate the unused load
        self.assertLess(count_statements(transformed), original_count)
        print("DCE unused load test passed!")

    def test_dce_unused_arithmetic(self):
        """Test that unused add/mul are eliminated."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val1 = const_ssa(b, 10, "val1")
        val2 = const_ssa(b, 20, "val2")
        unused_sum = b.add(val1, val2, "unused_sum")  # Never used
        unused_prod = b.mul(val1, val2, "unused_prod")  # Never used
        b.store(addr0, val1)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # DCE should eliminate both unused operations
        self.assertLess(count_statements(transformed), original_count)
        print("DCE unused arithmetic test passed!")

    def test_dce_chain_all_dead(self):
        """Test that a chain of unused ops is all eliminated."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val = const_ssa(b, 1, "val")
        b.store(addr0, val)

        # Chain of dead computations
        a = const_ssa(b, 10, "a")
        b_val = b.add(a, a, "b")
        c = b.mul(b_val, a, "c")
        d = b.sub(c, b_val, "d")

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # All the dead chain should be eliminated
        self.assertLess(count_statements(transformed), original_count)
        print("DCE chain all dead test passed!")

    def test_dce_partial_chain_kept(self):
        """Test that used ops in a chain are kept."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        addr1 = const_ssa(b, 1, "addr1")

        a = const_ssa(b, 10, "a")
        b_val = b.add(a, a, "b")      # Used
        c = b.mul(b_val, a, "c")       # Unused
        d = b.sub(b_val, a, "d")       # Used

        b.store(addr0, b_val)
        b.store(addr1, d)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # Only the unused 'c' should be eliminated
        self.assertLess(count_statements(transformed), original_count)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[0], 20)  # 10 + 10
        self.assertEqual(machine.mem[1], 10)  # 20 - 10
        print("DCE partial chain kept test passed!")

    # --- Side Effects Tests ---

    def test_dce_store_always_kept(self):
        """Test that store is never eliminated."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val = const_ssa(b, 42, "val")
        b.store(addr0, val)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # Store should be kept
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 42)
        print("DCE store always kept test passed!")

    def test_dce_store_operands_kept(self):
        """Test that values used by store are kept live."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val1 = const_ssa(b, 10, "val1")
        val2 = const_ssa(b, 20, "val2")
        sum_val = b.add(val1, val2, "sum")
        b.store(addr0, sum_val)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # All values needed for store should be kept
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 30)
        print("DCE store operands kept test passed!")

    def test_dce_halt_kept(self):
        """Test that Halt is never eliminated."""
        b = HIRBuilder()
        unused = const_ssa(b, 999, "unused")
        b.halt()

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # Halt should be in the output
        has_halt = any(isinstance(s, Halt) for s in transformed.body)
        self.assertTrue(has_halt)
        print("DCE halt kept test passed!")

    def test_dce_pause_kept(self):
        """Test that Pause is never eliminated."""
        b = HIRBuilder()
        unused = const_ssa(b, 999, "unused")
        b.pause()
        b.halt()

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # Pause should be in the output
        has_pause = any(isinstance(s, Pause) for s in transformed.body)
        self.assertTrue(has_pause)
        print("DCE pause kept test passed!")

    # --- ForLoop Tests ---

    def test_dce_dead_loop_eliminated(self):
        """Test that loop with unused results and no side effects is eliminated."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val = const_ssa(b, 1, "val")
        b.store(addr0, val)

        # Dead loop - no side effects, results not used
        init_sum = const_ssa(b, 0, "init_sum")

        def body(i, params):
            s = params[0]
            return [b.add(s, i, "sum")]

        results = b.for_loop(start=Const(0), end=Const(5), iter_args=[init_sum], body_fn=body)
        # Results not used!

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # Loop should be eliminated
        has_loop = any(isinstance(s, ForLoop) for s in transformed.body)
        self.assertFalse(has_loop)
        self.assertLess(count_statements(transformed), original_count)
        print("DCE dead loop eliminated test passed!")

    def test_dce_loop_with_store_kept(self):
        """Test that loop with store in body is kept."""
        b = HIRBuilder()

        def body(i, params):
            # Store has side effect
            b.store(i, i)
            return []

        b.for_loop(start=Const(0), end=Const(5), iter_args=[], body_fn=body)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # Loop should be kept (has side effects)
        has_loop = any(isinstance(s, ForLoop) for s in transformed.body)
        self.assertTrue(has_loop)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        for i in range(5):
            self.assertEqual(machine.mem[i], i)
        print("DCE loop with store kept test passed!")

    def test_dce_loop_result_used_kept(self):
        """Test that loop with used result is kept."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        init_sum = const_ssa(b, 0, "init_sum")

        def body(i, params):
            s = params[0]
            return [b.add(s, i, "sum")]

        results = b.for_loop(start=Const(0), end=Const(5), iter_args=[init_sum], body_fn=body)
        b.store(addr0, results[0])  # Result is used

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # Loop should be kept (result is used)
        has_loop = any(isinstance(s, ForLoop) for s in transformed.body)
        self.assertTrue(has_loop)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 10)  # 0+1+2+3+4
        print("DCE loop result used kept test passed!")

    def test_dce_nested_dead_loop(self):
        """Test nested loops where outer is dead."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val = const_ssa(b, 42, "val")
        b.store(addr0, val)

        # Dead outer loop with dead inner loop
        init_sum = const_ssa(b, 0, "init_sum")

        def outer_body(i, outer_params):
            outer_s = outer_params[0]

            def inner_body(j, inner_params):
                s = inner_params[0]
                return [b.add(s, j, "inner_sum")]

            inner_results = b.for_loop(start=Const(0), end=Const(3), iter_args=[outer_s], body_fn=inner_body)
            return [inner_results[0]]

        results = b.for_loop(start=Const(0), end=Const(2), iter_args=[init_sum], body_fn=outer_body)
        # Results not used!

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # Both loops should be eliminated
        has_loop = any(isinstance(s, ForLoop) for s in transformed.body)
        self.assertFalse(has_loop)
        self.assertLess(count_statements(transformed), original_count)
        print("DCE nested dead loop test passed!")

    # --- If Statement Tests ---

    def test_dce_dead_if_eliminated(self):
        """Test that if with unused results and no side effects is eliminated."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val = const_ssa(b, 1, "val")
        b.store(addr0, val)

        # Dead if - no side effects, results not used
        cond = const_ssa(b, 1, "cond")

        def then_fn():
            return [const_ssa(b, 100, "then_val")]

        def else_fn():
            return [const_ssa(b, 200, "else_val")]

        results = b.if_stmt(cond, then_fn, else_fn)
        # Results not used!

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # If should be eliminated
        has_if = any(isinstance(s, If) for s in transformed.body)
        self.assertFalse(has_if)
        self.assertLess(count_statements(transformed), original_count)
        print("DCE dead if eliminated test passed!")

    def test_dce_if_with_store_kept(self):
        """Test that if with store in branch is kept."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        addr1 = const_ssa(b, 1, "addr1")
        cond = b.load(addr0, "cond")

        def then_fn():
            b.store(addr1, const_ssa(b, 100, "then_val"))  # Side effect
            return []

        def else_fn():
            return []

        b.if_stmt(cond, then_fn, else_fn)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # If should be kept (has side effects)
        has_if = any(isinstance(s, If) for s in transformed.body)
        self.assertTrue(has_if)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [1, 0] + [0] * 98
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[1], 100)
        print("DCE if with store kept test passed!")

    def test_dce_if_result_used_kept(self):
        """Test that if with used result is kept."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        addr1 = const_ssa(b, 1, "addr1")
        cond = b.load(addr0, "cond")

        def then_fn():
            return [const_ssa(b, 100, "then_val")]

        def else_fn():
            return [const_ssa(b, 200, "else_val")]

        results = b.if_stmt(cond, then_fn, else_fn)
        b.store(addr1, results[0])  # Result is used

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)

        # If should be kept (result is used)
        has_if = any(isinstance(s, If) for s in transformed.body)
        self.assertTrue(has_if)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem1 = [1, 0] + [0] * 98
        machine1 = self._run_program(instrs, mem1)
        self.assertEqual(machine1.mem[1], 100)

        mem2 = [0, 0] + [0] * 98
        machine2 = self._run_program(instrs, mem2)
        self.assertEqual(machine2.mem[1], 200)
        print("DCE if result used kept test passed!")

    # --- Integration Tests ---

    def test_dce_preserves_semantics(self):
        """Test that DCE preserves program output."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        addr1 = const_ssa(b, 1, "addr1")
        init_sum = const_ssa(b, 0, "init_sum")

        # Some dead code mixed with live code
        dead1 = const_ssa(b, 999, "dead1")
        dead2 = b.add(dead1, dead1, "dead2")

        def body(i, params):
            s = params[0]
            unused = b.mul(i, i, "unused")  # Dead inside loop
            return [b.add(s, i, "sum")]

        results = b.for_loop(start=Const(0), end=Const(5), iter_args=[init_sum], body_fn=body)
        b.store(addr0, results[0])

        dead3 = const_ssa(b, 888, "dead3")

        hir = b.build()

        # Without DCE
        instrs_no_dce = compile_hir_to_vliw(hir)
        mem_no_dce = [0] * 100
        machine_no_dce = self._run_program(instrs_no_dce, mem_no_dce)

        # With DCE
        pm = PassManager()
        pm.add_pass(DCEPass())
        transformed = pm.run(hir)
        instrs_dce = compile_hir_to_vliw(transformed)
        mem_dce = [0] * 100
        machine_dce = self._run_program(instrs_dce, mem_dce)

        # Both should produce same result
        self.assertEqual(machine_no_dce.mem[0], 10)
        self.assertEqual(machine_dce.mem[0], 10)
        print("DCE preserves semantics test passed!")

    def test_dce_metrics_reported(self):
        """Test that DCE pass reports elimination counts."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val = const_ssa(b, 42, "val")
        b.store(addr0, val)

        # Dead code
        unused1 = const_ssa(b, 1, "unused1")
        unused2 = const_ssa(b, 2, "unused2")
        unused3 = b.add(unused1, unused2, "unused3")

        hir = b.build()

        dce_pass = DCEPass()
        pm = PassManager()
        pm.add_pass(dce_pass)
        pm.run(hir)

        metrics = dce_pass.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("ops_eliminated", metrics.custom)
        self.assertGreater(metrics.custom["ops_eliminated"], 0)
        print(f"DCE metrics: {metrics.custom}")
        print("DCE metrics reported test passed!")

    def test_dce_config_disable(self):
        """Test that DCE pass can be disabled via config."""
        b = HIRBuilder()
        addr0 = const_ssa(b, 0, "addr0")
        val = const_ssa(b, 42, "val")
        unused = const_ssa(b, 999, "unused")  # Dead
        b.store(addr0, val)

        hir = b.build()
        original_count = count_statements(hir)

        # With DCE enabled (default)
        pm_enabled = PassManager()
        pm_enabled.add_pass(DCEPass())
        result_enabled = pm_enabled.run(hir)

        # With DCE disabled
        pm_disabled = PassManager()
        pm_disabled.add_pass(DCEPass())
        pm_disabled.config["dce"] = PassConfig(name="dce", enabled=False)
        result_disabled = pm_disabled.run(hir)

        # Disabled should not eliminate anything
        self.assertEqual(count_statements(result_disabled), original_count)
        # Enabled should eliminate
        self.assertLess(count_statements(result_enabled), original_count)
        print("DCE config disable test passed!")


if __name__ == "__main__":
    unittest.main()
