"""Tests for pass manager and loop unrolling functionality."""

import unittest

from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    HIRBuilder,
    compile_hir_to_vliw,
    lower_to_lir,
    eliminate_phis,
    compile_to_vliw,
)
from compiler import (
    PassManager, PassConfig, LoopUnrollPass,
    Const, ForLoop, If, Op, SSAValue,
)


class TestPassManagerAndLoopUnroll(unittest.TestCase):
    """Test pass manager and loop unrolling functionality."""

    def test_full_unroll_simple_loop(self):
        """Test that a simple loop with static bounds gets fully unrolled."""
        b = HIRBuilder()
        zero = b.const(0)
        addr = b.const(10)

        # Simple loop: for i in 0..4: (no body, just count iterations)
        init_sum = b.const(0)

        def body(i, params):
            s = params[0]
            new_s = b.add(s, i, "sum")
            return [new_s]

        # Use Const for bounds so it can be unrolled
        results = b.for_loop(start=Const(0), end=Const(4), iter_args=[init_sum], body_fn=body)
        b.store(addr, results[0])

        hir = b.build()

        # Run loop unroll pass
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        # Set max_trip_count high enough for full unroll
        pm.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 100})
        unrolled = pm.run(hir)

        # After full unroll, there should be no ForLoop in the body
        has_for_loop = any(isinstance(s, ForLoop) for s in unrolled.body)
        self.assertFalse(has_for_loop, "Loop should be fully unrolled")

        # Compile directly without running pass manager again
        lir = lower_to_lir(unrolled)
        eliminate_phis(lir)
        instrs = compile_to_vliw(lir)

        mem = [0] * 20
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # Sum of 0+1+2+3 = 6
        self.assertEqual(machine.mem[10], 6)
        print("Full unroll simple loop test passed!")

    def test_partial_unroll(self):
        """Test partial unrolling with pragma_unroll."""
        b = HIRBuilder()
        addr = b.const(10)
        init_sum = b.const(0)

        def body(i, params):
            s = params[0]
            new_s = b.add(s, i, "sum")
            return [new_s]

        # Loop with 8 iterations, pragma_unroll=4 for partial unrolling
        results = b.for_loop(
            start=Const(0),
            end=Const(8),
            iter_args=[init_sum],
            body_fn=body,
            pragma_unroll=4  # Partial unroll by factor 4
        )
        b.store(addr, results[0])

        hir = b.build()

        # Run unroll pass (8/4 = 2 iterations remain)
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        unrolled = pm.run(hir)

        # After partial unroll, there should still be a ForLoop with 2 iterations
        for_loops = [s for s in unrolled.body if isinstance(s, ForLoop)]
        self.assertEqual(len(for_loops), 1, "Should have one loop remaining")
        loop = for_loops[0]
        self.assertEqual(loop.end.value, 2, "Loop should have 2 iterations after unroll by 4")

        # Compile directly without running pass manager again
        lir = lower_to_lir(unrolled)
        eliminate_phis(lir)
        instrs = compile_to_vliw(lir)

        mem = [0] * 20
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # Sum of 0+1+2+3+4+5+6+7 = 28
        self.assertEqual(machine.mem[10], 28)
        print("Partial unroll test passed!")

    def test_no_unroll_dynamic_bounds(self):
        """Test that loops with dynamic bounds are not unrolled."""
        b = HIRBuilder()
        # Load bound from memory (dynamic)
        addr0 = b.const(0)
        bound = b.load(addr0, "bound")
        addr10 = b.const(10)
        init_sum = b.const(0)

        def body(i, params):
            s = params[0]
            new_s = b.add(s, i, "sum")
            return [new_s]

        results = b.for_loop(start=b.const(0), end=bound, iter_args=[init_sum], body_fn=body)
        b.store(addr10, results[0])

        hir = b.build()

        # Try to unroll
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 100})
        unrolled = pm.run(hir)

        # Loop should still exist (not unrolled because bounds are dynamic)
        for_loops = [s for s in unrolled.body if isinstance(s, ForLoop)]
        self.assertEqual(len(for_loops), 1, "Dynamic loop should not be unrolled")
        print("No unroll dynamic bounds test passed!")

    def test_skip_unroll_bad_factor(self):
        """Test that unrolling is skipped when pragma factor doesn't divide trip count."""
        b = HIRBuilder()
        addr = b.const(10)
        init_sum = b.const(0)

        def body(i, params):
            s = params[0]
            new_s = b.add(s, i, "sum")
            return [new_s]

        # Loop with 10 iterations, pragma_unroll=3 (doesn't divide 10)
        results = b.for_loop(
            start=Const(0),
            end=Const(10),
            iter_args=[init_sum],
            body_fn=body,
            pragma_unroll=3  # 10 % 3 != 0, should skip unrolling
        )
        b.store(addr, results[0])

        hir = b.build()

        # Run unroll pass
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        unrolled = pm.run(hir)

        # Loop should still have 10 iterations (not unrolled because 10 % 3 != 0)
        for_loops = [s for s in unrolled.body if isinstance(s, ForLoop)]
        self.assertEqual(len(for_loops), 1, "Loop should not be unrolled")
        loop = for_loops[0]
        self.assertEqual(loop.end.value, 10, "Loop should still have 10 iterations")

        # But it should still produce correct result
        instrs = compile_hir_to_vliw(unrolled)
        mem = [0] * 20
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # Sum of 0+1+...+9 = 45
        self.assertEqual(machine.mem[10], 45)
        print("Skip unroll bad factor test passed!")

    def test_pass_config_set_config(self):
        """Test setting pass config from dict."""
        config_data = {
            "passes": {
                "loop-unroll": {
                    "enabled": True,
                    "options": {
                        "unroll_factor": 2,
                        "max_trip_count": 50
                    }
                }
            }
        }

        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.set_config(config_data)

        self.assertIn("loop-unroll", pm.config)
        self.assertEqual(pm.config["loop-unroll"].options["unroll_factor"], 2)
        self.assertEqual(pm.config["loop-unroll"].options["max_trip_count"], 50)
        print("Pass config set_config test passed!")

    def test_unroll_remaps_if_yields(self):
        """Ensure unrolled loop results are remapped through If yields."""
        b = HIRBuilder()
        addr0 = b.const(0)
        cond_true = b.const(1)
        init_sum = b.const(0)

        def then_fn():
            def body(i, params):
                s = params[0]
                return [b.add(s, i, "sum")]

            results = b.for_loop(start=Const(0), end=Const(4), iter_args=[init_sum], body_fn=body)
            return [results[0]]

        def else_fn():
            return [b.const(99)]

        out = b.if_stmt(cond_true, then_fn, else_fn)[0]
        b.store(addr0, out)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 100})
        unrolled = pm.run(hir)

        lir = lower_to_lir(unrolled)
        eliminate_phis(lir)
        instrs = compile_to_vliw(lir)

        mem = [0] * 20
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # then branch taken: sum of 0+1+2+3 = 6
        self.assertEqual(machine.mem[0], 6)
        print("Unroll remaps If yields test passed!")

    def test_unroll_remaps_forloop_yields(self):
        """Ensure unrolled loop results are remapped through enclosing ForLoop yields."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        bound = b.load(addr1, "bound")  # dynamic bound: outer loop should not be unrolled

        init_sum = b.const(0)

        def outer_body(i, params):
            outer_s = params[0]

            def inner_body(j, inner_params):
                s = inner_params[0]
                return [b.add(s, j, "inner_sum")]

            inner_results = b.for_loop(start=Const(0), end=Const(4), iter_args=[outer_s], body_fn=inner_body)
            return [inner_results[0]]

        results = b.for_loop(start=Const(0), end=bound, iter_args=[init_sum], body_fn=outer_body)
        b.store(addr0, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 100})
        unrolled = pm.run(hir)

        lir = lower_to_lir(unrolled)
        eliminate_phis(lir)
        instrs = compile_to_vliw(lir)

        mem = [0] * 20
        mem[1] = 1  # one outer iteration
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # one outer iter, inner sum of 0+1+2+3 = 6
        self.assertEqual(machine.mem[0], 6)
        print("Unroll remaps ForLoop yields test passed!")

    def test_unroll_propagates_const_iter_arg(self):
        """Ensure const iter_args are propagated into cloned nested loops."""
        b = HIRBuilder()

        def inner_body(j, inner_params):
            return [inner_params[0]]

        def outer_body(i, params):
            inner_results = b.for_loop(
                start=Const(0),
                end=Const(1),
                iter_args=[params[0]],
                body_fn=inner_body,
                pragma_unroll=1  # keep inner loop
            )
            return [inner_results[0]]

        # Pass Const directly to force remap -> Const during unroll
        b.for_loop(start=Const(0), end=Const(2), iter_args=[Const(7)], body_fn=outer_body)
        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        unrolled = pm.run(hir)

        loops = [(idx, s) for idx, s in enumerate(unrolled.body) if isinstance(s, ForLoop)]
        self.assertGreaterEqual(len(loops), 1)
        has_const_iter_arg = False
        for _, loop in loops:
            if any(isinstance(arg, Const) for arg in loop.iter_args):
                has_const_iter_arg = True
        self.assertTrue(has_const_iter_arg)

    def test_unroll_propagates_const_if_cond(self):
        """Ensure const If conditions are propagated into cloned Ifs."""
        b = HIRBuilder()

        def outer_body(i, params):
            def then_fn():
                return [params[0]]

            def else_fn():
                return [params[0]]

            results = b.if_stmt(params[0], then_fn, else_fn)
            return [results[0]]

        # Pass Const directly to force remap -> Const during unroll
        b.for_loop(start=Const(0), end=Const(2), iter_args=[Const(1)], body_fn=outer_body)
        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        unrolled = pm.run(hir)

        ifs = [(idx, s) for idx, s in enumerate(unrolled.body) if isinstance(s, If)]
        self.assertGreaterEqual(len(ifs), 1)
        has_const_cond = False
        for _, if_stmt in ifs:
            if isinstance(if_stmt.cond, Const):
                has_const_cond = True
        self.assertTrue(has_const_cond)


class TestPragmaUnroll(unittest.TestCase):
    """Tests for pragma_unroll loop directive."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def _count_loops(self, hir):
        """Count ForLoop statements in HIR."""
        count = 0
        def count_stmts(stmts):
            nonlocal count
            for stmt in stmts:
                if isinstance(stmt, ForLoop):
                    count += 1
                    count_stmts(stmt.body)
        count_stmts(hir.body)
        return count

    def test_pragma_unroll_disabled(self):
        """Test that pragma_unroll=1 prevents unrolling."""
        b = HIRBuilder()
        zero = b.const(0)
        one = b.const(1)
        addr0 = b.const(0)

        init_val = b.const(0)

        def loop_body(i, params):
            # Sum: val += 1
            new_val = b.add(params[0], one, "new_val")
            return [new_val]

        # Loop with pragma_unroll=1 (disabled)
        results = b.for_loop(
            start=Const(0),
            end=Const(4),
            iter_args=[init_val],
            body_fn=loop_body,
            pragma_unroll=1  # Disable unrolling
        )

        b.store(addr0, results[0])
        hir = b.build()

        # Apply loop unroll pass
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        result_hir = pm.run(hir)

        # Loop should still exist (not unrolled)
        loop_count = self._count_loops(result_hir)
        self.assertEqual(loop_count, 1, "Loop with pragma_unroll=1 should not be unrolled")

        # Verify correctness
        instrs = compile_hir_to_vliw(hir)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 4, "Loop result should be 4")

        print("pragma_unroll=1 disabled test passed!")

    def test_pragma_unroll_full(self):
        """Test that pragma_unroll=0 causes full unroll."""
        b = HIRBuilder()
        zero = b.const(0)
        one = b.const(1)
        addr0 = b.const(0)

        init_val = b.const(0)

        def loop_body(i, params):
            new_val = b.add(params[0], one, "new_val")
            return [new_val]

        # Loop with pragma_unroll=0 (full unroll, default)
        results = b.for_loop(
            start=Const(0),
            end=Const(4),
            iter_args=[init_val],
            body_fn=loop_body,
            pragma_unroll=0  # Full unroll
        )

        b.store(addr0, results[0])
        hir = b.build()

        # Apply loop unroll pass
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        result_hir = pm.run(hir)

        # Loop should be eliminated (fully unrolled)
        loop_count = self._count_loops(result_hir)
        self.assertEqual(loop_count, 0, "Loop with pragma_unroll=0 should be fully unrolled")

        # Verify correctness
        instrs = compile_hir_to_vliw(hir)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 4, "Loop result should be 4")

        print("pragma_unroll=0 full unroll test passed!")

    def test_pragma_unroll_partial(self):
        """Test that pragma_unroll=4 with 8 iters creates 2-iteration loop."""
        b = HIRBuilder()
        zero = b.const(0)
        one = b.const(1)
        addr0 = b.const(0)

        init_val = b.const(0)

        def loop_body(i, params):
            new_val = b.add(params[0], one, "new_val")
            return [new_val]

        # Loop with pragma_unroll=4 (partial unroll)
        results = b.for_loop(
            start=Const(0),
            end=Const(8),
            iter_args=[init_val],
            body_fn=loop_body,
            pragma_unroll=4  # Partial unroll by 4
        )

        b.store(addr0, results[0])
        hir = b.build()

        # Apply loop unroll pass
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        result_hir = pm.run(hir)

        # Should have 1 loop with 2 iterations (8/4=2)
        loop_count = self._count_loops(result_hir)
        self.assertEqual(loop_count, 1, "Partial unroll should leave 1 loop")

        # Check the loop has correct trip count
        def find_loop(stmts):
            for stmt in stmts:
                if isinstance(stmt, ForLoop):
                    return stmt
            return None
        loop = find_loop(result_hir.body)
        self.assertIsNotNone(loop)
        self.assertEqual(loop.start.value, 0)
        self.assertEqual(loop.end.value, 2)  # 8/4 = 2 iterations

        # Verify correctness
        instrs = compile_hir_to_vliw(hir)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 8, "Loop result should be 8")

        print("pragma_unroll=4 partial unroll test passed!")

    def test_pragma_unroll_nested_mixed(self):
        """Test outer=1 (disabled), inner=0 (full unroll) works correctly."""
        b = HIRBuilder()
        zero = b.const(0)
        one = b.const(1)
        addr0 = b.const(0)

        init_val = b.const(0)

        def inner_body(j, inner_params):
            new_val = b.add(inner_params[0], one, "inner_val")
            return [new_val]

        def outer_body(i, outer_params):
            # Inner loop with pragma_unroll=0 (full unroll)
            results = b.for_loop(
                start=Const(0),
                end=Const(2),
                iter_args=[outer_params[0]],
                body_fn=inner_body,
                pragma_unroll=0  # Full unroll inner
            )
            return results

        # Outer loop with pragma_unroll=1 (disabled)
        results = b.for_loop(
            start=Const(0),
            end=Const(3),
            iter_args=[init_val],
            body_fn=outer_body,
            pragma_unroll=1  # Don't unroll outer
        )

        b.store(addr0, results[0])
        hir = b.build()

        # Apply loop unroll pass
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        result_hir = pm.run(hir)

        # Should have 1 outer loop (inner is unrolled away)
        loop_count = self._count_loops(result_hir)
        self.assertEqual(loop_count, 1, "Only outer loop should remain")

        # Verify correctness: 3 outer iters * 2 inner iters = 6
        instrs = compile_hir_to_vliw(hir)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 6, "Result should be 6")

        print("pragma_unroll nested mixed test passed!")

    def test_pragma_unroll_non_divisible_skipped(self):
        """Test that partial unroll is skipped when factor doesn't divide trip count."""
        b = HIRBuilder()
        one = b.const(1)
        addr0 = b.const(0)

        init_val = b.const(0)

        def loop_body(i, params):
            new_val = b.add(params[0], one, "new_val")
            return [new_val]

        # Loop with pragma_unroll=4 but trip_count=7 (not divisible)
        results = b.for_loop(
            start=Const(0),
            end=Const(7),
            iter_args=[init_val],
            body_fn=loop_body,
            pragma_unroll=4  # 7 not divisible by 4
        )

        b.store(addr0, results[0])
        hir = b.build()

        # Apply loop unroll pass
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        result_hir = pm.run(hir)

        # Loop should remain (unroll skipped due to non-divisibility)
        loop_count = self._count_loops(result_hir)
        self.assertEqual(loop_count, 1, "Loop should not be unrolled when factor doesn't divide")

        # Check loop still has original trip count
        def find_loop(stmts):
            for stmt in stmts:
                if isinstance(stmt, ForLoop):
                    return stmt
            return None
        loop = find_loop(result_hir.body)
        self.assertIsNotNone(loop)
        self.assertEqual(loop.end.value, 7, "Loop should have original trip count")

        # Verify correctness
        instrs = compile_hir_to_vliw(hir)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 7, "Loop result should be 7")

        print("pragma_unroll non-divisible skipped test passed!")


if __name__ == "__main__":
    unittest.main()
