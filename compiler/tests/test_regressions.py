"""Regression tests for compiler bugs."""

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


class TestCompilerRegressions(unittest.TestCase):
    """Regression tests for compiler bugs."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_const_in_both_if_branches(self):
        """
        Regression test for P1: Const caching across control flow.

        If Const(1) is used in both branches of an if, the const load must
        be hoisted to dominate both branches. Previously, the const was only
        emitted in the first branch that requested it.
        """
        b = HIRBuilder()

        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        val = b.load(addr0, "val")

        # Compare with 5
        five = b.const_load(5, "five")
        cond = b.lt(five, val, "cond")

        # Both branches use a constant that's different from anything used before
        # Use 42 which is only used inside the if/else branches
        def then_fn():
            # Use 42 in then branch
            result = b.const_load(42, "const42_then")
            return [result]

        def else_fn():
            # Use 42 in else branch too (should reuse same scratch)
            result = b.const_load(42, "const42_else")
            return [result]

        results = b.if_stmt(cond, then_fn, else_fn)
        b.store(addr1, results[0])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        # Test then branch (mem[0] > 5)
        mem1 = [10] + [0] * 99
        machine1 = self._run_program(instrs, mem1)
        self.assertEqual(machine1.mem[1], 42, "Then branch should use const 42")

        # Test else branch (mem[0] <= 5)
        # This was failing before the fix because const 42 was only emitted in then block
        mem2 = [3] + [0] * 99
        machine2 = self._run_program(instrs, mem2)
        self.assertEqual(machine2.mem[1], 42, "Else branch should use const 42")

        print("Const in both branches test passed!")

    def test_phi_swap_two_values(self):
        """
        Regression test for P1: Phi elimination must preserve parallel copy semantics.

        Test swapping two values in a loop, which creates a cycle in the phi copies.
        for i in 0..n: a, b = b, a
        """
        b = HIRBuilder()

        zero = b.const_load(0, "zero")
        one = b.const_load(1, "one")
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        # Initial values: a=100, b=200
        init_a = b.const_load(100, "init_a")
        init_b = b.const_load(200, "init_b")

        def loop_body(i, params):
            a = params[0]
            b_val = params[1]
            # Swap: return [b, a]
            return [b_val, a]

        # Do 1 iteration of swap
        results = b.for_loop(
            start=zero,
            end=one,
            iter_args=[init_a, init_b],
            body_fn=loop_body
        )

        # After 1 swap: a should be 200, b should be 100
        b.store(addr0, results[0])
        b.store(addr1, results[1])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[0], 200, "After swap, a should be 200")
        self.assertEqual(machine.mem[1], 100, "After swap, b should be 100")

        print("Phi swap test passed!")

    def test_phi_swap_multiple_iterations(self):
        """
        Extended test: swap over multiple iterations to verify cycle handling.
        """
        b = HIRBuilder()

        zero = b.const_load(0, "zero")
        three = b.const_load(3, "three")
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        init_a = b.const_load(1, "init_a")
        init_b = b.const_load(2, "init_b")

        def loop_body(i, params):
            a = params[0]
            b_val = params[1]
            return [b_val, a]

        results = b.for_loop(
            start=zero,
            end=three,
            iter_args=[init_a, init_b],
            body_fn=loop_body
        )

        b.store(addr0, results[0])
        b.store(addr1, results[1])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # 3 swaps: (1,2) -> (2,1) -> (1,2) -> (2,1)
        self.assertEqual(machine.mem[0], 2, "After 3 swaps, a should be 2")
        self.assertEqual(machine.mem[1], 1, "After 3 swaps, b should be 1")

        print("Phi swap multiple iterations test passed!")

    def test_phi_three_way_rotation(self):
        """
        Test three-way rotation which forms a longer cycle.
        for i in 0..1: a, b, c = b, c, a
        """
        b = HIRBuilder()

        zero = b.const_load(0, "zero")
        one = b.const_load(1, "one")
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")

        init_a = b.const_load(1, "init_a")
        init_b = b.const_load(2, "init_b")
        init_c = b.const_load(3, "init_c")

        def loop_body(i, params):
            a = params[0]
            b_val = params[1]
            c = params[2]
            # Rotate: a <- b, b <- c, c <- a
            return [b_val, c, a]

        results = b.for_loop(
            start=zero,
            end=one,
            iter_args=[init_a, init_b, init_c],
            body_fn=loop_body
        )

        b.store(addr0, results[0])
        b.store(addr1, results[1])
        b.store(addr2, results[2])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # After 1 rotation: (1,2,3) -> (2,3,1)
        self.assertEqual(machine.mem[0], 2, "After rotation, a should be 2")
        self.assertEqual(machine.mem[1], 3, "After rotation, b should be 3")
        self.assertEqual(machine.mem[2], 1, "After rotation, c should be 1")

        print("Phi three-way rotation test passed!")

    def test_no_explicit_zero_constant(self):
        """
        Regression test for P1: Zero scratch fallback.

        Test a program that doesn't explicitly use const 0, but the compiler
        needs it internally for COPY operations. Previously this would use
        scratch[0] which might contain another value.
        """
        b = HIRBuilder()

        # Use only non-zero constants
        addr5 = b.const_load(5, "addr5")
        addr6 = b.const_load(6, "addr6")
        val100 = b.const_load(100, "val100")
        val200 = b.const_load(200, "val200")
        one = b.const_load(1, "one")
        two = b.const_load(2, "two")

        # Use a loop to force phi elimination (COPY needs zero)
        def loop_body(i, params):
            # Just carry a value through
            return [params[0]]

        results = b.for_loop(
            start=one,
            end=two,
            iter_args=[val100],
            body_fn=loop_body
        )

        b.store(addr5, results[0])
        b.store(addr6, val200)

        hir = b.build()

        # Manually compile to check zero handling
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        instrs = compile_to_vliw(lir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[5], 100, "Loop result should be 100")
        self.assertEqual(machine.mem[6], 200, "Direct store should be 200")

        print("No explicit zero constant test passed!")

    def test_const_after_zero_iteration_loop(self):
        """
        Test const used after a loop that may have 0 iterations.

        This tests that constants are available on all paths, including
        when a loop body never executes.
        """
        b = HIRBuilder()

        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        zero = b.const_load(0, "zero")

        # Load loop bound from memory (could be 0)
        bound = b.load(addr0, "bound")

        def loop_body(i, params):
            # Use const 99 inside loop
            val99 = b.const_load(99, "val99")
            b.store(addr1, val99)
            return []

        b.for_loop(start=zero, end=bound, iter_args=[], body_fn=loop_body)

        # Use const 77 after the loop (should work even if loop never ran)
        val77 = b.const_load(77, "val77")
        b.store(addr1, val77)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        # Test with bound=0 (loop never executes)
        mem = [0] + [0] * 99
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[1], 77, "Const after 0-iteration loop should work")

        print("Const after zero iteration loop test passed!")

    def test_large_const_value(self):
        """
        Regression test for P1: CONST immediates in zero_scratch.

        When computing max_scratch for the zero constant fallback, the code
        must not treat CONST immediate values as scratch indices. A large
        constant (e.g., 100000) should not inflate max_scratch.
        """
        b = HIRBuilder()

        # Use only non-zero constants, including a large one
        addr5 = b.const_load(5, "addr5")
        addr6 = b.const_load(6, "addr6")
        large_const = b.const_load(100000, "large_const")  # Large immediate value
        val42 = b.const_load(42, "val42")
        one = b.const_load(1, "one")
        two = b.const_load(2, "two")

        # Use a loop to trigger phi elimination (COPY needs zero)
        def loop_body(i, params):
            return [params[0]]

        results = b.for_loop(
            start=one,
            end=two,
            iter_args=[val42],
            body_fn=loop_body
        )

        # Store the large const and loop result
        b.store(addr5, large_const)
        b.store(addr6, results[0])

        hir = b.build()

        # Compile - this should NOT fail with assertion error about scratch space
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        instrs = compile_to_vliw(lir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[5], 100000, "Large const should be stored correctly")
        self.assertEqual(machine.mem[6], 42, "Loop result should be correct")

        print("Large const value test passed!")

    def test_phi_temp_scratch_safety(self):
        """
        Regression test for P2: Phi-cycle temp scratch reservation.

        Verify that the temp scratch used for phi cycle breaking doesn't
        collide with allocated scratch slots. This tests that max_scratch_used
        is tracked correctly and passed to eliminate_phis.
        """
        b = HIRBuilder()

        zero = b.const_load(0, "zero")
        one = b.const_load(1, "one")
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        # Create many intermediate values to use more scratch slots
        vals = [b.const_load(i * 10, f"val{i}") for i in range(10)]

        # Use all the values to prevent dead code elimination
        acc = vals[0]
        for v in vals[1:]:
            acc = b.add(acc, v)

        # Swap operation (phi cycle) - this needs temp scratch
        init_a = b.const_load(100, "init_a")
        init_b = b.const_load(200, "init_b")

        def loop_body(i, params):
            a = params[0]
            b_val = params[1]
            # Swap
            return [b_val, a]

        results = b.for_loop(
            start=zero,
            end=one,
            iter_args=[init_a, init_b],
            body_fn=loop_body
        )

        # Store results
        b.store(addr0, results[0])
        b.store(addr1, results[1])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # After 1 swap: a=200, b=100
        self.assertEqual(machine.mem[0], 200, "After swap, a should be 200")
        self.assertEqual(machine.mem[1], 100, "After swap, b should be 100")

        print("Phi temp scratch safety test passed!")

    def test_simplify_32bit_wrap(self):
        """
        Regression test for P1: SimplifyPass 32-bit wrap semantics.

        Constant folding must apply 32-bit wrap semantics (mask with 0xFFFFFFFF)
        because the VM uses mod 2**32 arithmetic. Without this, values that
        overflow 32 bits would be incorrect in subsequent operations.
        """
        b = HIRBuilder()
        # 0xFFFFFFFF + 1 should wrap to 0, then >> 1 = 0
        max_val = b.const_load(0xFFFFFFFF, "max")
        one = b.const_load(1, "one")
        added = b.add(max_val, one, "added")  # Should fold to 0
        shifted = b.shr(added, one, "shifted")  # 0 >> 1 = 0
        addr = b.const_load(0, "addr")
        b.store(addr, shifted)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 10
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 0, "32-bit wrap should make (0xFFFFFFFF+1)>>1 = 0")

        print("SimplifyPass 32-bit wrap test passed!")

    def test_cse_no_load_hoist_across_loop_stores(self):
        """
        Regression test for P1: CSE load hoisting across loop iterations.

        Loads inside loops must NOT CSE with pre-loop loads when the loop body
        contains stores. Each iteration may modify memory, so loads must be
        re-executed. This is achieved by incrementing the memory epoch when
        entering a loop body.
        """
        from compiler.hir import Const

        b = HIRBuilder()
        addr = b.const_load(0, "addr")
        zero = b.const_load(0, "zero")
        one = b.const_load(1, "one")
        two = b.const_load(2, "two")

        # store 0 to addr
        b.store(addr, zero)
        # pre_val = load(addr) -> 0
        pre_val = b.load(addr, "pre_val")

        def loop_body(i, params):
            # Each iteration: load, increment, store
            val = b.load(addr, "val")  # Should NOT CSE with pre_val
            new_val = b.add(val, one, "new_val")
            b.store(addr, new_val)
            return []

        b.for_loop(start=Const(0), end=two, iter_args=[], body_fn=loop_body)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 10
        machine = self._run_program(instrs, mem)
        # After 2 iterations: 0 -> 1 -> 2
        self.assertEqual(machine.mem[0], 2, "Loop should increment twice")

        print("CSE no load hoist across loop stores test passed!")


if __name__ == "__main__":
    unittest.main()
