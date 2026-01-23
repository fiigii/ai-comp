"""Tests for Common Subexpression Elimination pass."""

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
    CSEPass,
    LoopUnrollPass,
    Const,
    count_statements,
)


class TestCSEPass(unittest.TestCase):
    """Test Common Subexpression Elimination pass."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    # --- Basic CSE Tests ---

    def test_cse_redundant_add(self):
        """Test that two identical a + b expressions -> second eliminated."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        a = b.load(addr0, "a")
        c = b.load(addr1, "c")

        # First a + c
        result1 = b.add(a, c, "result1")
        # Second a + c (should be eliminated)
        result2 = b.add(a, c, "result2")

        # Store both to verify they're the same value
        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Check that an expression was eliminated
        # CSE should reduce statements
        self.assertLess(count_statements(transformed), count_statements(hir))

        print("CSE redundant add test passed!")

    def test_cse_redundant_mul(self):
        """Test that two identical a * b expressions -> second eliminated."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        a = b.load(addr0, "a")
        c = b.load(addr1, "c")

        # First a * c
        result1 = b.mul(a, c, "result1")
        # Second a * c (should be eliminated)
        result2 = b.mul(a, c, "result2")

        # Use both to verify they resolve to same value
        sum_results = b.add(result1, result2, "sum")
        b.store(addr0, sum_results)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Compile and run
        instrs = compile_hir_to_vliw(transformed)

        mem = [5, 7] + [0] * 98
        machine = self._run_program(instrs, mem)

        # 5 * 7 = 35, and result1 + result2 should be 35 + 35 = 70
        self.assertEqual(machine.mem[0], 70)
        print("CSE redundant mul test passed!")

    def test_cse_const_dedup(self):
        """Test that two const(42) -> second eliminated."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        # Two identical constants
        val42_1 = b.const_load(42, "val42_1")
        val42_2 = b.const_load(42, "val42_2")

        b.store(addr0, val42_1)
        b.store(addr1, val42_2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Compile and run
        instrs = compile_hir_to_vliw(transformed)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # Both should be 42
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[1], 42)
        print("CSE const dedup test passed!")

    def test_cse_different_operands_not_eliminated(self):
        """Test that a + b and a + c are both kept."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")
        c = b.load(addr2, "c")

        # a + b
        result1 = b.add(a, val_b, "result1")
        # a + c (different operands, should NOT be eliminated)
        result2 = b.add(a, c, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Statement count should be the same (no elimination for different operands)
        self.assertEqual(count_statements(transformed), original_count)
        print("CSE different operands not eliminated test passed!")

    def test_cse_different_opcodes_not_eliminated(self):
        """Test that a + b and a - b are both kept."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")

        # a + b
        result1 = b.add(a, val_b, "result1")
        # a - b (different opcode, should NOT be eliminated)
        result2 = b.sub(a, val_b, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Statement count should be the same
        self.assertEqual(count_statements(transformed), original_count)
        print("CSE different opcodes not eliminated test passed!")

    def test_cse_chain(self):
        """Test that a + b, then (a+b) + c reuses first result."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")
        c = b.load(addr2, "c")

        # First a + b
        ab = b.add(a, val_b, "ab")
        # Second a + b (should be eliminated)
        ab2 = b.add(a, val_b, "ab2")
        # (a+b) + c using the redundant ab2
        result = b.add(ab2, c, "result")

        b.store(addr0, result)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        mem = [10, 20, 5] + [0] * 97
        machine = self._run_program(instrs, mem)

        # (10 + 20) + 5 = 35
        self.assertEqual(machine.mem[0], 35)
        print("CSE chain test passed!")

    # --- Memory Safety Tests ---

    def test_cse_load_same_address(self):
        """Test that two loads from same address (no store between) -> second eliminated."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        # First load from addr0
        val1 = b.load(addr0, "val1")
        # Second load from addr0 (should be eliminated)
        val2 = b.load(addr0, "val2")

        result = b.add(val1, val2, "result")
        b.store(addr1, result)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # One load should be eliminated
        self.assertLess(count_statements(transformed), original_count)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)

        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)

        # 42 + 42 = 84
        self.assertEqual(machine.mem[1], 84)
        print("CSE load same address test passed!")

    def test_cse_load_after_store_not_eliminated(self):
        """Test that load, store, load -> second load NOT eliminated."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        val99 = b.const_load(99, "val99")

        # First load from addr0
        val1 = b.load(addr0, "val1")
        # Store to addr0 (clobbers memory)
        b.store(addr0, val99)
        # Second load from addr0 (should NOT be eliminated)
        val2 = b.load(addr0, "val2")

        b.store(addr1, val2)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # No load should be eliminated (store clobbered memory)
        # The const loads might be deduplicated but the loads shouldn't be
        self.assertEqual(count_statements(transformed), original_count)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)

        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)

        # After store, addr0 = 99, addr1 = val2 = 99
        self.assertEqual(machine.mem[0], 99)
        self.assertEqual(machine.mem[1], 99)
        print("CSE load after store not eliminated test passed!")

    def test_cse_load_different_address_after_store(self):
        """Test that store to X, load from Y -> load NOT eliminated (conservative)."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        val99 = b.const_load(99, "val99")

        # First load from addr1
        val1 = b.load(addr1, "val1")
        # Store to addr0 (different address, but clobbers memory conservatively)
        b.store(addr0, val99)
        # Second load from addr1 (should NOT be eliminated due to conservative approach)
        val2 = b.load(addr1, "val2")

        b.store(addr2, val2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)

        mem = [0, 42, 0] + [0] * 97
        machine = self._run_program(instrs, mem)

        # addr2 should have 42 (loaded from addr1)
        self.assertEqual(machine.mem[2], 42)
        print("CSE load different address after store test passed!")

    def test_cse_store_not_eliminated(self):
        """Test that store operations are always kept."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        val1 = b.const_load(100, "val1")
        val2 = b.const_load(200, "val2")

        # Two stores to same address
        b.store(addr0, val1)
        b.store(addr0, val2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # Second store should overwrite first
        self.assertEqual(machine.mem[0], 200)
        print("CSE store not eliminated test passed!")

    # --- Control Flow Tests ---

    def test_cse_in_loop_body(self):
        """Test that CSE works within a loop iteration."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        init_sum = b.const_load(0, "init_sum")

        def loop_body(i, params):
            s = params[0]
            # Two identical additions within the loop body
            inc1 = b.add(i, i, "inc1")  # i + i
            inc2 = b.add(i, i, "inc2")  # i + i (should be eliminated)
            total = b.add(inc1, inc2, "total")  # (i+i) + (i+i) = 4*i
            new_s = b.add(s, total, "new_s")
            return [new_s]

        results = b.for_loop(start=Const(0), end=Const(5), iter_args=[init_sum], body_fn=loop_body)
        b.store(addr0, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # Sum of 4*i for i in 0..5: 4*(0+1+2+3+4) = 4*10 = 40
        self.assertEqual(machine.mem[0], 40)
        print("CSE in loop body test passed!")

    def test_cse_across_loop_iterations_not_shared(self):
        """Test that loop body params have unique VNs per iteration."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        init_val = b.const_load(1, "init_val")

        def loop_body(i, params):
            # params[0] changes each iteration
            current = params[0]
            doubled = b.add(current, current, "doubled")
            return [doubled]

        results = b.for_loop(start=Const(0), end=Const(3), iter_args=[init_val], body_fn=loop_body)
        b.store(addr0, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # 1 -> 2 -> 4 -> 8 (3 iterations of doubling)
        self.assertEqual(machine.mem[0], 8)
        print("CSE across loop iterations not shared test passed!")

    def test_cse_in_if_then_branch(self):
        """Test that CSE works within then-branch."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        cond_true = b.const_load(1, "cond_true")

        def then_fn():
            # Two identical additions
            val1 = b.const_load(5, "val1")
            val2 = b.const_load(5, "val2")  # Should be eliminated
            return [b.add(val1, val2, "sum")]

        def else_fn():
            return [b.const_load(0, "zero")]

        results = b.if_stmt(cond_true, then_fn, else_fn)
        b.store(addr1, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # Then branch: 5 + 5 = 10
        self.assertEqual(machine.mem[1], 10)
        print("CSE in if then branch test passed!")

    def test_cse_in_if_else_branch(self):
        """Test that CSE works within else-branch."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        cond_false = b.const_load(0, "cond_false")

        def then_fn():
            return [b.const_load(0, "zero")]

        def else_fn():
            # Two identical additions
            val1 = b.const_load(7, "val1")
            val2 = b.const_load(7, "val2")  # Should be eliminated
            return [b.add(val1, val2, "sum")]

        results = b.if_stmt(cond_false, then_fn, else_fn)
        b.store(addr1, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # Else branch: 7 + 7 = 14
        self.assertEqual(machine.mem[1], 14)
        print("CSE in if else branch test passed!")

    def test_cse_across_if_branches_not_allowed(self):
        """Test that then-branch expr NOT reused in else-branch."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        cond = b.load(addr0, "cond")

        def then_fn():
            val = b.const_load(42, "val42_then")
            return [val]

        def else_fn():
            # Same constant as then branch, but should not be shared
            val = b.const_load(42, "val42_else")
            return [val]

        results = b.if_stmt(cond, then_fn, else_fn)
        b.store(addr1, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        # Test both paths
        mem1 = [1, 0] + [0] * 98  # cond = 1, take then
        machine1 = self._run_program(instrs, mem1)
        self.assertEqual(machine1.mem[1], 42)

        mem2 = [0, 0] + [0] * 98  # cond = 0, take else
        machine2 = self._run_program(instrs, mem2)
        self.assertEqual(machine2.mem[1], 42)

        print("CSE across if branches not allowed test passed!")

    def test_cse_before_if_reused_in_branches(self):
        """Test that expr before if CAN be reused in both branches."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        # Compute before if
        pre_val = b.const_load(100, "pre_val")

        cond = b.load(addr0, "cond")

        def then_fn():
            # Use same constant as pre_val
            val = b.const_load(100, "val100_then")  # Should be eliminated
            return [b.add(val, b.const_load(1, "one_then"), "then_result")]

        def else_fn():
            # Use same constant as pre_val
            val = b.const_load(100, "val100_else")  # Should be eliminated
            return [b.add(val, b.const_load(2, "two_else"), "else_result")]

        results = b.if_stmt(cond, then_fn, else_fn)
        b.store(addr1, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        # Test then path: 100 + 1 = 101
        mem1 = [1, 0] + [0] * 98
        machine1 = self._run_program(instrs, mem1)
        self.assertEqual(machine1.mem[1], 101)

        # Test else path: 100 + 2 = 102
        mem2 = [0, 0] + [0] * 98
        machine2 = self._run_program(instrs, mem2)
        self.assertEqual(machine2.mem[1], 102)

        print("CSE before if reused in branches test passed!")

    def test_cse_store_in_branch_clobbers_parent(self):
        """Test that store in branch invalidates loads after if."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        val99 = b.const_load(99, "val99")

        # Load before if
        val_before = b.load(addr0, "val_before")

        cond = b.load(addr1, "cond")

        def then_fn():
            # Store clobbers memory
            b.store(addr0, val99)
            return [val99]

        def else_fn():
            return [b.const_load(0, "zero")]

        b.if_stmt(cond, then_fn, else_fn)

        # Load after if - should NOT be eliminated even though same address
        val_after = b.load(addr0, "val_after")
        b.store(addr1, val_after)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)

        # When cond = 1 (then), addr0 gets 99
        mem1 = [42, 1] + [0] * 98
        machine1 = self._run_program(instrs, mem1)
        self.assertEqual(machine1.mem[1], 99)

        # When cond = 0 (else), addr0 stays 42
        mem2 = [42, 0] + [0] * 98
        machine2 = self._run_program(instrs, mem2)
        self.assertEqual(machine2.mem[1], 42)

        print("CSE store in branch clobbers parent test passed!")

    # --- Correctness Tests ---

    def test_cse_preserves_semantics_simple(self):
        """Test that CSE preserves program semantics for simple programs."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")

        # Multiple redundant operations
        sum1 = b.add(a, val_b, "sum1")
        sum2 = b.add(a, val_b, "sum2")
        sum3 = b.add(a, val_b, "sum3")
        total = b.add(sum1, b.add(sum2, sum3, "s23"), "total")
        b.store(addr2, total)

        hir = b.build()

        # Run without CSE
        instrs_no_cse = compile_hir_to_vliw(hir)
        mem_no_cse = [10, 20, 0] + [0] * 97
        machine_no_cse = self._run_program(instrs_no_cse, mem_no_cse)
        result_no_cse = machine_no_cse.mem[2]

        # Run with CSE
        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)
        instrs_cse = compile_hir_to_vliw(transformed)
        mem_cse = [10, 20, 0] + [0] * 97
        machine_cse = self._run_program(instrs_cse, mem_cse)
        result_cse = machine_cse.mem[2]

        # Both should produce same result: (10+20) + ((10+20) + (10+20)) = 30 + 60 = 90
        self.assertEqual(result_no_cse, 90)
        self.assertEqual(result_cse, 90)
        print("CSE preserves semantics simple test passed!")

    def test_cse_preserves_semantics_with_loops(self):
        """Test that CSE preserves semantics for programs with loops."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        init_sum = b.const_load(0, "init_sum")

        def loop_body(i, params):
            s = params[0]
            # Redundant computations
            twice_i = b.add(i, i, "twice_i")
            twice_i2 = b.add(i, i, "twice_i2")  # Should be eliminated by CSE
            combined = b.add(twice_i, twice_i2, "combined")
            new_s = b.add(s, combined, "new_s")
            return [new_s]

        results = b.for_loop(start=Const(0), end=Const(5), iter_args=[init_sum], body_fn=loop_body)
        b.store(addr0, results[0])

        hir = b.build()

        # Without CSE
        instrs_no_cse = compile_hir_to_vliw(hir)
        mem_no_cse = [0] * 100
        machine_no_cse = self._run_program(instrs_no_cse, mem_no_cse)
        result_no_cse = machine_no_cse.mem[0]

        # With CSE
        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)
        instrs_cse = compile_hir_to_vliw(transformed)
        mem_cse = [0] * 100
        machine_cse = self._run_program(instrs_cse, mem_cse)
        result_cse = machine_cse.mem[0]

        # Both should give same result
        self.assertEqual(result_no_cse, result_cse)
        # Sum of 4*i for i in 0..5 = 4*(0+1+2+3+4) = 40
        self.assertEqual(result_cse, 40)
        print("CSE preserves semantics with loops test passed!")

    def test_cse_preserves_semantics_with_memory(self):
        """Test that CSE preserves semantics with memory operations."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")

        # Load, compute, store, load again
        val1 = b.load(addr0, "val1")
        doubled = b.add(val1, val1, "doubled")
        b.store(addr1, doubled)
        val2 = b.load(addr1, "val2")
        result = b.add(val2, b.const_load(1, "one"), "result")
        b.store(addr2, result)

        hir = b.build()

        # Without CSE
        instrs_no_cse = compile_hir_to_vliw(hir)
        mem_no_cse = [10, 0, 0] + [0] * 97
        machine_no_cse = self._run_program(instrs_no_cse, mem_no_cse)

        # With CSE
        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)
        instrs_cse = compile_hir_to_vliw(transformed)
        mem_cse = [10, 0, 0] + [0] * 97
        machine_cse = self._run_program(instrs_cse, mem_cse)

        # Both should give same result
        self.assertEqual(machine_no_cse.mem[1], machine_cse.mem[1])  # 20
        self.assertEqual(machine_no_cse.mem[2], machine_cse.mem[2])  # 21
        print("CSE preserves semantics with memory test passed!")

    # --- Integration Tests ---

    def test_cse_after_unroll(self):
        """Test that CSE finds opportunities in unrolled code."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        init_sum = b.const_load(0, "init_sum")

        def loop_body(i, params):
            s = params[0]
            # When unrolled, multiple iterations will have const(1) loads
            one = b.const_load(1, "one")
            new_s = b.add(s, one, "new_s")
            return [new_s]

        results = b.for_loop(start=Const(0), end=Const(4), iter_args=[init_sum], body_fn=loop_body)
        b.store(addr0, results[0])

        hir = b.build()

        # Unroll only
        pm_unroll = PassManager()
        pm_unroll.add_pass(LoopUnrollPass())
        pm_unroll.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 100})
        unrolled = pm_unroll.run(hir)

        # Unroll + CSE
        pm_both = PassManager()
        pm_both.add_pass(LoopUnrollPass())
        pm_both.add_pass(CSEPass())
        pm_both.config["loop-unroll"] = PassConfig(name="loop-unroll", options={"max_trip_count": 100})
        with_cse = pm_both.run(hir)

        # CSE should reduce statements
        self.assertLess(count_statements(with_cse), count_statements(unrolled))

        # Both should produce correct result
        instrs_unroll = compile_hir_to_vliw(unrolled)
        mem_unroll = [0] * 100
        machine_unroll = self._run_program(instrs_unroll, mem_unroll)

        instrs_cse = compile_hir_to_vliw(with_cse)
        mem_cse = [0] * 100
        machine_cse = self._run_program(instrs_cse, mem_cse)

        # Both should give 4 (0 + 1 + 1 + 1 + 1)
        self.assertEqual(machine_unroll.mem[0], 4)
        self.assertEqual(machine_cse.mem[0], 4)

        print("CSE after unroll test passed!")

    def test_cse_metrics_reported(self):
        """Test that CSE pass reports correct metrics."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        # Create redundant expressions
        val42_1 = b.const_load(42, "val42_1")
        val42_2 = b.const_load(42, "val42_2")
        a = b.load(addr0, "a")
        c = b.load(addr1, "c")
        sum1 = b.add(a, c, "sum1")
        sum2 = b.add(a, c, "sum2")

        b.store(addr0, sum1)
        b.store(addr1, sum2)

        hir = b.build()

        cse_pass = CSEPass()
        pm = PassManager()
        pm.add_pass(cse_pass)
        pm.run(hir)

        metrics = cse_pass.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("expressions_eliminated", metrics.custom)
        self.assertGreater(metrics.custom["expressions_eliminated"], 0)
        self.assertIn("consts_eliminated", metrics.custom)

        print(f"CSE metrics: {metrics.custom}")
        print("CSE metrics reported test passed!")

    def test_cse_config_enabled_disabled(self):
        """Test that CSE pass can be disabled via config."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        val1 = b.const_load(42, "val1")
        val2 = b.const_load(42, "val2")  # Would be eliminated if CSE enabled

        b.store(addr0, val1)
        b.store(addr0, val2)

        hir = b.build()

        # With CSE enabled (default)
        pm_enabled = PassManager()
        pm_enabled.add_pass(CSEPass())
        result_enabled = pm_enabled.run(hir)

        # With CSE disabled
        pm_disabled = PassManager()
        pm_disabled.add_pass(CSEPass())
        pm_disabled.config["cse"] = PassConfig(name="cse", enabled=False)
        result_disabled = pm_disabled.run(hir)

        # Disabled should not eliminate anything
        self.assertEqual(count_statements(result_disabled), count_statements(hir))
        # Enabled should eliminate
        self.assertLess(count_statements(result_enabled), count_statements(hir))

        print("CSE config enabled/disabled test passed!")

    # --- Commutative Op Canonicalization Tests ---

    def test_cse_commutative_add(self):
        """Test that a + b and b + a are merged (commutative)."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")

        # a + b
        result1 = b.add(a, val_b, "result1")
        # b + a (same operation, different order - should be eliminated)
        result2 = b.add(val_b, a, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # One addition should be eliminated due to commutativity
        self.assertLess(count_statements(transformed), original_count)
        print("CSE commutative add test passed!")

    def test_cse_commutative_mul(self):
        """Test that a * b and b * a are merged (commutative)."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")

        # a * b
        result1 = b.mul(a, val_b, "result1")
        # b * a (should be eliminated)
        result2 = b.mul(val_b, a, "result2")

        sum_results = b.add(result1, result2, "sum")
        b.store(addr0, sum_results)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # One mul should be eliminated
        self.assertLess(count_statements(transformed), original_count)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [3, 4] + [0] * 98
        machine = self._run_program(instrs, mem)
        # 3 * 4 = 12, result1 + result2 = 12 + 12 = 24
        self.assertEqual(machine.mem[0], 24)
        print("CSE commutative mul test passed!")

    def test_cse_commutative_xor(self):
        """Test that a ^ b and b ^ a are merged (commutative)."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")

        # a ^ b
        result1 = b.xor(a, val_b, "result1")
        # b ^ a (should be eliminated)
        result2 = b.xor(val_b, a, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # One xor should be eliminated
        self.assertLess(count_statements(transformed), original_count)
        print("CSE commutative xor test passed!")

    def test_cse_non_commutative_sub_not_merged(self):
        """Test that a - b and b - a are NOT merged (non-commutative)."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")

        # a - b
        result1 = b.sub(a, val_b, "result1")
        # b - a (different operation, should NOT be eliminated)
        result2 = b.sub(val_b, a, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [10, 4, 0] + [0] * 97
        machine = self._run_program(instrs, mem)
        # a - b = 10 - 4 = 6, b - a = 4 - 10 = -6 (wraps in 32-bit)
        self.assertEqual(machine.mem[0], 6)
        self.assertEqual(machine.mem[1], (4 - 10) & 0xFFFFFFFF)
        print("CSE non-commutative sub not merged test passed!")

    def test_cse_non_commutative_div_not_merged(self):
        """Test that a // b and b // a are NOT merged (non-commutative)."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        a = b.load(addr0, "a")
        val_b = b.load(addr1, "b")

        # a // b
        result1 = b.div(a, val_b, "result1")
        # b // a (different operation, should NOT be eliminated)
        result2 = b.div(val_b, a, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [20, 4, 0] + [0] * 97
        machine = self._run_program(instrs, mem)
        # a // b = 20 // 4 = 5, b // a = 4 // 20 = 0
        self.assertEqual(machine.mem[0], 5)
        self.assertEqual(machine.mem[1], 0)
        print("CSE non-commutative div not merged test passed!")

    # --- Memory Epoch Tracking Tests ---

    def test_cse_multiple_loads_same_epoch(self):
        """Test that multiple loads with same address in same epoch are CSE'd."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        # Multiple loads from same address, no store between
        val1 = b.load(addr0, "val1")
        val2 = b.load(addr0, "val2")  # Same epoch, should be eliminated
        val3 = b.load(addr0, "val3")  # Same epoch, should be eliminated

        result = b.add(val1, b.add(val2, val3, "sum23"), "result")
        b.store(addr1, result)

        hir = b.build()
        original_count = count_statements(hir)

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Two loads should be eliminated
        self.assertLess(count_statements(transformed), original_count)
        print("CSE multiple loads same epoch test passed!")

    def test_cse_loads_different_epochs_not_merged(self):
        """Test that loads with different epochs are NOT merged."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        val99 = b.const_load(99, "val99")

        # Load, store, load - different epochs
        val1 = b.load(addr0, "val1")
        b.store(addr0, val99)  # Increments epoch
        val2 = b.load(addr0, "val2")  # Different epoch, should NOT be eliminated

        b.store(addr1, val2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)
        # After store, addr0 = 99, so val2 = 99
        self.assertEqual(machine.mem[1], 99)
        print("CSE loads different epochs not merged test passed!")

    def test_cse_epoch_propagates_from_loop(self):
        """Test that store in loop increments parent epoch."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        # Load before loop
        val_before = b.load(addr0, "val_before")

        def loop_body(i, params):
            # Store in loop should affect epoch
            b.store(addr0, i)
            return []

        b.for_loop(start=Const(0), end=Const(3), iter_args=[], body_fn=loop_body)

        # Load after loop - should NOT be CSE'd with val_before
        val_after = b.load(addr0, "val_after")
        b.store(addr1, val_after)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)
        # After loop, addr0 = 2 (last iteration), so val_after = 2
        self.assertEqual(machine.mem[1], 2)
        print("CSE epoch propagates from loop test passed!")

    def test_cse_epoch_propagates_from_if(self):
        """Test that store in branch increments parent epoch."""
        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        val99 = b.const_load(99, "val99")

        # Load before if
        val_before = b.load(addr0, "val_before")

        cond = b.load(addr1, "cond")

        def then_fn():
            b.store(addr0, val99)
            return []

        def else_fn():
            return []

        b.if_stmt(cond, then_fn, else_fn)

        # Load after if - should NOT be CSE'd with val_before
        val_after = b.load(addr0, "val_after")
        b.store(addr1, val_after)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(CSEPass())
        transformed = pm.run(hir)

        # Verify correctness with cond = 1 (then branch)
        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 1] + [0] * 98
        machine = self._run_program(instrs, mem)
        # After then branch, addr0 = 99, so val_after = 99
        self.assertEqual(machine.mem[1], 99)
        print("CSE epoch propagates from if test passed!")


if __name__ == "__main__":
    unittest.main()
