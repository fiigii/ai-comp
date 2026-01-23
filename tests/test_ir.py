"""Tests for the IR compiler."""

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import unittest
import random
import json

from problem import (
    Machine,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    N_CORES,
    DebugInfo,
)
from perf_takehome import KernelBuilder
from compiler import (
    HIRBuilder,
    compile_hir_to_vliw,
    lower_to_lir,
    eliminate_phis,
    compile_to_vliw,
)


class TestIRCompiler(unittest.TestCase):
    """Test IR compiler correctness."""

    def test_ir_kernel_small(self):
        """Test IR kernel on a small example."""
        random.seed(42)
        forest = Tree.generate(4)
        inp = Input.generate(forest, 8, 4)  # small: 8 batch, 4 rounds
        mem = build_mem_image(forest, inp)

        kb = KernelBuilder()
        kb.build_kernel_ir(forest.height, len(forest.values), len(inp.indices), inp.rounds)

        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # Get reference result
        ref_mem = list(mem)
        for _ in reference_kernel2(ref_mem):
            pass

        inp_values_p = ref_mem[6]
        self.assertEqual(
            machine.mem[inp_values_p:inp_values_p + len(inp.values)],
            ref_mem[inp_values_p:inp_values_p + len(inp.values)],
            "IR kernel output doesn't match reference"
        )
        print(f"Small test passed! Cycles: {machine.cycle}")

    def _create_config_file(self, unroll_factor=None, max_trip_count=None):
        """Create a temp config file for pass manager."""
        import tempfile
        config = {"passes": {"loop-unroll": {"enabled": True, "options": {}}}}
        if unroll_factor is not None:
            config["passes"]["loop-unroll"]["options"]["unroll_factor"] = unroll_factor
        if max_trip_count is not None:
            config["passes"]["loop-unroll"]["options"]["max_trip_count"] = max_trip_count
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, f)
        f.close()
        return f.name

    def test_ir_kernel_medium(self):
        """Test IR kernel on a medium example."""
        random.seed(123)
        forest = Tree.generate(8)
        inp = Input.generate(forest, 32, 8)
        mem = build_mem_image(forest, inp)

        # Use config to limit unrolling (32*8=256 iterations would exhaust scratch)
        config_path = self._create_config_file(max_trip_count=8)
        kb = KernelBuilder()
        kb.build_kernel_ir(forest.height, len(forest.values), len(inp.indices), inp.rounds,
                           pass_config=config_path)
        os.unlink(config_path)

        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # Get reference result
        ref_mem = list(mem)
        for _ in reference_kernel2(ref_mem):
            pass

        inp_values_p = ref_mem[6]
        self.assertEqual(
            machine.mem[inp_values_p:inp_values_p + len(inp.values)],
            ref_mem[inp_values_p:inp_values_p + len(inp.values)],
            "IR kernel output doesn't match reference"
        )
        print(f"Medium test passed! Cycles: {machine.cycle}")

    def test_ir_kernel_full(self):
        """Test IR kernel on the full benchmark size."""
        random.seed(123)
        forest = Tree.generate(10)
        inp = Input.generate(forest, 256, 16)
        mem = build_mem_image(forest, inp)

        # Use config to limit unrolling (256*16=4096 iterations would exhaust scratch)
        config_path = self._create_config_file(max_trip_count=8)
        kb = KernelBuilder()
        kb.build_kernel_ir(forest.height, len(forest.values), len(inp.indices), inp.rounds,
                           pass_config=config_path)
        os.unlink(config_path)

        print(f"Generated {len(kb.instrs)} instructions")

        machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # Get reference result
        ref_mem = list(mem)
        for _ in reference_kernel2(ref_mem):
            pass

        inp_values_p = ref_mem[6]
        self.assertEqual(
            machine.mem[inp_values_p:inp_values_p + len(inp.values)],
            ref_mem[inp_values_p:inp_values_p + len(inp.values)],
            "IR kernel output doesn't match reference"
        )
        print(f"Full test passed! Cycles: {machine.cycle}")

    def test_ir_vs_original_kernel(self):
        """Compare IR kernel output with original build_kernel."""
        random.seed(456)
        forest = Tree.generate(6)
        inp = Input.generate(forest, 16, 4)

        # Run original unrolled kernel (use_ir=False)
        mem1 = build_mem_image(forest, inp)
        kb1 = KernelBuilder()
        kb1.build_kernel(forest.height, len(forest.values), len(inp.indices), inp.rounds, use_ir=False)
        machine1 = Machine(mem1, kb1.instrs, kb1.debug_info(), n_cores=N_CORES)
        machine1.enable_pause = False
        machine1.enable_debug = False
        machine1.run()

        # Run IR kernel (use_ir=True) with limited unrolling (16*4=64 iterations)
        config_path = self._create_config_file(max_trip_count=8)
        mem2 = build_mem_image(forest, inp)
        kb2 = KernelBuilder()
        kb2.build_kernel(forest.height, len(forest.values), len(inp.indices), inp.rounds,
                         use_ir=True, pass_config=config_path)
        os.unlink(config_path)
        machine2 = Machine(mem2, kb2.instrs, kb2.debug_info(), n_cores=N_CORES)
        machine2.enable_pause = False
        machine2.enable_debug = False
        machine2.run()

        inp_values_p = mem1[6]
        self.assertEqual(
            machine1.mem[inp_values_p:inp_values_p + len(inp.values)],
            machine2.mem[inp_values_p:inp_values_p + len(inp.values)],
            "IR kernel output doesn't match original kernel"
        )
        print(f"Comparison test passed!")
        print(f"Original kernel: {machine1.cycle} cycles, {len(kb1.instrs)} instructions")
        print(f"IR kernel: {machine2.cycle} cycles, {len(kb2.instrs)} instructions")


class TestIRCompilerSimplePrograms(unittest.TestCase):
    """Test IR compiler with simple programs."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_simple_arithmetic(self):
        """Test simple arithmetic: mem[0] = 10, mem[1] = 20, mem[2] = mem[0] + mem[1]"""
        b = HIRBuilder()

        # Load constants
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        val10 = b.const_load(10, "val10")
        val20 = b.const_load(20, "val20")

        # Store 10 to mem[0], 20 to mem[1]
        b.store(addr0, val10)
        b.store(addr1, val20)

        # Load them back and add
        a = b.load(addr0, "a")
        c = b.load(addr1, "c")
        result = b.add(a, c, "result")

        # Store result to mem[2]
        b.store(addr2, result)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[0], 10)
        self.assertEqual(machine.mem[1], 20)
        self.assertEqual(machine.mem[2], 30)
        print("Simple arithmetic test passed!")

    def test_simple_loop_sum(self):
        """Test simple loop: sum = 0; for i in 0..5: sum += i; mem[0] = sum"""
        b = HIRBuilder()

        zero = b.const_load(0, "zero")
        five = b.const_load(5, "five")
        addr0 = b.const_load(0, "addr0")

        # Sum loop with carried value
        init_sum = b.const_load(0, "init_sum")

        def loop_body(i, params):
            # params[0] is the running sum
            current_sum = params[0]
            new_sum = b.add(current_sum, i, "new_sum")
            return [new_sum]

        results = b.for_loop(
            start=zero,
            end=five,
            iter_args=[init_sum],
            body_fn=loop_body
        )

        # Store final sum (0+1+2+3+4 = 10)
        final_sum = results[0]
        b.store(addr0, final_sum)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[0], 10)  # 0+1+2+3+4 = 10
        print("Simple loop sum test passed!")

    def test_loop_array_init(self):
        """Test loop that initializes array: for i in 0..5: mem[i] = i * 2"""
        b = HIRBuilder()

        zero = b.const_load(0, "zero")
        five = b.const_load(5, "five")
        two = b.const_load(2, "two")

        def loop_body(i, params):
            val = b.mul(i, two, "val")
            b.store(i, val)
            return []

        b.for_loop(
            start=zero,
            end=five,
            iter_args=[],
            body_fn=loop_body
        )

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        for i in range(5):
            self.assertEqual(machine.mem[i], i * 2)
        print("Loop array init test passed!")

    def test_nested_loops(self):
        """Test nested loops: for i in 0..3: for j in 0..3: mem[i*3+j] = i + j"""
        b = HIRBuilder()

        zero = b.const_load(0, "zero")
        three = b.const_load(3, "three")

        def outer_body(i, outer_params):
            def inner_body(j, inner_params):
                # addr = i * 3 + j
                i_times_3 = b.mul(i, three, "i_times_3")
                addr = b.add(i_times_3, j, "addr")
                # val = i + j
                val = b.add(i, j, "val")
                b.store(addr, val)
                return []

            b.for_loop(start=zero, end=three, iter_args=[], body_fn=inner_body)
            return []

        b.for_loop(start=zero, end=three, iter_args=[], body_fn=outer_body)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        for i in range(3):
            for j in range(3):
                self.assertEqual(machine.mem[i * 3 + j], i + j)
        print("Nested loops test passed!")

    def test_if_else(self):
        """Test if/else: if mem[0] > 5: mem[1] = 100 else mem[1] = 200"""
        b = HIRBuilder()

        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        five = b.const_load(5, "five")
        val100 = b.const_load(100, "val100")
        val200 = b.const_load(200, "val200")

        # Load value from mem[0]
        val = b.load(addr0, "val")

        # Check if val > 5 (use val > 5 is same as 5 < val)
        cond = b.lt(five, val, "cond")

        # if/else
        def then_fn():
            return [val100]

        def else_fn():
            return [val200]

        results = b.if_stmt(cond, then_fn, else_fn)
        b.store(addr1, results[0])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        # Test with mem[0] = 10 (> 5), should store 100
        mem1 = [10] + [0] * 99
        machine1 = self._run_program(instrs, mem1)
        self.assertEqual(machine1.mem[1], 100)

        # Test with mem[0] = 3 (<= 5), should store 200
        mem2 = [3] + [0] * 99
        machine2 = self._run_program(instrs, mem2)
        self.assertEqual(machine2.mem[1], 200)

        print("If/else test passed!")

    def test_select_operation(self):
        """Test select operation: mem[1] = (mem[0] == 0) ? 42 : 99"""
        b = HIRBuilder()

        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        zero = b.const_load(0, "zero")
        val42 = b.const_load(42, "val42")
        val99 = b.const_load(99, "val99")

        val = b.load(addr0, "val")
        cond = b.eq(val, zero, "cond")
        result = b.select(cond, val42, val99, "result")
        b.store(addr1, result)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        # Test with mem[0] = 0, should select 42
        mem1 = [0] + [0] * 99
        machine1 = self._run_program(instrs, mem1)
        self.assertEqual(machine1.mem[1], 42)

        # Test with mem[0] = 5, should select 99
        mem2 = [5] + [0] * 99
        machine2 = self._run_program(instrs, mem2)
        self.assertEqual(machine2.mem[1], 99)

        print("Select operation test passed!")

    def test_fibonacci(self):
        """Test computing Fibonacci: fib(10) = 55"""
        b = HIRBuilder()

        zero = b.const_load(0, "zero")
        one = b.const_load(1, "one")
        ten = b.const_load(10, "ten")
        addr0 = b.const_load(0, "addr0")

        # fib_prev = 0, fib_curr = 1
        # for i in 0..10: fib_prev, fib_curr = fib_curr, fib_prev + fib_curr
        init_prev = b.const_load(0, "init_prev")
        init_curr = b.const_load(1, "init_curr")

        def loop_body(i, params):
            fib_prev = params[0]
            fib_curr = params[1]
            new_prev = fib_curr
            new_curr = b.add(fib_prev, fib_curr, "new_curr")
            return [new_prev, new_curr]

        results = b.for_loop(
            start=zero,
            end=ten,
            iter_args=[init_prev, init_curr],
            body_fn=loop_body
        )

        # After 10 iterations: results[0]=fib(10)=55, results[1]=fib(11)=89
        # Sequence: 0,1,1,2,3,5,8,13,21,34,55
        b.store(addr0, results[0])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[0], 55)
        print("Fibonacci test passed!")

    def test_bitwise_operations(self):
        """Test bitwise operations: xor, and, or, shifts"""
        b = HIRBuilder()

        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        addr2 = b.const_load(2, "addr2")
        addr3 = b.const_load(3, "addr3")
        addr4 = b.const_load(4, "addr4")

        val_a = b.const_load(0b11110000, "val_a")  # 240
        val_b = b.const_load(0b10101010, "val_b")  # 170
        shift = b.const_load(2, "shift")

        # XOR
        xor_result = b.xor(val_a, val_b, "xor")
        b.store(addr0, xor_result)

        # AND
        and_result = b.and_(val_a, val_b, "and")
        b.store(addr1, and_result)

        # OR
        or_result = b.or_(val_a, val_b, "or")
        b.store(addr2, or_result)

        # Left shift
        shl_result = b.shl(val_a, shift, "shl")
        b.store(addr3, shl_result)

        # Right shift
        shr_result = b.shr(val_a, shift, "shr")
        b.store(addr4, shr_result)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[0], 0b11110000 ^ 0b10101010)  # 90
        self.assertEqual(machine.mem[1], 0b11110000 & 0b10101010)  # 160
        self.assertEqual(machine.mem[2], 0b11110000 | 0b10101010)  # 250
        self.assertEqual(machine.mem[3], 0b11110000 << 2)  # 960
        self.assertEqual(machine.mem[4], 0b11110000 >> 2)  # 60

        print("Bitwise operations test passed!")

    def test_modulo_and_division(self):
        """Test modulo and division operations"""
        b = HIRBuilder()

        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")

        val17 = b.const_load(17, "val17")
        val5 = b.const_load(5, "val5")

        # 17 // 5 = 3
        div_result = b.div(val17, val5, "div")
        b.store(addr0, div_result)

        # 17 % 5 = 2
        mod_result = b.mod(val17, val5, "mod")
        b.store(addr1, mod_result)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[0], 3)  # 17 // 5
        self.assertEqual(machine.mem[1], 2)  # 17 % 5

        print("Modulo and division test passed!")

    def test_loop_with_early_computed_bound(self):
        """Test loop with bound computed from memory: for i in 0..mem[0]: mem[i+1] = i"""
        b = HIRBuilder()

        addr0 = b.const_load(0, "addr0")
        one = b.const_load(1, "one")
        zero = b.const_load(0, "zero")

        # Load loop bound from memory
        bound = b.load(addr0, "bound")

        def loop_body(i, params):
            addr = b.add(i, one, "addr")
            b.store(addr, i)
            return []

        b.for_loop(start=zero, end=bound, iter_args=[], body_fn=loop_body)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        # Set mem[0] = 5, so loop runs 0..5
        mem = [5] + [0] * 99
        machine = self._run_program(instrs, mem)

        for i in range(5):
            self.assertEqual(machine.mem[i + 1], i)

        print("Loop with computed bound test passed!")


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
        from compiler import lower_to_lir, eliminate_phis, compile_to_vliw

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
        from compiler import lower_to_lir, eliminate_phis, compile_to_vliw

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


class TestPassManagerAndLoopUnroll(unittest.TestCase):
    """Test pass manager and loop unrolling functionality."""

    def test_full_unroll_simple_loop(self):
        """Test that a simple loop with static bounds gets fully unrolled."""
        from compiler import (
            PassManager, PassConfig, LoopUnrollPass,
            Const, ForLoop, lower_to_lir, eliminate_phis, compile_to_vliw
        )

        b = HIRBuilder()
        zero = b.const_load(0, "zero")
        addr = b.const_load(10, "addr")

        # Simple loop: for i in 0..4: (no body, just count iterations)
        init_sum = b.const_load(0, "init_sum")

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
        """Test partial unrolling with a specific factor."""
        from compiler import (
            PassManager, PassConfig, LoopUnrollPass,
            Const, ForLoop, lower_to_lir, eliminate_phis, compile_to_vliw
        )

        b = HIRBuilder()
        addr = b.const_load(10, "addr")
        init_sum = b.const_load(0, "init_sum")

        def body(i, params):
            s = params[0]
            new_s = b.add(s, i, "sum")
            return [new_s]

        # Loop with 8 iterations
        results = b.for_loop(start=Const(0), end=Const(8), iter_args=[init_sum], body_fn=body)
        b.store(addr, results[0])

        hir = b.build()

        # Partial unroll with factor 4 (8/4 = 2 iterations remain)
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.config["loop-unroll"] = PassConfig(
            name="loop-unroll",
            options={"unroll_factor": 4, "max_trip_count": 100}
        )
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
        from compiler import (
            PassManager, PassConfig, LoopUnrollPass,
            ForLoop
        )

        b = HIRBuilder()
        # Load bound from memory (dynamic)
        addr0 = b.const_load(0, "addr0")
        bound = b.load(addr0, "bound")
        addr10 = b.const_load(10, "addr10")
        init_sum = b.const_load(0, "init_sum")

        def body(i, params):
            s = params[0]
            new_s = b.add(s, i, "sum")
            return [new_s]

        results = b.for_loop(start=b.const_load(0), end=bound, iter_args=[init_sum], body_fn=body)
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
        """Test that unrolling is skipped when factor doesn't divide trip count."""
        from compiler import (
            PassManager, PassConfig, LoopUnrollPass,
            Const, ForLoop
        )

        b = HIRBuilder()
        addr = b.const_load(10, "addr")
        init_sum = b.const_load(0, "init_sum")

        def body(i, params):
            s = params[0]
            new_s = b.add(s, i, "sum")
            return [new_s]

        # Loop with 10 iterations
        results = b.for_loop(start=Const(0), end=Const(10), iter_args=[init_sum], body_fn=body)
        b.store(addr, results[0])

        hir = b.build()

        # Try partial unroll with factor 3 (10 % 3 != 0, should skip)
        pm = PassManager()
        pm.add_pass(LoopUnrollPass())
        pm.config["loop-unroll"] = PassConfig(
            name="loop-unroll",
            options={"unroll_factor": 3, "max_trip_count": 100}
        )
        unrolled = pm.run(hir)

        # Loop should still have 10 iterations (not unrolled)
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

    def test_pass_config_from_json(self):
        """Test loading pass config from JSON."""
        import tempfile
        import os
        from compiler import PassManager, PassConfig, LoopUnrollPass

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

        # Write temp config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            pm = PassManager()
            pm.add_pass(LoopUnrollPass())
            pm.load_config(config_path)

            self.assertIn("loop-unroll", pm.config)
            self.assertEqual(pm.config["loop-unroll"].options["unroll_factor"], 2)
            self.assertEqual(pm.config["loop-unroll"].options["max_trip_count"], 50)
            print("Pass config from JSON test passed!")
        finally:
            os.unlink(config_path)

    def test_unroll_remaps_if_yields(self):
        """Ensure unrolled loop results are remapped through If yields."""
        from compiler import (
            PassManager, PassConfig, LoopUnrollPass,
            Const, lower_to_lir, eliminate_phis, compile_to_vliw
        )

        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        cond_true = b.const_load(1, "cond_true")
        init_sum = b.const_load(0, "init_sum")

        def then_fn():
            def body(i, params):
                s = params[0]
                return [b.add(s, i, "sum")]

            results = b.for_loop(start=Const(0), end=Const(4), iter_args=[init_sum], body_fn=body)
            return [results[0]]

        def else_fn():
            return [b.const_load(99, "else_val")]

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
        from compiler import (
            PassManager, PassConfig, LoopUnrollPass,
            Const, lower_to_lir, eliminate_phis, compile_to_vliw
        )

        b = HIRBuilder()
        addr0 = b.const_load(0, "addr0")
        addr1 = b.const_load(1, "addr1")
        bound = b.load(addr1, "bound")  # dynamic bound: outer loop should not be unrolled

        init_sum = b.const_load(0, "init_sum")

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass

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
        from compiler import count_statements
        # CSE should reduce statements
        self.assertLess(count_statements(transformed), count_statements(hir))

        print("CSE redundant add test passed!")

    def test_cse_redundant_mul(self):
        """Test that two identical a * b expressions -> second eliminated."""
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass

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
        from compiler import compile_hir_to_vliw
        instrs = compile_hir_to_vliw(transformed)

        mem = [5, 7] + [0] * 98
        machine = self._run_program(instrs, mem)

        # 5 * 7 = 35, and result1 + result2 should be 35 + 35 = 70
        self.assertEqual(machine.mem[0], 70)
        print("CSE redundant mul test passed!")

    def test_cse_const_dedup(self):
        """Test that two const(42) -> second eliminated."""
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass

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
        from compiler import compile_hir_to_vliw
        instrs = compile_hir_to_vliw(transformed)

        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        # Both should be 42
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[1], 42)
        print("CSE const dedup test passed!")

    def test_cse_different_operands_not_eliminated(self):
        """Test that a + b and a + c are both kept."""
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, count_statements

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, count_statements

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, count_statements

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
        from compiler import compile_hir_to_vliw
        instrs = compile_hir_to_vliw(transformed)

        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)

        # 42 + 42 = 84
        self.assertEqual(machine.mem[1], 84)
        print("CSE load same address test passed!")

    def test_cse_load_after_store_not_eliminated(self):
        """Test that load, store, load -> second load NOT eliminated."""
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, count_statements

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
        from compiler import compile_hir_to_vliw
        instrs = compile_hir_to_vliw(transformed)

        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)

        # After store, addr0 = 99, addr1 = val2 = 99
        self.assertEqual(machine.mem[0], 99)
        self.assertEqual(machine.mem[1], 99)
        print("CSE load after store not eliminated test passed!")

    def test_cse_load_different_address_after_store(self):
        """Test that store to X, load from Y -> load NOT eliminated (conservative)."""
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, count_statements

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
        from compiler import compile_hir_to_vliw
        instrs = compile_hir_to_vliw(transformed)

        mem = [0, 42, 0] + [0] * 97
        machine = self._run_program(instrs, mem)

        # addr2 should have 42 (loaded from addr1)
        self.assertEqual(machine.mem[2], 42)
        print("CSE load different address after store test passed!")

    def test_cse_store_not_eliminated(self):
        """Test that store operations are always kept."""
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, count_statements, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, Const, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, Const, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, compile_hir_to_vliw, count_statements

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, Const, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, LoopUnrollPass, CSEPass, Const, count_statements

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
        from compiler import compile_hir_to_vliw

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass

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
        from compiler import HIRBuilder, PassManager, PassConfig, CSEPass, count_statements

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


if __name__ == "__main__":
    unittest.main()
