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
from ir_compiler import (
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
        from ir_compiler import lower_to_lir, eliminate_phis, compile_to_vliw

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
        from ir_compiler import lower_to_lir, eliminate_phis, compile_to_vliw

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
        from ir_compiler import (
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
        from ir_compiler import (
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
        from ir_compiler import (
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
        from ir_compiler import (
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
        from ir_compiler import PassManager, PassConfig, LoopUnrollPass

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


if __name__ == "__main__":
    unittest.main()
