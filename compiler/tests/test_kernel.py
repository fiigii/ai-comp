"""Tests for IR compiler end-to-end kernel correctness."""

import unittest
import random

from compiler.tests.conftest import (
    Machine,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    N_CORES,
    KernelBuilder,
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
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), inp.rounds)

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

    def test_ir_kernel_medium(self):
        """Test IR kernel on a medium example."""
        random.seed(123)
        forest = Tree.generate(8)
        inp = Input.generate(forest, 32, 8)
        mem = build_mem_image(forest, inp)

        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), inp.rounds)

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

        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), inp.rounds)

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

    def test_ir_vs_reference_kernel(self):
        """Compare IR kernel output with reference_kernel2."""
        random.seed(456)
        forest = Tree.generate(6)
        inp = Input.generate(forest, 16, 4)

        # Get reference result
        mem_ref = build_mem_image(forest, inp)
        for _ in reference_kernel2(mem_ref):
            pass

        # Run IR kernel
        mem_ir = build_mem_image(forest, inp)
        kb = KernelBuilder()
        kb.build_kernel(forest.height, len(forest.values), len(inp.indices), inp.rounds)
        machine = Machine(mem_ir, kb.instrs, kb.debug_info(), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        inp_values_p = mem_ref[6]
        self.assertEqual(
            machine.mem[inp_values_p:inp_values_p + len(inp.values)],
            mem_ref[inp_values_p:inp_values_p + len(inp.values)],
            "IR kernel output doesn't match reference kernel"
        )
        print(f"Comparison test passed! IR kernel: {machine.cycle} cycles")


if __name__ == "__main__":
    unittest.main()
