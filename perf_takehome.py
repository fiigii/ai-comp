"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import json
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)

from ir_compiler import (
    HIRBuilder,
    Const,
    lower_to_lir,
    compile_to_vliw,
    compile_hir_to_vliw,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
        use_ir: bool = True, print_after_all: bool = False, pass_config: str = None
    ):
        """
        Build kernel instructions.

        Args:
            use_ir: If True (default), use the IR compiler with control flow.
                    If False, use the original unrolled implementation.
            print_after_all: If True, print IR after each compilation pass (only for IR mode).
            pass_config: Optional path to JSON config file for pass options.
        """
        if use_ir:
            return self.build_kernel_ir(forest_height, n_nodes, batch_size, rounds,
                                        print_after_all, pass_config)
        return self._build_kernel_unrolled(forest_height, n_nodes, batch_size, rounds)

    def _build_kernel_unrolled(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Original unrolled kernel implementation.
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr = self.alloc_scratch("tmp_addr")

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)
                # idx = mem[inp_indices_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("load", tmp_idx, tmp_addr)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "idx"))))
                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_val, (round, i, "val"))))
                # node_val = mem[forest_values_p + idx]
                body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
                body.append(("load", ("load", tmp_node_val, tmp_addr)))
                body.append(("debug", ("compare", tmp_node_val, (round, i, "node_val"))))
                # val = myhash(val ^ node_val)
                body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
                body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
                body.append(("debug", ("compare", tmp_val, (round, i, "hashed_val"))))
                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("alu", ("%", tmp1, tmp_val, two_const)))
                body.append(("alu", ("==", tmp1, tmp1, zero_const)))
                body.append(("flow", ("select", tmp3, tmp1, one_const, two_const)))
                body.append(("alu", ("*", tmp_idx, tmp_idx, two_const)))
                body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "next_idx"))))
                # idx = 0 if idx >= n_nodes else idx
                body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
                body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
                body.append(("debug", ("compare", tmp_idx, (round, i, "wrapped_idx"))))
                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx)))
                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_val)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

    def build_kernel_ir(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int,
        print_after_all: bool = False, pass_config: str = None
    ):
        """
        Build kernel using the IR compiler with control flow (loops).
        This uses SSA-form HIR which is lowered to LIR and then to VLIW.

        Args:
            print_after_all: If True, print IR after each compilation pass.
            pass_config: Optional path to JSON config file for pass options.
        """
        self._pass_config = pass_config  # Store for compile call
        b = HIRBuilder()

        # Load header values from memory (addresses 0-6)
        def load_header(idx: int, name: str):
            addr = b.const_load(idx, f"addr_{name}")
            return b.load(addr, name)

        rounds_val = load_header(0, "rounds")
        n_nodes_val = load_header(1, "n_nodes")
        batch_size_val = load_header(2, "batch_size")
        forest_height_val = load_header(3, "forest_height")
        forest_values_p = load_header(4, "forest_values_p")
        inp_indices_p = load_header(5, "inp_indices_p")
        inp_values_p = load_header(6, "inp_values_p")

        # Constants (as SSAValues for use in computations)
        zero = b.const_load(0, "zero")
        one = b.const_load(1, "one")
        two = b.const_load(2, "two")

        # Compile-time constants for loop bounds (as Const for unrolling)
        rounds_const = Const(rounds)
        batch_const = Const(batch_size)
        zero_const = Const(0)

        # Hash stage constants
        hash_consts = []
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = b.const_load(val1, f"hash_c1_{val1:x}")
            c3 = b.const_load(val3, f"hash_c3_{val3}")
            hash_consts.append((c1, c3))

        # First pause (sync with reference_kernel2 first yield)
        b.pause()

        # Outer loop: rounds
        def round_body(round_i, round_params):
            # Inner loop: batch elements
            def batch_body(batch_i, batch_params):
                # idx = mem[inp_indices_p + i]
                idx_addr = b.add(inp_indices_p, batch_i, "idx_addr")
                idx = b.load(idx_addr, "idx")

                # val = mem[inp_values_p + i]
                val_addr = b.add(inp_values_p, batch_i, "val_addr")
                val = b.load(val_addr, "val")

                # node_val = mem[forest_values_p + idx]
                node_addr = b.add(forest_values_p, idx, "node_addr")
                node_val = b.load(node_addr, "node_val")

                # val = val ^ node_val
                val = b.xor(val, node_val, "xored")

                # Hash computation (6 stages)
                for i, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                    c1, c3 = hash_consts[i]
                    t1 = b.alu(op1, val, c1, f"h{i}_t1")
                    t2 = b.alu(op3, val, c3, f"h{i}_t2")
                    val = b.alu(op2, t1, t2, f"h{i}_val")

                # idx = 2*idx + (1 if val%2==0 else 2)
                mod_val = b.mod(val, two, "mod_val")
                is_even = b.eq(mod_val, zero, "is_even")
                offset = b.select(is_even, one, two, "offset")
                idx_doubled = b.mul(idx, two, "idx_doubled")
                next_idx = b.add(idx_doubled, offset, "next_idx")

                # Wrap: idx = 0 if idx >= n_nodes else idx
                in_bounds = b.lt(next_idx, n_nodes_val, "in_bounds")
                final_idx = b.select(in_bounds, next_idx, zero, "final_idx")

                # Store back
                idx_store_addr = b.add(inp_indices_p, batch_i, "idx_store_addr")
                b.store(idx_store_addr, final_idx)

                val_store_addr = b.add(inp_values_p, batch_i, "val_store_addr")
                b.store(val_store_addr, val)

                return []  # No loop-carried values

            # Batch loop (use Const bounds for unrolling)
            b.for_loop(
                start=zero_const,
                end=batch_const,
                iter_args=[],
                body_fn=batch_body
            )
            return []  # No loop-carried values

        # Round loop (use Const bounds for unrolling)
        b.for_loop(
            start=zero_const,
            end=rounds_const,
            iter_args=[],
            body_fn=round_body
        )

        # Final pause (sync with reference_kernel2 second yield)
        b.pause()

        # Compile HIR -> LIR -> VLIW
        hir = b.build()
        self.instrs = compile_hir_to_vliw(hir, print_after_all=print_after_all,
                                          config_path=self._pass_config)

        # Note: scratch_debug won't be populated with IR compiler
        # For now, leave it empty (debug info not critical for correctness)


BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
    print_vliw: bool = False,
    print_after_all: bool = False,
    use_ir: bool = True,
    pass_config: str = None,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds,
                    use_ir=use_ir, print_after_all=print_after_all, pass_config=pass_config)
    if print_vliw:
        for i, instr in enumerate(kb.instrs):
            print(f"[{i:4d}] {json.dumps(instr)}")

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

# Command line flags:
#    --print-vliw        Print the final VLIW instructions
#    --print-after-all   Print IR after each compilation pass
#    --no-ir             Use the original unrolled kernel instead of IR

if __name__ == "__main__":
    import sys
    import argparse

    # Check if running with custom flags (not unittest flags)
    # Only route to argparse when a known custom flag is present
    custom_flags = {'--print-vliw', '--print-after-all', '--no-ir', '--trace',
                    '--forest-height', '--rounds', '--batch-size'}
    has_custom_flag = any(arg.split('=')[0] in custom_flags for arg in sys.argv[1:])

    if len(sys.argv) > 1 and sys.argv[1].startswith("Tests."):
        # Running a specific test
        unittest.main()
    elif has_custom_flag:
        # Custom flags - run do_kernel_test directly
        parser = argparse.ArgumentParser(description="Performance engineering take-home")
        parser.add_argument("--print-vliw", action="store_true",
                            help="Print the final VLIW instructions")
        parser.add_argument("--print-after-all", action="store_true",
                            help="Print IR after each compilation pass")
        parser.add_argument("--no-ir", action="store_true",
                            help="Use the original unrolled kernel instead of IR")
        parser.add_argument("--trace", action="store_true",
                            help="Enable execution trace")
        parser.add_argument("--pass-config", type=str, default=None,
                            help="Path to JSON config file for pass options")
        parser.add_argument("--forest-height", type=int, default=10,
                            help="Forest height (default: 10)")
        parser.add_argument("--rounds", type=int, default=16,
                            help="Number of rounds (default: 16)")
        parser.add_argument("--batch-size", type=int, default=256,
                            help="Batch size (default: 256)")
        args = parser.parse_args()

        do_kernel_test(
            args.forest_height,
            args.rounds,
            args.batch_size,
            trace=args.trace,
            print_vliw=args.print_vliw,
            print_after_all=args.print_after_all,
            use_ir=not args.no_ir,
            pass_config=args.pass_config,
        )
    else:
        # Run unittest by default
        unittest.main()
