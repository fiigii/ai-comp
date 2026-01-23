"""
Tree Hash Kernel

Implements the tree traversal + hash computation kernel using the IR compiler.
"""

from typing import Optional

from problem import HASH_STAGES

from compiler import HIRBuilder, Const, compile_hir_to_vliw


def build_tree_hash_kernel(
    forest_height: int,
    n_nodes: int,
    batch_size: int,
    rounds: int,
    print_after_all: bool = False,
    print_metrics: bool = False,
    pass_config: Optional[str] = None
) -> tuple[list[dict], dict]:
    """
    Build IR-based tree hash kernel.

    This kernel performs:
    1. Load batch of indices and values from memory
    2. For each round and batch element:
       - Look up node value at current index
       - Compute val = myhash(val ^ node_val) (6-stage hash)
       - Compute next index: 2*idx + (1 if val%2==0 else 2)
       - Wrap index if out of bounds
       - Store updated values back

    Args:
        forest_height: Height of the forest tree
        n_nodes: Number of nodes in the forest
        batch_size: Number of elements in a batch
        rounds: Number of rounds to process
        print_after_all: If True, print IR after each compilation pass
        print_metrics: If True, print pass metrics and diagnostics
        pass_config: Optional path to JSON config file for pass options

    Returns:
        Tuple of (instructions, debug_info) where:
        - instructions: List of VLIW instruction bundles
        - debug_info: Dictionary with debug information (currently empty for IR kernel)
    """
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

        # Batch loop: pragma_unroll=4 for partial unroll by factor 4
        b.for_loop(
            start=zero_const,
            end=batch_const,
            iter_args=[],
            body_fn=batch_body,
            pragma_unroll=16
        )
        return []  # No loop-carried values

    # Round loop: pragma_unroll=1 to disable unrolling on outer loop
    b.for_loop(
        start=zero_const,
        end=rounds_const,
        iter_args=[],
        body_fn=round_body,
        pragma_unroll=1
    )

    # Final pause (sync with reference_kernel2 second yield)
    b.pause()

    # Compile HIR -> LIR -> VLIW
    hir = b.build()
    instrs = compile_hir_to_vliw(
        hir,
        print_after_all=print_after_all,
        config_path=pass_config,
        print_metrics=print_metrics
    )

    # Debug info is empty for IR-compiled kernel
    debug_info = {}

    return instrs, debug_info
