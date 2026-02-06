"""
Tree Hash Kernel

Implements the tree traversal + hash computation kernel using the IR compiler.
"""

from problem import HASH_STAGES

from compiler import HIRBuilder, Const, compile_hir_to_vliw


def build_tree_hash_kernel(
    forest_height: int,
    n_nodes: int,
    batch_size: int,
    rounds: int,
    print_after_all: bool = False,
    print_metrics: bool = False,
    print_ddg_after_all: bool = False,
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
        print_ddg_after_all: If True, print DDGs after each compilation pass

    Returns:
        Tuple of (instructions, debug_info) where:
        - instructions: List of VLIW instruction bundles
        - debug_info: Dictionary with debug information (currently empty for IR kernel)
    """
    b = HIRBuilder()

    # Load header values from memory (addresses 0-6)
    def load_header(idx: int, name: str):
        addr = b.const(idx)
        return b.load(addr, name)

    rounds_val = load_header(0, "rounds")
    _n_nodes_loaded = load_header(1, "n_nodes")
    batch_size_val = load_header(2, "batch_size")
    forest_height_val = load_header(3, "forest_height")
    forest_values_p = load_header(4, "forest_values_p")
    inp_indices_p = load_header(5, "inp_indices_p")
    inp_values_p = load_header(6, "inp_values_p")

    # n_nodes is a compile-time kernel parameter. Materializing it as Const
    # enables downstream passes (e.g. periodic tree-level-cache round analysis).
    n_nodes_val = Const(n_nodes)

    # Constants (as SSAValues for use in computations)
    zero = b.const(0)
    one = b.const(1)
    two = b.const(2)

    # Compile-time constants for loop bounds (as Const for unrolling)
    rounds_const = Const(rounds)
    batch_const = Const(batch_size)
    zero_const = Const(0)

    # Hash stage constants
    hash_consts = []
    for op1, val1, op2, op3, val3 in HASH_STAGES:
        c1 = b.const(val1)
        c3 = b.const(val3)
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

        # Batch loop
        b.for_loop(
            start=zero_const,
            end=batch_const,
            iter_args=[],
            body_fn=batch_body,
            pragma_unroll=0
        )
        return []  # No loop-carried values

    # Round loop
    b.for_loop(
        start=zero_const,
        end=rounds_const,
        iter_args=[],
        body_fn=round_body,
        pragma_unroll=0
    )

    # Final pause (sync with reference_kernel2 second yield)
    b.pause()

    # Compile HIR -> LIR -> VLIW
    hir = b.build()
    instrs = compile_hir_to_vliw(
        hir,
        print_after_all=print_after_all,
        print_metrics=print_metrics,
        print_ddg_after_all=print_ddg_after_all
    )

    # Debug info is empty for IR-compiled kernel
    debug_info = {}

    return instrs, debug_info
