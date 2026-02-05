"""
Tree Hash Kernel

Implements the tree traversal + hash computation kernel using the IR compiler.
"""

from problem import HASH_STAGES, VLEN

from compiler import HIRBuilder, Const, compile_hir_to_vliw


def build_tree_hash_kernel(
    forest_height: int,
    n_nodes: int,
    batch_size: int,
    rounds: int,
    print_after_all: bool = False,
    print_metrics: bool = False,
    print_ddg_after_all: bool = False,
    manual_opt_load: bool = False,
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
        manual_opt_load: If True, use manually optimized load/store version

    Returns:
        Tuple of (instructions, debug_info) where:
        - instructions: List of VLIW instruction bundles
        - debug_info: Dictionary with debug information (currently empty for IR kernel)
    """
    if manual_opt_load:
        return build_tree_hash_kernel_manual_opt_load(
            forest_height, n_nodes, batch_size, rounds,
            print_after_all, print_metrics, print_ddg_after_all
        )

    b = HIRBuilder()

    # Load header values from memory (addresses 0-6)
    def load_header(idx: int, name: str):
        addr = b.const(idx)
        return b.load(addr, name)

    rounds_val = load_header(0, "rounds")
    n_nodes_val = load_header(1, "n_nodes")
    batch_size_val = load_header(2, "batch_size")
    forest_height_val = load_header(3, "forest_height")
    forest_values_p = load_header(4, "forest_values_p")
    inp_indices_p = load_header(5, "inp_indices_p")
    inp_values_p = load_header(6, "inp_values_p")

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


def build_tree_hash_kernel_manual_opt_load(
    forest_height: int,
    n_nodes: int,
    batch_size: int,
    rounds: int,
    print_after_all: bool = False,
    print_metrics: bool = False,
    print_ddg_after_all: bool = False,
) -> tuple[list[dict], dict]:
    """
    Build IR-based tree hash kernel with manual load/store optimization.

    This optimized version:
    1. Loads all batch idx/val vectors from memory ONCE at start
    2. For each round:
       - For each vector batch:
         - Look up node values (vgather)
         - Compute val = myhash(val ^ node_val) (6-stage hash)
         - Compute next index: 2*idx + (1 if val%2==0 else 2)
         - Wrap index if out of bounds
       - Keep all values as SSA values (no memory ops)
    3. Stores all results ONCE at end

    This reduces memory operations from 2048 to 128 (16x improvement).
    Cycle count: ~2607 -> ~2135 (18% improvement)

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
    n_nodes_val = load_header(1, "n_nodes")
    batch_size_val = load_header(2, "batch_size")
    forest_height_val = load_header(3, "forest_height")
    forest_values_p = load_header(4, "forest_values_p")
    inp_indices_p = load_header(5, "inp_indices_p")
    inp_values_p = load_header(6, "inp_values_p")

    # Constants (as SSAValues for use in computations)
    zero = b.const(0)
    one = b.const(1)
    two = b.const(2)

    # Hash stage constants
    hash_consts = []
    for op1, val1, op2, op3, val3 in HASH_STAGES:
        c1 = b.const(val1)
        c3 = b.const(val3)
        hash_consts.append((c1, c3))

    # First pause (sync with reference_kernel2 first yield)
    b.pause()

    # Calculate number of vectors needed
    num_vectors = batch_size // VLEN  # 256 // 8 = 32

    # ===== PHASE 1: Load all vectors at start (once) =====
    idx_vecs = []
    val_vecs = []
    for vec_i in range(num_vectors):
        addr_offset = b.const(vec_i * VLEN)
        idx_addr = b.add(inp_indices_p, addr_offset, f"idx_addr_{vec_i}")
        val_addr = b.add(inp_values_p, addr_offset, f"val_addr_{vec_i}")
        idx_vecs.append(b.vload(idx_addr, f"idx_vec_{vec_i}"))
        val_vecs.append(b.vload(val_addr, f"val_vec_{vec_i}"))

    # Broadcast scalar constants to vectors
    zero_vec = b.vbroadcast(zero, "zero_vec")
    one_vec = b.vbroadcast(one, "one_vec")
    two_vec = b.vbroadcast(two, "two_vec")
    n_nodes_vec = b.vbroadcast(n_nodes_val, "n_nodes_vec")
    forest_values_p_vec = b.vbroadcast(forest_values_p, "forest_values_p_vec")

    # Map operators to vector operations
    vop_map = {
        "+": b.vadd,
        "^": b.vxor,
        "<<": b.vshl,
        ">>": b.vshr,
    }

    # ===== PHASE 2: Process all rounds (Python-level unroll) =====
    for round_i in range(rounds):
        new_idx_vecs = []
        new_val_vecs = []

        for vec_i in range(num_vectors):
            idx_vec = idx_vecs[vec_i]
            val_vec = val_vecs[vec_i]

            # Tree lookup: vgather(forest_values_p, idx_vec)
            # Compute addresses: forest_values_p + idx_vec
            gather_addr = b.vadd(forest_values_p_vec, idx_vec, f"gather_addr_r{round_i}_v{vec_i}")
            node_val_vec = b.vgather(gather_addr, f"node_r{round_i}_v{vec_i}")

            # val = val ^ node_val
            val_vec = b.vxor(val_vec, node_val_vec, f"xored_r{round_i}_v{vec_i}")

            # Hash computation (6 stages, all vector ops)
            for stage_i, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                c1, c3 = hash_consts[stage_i]
                c1_vec = b.vbroadcast(c1, f"c1_r{round_i}_v{vec_i}_s{stage_i}")
                c3_vec = b.vbroadcast(c3, f"c3_r{round_i}_v{vec_i}_s{stage_i}")
                t1 = vop_map[op1](val_vec, c1_vec, f"h{stage_i}_t1_r{round_i}_v{vec_i}")
                t2 = vop_map[op3](val_vec, c3_vec, f"h{stage_i}_t2_r{round_i}_v{vec_i}")
                val_vec = vop_map[op2](t1, t2, f"h{stage_i}_val_r{round_i}_v{vec_i}")

            # idx = 2*idx + (1 if val%2==0 else 2)
            mod_vec = b.vmod(val_vec, two_vec, f"mod_r{round_i}_v{vec_i}")
            is_even_vec = b.veq(mod_vec, zero_vec, f"is_even_r{round_i}_v{vec_i}")
            offset_vec = b.vselect(is_even_vec, one_vec, two_vec, f"offset_r{round_i}_v{vec_i}")
            idx_doubled_vec = b.vmul(idx_vec, two_vec, f"idx_doubled_r{round_i}_v{vec_i}")
            next_idx_vec = b.vadd(idx_doubled_vec, offset_vec, f"next_idx_r{round_i}_v{vec_i}")

            # Wrap: idx = 0 if idx >= n_nodes else idx
            in_bounds_vec = b.vlt(next_idx_vec, n_nodes_vec, f"in_bounds_r{round_i}_v{vec_i}")
            final_idx_vec = b.vselect(in_bounds_vec, next_idx_vec, zero_vec, f"final_idx_r{round_i}_v{vec_i}")

            new_idx_vecs.append(final_idx_vec)
            new_val_vecs.append(val_vec)

        idx_vecs = new_idx_vecs
        val_vecs = new_val_vecs

    # ===== PHASE 3: Store all results at end (once) =====
    for vec_i in range(num_vectors):
        addr_offset = b.const(vec_i * VLEN)
        idx_addr = b.add(inp_indices_p, addr_offset, f"idx_store_addr_{vec_i}")
        val_addr = b.add(inp_values_p, addr_offset, f"val_store_addr_{vec_i}")
        b.vstore(idx_addr, idx_vecs[vec_i])
        b.vstore(val_addr, val_vecs[vec_i])

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
