"""
Instruction Scheduling Pass (LIR -> MIR)

Schedules LIR instructions into VLIW bundles using a delay-aware list
scheduler and constructs MIR bundles per basic block.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

from ..pass_manager import LIRToMIRLoweringPass, PassConfig
from ..lir import LIRFunction, LIROpcode, LIRInst
from ..mir import MachineInst, MBundle, MachineBasicBlock, MachineFunction
from .mir_lowering_utils import lir_inst_to_machine_inst, get_successors, get_block_order
from vm import SCRATCH_SIZE, VLEN


@dataclass
class ScheduleNode:
    """Node in the scheduling dependency graph."""
    inst: MachineInst
    index: int
    succs: dict[int, int] = field(default_factory=dict)  # succ index -> delay
    preds: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class AddrExpr:
    """Simple address expression: base pointer + optional constant offset."""
    base: int
    offset: Optional[int]


@dataclass(frozen=True)
class MemoryKey:
    """Tracked address expression plus its scalar-word access width."""

    address: AddrExpr
    width: int


_ADDRESS_SPACE = 1 << 32
_ADDRESS_MASK = _ADDRESS_SPACE - 1


ENGINE_PRIORITY = {
    "flow": 4,
    "load": 3,
    "store": 3,
    "valu": 2,
    "alu": 1,
}

# A node whose dependence-graph degree exceeds what even the widest engine can
# consume in one bundle behaves as shared scheduling infrastructure rather
# than one stream's private work.  Derive the cutoff from the target instead
# of a workload-tuned number.
DEFAULT_SHARED_DEGREE_THRESHOLD = max(MBundle.SLOT_LIMITS.values())
DEFAULT_AUTO_STAGGER_PRESSURE_HEADROOM = VLEN * (
    MBundle.SLOT_LIMITS["valu"] + MBundle.SLOT_LIMITS["load"]
)
DEFAULT_AUTO_STAGGER_MIN_GAP_PCT = 25
DEFAULT_AUTO_STAGGER_CANDIDATE_START = 1
DEFAULT_AUTO_STAGGER_CANDIDATE_MULTIPLIER = 2
AUTO_STAGGER_DIRECTIONS = {
    "auto", "bidirectional", "both", "unidirectional"
}


@dataclass(frozen=True)
class AutoStaggerOptions:
    min_gap_pct: int
    pressure_headroom: int
    candidate_start: int
    candidate_multiplier: int
    candidate_max: Optional[int]
    direction: str


def _parse_auto_stagger_options(options: dict[str, object]) -> AutoStaggerOptions:
    """Parse and validate the auto-stagger heuristic controls."""

    def int_option(name: str, default: int, minimum: int = 0) -> int:
        value = options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        return value

    min_gap_pct = int_option(
        "stream_stagger_auto_min_gap_pct",
        DEFAULT_AUTO_STAGGER_MIN_GAP_PCT,
    )
    pressure_headroom = int_option(
        "stream_stagger_auto_pressure_headroom",
        DEFAULT_AUTO_STAGGER_PRESSURE_HEADROOM,
    )
    if pressure_headroom >= SCRATCH_SIZE:
        raise ValueError(
            "stream_stagger_auto_pressure_headroom must be smaller than "
            f"SCRATCH_SIZE ({SCRATCH_SIZE}), got {pressure_headroom}"
        )

    candidate_start = int_option(
        "stream_stagger_auto_candidate_start",
        DEFAULT_AUTO_STAGGER_CANDIDATE_START,
        minimum=1,
    )
    candidate_multiplier = int_option(
        "stream_stagger_auto_candidate_multiplier",
        DEFAULT_AUTO_STAGGER_CANDIDATE_MULTIPLIER,
        minimum=2,
    )
    candidate_max_value = options.get("stream_stagger_auto_candidate_max")
    if candidate_max_value is None:
        candidate_max = None
    elif (isinstance(candidate_max_value, bool)
          or not isinstance(candidate_max_value, int)
          or candidate_max_value < candidate_start):
        raise ValueError(
            "stream_stagger_auto_candidate_max must be null or an integer "
            f">= candidate_start ({candidate_start}), got {candidate_max_value!r}"
        )
    else:
        candidate_max = candidate_max_value

    direction = options.get("stream_stagger_auto_direction", "auto")
    if (not isinstance(direction, str)
            or direction not in AUTO_STAGGER_DIRECTIONS):
        choices = ", ".join(sorted(AUTO_STAGGER_DIRECTIONS))
        raise ValueError(
            f"stream_stagger_auto_direction must be one of {choices}, "
            f"got {direction!r}"
        )

    return AutoStaggerOptions(
        min_gap_pct=min_gap_pct,
        pressure_headroom=pressure_headroom,
        candidate_start=candidate_start,
        candidate_multiplier=candidate_multiplier,
        candidate_max=candidate_max,
        direction=direction,
    )

VECTOR_TO_SCALAR_OPCODE = {
    LIROpcode.VADD: LIROpcode.ADD,
    LIROpcode.VSUB: LIROpcode.SUB,
    LIROpcode.VMUL: LIROpcode.MUL,
    LIROpcode.VDIV: LIROpcode.DIV,
    LIROpcode.VMOD: LIROpcode.MOD,
    LIROpcode.VXOR: LIROpcode.XOR,
    LIROpcode.VAND: LIROpcode.AND,
    LIROpcode.VOR: LIROpcode.OR,
    LIROpcode.VSHL: LIROpcode.SHL,
    LIROpcode.VSHR: LIROpcode.SHR,
    LIROpcode.VLT: LIROpcode.LT,
    LIROpcode.VEQ: LIROpcode.EQ,
}


def _engine_priority(engine: str) -> int:
    return ENGINE_PRIORITY.get(engine, 0)


def _is_memory_load(inst: MachineInst) -> bool:
    """Check if instruction reads from main memory."""
    return inst.opcode in (LIROpcode.LOAD, LIROpcode.VLOAD, LIROpcode.LOAD_OFFSET)


def _is_memory_store(inst: MachineInst) -> bool:
    """Check if instruction writes to main memory."""
    return inst.opcode in (LIROpcode.STORE, LIROpcode.VSTORE)


def _is_barrier(inst: MachineInst) -> bool:
    """Check if instruction is a barrier (must not reorder across)."""
    return inst.opcode in (LIROpcode.PAUSE, LIROpcode.HALT)


def _add_edge(nodes: list[ScheduleNode], pred: int, succ: int, delay: int) -> None:
    """Add a dependency edge pred -> succ with the given delay."""
    if pred == succ:
        return
    existing = nodes[pred].succs.get(succ)
    if existing is None or delay > existing:
        nodes[pred].succs[succ] = delay
    nodes[succ].preds.add(pred)


def _slot_root(base: object) -> Optional[int]:
    """The memory-slot provenance root of a base, if it has one."""
    if isinstance(base, tuple) and len(base) == 3 and base[0] == "slot":
        return base[1]
    return None


def _keys_alias(
    a: Optional[MemoryKey],
    b: Optional[MemoryKey],
    restrict_ptr: bool = False,
) -> bool:
    """Check if two memory keys may alias."""
    if a is None or b is None:
        return True
    if a.address.base != b.address.base:
        # Two loads of the SAME slot are different pointer identities that
        # share a provenance root: the slot may have been overwritten in
        # between, so constant offsets prove nothing and restrict_ptr does
        # not apply (it separates ROOTS, not identities).
        slot_a = _slot_root(a.address.base)
        slot_b = _slot_root(b.address.base)
        if slot_a is not None and slot_a == slot_b:
            return True
        return not restrict_ptr
    if a.address.offset is None or b.address.offset is None:
        return True

    def intervals(start: int, width: int) -> tuple[tuple[int, int], ...]:
        start &= _ADDRESS_MASK
        if width >= _ADDRESS_SPACE:
            return ((0, _ADDRESS_SPACE),)
        end = start + width
        if end <= _ADDRESS_SPACE:
            return ((start, end),)
        return ((start, _ADDRESS_SPACE), (0, end - _ADDRESS_SPACE))

    for a_lo, a_hi in intervals(a.address.offset, a.width):
        for b_lo, b_hi in intervals(b.address.offset, b.width):
            if max(a_lo, b_lo) < min(a_hi, b_hi):
                return True
    return False


def _clear_value_info(dest: int, const_val: dict[int, int], addr_expr: dict[int, AddrExpr]) -> None:
    const_val.pop(dest, None)
    addr_expr.pop(dest, None)


def _set_const(dest: int, value: int, const_val: dict[int, int], addr_expr: dict[int, AddrExpr]) -> None:
    const_val[dest] = value & _ADDRESS_MASK
    addr_expr.pop(dest, None)


def _set_addr(dest: int, base: int, offset: Optional[int],
              const_val: dict[int, int], addr_expr: dict[int, AddrExpr]) -> None:
    addr_expr[dest] = AddrExpr(
        base=base,
        offset=None if offset is None else offset & _ADDRESS_MASK,
    )
    const_val.pop(dest, None)


def _try_compute_binop(dest: int, op: LIROpcode, a: int, b: int,
                       const_val: dict[int, int], addr_expr: dict[int, AddrExpr]) -> None:
    const_a = const_val.get(a)
    const_b = const_val.get(b)
    addr_a = addr_expr.get(a)
    addr_b = addr_expr.get(b)

    if const_a is not None and const_b is not None:
        if op == LIROpcode.ADD:
            _set_const(dest, const_a + const_b, const_val, addr_expr)
            return
        if op == LIROpcode.SUB:
            _set_const(dest, const_a - const_b, const_val, addr_expr)
            return

    if addr_a is not None and const_b is not None:
        if op == LIROpcode.ADD:
            _set_addr(dest, addr_a.base,
                      None if addr_a.offset is None else addr_a.offset + const_b,
                      const_val, addr_expr)
            return
        if op == LIROpcode.SUB:
            _set_addr(dest, addr_a.base,
                      None if addr_a.offset is None else addr_a.offset - const_b,
                      const_val, addr_expr)
            return

    if addr_a is not None and const_b is None and addr_b is None:
        if op in (LIROpcode.ADD, LIROpcode.SUB):
            _set_addr(dest, addr_a.base, None, const_val, addr_expr)
            return

    if const_a is not None and addr_b is not None:
        if op == LIROpcode.ADD:
            _set_addr(dest, addr_b.base,
                      None if addr_b.offset is None else addr_b.offset + const_a,
                      const_val, addr_expr)
            return
        # const - addr => unknown

    if addr_b is not None and const_a is None and addr_a is None:
        if op == LIROpcode.ADD:
            _set_addr(dest, addr_b.base, None, const_val, addr_expr)
            return

    _clear_value_info(dest, const_val, addr_expr)


def _update_value_info(inst: MachineInst,
                       const_val: dict[int, int],
                       addr_expr: dict[int, AddrExpr]) -> None:
    """Update constant/address information for instruction defs."""
    dests = sorted(inst.get_defs())

    if not dests:
        return

    # CONST
    if inst.opcode == LIROpcode.CONST and isinstance(inst.dest, int):
        _set_const(inst.dest, int(inst.operands[0]), const_val, addr_expr)
        return

    # ADD_IMM: dest = src + imm (flow-engine constant materialization)
    if inst.opcode == LIROpcode.ADD_IMM and isinstance(inst.dest, int):
        src, imm = inst.operands[0], int(inst.operands[1])
        if isinstance(src, int) and src in const_val:
            _set_const(inst.dest, (const_val[src] + imm) & 0xFFFFFFFF,
                       const_val, addr_expr)
            return
        if isinstance(src, int) and src in addr_expr:
            src_expr = addr_expr[src]
            _set_addr(inst.dest, src_expr.base,
                      None if src_expr.offset is None else src_expr.offset + imm,
                      const_val, addr_expr)
            return
        _clear_value_info(inst.dest, const_val, addr_expr)
        return

    # COPY propagation (if any survive)
    if inst.opcode == LIROpcode.COPY and isinstance(inst.dest, int):
        src = inst.operands[0]
        if isinstance(src, int):
            if src in const_val:
                _set_const(inst.dest, const_val[src], const_val, addr_expr)
                return
            if src in addr_expr:
                src_expr = addr_expr[src]
                _set_addr(inst.dest, src_expr.base, src_expr.offset, const_val, addr_expr)
                return
        _clear_value_info(inst.dest, const_val, addr_expr)
        return

    # Scalar add/sub
    if inst.opcode in (LIROpcode.ADD, LIROpcode.SUB) and isinstance(inst.dest, int):
        if len(inst.operands) >= 2:
            a, b = inst.operands[0], inst.operands[1]
            if isinstance(a, int) and isinstance(b, int):
                _try_compute_binop(inst.dest, inst.opcode, a, b, const_val, addr_expr)
                return

    # Vector add/sub (lane-wise)
    if inst.opcode in (LIROpcode.VADD, LIROpcode.VSUB) and isinstance(inst.dest, list):
        if len(inst.operands) >= 2 and isinstance(inst.operands[0], list) and isinstance(inst.operands[1], list):
            for lane, d in enumerate(inst.dest):
                a = inst.operands[0][lane]
                b = inst.operands[1][lane]
                if isinstance(a, int) and isinstance(b, int):
                    _try_compute_binop(d, LIROpcode.ADD if inst.opcode == LIROpcode.VADD else LIROpcode.SUB,
                                       a, b, const_val, addr_expr)
                else:
                    _clear_value_info(d, const_val, addr_expr)
            return

    # VBROADCAST: replicate scalar const/addr to lanes
    if inst.opcode == LIROpcode.VBROADCAST and isinstance(inst.dest, list):
        src = inst.operands[0]
        if isinstance(src, int):
            if src in const_val:
                for d in inst.dest:
                    _set_const(d, const_val[src], const_val, addr_expr)
                return
            if src in addr_expr:
                src_expr = addr_expr[src]
                for d in inst.dest:
                    _set_addr(d, src_expr.base, src_expr.offset, const_val, addr_expr)
                return
        for d in inst.dest:
            _clear_value_info(d, const_val, addr_expr)
        return

    # A load from a constant memory slot establishes a base pointer symbol.
    # The base carries the LOAD IDENTITY (its dest scratch), not just the
    # slot: two loads of the same slot may see different values when the
    # slot is overwritten in between, so only offsets derived from the SAME
    # load may be compared exactly. The slot is kept as a provenance root:
    # different identities sharing a slot stay MAY_ALIAS even under
    # restrict_ptr (see _keys_alias).
    if inst.opcode == LIROpcode.LOAD and isinstance(inst.dest, int):
        addr = inst.operands[0]
        if isinstance(addr, int):
            const_addr = const_val.get(addr)
            if const_addr is not None:
                _set_addr(inst.dest, ("slot", const_addr, inst.dest),
                          0, const_val, addr_expr)
                return
        _clear_value_info(inst.dest, const_val, addr_expr)
        return

    # For all other defs, clear any tracked info.
    for d in dests:
        _clear_value_info(d, const_val, addr_expr)


def _compute_exit_value_state(
    instructions: list[MachineInst],
    initial_const_val: dict[int, int] | None = None,
    initial_addr_expr: dict[int, AddrExpr] | None = None,
) -> tuple[dict[int, int], dict[int, AddrExpr]]:
    """Simulate _update_value_info for all instructions, returning final state.

    Used during inter-block propagation to compute exit state for each block
    without building the full dependency graph.
    """
    const_val: dict[int, int] = dict(initial_const_val) if initial_const_val else {}
    addr_expr: dict[int, AddrExpr] = dict(initial_addr_expr) if initial_addr_expr else {}
    for inst in instructions:
        _update_value_info(inst, const_val, addr_expr)
    return const_val, addr_expr


def _memory_key(inst: MachineInst,
                const_val: dict[int, int],
                addr_expr: dict[int, AddrExpr]) -> Optional[MemoryKey]:
    """Compute a conservative alias key for a memory instruction."""
    if inst.opcode == LIROpcode.LOAD_OFFSET:
        uses = inst.get_uses()
        if len(uses) == 1:
            lane_addr = next(iter(uses))
            address = addr_expr.get(lane_addr)
            return None if address is None else MemoryKey(address, 1)
        return None

    addr_op = None
    if inst.opcode in (LIROpcode.LOAD, LIROpcode.VLOAD):
        addr_op = inst.operands[0]
    elif inst.opcode in (LIROpcode.STORE, LIROpcode.VSTORE):
        addr_op = inst.operands[0]

    if isinstance(addr_op, int):
        address = addr_expr.get(addr_op)
        if address is None:
            return None
        width = VLEN if inst.opcode in (LIROpcode.VLOAD, LIROpcode.VSTORE) else 1
        return MemoryKey(address, width)

    return None


def _build_dep_graph(
    instructions: list[MachineInst],
    initial_const_val: dict[int, int] | None = None,
    initial_addr_expr: dict[int, AddrExpr] | None = None,
    restrict_ptr: bool = False,
) -> list[ScheduleNode]:
    """Build a dependency graph with delay-annotated edges."""
    nodes = [ScheduleNode(inst=inst, index=i) for i, inst in enumerate(instructions)]

    last_def: dict[int, int] = {}
    last_barrier: Optional[int] = None

    # Conservative constant/address tracking for simple alias analysis
    const_val: dict[int, int] = dict(initial_const_val) if initial_const_val else {}
    addr_expr: dict[int, AddrExpr] = dict(initial_addr_expr) if initial_addr_expr else {}

    # Track memory ops by alias key
    last_store_by_key: dict[Optional[MemoryKey], int] = {}
    loads_since_store_by_key: dict[Optional[MemoryKey], list[int]] = {}

    for i, inst in enumerate(instructions):
        uses = inst.get_uses()
        defs = inst.get_defs()

        # RAW dependencies (no same-bundle forwarding)
        for use in uses:
            if use in last_def:
                _add_edge(nodes, last_def[use], i, 1)

        # Memory ordering with simple alias disambiguation
        if _is_memory_load(inst) or _is_memory_store(inst):
            key = _memory_key(inst, const_val, addr_expr)

            if _is_memory_load(inst):
                for k, store_idx in list(last_store_by_key.items()):
                    if _keys_alias(key, k, restrict_ptr):
                        _add_edge(nodes, store_idx, i, 1)
                loads_since_store_by_key.setdefault(key, []).append(i)

            if _is_memory_store(inst):
                for k, store_idx in list(last_store_by_key.items()):
                    if _keys_alias(key, k, restrict_ptr):
                        _add_edge(nodes, store_idx, i, 0)
                for k, load_list in list(loads_since_store_by_key.items()):
                    if _keys_alias(key, k, restrict_ptr):
                        for load_idx in load_list:
                            _add_edge(nodes, load_idx, i, 0)
                last_store_by_key[key] = i
                if key is None:
                    # Unknown store aliases everything: reset all load tracking
                    for k in list(loads_since_store_by_key.keys()):
                        loads_since_store_by_key[k] = []
                else:
                    loads_since_store_by_key[key] = []

        # Barriers: must remain after all previous and before all following
        if _is_barrier(inst):
            for pred_idx in range(i):
                _add_edge(nodes, pred_idx, i, 1)
            last_barrier = i
        elif last_barrier is not None:
            _add_edge(nodes, last_barrier, i, 1)

        # Update last_def map
        for d in defs:
            last_def[d] = i

        # Update constant/address tracking
        _update_value_info(inst, const_val, addr_expr)

    return nodes


def _compute_critical_path_heights(nodes: list[ScheduleNode]) -> list[int]:
    """Compute critical path height for each node (longest delayed path to a sink)."""
    heights = [0] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):
        if nodes[i].succs:
            heights[i] = max(
                delay + heights[succ] for succ, delay in nodes[i].succs.items()
            )
    return heights


def _compute_stream_stagger(
    nodes: list[ScheduleNode],
    stagger: int,
    shared_degree_threshold: int = DEFAULT_SHARED_DEGREE_THRESHOLD,
    bidirectional: bool = False,
) -> list[int]:
    """Compute per-node height offsets that stagger independent streams.

    Independent dependence-graph components (e.g. per-vector computation
    chains in a fully unrolled data-parallel kernel) otherwise have identical
    critical-path heights, so the scheduler advances them in lockstep. That
    starves engines whose work is phase-local (e.g. the load engine during
    cached rounds). Offsetting each component's heights by a decreasing
    amount makes earlier components run ahead, overlapping load-heavy and
    compute-heavy phases.

    Widely-shared nodes (broadcast constants, preloaded values) are excluded
    from component merging via a degree threshold and get the maximum offset
    so they are available to every stream from the start.
    """
    n = len(nodes)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # High out-degree = shared producer (broadcasts); high in-degree =
    # barrier-like sink (e.g. final pause with edges from every node).
    shared = [
        len(node.succs) > shared_degree_threshold
        or len(node.preds) > shared_degree_threshold
        for node in nodes
    ]
    for i, node in enumerate(nodes):
        if shared[i]:
            continue
        for succ in node.succs:
            if not shared[succ]:
                union(i, succ)

    # Rank components by first (minimum) node index
    comp_first: dict[int, int] = {}
    for i in range(n):
        if shared[i]:
            continue
        root = find(i)
        if root not in comp_first:
            comp_first[root] = i
    ranked = sorted(comp_first, key=lambda r: comp_first[r])
    comp_rank = {root: rank for rank, root in enumerate(ranked)}
    n_comps = len(ranked)

    def rank_offset(rank: int) -> int:
        if bidirectional:
            # First and last components run ahead, middle components trail:
            # halves the ramp depth for the same total spread and shortens
            # the drain tail (the middle finishes last from both sides).
            d = min(rank, n_comps - 1 - rank)
            return (n_comps - 2 * d) * stagger
        return (n_comps - rank) * stagger

    offsets = [n_comps * stagger] * n  # shared nodes: max offset
    for i in range(n):
        if not shared[i]:
            offsets[i] = rank_offset(comp_rank[find(i)])
    return offsets


def _compute_load_unblock_scores(nodes: list[ScheduleNode]) -> list[int]:
    """Score nodes by how directly they unblock pending loads."""
    scores = [0] * len(nodes)
    for i, node in enumerate(nodes):
        score = 0
        for succ in node.succs:
            succ_node = nodes[succ]
            if succ_node.inst.engine == "load":
                score += 4
            # One-hop lookahead: prioritize producers of load producers.
            for succ2 in succ_node.succs:
                if nodes[succ2].inst.engine == "load":
                    score += 1
        scores[i] = score
    return scores


def _compute_distance_to_load(nodes: list[ScheduleNode]) -> list[int]:
    """Compute minimal dependency-delay distance from each node to any load node."""
    inf = 10**9
    dist = [inf] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):
        if nodes[i].inst.engine == "load":
            dist[i] = 0
            continue
        best = inf
        for succ, delay in nodes[i].succs.items():
            succ_dist = dist[succ]
            if succ_dist == inf:
                continue
            cand = delay + succ_dist
            if cand < best:
                best = cand
        dist[i] = best
    return dist


def _devectorize_valu_to_alu(inst: MachineInst) -> Optional[list[MachineInst]]:
    """Try to expand a vector-ALU op to scalar ALU lane ops."""
    if inst.opcode == LIROpcode.MULTIPLY_ADD:
        return None

    if inst.opcode == LIROpcode.VBROADCAST:
        if not isinstance(inst.dest, list) or len(inst.operands) < 1:
            return None
        src = inst.operands[0]
        if not isinstance(src, int):
            return None
        scalar_insts: list[MachineInst] = []
        for lane_dest in inst.dest:
            if not isinstance(lane_dest, int):
                return None
            # Copy via OR(x, x): avoids needing a dedicated zero scratch.
            scalar_insts.append(
                MachineInst(opcode=LIROpcode.OR, dest=lane_dest, operands=[src, src], engine="alu")
            )
        return scalar_insts

    scalar_opcode = VECTOR_TO_SCALAR_OPCODE.get(inst.opcode)
    if scalar_opcode is None:
        return None
    if not isinstance(inst.dest, list) or len(inst.operands) < 2:
        return None
    lhs = inst.operands[0]
    rhs = inst.operands[1]
    if not isinstance(lhs, list) or not isinstance(rhs, list):
        return None
    if len(inst.dest) != len(lhs) or len(inst.dest) != len(rhs):
        return None

    scalar_insts = []
    for lane_dest, lane_lhs, lane_rhs in zip(inst.dest, lhs, rhs):
        if not isinstance(lane_dest, int) or not isinstance(lane_lhs, int) or not isinstance(lane_rhs, int):
            return None
        scalar_insts.append(
            MachineInst(opcode=scalar_opcode, dest=lane_dest, operands=[lane_lhs, lane_rhs], engine="alu")
        )
    return scalar_insts


def _devectorize_valu_to_alu_with_knobs(
    inst: MachineInst,
    *,
    devectorize_vector_ops_to_alu: bool,
    devectorize_vbroadcast_to_alu: bool,
) -> Optional[list[MachineInst]]:
    if inst.opcode == LIROpcode.MULTIPLY_ADD:
        return None
    if inst.opcode == LIROpcode.VBROADCAST and not devectorize_vbroadcast_to_alu:
        return None
    if inst.opcode != LIROpcode.VBROADCAST and not devectorize_vector_ops_to_alu:
        return None
    return _devectorize_valu_to_alu(inst)



def _schedule_block(
    instructions: list[MachineInst],
    terminator: Optional[LIRInst],
    prefer_load_fill: bool = False,
    devectorize_valu_to_alu: bool = False,
    devectorize_vector_ops_to_alu: bool = True,
    devectorize_vbroadcast_to_alu: bool = True,
    devectorize_partial_alu_fill: bool = False,
    prioritize_load_unblock: bool = False,
    register_pressure_limit: int = 0,
    stream_stagger: int = 0,
    stream_stagger_bidirectional: bool = False,
    stream_stagger_threshold: int = DEFAULT_SHARED_DEGREE_THRESHOLD,
    initial_const_val: dict[int, int] | None = None,
    initial_addr_expr: dict[int, AddrExpr] | None = None,
    restrict_ptr: bool = False,
    _prebuilt_nodes: list[ScheduleNode] | None = None,
) -> tuple[list[MBundle], dict[str, dict[str, int]]]:
    """Schedule a block's instructions into MIR bundles.

    Returns:
        (bundles, stats) where stats contains per-engine utilization and
        bundle-end reason counts.
    """
    if not instructions and terminator is None:
        return [], {
            "engine_used_slots": {},
            "engine_idle_no_ready": {},
            "bundle_end_reasons": {"deps": 0, "slot_limit": 0},
            "engine_bundle_hist": {},
            "slot_limit_blocked_bundles_by_engine": {},
            "slot_limit_ready_left_by_engine": {},
            "slot_limit_bundles": 0,
            "devectorized_valu_ops": 0,
            "devectorized_alu_ops": 0,
        }

    slot_limits = MBundle.SLOT_LIMITS
    nodes = _prebuilt_nodes
    if nodes is None:
        nodes = _build_dep_graph(
            instructions,
            initial_const_val,
            initial_addr_expr,
            restrict_ptr=restrict_ptr,
        )
    heights = _compute_critical_path_heights(nodes)
    if stream_stagger > 0:
        offsets = _compute_stream_stagger(
            nodes, stream_stagger,
            shared_degree_threshold=stream_stagger_threshold,
            bidirectional=stream_stagger_bidirectional,
        )
        heights = [h + off for h, off in zip(heights, offsets)]
    load_unblock_scores = _compute_load_unblock_scores(nodes) if prioritize_load_unblock else [0] * len(nodes)
    load_distance = _compute_distance_to_load(nodes) if prioritize_load_unblock else [10**9] * len(nodes)
    n = len(nodes)
    remaining_preds = [len(node.preds) for node in nodes]
    earliest_bundle = [0] * n
    scheduled = [False] * n
    ready: set[int] = set()
    delayed_ready: list[tuple[int, int]] = []

    bundles: list[MBundle] = []
    current_bundle = 0
    scheduled_count = 0

    # Register pressure tracking
    pressure_aware = register_pressure_limit > 0
    remaining_uses: dict[int, int] = {}  # scratch addr -> remaining use count
    live_regs: set[int] = set()          # currently live scratch addresses
    live_reg_count = 0                   # weighted count (vectors count as VLEN)

    # Pre-compute use counts for pressure tracking
    if pressure_aware:
        from collections import Counter
        from vm import VLEN as _VLEN
        use_counter: Counter[int] = Counter()
        for node in nodes:
            for u in node.inst.get_uses():
                use_counter[u] += 1
        remaining_uses = dict(use_counter)

        # Detect vector bases (contiguous groups of VLEN addresses)
        all_defs: set[int] = set()
        _vec_bases: set[int] = set()
        for node in nodes:
            inst = node.inst
            if inst.dest is not None and isinstance(inst.dest, list) and inst.dest:
                base = inst.dest[0]
                if isinstance(base, int):
                    _vec_bases.add(base)
            all_defs.update(inst.get_defs())
        # For pressure estimation: vector defs count as VLEN regs
        _vec_addrs: set[int] = set()
        for base in _vec_bases:
            for i in range(_VLEN):
                _vec_addrs.add(base + i)

        def _reg_size(addr: int) -> int:
            return _VLEN if addr in _vec_bases else (0 if addr in _vec_addrs else 1)

        def _kill_score(node_idx: int) -> int:
            """How many registers this instruction frees (last use of values)."""
            freed = 0
            for u in nodes[node_idx].inst.get_uses():
                if u in remaining_uses and remaining_uses[u] == 1:
                    freed += _reg_size(u)
            return freed

        def _def_cost(node_idx: int) -> int:
            """How many new live registers this instruction creates."""
            cost = 0
            for d in nodes[node_idx].inst.get_defs():
                if d not in live_regs:
                    cost += _reg_size(d)
            return cost

    # Bundle-level utilization diagnostics
    bundle_end_reasons = {"deps": 0, "slot_limit": 0}
    engine_used_slots = {engine: 0 for engine in slot_limits}
    engine_idle_no_ready = {engine: 0 for engine in slot_limits}
    engine_bundle_hist: dict[str, dict[int, int]] = {
        engine: {i: 0 for i in range(limit + 1)} for engine, limit in slot_limits.items()
    }
    slot_limit_blocked_bundles_by_engine = {engine: 0 for engine in slot_limits}
    slot_limit_ready_left_by_engine = {engine: 0 for engine in slot_limits}
    slot_limit_bundles = 0
    devectorized_valu_ops = 0
    devectorized_alu_ops = 0
    # Node idx -> (expanded scalar insts, next scalar idx to emit)
    pending_devectorized: dict[int, tuple[list[MachineInst], int]] = {}
    # Preserve deterministic emission order for partially emitted expansions
    pending_devectorized_order: list[int] = []

    def enqueue_ready(node_idx: int) -> None:
        if earliest_bundle[node_idx] <= current_bundle:
            ready.add(node_idx)
        else:
            heapq.heappush(
                delayed_ready, (earliest_bundle[node_idx], node_idx)
            )

    def refresh_ready() -> None:
        while delayed_ready and delayed_ready[0][0] <= current_bundle:
            _, node_idx = heapq.heappop(delayed_ready)
            if not scheduled[node_idx] and remaining_preds[node_idx] == 0:
                ready.add(node_idx)

    for i in range(n):
        if remaining_preds[i] == 0:
            enqueue_ready(i)

    while scheduled_count < n:
        refresh_ready()
        if not ready:
            if delayed_ready:
                current_bundle = max(current_bundle, delayed_ready[0][0])
            refresh_ready()
            if not ready:
                raise RuntimeError("dependency graph contains an unschedulable cycle")

        bundle = MBundle()
        used_in_bundle_by_engine = {engine: 0 for engine in slot_limits}

        def pick_best(only_engine: Optional[str] = None) -> Optional[int]:
            best_idx = None
            best_key = None
            load_slot_open = bundle.has_slot_available("load")
            any_ready_load = False
            if prioritize_load_unblock and only_engine is None and load_slot_open:
                for ridx in ready:
                    rinst = nodes[ridx].inst
                    if rinst.engine == "load" and bundle.has_slot_available("load"):
                        any_ready_load = True
                        break

            # Under high pressure, prefer instructions that free registers.
            high_pressure = pressure_aware and live_reg_count > register_pressure_limit

            for idx in ready:
                if idx in pending_devectorized:
                    continue
                inst = nodes[idx].inst
                if only_engine is not None and inst.engine != only_engine:
                    continue
                if not bundle.has_slot_available(inst.engine):
                    continue

                if high_pressure:
                    # Under pressure: maximize freed regs, minimize new defs
                    kill = _kill_score(idx)
                    cost = _def_cost(idx)
                    pressure_key = kill - cost  # net register freedom
                    key = (
                        pressure_key,
                        heights[idx],
                        load_unblock_scores[idx],
                        _engine_priority(inst.engine),
                        -nodes[idx].index,
                    )
                elif (prioritize_load_unblock and only_engine is None
                      and load_slot_open and not any_ready_load):
                    dist = load_distance[idx]
                    key = (
                        -dist if dist < 10**9 else -10**9,
                        load_unblock_scores[idx],
                        heights[idx],
                        _engine_priority(inst.engine),
                        -nodes[idx].index,
                    )
                else:
                    key = (
                        heights[idx],
                        load_unblock_scores[idx],
                        _engine_priority(inst.engine),
                        -nodes[idx].index,
                    )
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
            return best_idx

        def available_slots(engine: str) -> int:
            return slot_limits[engine] - used_in_bundle_by_engine[engine]

        def mark_node_scheduled(node_idx: int) -> None:
            nonlocal scheduled_count, live_reg_count
            ready.remove(node_idx)
            scheduled[node_idx] = True
            scheduled_count += 1
            if pressure_aware:
                # Update live register tracking
                inst = nodes[node_idx].inst
                # Process uses: decrement remaining_uses, kill if last use
                for u in inst.get_uses():
                    if u in remaining_uses:
                        remaining_uses[u] -= 1
                        if remaining_uses[u] == 0:
                            if u in live_regs:
                                live_regs.discard(u)
                                live_reg_count -= _reg_size(u)
                # Process defs: add to live set
                for d in inst.get_defs():
                    if d not in live_regs:
                        live_regs.add(d)
                        live_reg_count += _reg_size(d)
            for succ, delay in nodes[node_idx].succs.items():
                remaining_preds[succ] -= 1
                target_bundle = current_bundle + delay
                if target_bundle > earliest_bundle[succ]:
                    earliest_bundle[succ] = target_bundle
                if remaining_preds[succ] == 0:
                    enqueue_ready(succ)

        def try_schedule_devectorized_valu() -> bool:
            nonlocal devectorized_valu_ops, devectorized_alu_ops
            if not devectorize_valu_to_alu:
                return False
            if available_slots("alu") <= 0:
                return False

            def emit_pending(idx: int) -> bool:
                nonlocal devectorized_alu_ops
                scalar_insts, next_pos = pending_devectorized[idx]
                emitted = 0
                while next_pos < len(scalar_insts) and available_slots("alu") > 0:
                    scalar_inst = scalar_insts[next_pos]
                    if not bundle.add_instruction(scalar_inst):
                        raise RuntimeError("unexpected ALU slot exhaustion during devectorization")
                    used_in_bundle_by_engine["alu"] += 1
                    next_pos += 1
                    emitted += 1
                pending_devectorized[idx] = (scalar_insts, next_pos)
                devectorized_alu_ops += emitted
                if next_pos >= len(scalar_insts):
                    pending_devectorized.pop(idx, None)
                    pending_devectorized_order.remove(idx)
                    mark_node_scheduled(idx)
                return emitted > 0

            # First, drain any previously-started expansion(s).
            if devectorize_partial_alu_fill and pending_devectorized_order:
                for idx in list(pending_devectorized_order):
                    if available_slots("alu") <= 0:
                        break
                    if emit_pending(idx):
                        return True

            # Only start a new expansion when the VALU engine is saturated.
            if available_slots("valu") > 0:
                return False

            best_idx = None
            best_key = None
            best_scalar_insts: Optional[list[MachineInst]] = None
            for idx in ready:
                if idx in pending_devectorized:
                    continue
                inst = nodes[idx].inst
                if inst.engine != "valu":
                    continue
                scalar_insts = _devectorize_valu_to_alu_with_knobs(
                    inst,
                    devectorize_vector_ops_to_alu=devectorize_vector_ops_to_alu,
                    devectorize_vbroadcast_to_alu=devectorize_vbroadcast_to_alu,
                )
                if scalar_insts is None:
                    continue
                if not devectorize_partial_alu_fill and len(scalar_insts) > available_slots("alu"):
                    continue
                key = (
                    load_unblock_scores[idx],
                    heights[idx],
                    -nodes[idx].index,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
                    best_scalar_insts = scalar_insts

            if best_idx is None or best_scalar_insts is None:
                return False

            pending_devectorized[best_idx] = (best_scalar_insts, 0)
            pending_devectorized_order.append(best_idx)
            devectorized_valu_ops += 1
            return emit_pending(best_idx)

        while True:
            best_idx = None
            if prefer_load_fill and bundle.has_slot_available("load"):
                best_idx = pick_best("load")
            if best_idx is None:
                best_idx = pick_best()

            if best_idx is None:
                if try_schedule_devectorized_valu():
                    continue
                break

            inst = nodes[best_idx].inst

            if not bundle.add_instruction(inst):
                ready.remove(best_idx)
                continue

            used_in_bundle_by_engine[inst.engine] += 1
            mark_node_scheduled(best_idx)

        if bundle.instructions:
            bundles.append(bundle)

            # Utilization accounting and idle-slot reasons
            ready_by_engine = {engine: 0 for engine in slot_limits}
            for idx in ready:
                eng = nodes[idx].inst.engine
                if eng in ready_by_engine:
                    ready_by_engine[eng] += 1

            used_by_engine = {}
            for engine, limit in slot_limits.items():
                used = sum(1 for inst in bundle.instructions if inst.engine == engine)
                used_by_engine[engine] = used
                engine_used_slots[engine] += used
                if used <= limit:
                    engine_bundle_hist[engine][used] += 1
                if used < limit and ready_by_engine[engine] == 0:
                    engine_idle_no_ready[engine] += (limit - used)

            if ready:
                bundle_end_reasons["slot_limit"] += 1
                slot_limit_bundles += 1
                for engine, limit in slot_limits.items():
                    if ready_by_engine[engine] > 0:
                        slot_limit_ready_left_by_engine[engine] += ready_by_engine[engine]
                        if used_by_engine[engine] >= limit:
                            slot_limit_blocked_bundles_by_engine[engine] += 1
            else:
                bundle_end_reasons["deps"] += 1

            current_bundle += 1
        else:
            # No instruction fit; advance to avoid infinite loops.
            current_bundle += 1

    if terminator is not None:
        term_inst = lir_inst_to_machine_inst(terminator)
        term_bundle = MBundle()
        term_bundle.add_instruction(term_inst)
        bundles.append(term_bundle)

    stats = {
        "engine_used_slots": engine_used_slots,
        "engine_idle_no_ready": engine_idle_no_ready,
        "bundle_end_reasons": bundle_end_reasons,
        "engine_bundle_hist": engine_bundle_hist,
        "slot_limit_blocked_bundles_by_engine": slot_limit_blocked_bundles_by_engine,
        "slot_limit_ready_left_by_engine": slot_limit_ready_left_by_engine,
        "slot_limit_bundles": slot_limit_bundles,
        "devectorized_valu_ops": devectorized_valu_ops,
        "devectorized_alu_ops": devectorized_alu_ops,
    }
    return bundles, stats


def _estimate_peak_live_words(bundles: list[MBundle]) -> int:
    """Estimate block-local peak pressure in scalar scratch words.

    Scratch addresses are already virtual registers here.  Counting each
    address separately also gives vector values their natural VLEN weight.
    The estimate is conservative for redefined addresses and intentionally
    ignores allocator coalescing; it is used only to reject risky schedule
    candidates, never to establish legality.
    """
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}

    for bundle_idx, bundle in enumerate(bundles):
        for inst in bundle.instructions:
            for use in inst.get_uses():
                starts.setdefault(use, 0)
                ends[use] = max(ends.get(use, bundle_idx), bundle_idx)
            for dest in inst.get_defs():
                starts.setdefault(dest, bundle_idx)
                ends[dest] = max(ends.get(dest, bundle_idx), bundle_idx)

    events: dict[int, int] = {}
    for addr, start in starts.items():
        end = ends.get(addr, start)
        events[start] = events.get(start, 0) + 1
        events[end + 1] = events.get(end + 1, 0) - 1

    live = 0
    peak = 0
    for bundle_idx in sorted(events):
        live += events[bundle_idx]
        peak = max(peak, live)
    return peak


def _schedule_lower_bound(nodes: list[ScheduleNode]) -> int:
    """Return a resource/critical-path lower bound for one basic block."""
    engine_work = {engine: 0 for engine in MBundle.SLOT_LIMITS}
    for node in nodes:
        if node.inst.engine in engine_work:
            engine_work[node.inst.engine] += 1

    resource_bound = max(
        (work + MBundle.SLOT_LIMITS[engine] - 1)
        // MBundle.SLOT_LIMITS[engine]
        for engine, work in engine_work.items()
    )
    heights = _compute_critical_path_heights(nodes)
    critical_path_bound = max(heights, default=-1) + 1
    return max(resource_bound, critical_path_bound)


def _schedule_block_auto_stagger(
    instructions: list[MachineInst],
    terminator: Optional[LIRInst],
    *,
    prefer_load_fill: bool,
    devectorize_valu_to_alu: bool,
    devectorize_vector_ops_to_alu: bool,
    devectorize_vbroadcast_to_alu: bool,
    devectorize_partial_alu_fill: bool,
    prioritize_load_unblock: bool,
    register_pressure_limit: int,
    stream_stagger_threshold: int,
    auto_options: AutoStaggerOptions,
    initial_const_val: dict[int, int] | None,
    initial_addr_expr: dict[int, AddrExpr] | None,
    restrict_ptr: bool,
) -> tuple[list[MBundle], dict[str, dict[str, int]]]:
    """Search stagger strengths and keep the best measured schedule.

    The baseline is always a candidate.  Configurable resource-gap and scratch
    guards decide whether to search, while configurable geometric candidates
    bracket useful offsets without baking one workload-specific strength into
    the compiler.
    """
    nodes = _build_dep_graph(
        instructions,
        initial_const_val,
        initial_addr_expr,
        restrict_ptr=restrict_ptr,
    )

    def schedule(stagger: int, bidirectional: bool):
        return _schedule_block(
            instructions,
            terminator,
            prefer_load_fill=prefer_load_fill,
            devectorize_valu_to_alu=devectorize_valu_to_alu,
            devectorize_vector_ops_to_alu=devectorize_vector_ops_to_alu,
            devectorize_vbroadcast_to_alu=devectorize_vbroadcast_to_alu,
            devectorize_partial_alu_fill=devectorize_partial_alu_fill,
            prioritize_load_unblock=prioritize_load_unblock,
            register_pressure_limit=register_pressure_limit,
            stream_stagger=stagger,
            stream_stagger_bidirectional=bidirectional,
            stream_stagger_threshold=stream_stagger_threshold,
            restrict_ptr=restrict_ptr,
            _prebuilt_nodes=nodes,
        )

    baseline = schedule(0, False)
    baseline_bundles = len(baseline[0]) - int(terminator is not None)
    baseline_peak = _estimate_peak_live_words(baseline[0])
    lower_bound = _schedule_lower_bound(nodes)

    # Linear-scan allocation needs some room for vector alignment and holes.
    # This is a rejection guard, not a pressure model used for ordering.
    pressure_budget = SCRATCH_SIZE - auto_options.pressure_headroom
    if (baseline_bundles * 100
            <= lower_bound * (100 + auto_options.min_gap_pct)
            and baseline_peak <= pressure_budget):
        baseline[1]["auto_stagger_candidates"] = 1
        baseline[1]["selected_stream_stagger"] = 0
        baseline[1]["selected_stagger_bidirectional"] = 0
        baseline[1]["selected_stagger_peak_live"] = baseline_peak
        return baseline

    def score(candidate):
        bundles, _ = candidate
        peak = _estimate_peak_live_words(bundles)
        if peak <= pressure_budget:
            return (0, len(bundles), peak)
        return (1, peak, len(bundles))

    best = baseline
    best_score = score(best)
    best_stagger = 0
    best_bidirectional = False
    candidate_count = 1

    if auto_options.direction == "auto":
        # For two streams, the symmetric offsets are identical; use a
        # one-sided ramp instead. Larger sets use the lower-pressure wave.
        probe_offsets = _compute_stream_stagger(
            nodes,
            auto_options.candidate_start,
            shared_degree_threshold=stream_stagger_threshold,
            bidirectional=True,
        )
        directions = [len(set(probe_offsets)) > 1]
    elif auto_options.direction == "both":
        directions = [False, True]
    else:
        directions = [auto_options.direction == "bidirectional"]

    stagger = auto_options.candidate_start
    critical_path_height = max(_compute_critical_path_heights(nodes), default=1)
    max_stagger = (
        critical_path_height
        if auto_options.candidate_max is None
        else auto_options.candidate_max
    )
    while stagger <= max_stagger:
        for bidirectional in directions:
            candidate = schedule(stagger, bidirectional)
            candidate_count += 1
            candidate_score = score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
                best_stagger = stagger
                best_bidirectional = bidirectional

        stagger *= auto_options.candidate_multiplier

    best[1]["auto_stagger_candidates"] = candidate_count
    best[1]["selected_stream_stagger"] = best_stagger
    best[1]["selected_stagger_bidirectional"] = int(best_bidirectional)
    best[1]["selected_stagger_peak_live"] = _estimate_peak_live_words(best[0])
    return best


class InstSchedulingPass(LIRToMIRLoweringPass):
    """
    LIR -> MIR lowering with instruction scheduling and bundling.
    """

    @property
    def name(self) -> str:
        return "inst-scheduling"

    def run(self, lir: LIRFunction, config: PassConfig) -> MachineFunction:
        """Lower LIR to MIR using the scheduling algorithm."""
        self._check_no_remaining_phis(lir)
        self._init_metrics()
        prefer_load_fill = bool(config.options.get("prefer_load_fill", False))
        devectorize_valu_to_alu = bool(config.options.get("devectorize_valu_to_alu", False))
        devectorize_vector_ops_to_alu = bool(config.options.get("devectorize_vector_ops_to_alu", True))
        devectorize_vbroadcast_to_alu = bool(config.options.get("devectorize_vbroadcast_to_alu", True))
        devectorize_partial_alu_fill = bool(config.options.get("devectorize_partial_alu_fill", False))
        prioritize_load_unblock = bool(config.options.get("prioritize_load_unblock", False))
        register_pressure_limit = int(config.options.get("register_pressure_limit", 0))
        stream_stagger_option = config.options.get("stream_stagger", 0)
        stream_stagger_auto = stream_stagger_option == "auto"
        stream_stagger = 0 if stream_stagger_auto else int(stream_stagger_option)
        auto_stagger_options = (
            _parse_auto_stagger_options(config.options)
            if stream_stagger_auto else None
        )
        stream_stagger_bidirectional = bool(
            config.options.get("stream_stagger_bidirectional", False))
        stream_stagger_threshold = config.options.get(
            "stream_stagger_threshold", DEFAULT_SHARED_DEGREE_THRESHOLD
        )
        if (isinstance(stream_stagger_threshold, bool)
                or not isinstance(stream_stagger_threshold, int)
                or stream_stagger_threshold < 1):
            raise ValueError(
                "stream_stagger_threshold must be an integer >= 1, got "
                f"{stream_stagger_threshold!r}"
            )
        restrict_ptr = bool(config.options.get("restrict_ptr", False))
        # Statically pre-expand this percentage of devectorizable valu ops
        # into scalar alu ops before scheduling. Unlike the reactive
        # per-bundle expansion (which only fires when the scheduler is
        # stuck), pre-expansion exposes the scalar work in the dependency
        # graph from the start, so idle alu slots can absorb it whenever it
        # is ready. Balances valu/alu when valu is the binding engine.
        static_devectorize_pct = int(config.options.get("static_devectorize_pct", 0))
        # Materialize constants on the flow engine (add_imm from an anchor
        # const) instead of the load engine. Frees load slots when the load
        # engine is the binding resource.
        const_via_flow = bool(config.options.get("const_via_flow", False))
        # Keep the first K non-anchor consts on the load engine (they tend
        # to feed the critical early rounds; flow materializes 1/cycle).
        const_via_flow_skip = int(config.options.get("const_via_flow_skip", 0))

        mfunc = MachineFunction(entry=lir.entry, max_scratch_used=lir.max_scratch_used,
                                phi_eliminated=True)

        block_order = get_block_order(lir)

        # --- Pre-compute predecessors and MachineInsts for all blocks ---
        block_predecessors: dict[str, list[str]] = {name: [] for name in lir.blocks}
        for name, block in lir.blocks.items():
            for succ in get_successors(block):
                if succ in block_predecessors:
                    block_predecessors[succ].append(name)

        block_machine_insts: dict[str, list[MachineInst]] = {}
        static_devectorized = 0
        consts_via_flow = 0
        expand_counter = 0
        for name in block_order:
            insts = [
                lir_inst_to_machine_inst(inst) for inst in lir.blocks[name].instructions
            ]
            if const_via_flow:
                anchor_dest = None
                anchor_val = None
                skipped = 0
                converted: list[MachineInst] = []
                for inst in insts:
                    if inst.opcode == LIROpcode.CONST and isinstance(inst.dest, int):
                        if anchor_dest is None:
                            anchor_dest = inst.dest
                            anchor_val = int(inst.operands[0])
                        elif skipped < const_via_flow_skip:
                            skipped += 1
                        else:
                            imm = (int(inst.operands[0]) - anchor_val) & 0xFFFFFFFF
                            converted.append(MachineInst(
                                opcode=LIROpcode.ADD_IMM,
                                dest=inst.dest,
                                operands=[anchor_dest, imm],
                                engine="flow",
                            ))
                            consts_via_flow += 1
                            continue
                    converted.append(inst)
                insts = converted
            if static_devectorize_pct > 0 and devectorize_valu_to_alu:
                expanded: list[MachineInst] = []
                for inst in insts:
                    # Cheap eligibility gate first; only ops the counter
                    # selects pay for building the 8-instruction expansion.
                    eligible = (
                        inst.engine == "valu"
                        and inst.opcode != LIROpcode.MULTIPLY_ADD
                        and (inst.opcode in VECTOR_TO_SCALAR_OPCODE
                             or (inst.opcode == LIROpcode.VBROADCAST
                                 and devectorize_vbroadcast_to_alu))
                        and (inst.opcode == LIROpcode.VBROADCAST
                             or devectorize_vector_ops_to_alu)
                    )
                    if eligible:
                        expand_counter += static_devectorize_pct
                        if expand_counter >= 100:
                            expand_counter -= 100
                            scalar_insts = _devectorize_valu_to_alu_with_knobs(
                                inst,
                                devectorize_vector_ops_to_alu=devectorize_vector_ops_to_alu,
                                devectorize_vbroadcast_to_alu=devectorize_vbroadcast_to_alu,
                            )
                            if scalar_insts is not None:
                                expanded.extend(scalar_insts)
                                static_devectorized += 1
                                continue
                    expanded.append(inst)
                insts = expanded
            block_machine_insts[name] = insts

        # --- Forward dataflow of const_val/addr_expr to a fixpoint ---
        # A single RPO pass would ignore back edges: a loop-carried
        # redefinition (a phi turned into copies) would inherit the
        # preheader's facts as if they were loop-invariant, and a "provably
        # disjoint" access could be reordered across a second-iteration
        # conflict. Iterating meets every REACHABLE predecessor; facts only
        # ever shrink at a meet, so the sweep count is bounded.
        block_entry_const: dict[str, dict[int, int]] = {}
        block_entry_addr: dict[str, dict[int, AddrExpr]] = {}
        block_exit_const: dict[str, dict[int, int]] = {}
        block_exit_addr: dict[str, dict[int, AddrExpr]] = {}

        max_sweeps = 4 * len(block_order) + 8  # safety net; typical is 2-3
        for _sweep in range(max_sweeps):
            changed = False
            for block_name in block_order:
                preds = block_predecessors[block_name]
                # Meet over predecessors whose exit state has been computed
                # in SOME sweep; unvisited preds are optimistic TOP and drop
                # out once computed (their facts can only remove entries).
                computed = [p for p in preds if p in block_exit_const]
                if not computed:
                    entry_const: dict[int, int] = {}
                    entry_addr: dict[int, AddrExpr] = {}
                elif len(computed) == 1:
                    entry_const = dict(block_exit_const[computed[0]])
                    entry_addr = dict(block_exit_addr[computed[0]])
                else:
                    first = computed[0]
                    entry_const = {}
                    for k, v in block_exit_const[first].items():
                        if all(block_exit_const[p].get(k) == v for p in computed[1:]):
                            entry_const[k] = v
                    entry_addr = {}
                    for k, v in block_exit_addr[first].items():
                        if all(block_exit_addr[p].get(k) == v for p in computed[1:]):
                            entry_addr[k] = v

                if (block_entry_const.get(block_name) != entry_const
                        or block_entry_addr.get(block_name) != entry_addr):
                    changed = True
                block_entry_const[block_name] = entry_const
                block_entry_addr[block_name] = entry_addr

                exit_const, exit_addr = _compute_exit_value_state(
                    block_machine_insts[block_name], entry_const, entry_addr
                )
                if (block_exit_const.get(block_name) != exit_const
                        or block_exit_addr.get(block_name) != exit_addr):
                    changed = True
                block_exit_const[block_name] = exit_const
                block_exit_addr[block_name] = exit_addr
            if not changed:
                break

        # --- Main scheduling loop ---
        # Global diagnostics aggregation
        slot_limits = MBundle.SLOT_LIMITS
        engine_used_slots = {engine: 0 for engine in slot_limits}
        engine_idle_no_ready = {engine: 0 for engine in slot_limits}
        bundle_end_reasons = {"deps": 0, "slot_limit": 0}
        engine_bundle_hist: dict[str, dict[int, int]] = {
            engine: {i: 0 for i in range(limit + 1)} for engine, limit in slot_limits.items()
        }
        slot_limit_blocked_bundles_by_engine = {engine: 0 for engine in slot_limits}
        slot_limit_ready_left_by_engine = {engine: 0 for engine in slot_limits}
        slot_limit_bundles = 0
        total_devectorized_valu_ops = 0
        total_devectorized_alu_ops = 0
        auto_stagger_candidates = 0
        auto_stagger_selection: dict[str, int] = {}
        auto_stagger_peak_live = 0

        for block_name in block_order:
            lir_block = lir.blocks[block_name]
            machine_insts = block_machine_insts[block_name]
            terminator = lir_block.terminator

            schedule_kwargs = {
                "prefer_load_fill": prefer_load_fill,
                "devectorize_valu_to_alu": devectorize_valu_to_alu,
                "devectorize_vector_ops_to_alu": devectorize_vector_ops_to_alu,
                "devectorize_vbroadcast_to_alu": devectorize_vbroadcast_to_alu,
                "devectorize_partial_alu_fill": devectorize_partial_alu_fill,
                "prioritize_load_unblock": prioritize_load_unblock,
                "register_pressure_limit": register_pressure_limit,
                "stream_stagger_threshold": stream_stagger_threshold,
                "initial_const_val": block_entry_const[block_name],
                "initial_addr_expr": block_entry_addr[block_name],
                "restrict_ptr": restrict_ptr,
            }
            if stream_stagger_auto:
                assert auto_stagger_options is not None
                bundles, stats = _schedule_block_auto_stagger(
                    machine_insts,
                    terminator,
                    auto_options=auto_stagger_options,
                    **schedule_kwargs,
                )
            else:
                bundles, stats = _schedule_block(
                    machine_insts,
                    terminator,
                    stream_stagger=stream_stagger,
                    stream_stagger_bidirectional=stream_stagger_bidirectional,
                    **schedule_kwargs,
                )

            for engine in slot_limits:
                engine_used_slots[engine] += stats["engine_used_slots"].get(engine, 0)
                engine_idle_no_ready[engine] += stats["engine_idle_no_ready"].get(engine, 0)
                for used, count in stats["engine_bundle_hist"].get(engine, {}).items():
                    engine_bundle_hist[engine][used] = engine_bundle_hist[engine].get(used, 0) + count
                slot_limit_blocked_bundles_by_engine[engine] += stats[
                    "slot_limit_blocked_bundles_by_engine"
                ].get(engine, 0)
                slot_limit_ready_left_by_engine[engine] += stats[
                    "slot_limit_ready_left_by_engine"
                ].get(engine, 0)
            for key in bundle_end_reasons:
                bundle_end_reasons[key] += stats["bundle_end_reasons"].get(key, 0)
            slot_limit_bundles += stats.get("slot_limit_bundles", 0)
            total_devectorized_valu_ops += int(stats.get("devectorized_valu_ops", 0))
            total_devectorized_alu_ops += int(stats.get("devectorized_alu_ops", 0))
            auto_stagger_candidates += int(stats.get("auto_stagger_candidates", 0))
            if "selected_stream_stagger" in stats:
                selected = int(stats["selected_stream_stagger"])
                bidirectional = bool(stats["selected_stagger_bidirectional"])
                label = f"{selected}{'b' if bidirectional else 'u'}"
                auto_stagger_selection[label] = (
                    auto_stagger_selection.get(label, 0) + 1
                )
                auto_stagger_peak_live = max(
                    auto_stagger_peak_live,
                    int(stats["selected_stagger_peak_live"]),
                )

            successors = get_successors(lir_block)
            predecessors = block_predecessors[block_name]

            mbb = MachineBasicBlock(
                name=block_name,
                bundles=bundles,
                predecessors=predecessors,
                successors=successors,
            )
            mfunc.blocks[block_name] = mbb

        if self._metrics:
            total_bundles = mfunc.total_bundles()
            total_insts = mfunc.total_instructions()
            avg_insts_per_bundle = total_insts / total_bundles if total_bundles > 0 else 0

            bundle_size_histogram: dict[int, int] = {}
            for block in mfunc.blocks.values():
                for bundle in block.bundles:
                    size = len(bundle.instructions)
                    bundle_size_histogram[size] = bundle_size_histogram.get(size, 0) + 1

            multi_inst_bundles = sum(
                count for size, count in bundle_size_histogram.items() if size > 1
            )

            self._metrics.custom = {
                "bundles": total_bundles,
                "instructions": total_insts,
                "avg_insts_per_bundle": round(avg_insts_per_bundle, 2),
                "multi_inst_bundles": multi_inst_bundles,
                "single_inst_bundles": bundle_size_histogram.get(1, 0),
                "packing_ratio": round(avg_insts_per_bundle, 2),
                "devectorized_valu_ops": total_devectorized_valu_ops,
                "devectorized_alu_ops": total_devectorized_alu_ops,
                "static_devectorized": static_devectorized,
                "consts_via_flow": consts_via_flow,
                "auto_stagger_candidates": auto_stagger_candidates,
                "auto_stagger_selection": auto_stagger_selection,
                "auto_stagger_peak_live": auto_stagger_peak_live,
            }

            self._add_metric_message(
                f"Bundle size distribution: {dict(sorted(bundle_size_histogram.items()))}"
            )

            # Slot utilization summary
            if total_bundles > 0:
                engine_capacity = {
                    engine: total_bundles * limit for engine, limit in slot_limits.items()
                }
                engine_util = {
                    engine: {
                        "used": engine_used_slots[engine],
                        "capacity": engine_capacity[engine],
                        "util": round(engine_used_slots[engine] / engine_capacity[engine], 3)
                        if engine_capacity[engine] > 0 else 0.0,
                    }
                    for engine in slot_limits
                }

                # Saturation/empty rates per engine
                engine_saturation = {}
                engine_empty = {}
                for engine, limit in slot_limits.items():
                    total = sum(engine_bundle_hist[engine].values())
                    if total == 0:
                        engine_saturation[engine] = 0.0
                        engine_empty[engine] = 0.0
                        continue
                    engine_saturation[engine] = round(engine_bundle_hist[engine].get(limit, 0) / total, 3)
                    engine_empty[engine] = round(engine_bundle_hist[engine].get(0, 0) / total, 3)

                # Readable diagnostics (multi-line message)
                lines = []
                lines.append(
                    f"Bundle end reasons: deps={bundle_end_reasons['deps']}, slot_limit={bundle_end_reasons['slot_limit']}"
                )
                if slot_limit_bundles > 0:
                    lines.append("Slot-limit details (bundles blocked by engine, avg ready left):")
                    for engine in slot_limits:
                        blocked = slot_limit_blocked_bundles_by_engine[engine]
                        avg_ready = slot_limit_ready_left_by_engine[engine] / slot_limit_bundles
                        lines.append(
                            f"  {engine}: blocked_bundles={blocked} "
                            f"({blocked/slot_limit_bundles*100:.1f}%), "
                            f"avg_ready_left={avg_ready:.2f}"
                        )
                lines.append("Engine utilization (used/capacity, util, empty, saturated, idle_no_ready):")
                for engine in slot_limits:
                    util = engine_util[engine]
                    lines.append(
                        f"  {engine}: {util['used']}/{util['capacity']} "
                        f"({util['util']*100:.1f}%), "
                        f"empty={engine_empty[engine]*100:.1f}%, "
                        f"saturated={engine_saturation[engine]*100:.1f}%, "
                        f"idle_no_ready={engine_idle_no_ready[engine]}"
                    )
                lines.append("Engine slots per bundle (used_slots: bundle_count):")
                for engine, limit in slot_limits.items():
                    non_zero = {
                        k: v for k, v in engine_bundle_hist[engine].items() if v != 0
                    }
                    ordered = ", ".join(f"{k}:{v}" for k, v in sorted(non_zero.items()))
                    lines.append(f"  {engine} (limit {limit}): {ordered}")
                if devectorize_valu_to_alu:
                    lines.append(
                        "Devectorize valu->alu: "
                        f"valu_ops={total_devectorized_valu_ops}, "
                        f"alu_ops_emitted={total_devectorized_alu_ops}"
                    )
                    lines.append(
                        "Devectorize knobs: "
                        f"vector_ops={devectorize_vector_ops_to_alu}, "
                        f"vbroadcast={devectorize_vbroadcast_to_alu}, "
                        f"partial_alu_fill={devectorize_partial_alu_fill}"
                    )
                self._add_metric_message("\n".join(lines))

        return mfunc
