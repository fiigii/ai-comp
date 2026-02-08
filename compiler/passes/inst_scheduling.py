"""
Instruction Scheduling Pass (LIR -> MIR)

Schedules LIR instructions into VLIW bundles using a delay-aware list
scheduler and constructs MIR bundles per basic block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..pass_manager import LIRToMIRLoweringPass, PassConfig
from ..lir import LIRFunction, LIROpcode, LIRInst, BasicBlock
from ..mir import MachineInst, MBundle, MachineBasicBlock, MachineFunction


def _lir_inst_to_machine_inst(inst: LIRInst) -> MachineInst:
    """Convert a LIRInst to a MachineInst."""
    return MachineInst(
        opcode=inst.opcode,
        dest=inst.dest,
        operands=list(inst.operands),
        engine=inst.engine,
    )


def _get_successors(block: BasicBlock) -> list[str]:
    """Get successor block names from a LIR basic block."""
    if block.terminator is None:
        return []
    if block.terminator.opcode == LIROpcode.JUMP:
        return [block.terminator.operands[0]]
    if block.terminator.opcode == LIROpcode.COND_JUMP:
        return [block.terminator.operands[1], block.terminator.operands[2]]
    return []


def _get_block_order(lir: LIRFunction) -> list[str]:
    """Get blocks in reverse postorder for scheduling."""
    visited: set[str] = set()
    postorder: list[str] = []

    def dfs(name: str):
        if name in visited:
            return
        visited.add(name)
        block = lir.blocks.get(name)
        if block:
            for succ_name in _get_successors(block):
                dfs(succ_name)
            postorder.append(name)

    dfs(lir.entry)
    return list(reversed(postorder))


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


ENGINE_PRIORITY = {
    "flow": 4,
    "load": 3,
    "store": 3,
    "valu": 2,
    "alu": 1,
}

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


def _keys_alias(a: Optional[AddrExpr], b: Optional[AddrExpr]) -> bool:
    """Check if two memory keys may alias."""
    if a is None or b is None:
        return True
    if a.base != b.base:
        return False
    if a.offset is None or b.offset is None:
        return True
    return a.offset == b.offset


def _clear_value_info(dest: int, const_val: dict[int, int], addr_expr: dict[int, AddrExpr]) -> None:
    const_val.pop(dest, None)
    addr_expr.pop(dest, None)


def _set_const(dest: int, value: int, const_val: dict[int, int], addr_expr: dict[int, AddrExpr]) -> None:
    const_val[dest] = value
    addr_expr.pop(dest, None)


def _set_addr(dest: int, base: int, offset: Optional[int],
              const_val: dict[int, int], addr_expr: dict[int, AddrExpr]) -> None:
    addr_expr[dest] = AddrExpr(base=base, offset=offset)
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
    dests = []
    if isinstance(inst.dest, int):
        dests = [inst.dest]
    elif isinstance(inst.dest, list):
        dests = list(inst.dest)

    if not dests:
        return

    # CONST
    if inst.opcode == LIROpcode.CONST and isinstance(inst.dest, int):
        _set_const(inst.dest, int(inst.operands[0]), const_val, addr_expr)
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

    # LOAD from header pointers (mem[4..6]) establishes base pointer symbols
    if inst.opcode == LIROpcode.LOAD and isinstance(inst.dest, int):
        addr = inst.operands[0]
        if isinstance(addr, int):
            const_addr = const_val.get(addr)
            if const_addr in (4, 5, 6):
                _set_addr(inst.dest, const_addr, 0, const_val, addr_expr)
                return
        _clear_value_info(inst.dest, const_val, addr_expr)
        return

    # For all other defs, clear any tracked info.
    for d in dests:
        _clear_value_info(d, const_val, addr_expr)


def _memory_key(inst: MachineInst,
                const_val: dict[int, int],
                addr_expr: dict[int, AddrExpr]) -> Optional[AddrExpr]:
    """Compute a conservative alias key for a memory instruction."""
    if inst.opcode == LIROpcode.LOAD_OFFSET:
        addr_base = inst.operands[0]
        offset = inst.operands[1] if len(inst.operands) > 1 else None
        if isinstance(addr_base, int) and isinstance(offset, int):
            lane_addr = addr_base + offset
            return addr_expr.get(lane_addr)
        return None

    addr_op = None
    if inst.opcode in (LIROpcode.LOAD, LIROpcode.VLOAD):
        addr_op = inst.operands[0]
    elif inst.opcode in (LIROpcode.STORE, LIROpcode.VSTORE):
        addr_op = inst.operands[0]

    if isinstance(addr_op, int):
        return addr_expr.get(addr_op)

    return None


def _build_dep_graph(instructions: list[MachineInst]) -> list[ScheduleNode]:
    """Build a dependency graph with delay-annotated edges."""
    nodes = [ScheduleNode(inst=inst, index=i) for i, inst in enumerate(instructions)]

    last_def: dict[int, int] = {}
    last_barrier: Optional[int] = None

    # Conservative constant/address tracking for simple alias analysis
    const_val: dict[int, int] = {}
    addr_expr: dict[int, AddrExpr] = {}

    # Track memory ops by alias key
    last_store_by_key: dict[Optional[AddrExpr], int] = {}
    loads_since_store_by_key: dict[Optional[AddrExpr], list[int]] = {}

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
                    if _keys_alias(key, k):
                        _add_edge(nodes, store_idx, i, 1)
                loads_since_store_by_key.setdefault(key, []).append(i)

            if _is_memory_store(inst):
                for k, store_idx in list(last_store_by_key.items()):
                    if _keys_alias(key, k):
                        _add_edge(nodes, store_idx, i, 0)
                for k, load_list in list(loads_since_store_by_key.items()):
                    if _keys_alias(key, k):
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


def _expand_multiply_add_to_alu(inst: MachineInst) -> Optional[list[MachineInst]]:
    """Lower vector multiply_add lanes into scalar MUL+ADD pairs."""
    if inst.opcode != LIROpcode.MULTIPLY_ADD:
        return None
    if not isinstance(inst.dest, list) or len(inst.operands) < 3:
        return None
    lhs = inst.operands[0]
    rhs = inst.operands[1]
    addend = inst.operands[2]
    if not isinstance(lhs, list) or not isinstance(rhs, list) or not isinstance(addend, list):
        return None
    if len(inst.dest) != len(lhs) or len(inst.dest) != len(rhs) or len(inst.dest) != len(addend):
        return None

    scalar_insts: list[MachineInst] = []
    for lane_dest, lane_lhs, lane_rhs, lane_addend in zip(inst.dest, lhs, rhs, addend):
        if (
            not isinstance(lane_dest, int)
            or not isinstance(lane_lhs, int)
            or not isinstance(lane_rhs, int)
            or not isinstance(lane_addend, int)
        ):
            return None
        scalar_insts.append(
            MachineInst(opcode=LIROpcode.MUL, dest=lane_dest, operands=[lane_lhs, lane_rhs], engine="alu")
        )
        scalar_insts.append(
            MachineInst(opcode=LIROpcode.ADD, dest=lane_dest, operands=[lane_dest, lane_addend], engine="alu")
        )
    return scalar_insts


def _devectorize_valu_to_alu_with_knobs(
    inst: MachineInst,
    *,
    devectorize_vector_ops_to_alu: bool,
    devectorize_vbroadcast_to_alu: bool,
    devectorize_multiply_add_to_alu: bool,
) -> Optional[list[MachineInst]]:
    if inst.opcode == LIROpcode.MULTIPLY_ADD:
        if not devectorize_multiply_add_to_alu:
            return None
        return _expand_multiply_add_to_alu(inst)
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
    devectorize_multiply_add_to_alu: bool = False,
    prioritize_load_unblock: bool = False,
    register_pressure_limit: int = 0,
) -> tuple[list[MBundle], dict[str, dict[str, int]]]:
    """Schedule a block's instructions into MIR bundles.

    Returns:
        (bundles, stats) where stats contains per-engine utilization and
        bundle-end reason counts.
    """
    if not instructions and terminator is None:
        return []

    slot_limits = MBundle.SLOT_LIMITS
    nodes = _build_dep_graph(instructions)
    heights = _compute_critical_path_heights(nodes)
    load_unblock_scores = _compute_load_unblock_scores(nodes) if prioritize_load_unblock else [0] * len(nodes)
    load_distance = _compute_distance_to_load(nodes) if prioritize_load_unblock else [10**9] * len(nodes)

    n = len(nodes)
    remaining_preds = [len(node.preds) for node in nodes]
    earliest_bundle = [0] * n
    scheduled = [False] * n
    ready: set[int] = set()

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
    devectorized_multiply_add_ops = 0

    def refresh_ready() -> None:
        for i in range(n):
            if scheduled[i] or remaining_preds[i] != 0:
                continue
            if earliest_bundle[i] <= current_bundle:
                ready.add(i)

    refresh_ready()

    while scheduled_count < n:
        refresh_ready()
        if not ready:
            next_bundle = min(
                earliest_bundle[i] for i in range(n) if not scheduled[i]
            )
            if next_bundle > current_bundle:
                current_bundle = next_bundle
            refresh_ready()
            if not ready:
                for i in range(n):
                    if not scheduled[i]:
                        ready.add(i)
                        break

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

            # Under high pressure, prefer instructions that free registers
            high_pressure = pressure_aware and live_reg_count > register_pressure_limit

            for idx in sorted(ready):
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
                elif prioritize_load_unblock and only_engine is None and load_slot_open and not any_ready_load:
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
                if remaining_preds[succ] == 0 and earliest_bundle[succ] <= current_bundle:
                    ready.add(succ)

        def try_schedule_devectorized_valu() -> bool:
            nonlocal devectorized_valu_ops, devectorized_alu_ops, devectorized_multiply_add_ops
            if not devectorize_valu_to_alu:
                return False
            if available_slots("valu") > 0:
                return False
            if available_slots("alu") <= 0:
                return False

            best_idx = None
            best_key = None
            best_scalar_insts: Optional[list[MachineInst]] = None
            for idx in sorted(ready):
                inst = nodes[idx].inst
                if inst.engine != "valu":
                    continue
                scalar_insts = _devectorize_valu_to_alu_with_knobs(
                    inst,
                    devectorize_vector_ops_to_alu=devectorize_vector_ops_to_alu,
                    devectorize_vbroadcast_to_alu=devectorize_vbroadcast_to_alu,
                    devectorize_multiply_add_to_alu=devectorize_multiply_add_to_alu,
                )
                if scalar_insts is None or len(scalar_insts) > available_slots("alu"):
                    continue
                key = (heights[idx], -nodes[idx].index)
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx
                    best_scalar_insts = scalar_insts

            if best_idx is None or best_scalar_insts is None:
                return False

            for scalar_inst in best_scalar_insts:
                if not bundle.add_instruction(scalar_inst):
                    raise RuntimeError("unexpected ALU slot exhaustion during devectorization")
                used_in_bundle_by_engine["alu"] += 1

            devectorized_valu_ops += 1
            devectorized_alu_ops += len(best_scalar_insts)
            if nodes[best_idx].inst.opcode == LIROpcode.MULTIPLY_ADD:
                devectorized_multiply_add_ops += 1
            mark_node_scheduled(best_idx)
            return True

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
        term_inst = _lir_inst_to_machine_inst(terminator)
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
        "devectorized_multiply_add_ops": devectorized_multiply_add_ops,
    }
    return bundles, stats


class InstSchedulingPass(LIRToMIRLoweringPass):
    """
    LIR -> MIR lowering with instruction scheduling and bundling.
    """

    @property
    def name(self) -> str:
        return "inst-scheduling"

    def run(self, lir: LIRFunction, config: PassConfig) -> MachineFunction:
        """Lower LIR to MIR using the scheduling algorithm."""
        self._init_metrics()
        prefer_load_fill = bool(config.options.get("prefer_load_fill", False))
        devectorize_valu_to_alu = bool(config.options.get("devectorize_valu_to_alu", False))
        devectorize_vector_ops_to_alu = bool(config.options.get("devectorize_vector_ops_to_alu", True))
        devectorize_vbroadcast_to_alu = bool(config.options.get("devectorize_vbroadcast_to_alu", True))
        devectorize_multiply_add_to_alu = bool(config.options.get("devectorize_multiply_add_to_alu", False))
        prioritize_load_unblock = bool(config.options.get("prioritize_load_unblock", False))
        register_pressure_limit = int(config.options.get("register_pressure_limit", 0))

        mfunc = MachineFunction(entry=lir.entry, max_scratch_used=lir.max_scratch_used)

        block_order = _get_block_order(lir)
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
        total_devectorized_multiply_add_ops = 0

        for block_name in block_order:
            lir_block = lir.blocks[block_name]

            machine_insts = [_lir_inst_to_machine_inst(inst) for inst in lir_block.instructions]
            terminator = lir_block.terminator

            bundles, stats = _schedule_block(
                machine_insts,
                terminator,
                prefer_load_fill=prefer_load_fill,
                devectorize_valu_to_alu=devectorize_valu_to_alu,
                devectorize_vector_ops_to_alu=devectorize_vector_ops_to_alu,
                devectorize_vbroadcast_to_alu=devectorize_vbroadcast_to_alu,
                devectorize_multiply_add_to_alu=devectorize_multiply_add_to_alu,
                prioritize_load_unblock=prioritize_load_unblock,
                register_pressure_limit=register_pressure_limit,
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
            total_devectorized_multiply_add_ops += int(stats.get("devectorized_multiply_add_ops", 0))

            successors = _get_successors(lir_block)
            predecessors: list[str] = []
            for other_name, other_block in lir.blocks.items():
                if block_name in _get_successors(other_block):
                    predecessors.append(other_name)

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
                "devectorized_multiply_add_ops": total_devectorized_multiply_add_ops,
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
                        f"multiply_add={devectorize_multiply_add_to_alu}, "
                        f"devectorized_multiply_add={total_devectorized_multiply_add_ops}"
                    )
                self._add_metric_message("\n".join(lines))

        return mfunc
