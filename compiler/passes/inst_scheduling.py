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


def _schedule_block(
    instructions: list[MachineInst],
    terminator: Optional[LIRInst],
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

    n = len(nodes)
    remaining_preds = [len(node.preds) for node in nodes]
    earliest_bundle = [0] * n
    scheduled = [False] * n
    ready: set[int] = set()

    bundles: list[MBundle] = []
    current_bundle = 0
    scheduled_count = 0

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

        while True:
            best_idx = None
            best_key = None
            for idx in sorted(ready):
                inst = nodes[idx].inst
                if not bundle.has_slot_available(inst.engine):
                    continue
                key = (heights[idx], _engine_priority(inst.engine), -nodes[idx].index)
                if best_key is None or key > best_key:
                    best_key = key
                    best_idx = idx

            if best_idx is None:
                break

            inst = nodes[best_idx].inst
            if not bundle.add_instruction(inst):
                ready.remove(best_idx)
                continue

            ready.remove(best_idx)
            scheduled[best_idx] = True
            scheduled_count += 1

            for succ, delay in nodes[best_idx].succs.items():
                remaining_preds[succ] -= 1
                target_bundle = current_bundle + delay
                if target_bundle > earliest_bundle[succ]:
                    earliest_bundle[succ] = target_bundle
                if remaining_preds[succ] == 0 and earliest_bundle[succ] <= current_bundle:
                    ready.add(succ)

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

        for block_name in block_order:
            lir_block = lir.blocks[block_name]

            machine_insts = [_lir_inst_to_machine_inst(inst) for inst in lir_block.instructions]
            terminator = lir_block.terminator

            bundles, stats = _schedule_block(machine_insts, terminator)

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
                self._add_metric_message("\n".join(lines))

        return mfunc
