"""
LIR to MIR Lowering Pass

Converts LIR (after phi-elimination) to MIR with instruction scheduling.
Uses a list scheduling algorithm to pack instructions into VLIW bundles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from problem import VLEN

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
class DepGraphNode:
    """Node in the data dependency graph."""
    inst: MachineInst
    index: int  # Original instruction index
    preds: set[int] = field(default_factory=set)   # Indices of predecessors
    succs: set[int] = field(default_factory=set)   # Indices of successors
    in_degree: int = 0  # Number of unsatisfied predecessors


def _is_memory_load(inst: MachineInst) -> bool:
    """Check if instruction is a memory load (reads from main memory)."""
    return inst.opcode in (LIROpcode.LOAD, LIROpcode.VLOAD)


def _is_memory_store(inst: MachineInst) -> bool:
    """Check if instruction is a memory store (writes to main memory)."""
    return inst.opcode in (LIROpcode.STORE, LIROpcode.VSTORE)


def _is_barrier(inst: MachineInst) -> bool:
    """Check if instruction is a barrier (must execute in program order).

    PAUSE and HALT are barriers - they must execute after all preceding
    instructions and before all following instructions.
    """
    return inst.opcode in (LIROpcode.PAUSE, LIROpcode.HALT)


def _build_dep_graph(instructions: list[MachineInst]) -> list[DepGraphNode]:
    """Build data dependency graph for a list of instructions.

    Dependencies are based on def-use chains:
    - RAW (Read After Write): instruction uses a value defined by an earlier instruction
    - WAW (Write After Write): instruction defines a value defined by an earlier instruction
    - WAR (Write After Read): instruction defines a value used by an earlier instruction

    Memory dependencies (conservative):
    - LOAD after STORE: must wait for all preceding stores
    - STORE after LOAD: must wait for all preceding loads
    - STORE after STORE: must wait for all preceding stores

    For VLIW, all instructions in a bundle read pre-bundle state, so we only need
    to track cross-bundle dependencies.
    """
    nodes = [DepGraphNode(inst=inst, index=i) for i, inst in enumerate(instructions)]

    # Map from scratch address to last instruction that defined it
    last_def: dict[int, int] = {}
    # Map from scratch address to all instructions that use it (for WAR)
    all_uses: dict[int, list[int]] = {}

    # Track memory operations for memory dependency ordering
    # We use conservative ordering: all memory ops are ordered relative to each other
    last_memory_store: Optional[int] = None
    all_memory_loads: list[int] = []

    # Track barriers (PAUSE, HALT) to ensure program order
    last_barrier: Optional[int] = None

    for i, inst in enumerate(instructions):
        uses = inst.get_uses()
        defs = inst.get_defs()

        # RAW dependencies: this instruction uses values from earlier defs
        for use in uses:
            if use in last_def:
                pred_idx = last_def[use]
                nodes[i].preds.add(pred_idx)
                nodes[pred_idx].succs.add(i)

        # WAW dependencies: this instruction defines a value defined earlier
        for d in defs:
            if d in last_def:
                pred_idx = last_def[d]
                nodes[i].preds.add(pred_idx)
                nodes[pred_idx].succs.add(i)

        # WAR dependencies: this instruction defines a value used earlier
        for d in defs:
            if d in all_uses:
                for pred_idx in all_uses[d]:
                    if pred_idx != i:  # Don't add self-loop
                        nodes[i].preds.add(pred_idx)
                        nodes[pred_idx].succs.add(i)

        # Memory dependencies (conservative)
        if _is_memory_load(inst):
            # LOAD depends on all preceding STOREs
            if last_memory_store is not None:
                nodes[i].preds.add(last_memory_store)
                nodes[last_memory_store].succs.add(i)
            all_memory_loads.append(i)

        if _is_memory_store(inst):
            # STORE depends on the last STORE (to preserve store order)
            if last_memory_store is not None:
                nodes[i].preds.add(last_memory_store)
                nodes[last_memory_store].succs.add(i)
            # STORE depends on all preceding LOADs (WAR for memory)
            for load_idx in all_memory_loads:
                nodes[i].preds.add(load_idx)
                nodes[load_idx].succs.add(i)
            last_memory_store = i
            all_memory_loads = []  # Reset loads since they're now covered

        # Barrier dependencies (PAUSE, HALT)
        # Barriers must execute after all preceding instructions
        # and all following instructions must wait for the barrier
        if _is_barrier(inst):
            for pred_idx in range(i):
                nodes[i].preds.add(pred_idx)
                nodes[pred_idx].succs.add(i)
            # Track this as the last barrier for subsequent instructions
            last_barrier = i
        elif last_barrier is not None:
            # Non-barrier instruction depends on the last barrier
            nodes[i].preds.add(last_barrier)
            nodes[last_barrier].succs.add(i)

        # Update last_def
        for d in defs:
            last_def[d] = i

        # Update all_uses
        for use in uses:
            if use not in all_uses:
                all_uses[use] = []
            all_uses[use].append(i)

    # Compute initial in-degrees
    for node in nodes:
        node.in_degree = len(node.preds)

    return nodes


def _schedule_block_no_packing(instructions: list[MachineInst], terminator: Optional[MachineInst]) -> list[MBundle]:
    """Create one bundle per instruction (no scheduling/packing).

    This is the simple mode that preserves original instruction order
    with each instruction in its own bundle.
    """
    bundles: list[MBundle] = []

    for inst in instructions:
        bundle = MBundle()
        bundle.add_instruction(inst)
        bundles.append(bundle)

    if terminator is not None:
        term_inst = _lir_inst_to_machine_inst(terminator)
        term_bundle = MBundle()
        term_bundle.add_instruction(term_inst)
        bundles.append(term_bundle)

    return bundles


def _schedule_block_with_packing(instructions: list[MachineInst], terminator: Optional[MachineInst]) -> list[MBundle]:
    """Schedule instructions into bundles using list scheduling.

    Algorithm:
    1. Build data dependency graph
    2. Start with ready set = instructions with no predecessors
    3. Greedily pack ready instructions into current bundle
    4. When bundle is full or no more ready instructions fit, advance to next bundle
    5. Update ready set based on completed instructions
    6. Terminator is always scheduled last in its own bundle
    """
    if not instructions and terminator is None:
        return []

    # Build dependency graph (excluding terminator)
    nodes = _build_dep_graph(instructions)

    bundles: list[MBundle] = []
    scheduled: set[int] = set()

    # Initialize ready set with instructions that have no predecessors
    ready: set[int] = {i for i, node in enumerate(nodes) if node.in_degree == 0}

    while len(scheduled) < len(instructions):
        if not ready:
            # This shouldn't happen if there are no cycles
            # But handle it gracefully by finding unscheduled nodes
            unscheduled = [i for i in range(len(nodes)) if i not in scheduled]
            if unscheduled:
                ready.add(unscheduled[0])
            else:
                break

        bundle = MBundle()

        # Try to add as many ready instructions as possible to the bundle
        # Prioritize by original order for determinism
        ready_list = sorted(ready)
        added = []

        for idx in ready_list:
            inst = nodes[idx].inst
            if bundle.add_instruction(inst):
                added.append(idx)

        # Update scheduled set and ready set
        for idx in added:
            ready.remove(idx)
            scheduled.add(idx)

            # Update successors' in-degrees
            for succ_idx in nodes[idx].succs:
                nodes[succ_idx].in_degree -= 1
                if nodes[succ_idx].in_degree == 0 and succ_idx not in scheduled:
                    ready.add(succ_idx)

        if bundle.instructions:
            bundles.append(bundle)

    # Add terminator in its own bundle
    if terminator is not None:
        term_inst = _lir_inst_to_machine_inst(terminator)
        term_bundle = MBundle()
        term_bundle.add_instruction(term_inst)
        bundles.append(term_bundle)

    return bundles


def _schedule_block(instructions: list[MachineInst], terminator: Optional[MachineInst],
                    enable_scheduling: bool = True) -> list[MBundle]:
    """Schedule instructions into bundles.

    Args:
        instructions: List of machine instructions to schedule
        terminator: Optional terminator instruction
        enable_scheduling: If True, use list scheduling to pack multiple
                          instructions per bundle. If False, one instruction
                          per bundle.
    """
    if enable_scheduling:
        return _schedule_block_with_packing(instructions, terminator)
    else:
        return _schedule_block_no_packing(instructions, terminator)


class LIRToMIRPass(LIRToMIRLoweringPass):
    """
    Pass that lowers LIR to MIR with optional instruction scheduling.

    Takes LIR after phi-elimination and produces MIR with instructions
    packed into VLIW bundles.

    Config options:
        enable_scheduling: If True (default), use list scheduling to pack
                          multiple instructions per bundle. If False, each
                          instruction gets its own bundle.
    """

    @property
    def name(self) -> str:
        return "lir-to-mir"

    def run(self, lir: LIRFunction, config: PassConfig) -> MachineFunction:
        """Lower LIR to MIR with optional instruction scheduling."""
        self._init_metrics()

        # Read config option
        enable_scheduling = config.options.get("enable_scheduling", True)

        mfunc = MachineFunction(entry=lir.entry, max_scratch_used=lir.max_scratch_used)

        # Process blocks in order
        block_order = _get_block_order(lir)

        for block_name in block_order:
            lir_block = lir.blocks[block_name]

            # Convert instructions to MachineInsts
            machine_insts = [_lir_inst_to_machine_inst(inst) for inst in lir_block.instructions]
            terminator = lir_block.terminator

            # Schedule into bundles
            bundles = _schedule_block(machine_insts, terminator, enable_scheduling)

            # Compute predecessors/successors
            successors = _get_successors(lir_block)
            predecessors: list[str] = []

            # Find predecessors by checking which blocks have this as a successor
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

        # Record metrics
        if self._metrics:
            total_bundles = mfunc.total_bundles()
            total_insts = mfunc.total_instructions()
            avg_insts_per_bundle = total_insts / total_bundles if total_bundles > 0 else 0

            # Count bundles by size (how many instructions per bundle)
            bundle_size_histogram: dict[int, int] = {}
            for block in mfunc.blocks.values():
                for bundle in block.bundles:
                    size = len(bundle.instructions)
                    bundle_size_histogram[size] = bundle_size_histogram.get(size, 0) + 1

            # Count multi-instruction bundles
            multi_inst_bundles = sum(count for size, count in bundle_size_histogram.items() if size > 1)

            # Calculate scheduling effectiveness
            # If no scheduling, bundles == instructions
            # Packing ratio = instructions / bundles (higher is better)
            packing_ratio = total_insts / total_bundles if total_bundles > 0 else 0

            self._metrics.custom = {
                "scheduling_enabled": enable_scheduling,
                "bundles": total_bundles,
                "instructions": total_insts,
                "avg_insts_per_bundle": round(avg_insts_per_bundle, 2),
                "multi_inst_bundles": multi_inst_bundles,
                "single_inst_bundles": bundle_size_histogram.get(1, 0),
                "packing_ratio": round(packing_ratio, 2),
            }

            # Add bundle size distribution to messages for detailed output
            if enable_scheduling:
                self._add_metric_message(f"Bundle size distribution: {dict(sorted(bundle_size_histogram.items()))}")

        return mfunc
