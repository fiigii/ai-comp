"""
LIR to MIR Lowering Pass

Converts LIR (after phi-elimination) to MIR without instruction scheduling.
Each instruction is placed in its own bundle, preserving program order.
"""

from __future__ import annotations

from typing import Optional

from ..pass_manager import LIRToMIRLoweringPass, PassConfig
from ..lir import LIRFunction
from ..mir import MachineInst, MBundle, MachineBasicBlock, MachineFunction
from .mir_lowering_utils import lir_inst_to_machine_inst, get_successors, get_block_order


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
        term_inst = lir_inst_to_machine_inst(terminator)
        term_bundle = MBundle()
        term_bundle.add_instruction(term_inst)
        bundles.append(term_bundle)

    return bundles


class LIRToMIRPass(LIRToMIRLoweringPass):
    """
    Pass that lowers LIR to MIR without instruction scheduling.

    Takes LIR after phi-elimination and produces MIR with one instruction
    per bundle (terminators in their own bundle).
    """

    @property
    def name(self) -> str:
        return "lir-to-mir"

    def run(self, lir: LIRFunction, config: PassConfig) -> MachineFunction:
        """Lower LIR to MIR without instruction scheduling."""
        self._init_metrics()

        mfunc = MachineFunction(entry=lir.entry, max_scratch_used=lir.max_scratch_used)

        # Process blocks in order
        block_order = get_block_order(lir)

        for block_name in block_order:
            lir_block = lir.blocks[block_name]

            # Convert instructions to MachineInsts
            machine_insts = [lir_inst_to_machine_inst(inst) for inst in lir_block.instructions]
            terminator = lir_block.terminator

            # Schedule into bundles (no packing)
            bundles = _schedule_block_no_packing(machine_insts, terminator)

            # Compute predecessors/successors
            successors = get_successors(lir_block)
            predecessors: list[str] = []

            # Find predecessors by checking which blocks have this as a successor
            for other_name, other_block in lir.blocks.items():
                if block_name in get_successors(other_block):
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

            self._metrics.custom = {
                "bundles": total_bundles,
                "instructions": total_insts,
                "avg_insts_per_bundle": round(avg_insts_per_bundle, 2),
                "multi_inst_bundles": multi_inst_bundles,
                "single_inst_bundles": bundle_size_histogram.get(1, 0),
                "packing_ratio": round(avg_insts_per_bundle, 2),
            }
            self._add_metric_message(f"Bundle size distribution: {dict(sorted(bundle_size_histogram.items()))}")

        return mfunc
