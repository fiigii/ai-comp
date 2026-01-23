"""
HIR to LIR Lowering Pass

Wraps the lowering stage as a pass for the CompilerPipeline.
"""

from ..pass_manager import LoweringPass, PassConfig
from ..hir import HIRFunction
from ..lir import LIRFunction
from ..lowering import lower_to_lir


class HIRToLIRPass(LoweringPass):
    """
    Pass that lowers HIR to LIR.

    Converts high-level IR with loops and branches to low-level IR
    with basic blocks and explicit jumps.
    """

    @property
    def name(self) -> str:
        return "lowering"

    def run(self, hir: HIRFunction, config: PassConfig) -> LIRFunction:
        """Lower HIR to LIR."""
        self._init_metrics()

        lir = lower_to_lir(hir)

        # Record custom metrics
        if self._metrics:
            self._metrics.custom = {
                "blocks": len(lir.blocks),
                "max_scratch": lir.max_scratch_used,
            }

        return lir
