"""
Phi Elimination Pass

Wraps the phi elimination stage as a pass for the CompilerPipeline.
"""

from problem import SCRATCH_SIZE

from ..pass_manager import LIRPass, PassConfig
from ..lir import LIRFunction
from ..phi_elimination import eliminate_phis


class PhiEliminationPass(LIRPass):
    """
    Pass that eliminates phi nodes from LIR.

    Replaces phi nodes with copies at the end of predecessor blocks.
    Uses a parallel copy algorithm to handle cycles (e.g., swaps) correctly.

    Options:
        temp_scratch: Scratch location for breaking cycles.
                      If not specified, uses max_scratch_used + 1.
    """

    @property
    def name(self) -> str:
        return "phi-elimination"

    def run(self, lir: LIRFunction, config: PassConfig) -> LIRFunction:
        """Eliminate phi nodes from LIR."""
        self._init_metrics()

        # Count phis before elimination
        phi_count = sum(len(b.phis) for b in lir.blocks.values())

        # Determine temp scratch location
        temp_scratch = config.options.get("temp_scratch", None)
        if temp_scratch is None:
            temp_scratch = lir.max_scratch_used + 1 if lir.max_scratch_used >= 0 else 0
        assert temp_scratch < SCRATCH_SIZE, "No room for phi temp scratch"

        eliminate_phis(lir, temp_scratch=temp_scratch)

        # Record custom metrics
        if self._metrics:
            self._metrics.custom = {
                "phis_eliminated": phi_count,
                "temp_scratch": temp_scratch,
            }

        return lir
