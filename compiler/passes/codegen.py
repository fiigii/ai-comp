"""
LIR to VLIW Codegen Pass

Wraps the codegen stage as a pass for the CompilerPipeline.
"""

from ..pass_manager import CodegenPass, PassConfig
from ..lir import LIRFunction
from ..codegen import compile_to_vliw


class LIRToVLIWPass(CodegenPass):
    """
    Pass that generates VLIW bundles from LIR.

    Linearizes the LIR control flow graph and generates VLIW instruction
    bundles. Currently uses a simple strategy of one instruction per bundle.

    Note: Phis must be eliminated before this pass runs.
    """

    @property
    def name(self) -> str:
        return "codegen"

    def run(self, lir: LIRFunction, config: PassConfig) -> list[dict]:
        """Generate VLIW bundles from LIR."""
        self._init_metrics()

        bundles = compile_to_vliw(lir)

        # Record custom metrics
        if self._metrics:
            self._metrics.custom = {
                "bundles": len(bundles),
            }

        return bundles
