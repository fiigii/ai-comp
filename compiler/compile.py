"""
Main Compilation Entry Point

Provides the compile_hir_to_vliw function that orchestrates the full
compilation pipeline from HIR to VLIW bundles using the CompilerPipeline.
"""

import json
import os

from .hir import HIRFunction
from .pass_manager import CompilerPipeline
from .passes import (
    DCEPass, LoopUnrollPass, CSEPass, SimplifyPass, HIRToLIRPass,
    SimplifyCFGPass, CopyPropagationPass, LIRDCEPass, PhiEliminationPass,
    SLPVectorizationPass, MADSynthesisPass,
    LIRToMIRPass, MIRRegisterAllocationPass, MIRToVLIWPass
)


def compile_hir_to_vliw(
    hir: HIRFunction,
    print_after_all: bool = False,
    print_metrics: bool = False,
    print_ddg_after_all: bool = False
) -> list[dict]:
    """
    Full compilation from HIR to VLIW with optional debug printing.

    Args:
        hir: The HIR function to compile
        print_after_all: If True, print IR after each compilation phase
        print_metrics: If True, print pass metrics and diagnostics
        print_ddg_after_all: If True, print DDGs after each compilation pass

    Returns:
        List of VLIW instruction bundles
    """
    # Load config from compiler/pass_config.json
    config_path = os.path.join(os.path.dirname(__file__), "pass_config.json")
    with open(config_path) as f:
        config_data = json.load(f)

    # Create pipeline with all passes in order
    pipeline = CompilerPipeline(
        print_after_all=print_after_all,
        print_metrics=print_metrics,
        print_ddg_after_all=print_ddg_after_all
    )
    pipeline.set_config(config_data)

    # Register all passes in pipeline order
    pipeline.add_pass(DCEPass())             # HIR -> HIR (pre-unroll cleanup)
    pipeline.add_pass(LoopUnrollPass())      # HIR -> HIR
    pipeline.add_pass(SimplifyPass())        # HIR -> HIR (constant fold & identities)
    pipeline.add_pass(DCEPass())             # HIR -> HIR (post-peephole cleanup)
    pipeline.add_pass(CSEPass())             # HIR -> HIR
    pipeline.add_pass(SLPVectorizationPass())  # HIR -> HIR (vectorization)
    pipeline.add_pass(CSEPass())             # HIR -> HIR (deduplicate SLP-generated broadcasts)
    pipeline.add_pass(MADSynthesisPass())    # HIR -> HIR (fuse v* + v+ into multiply_add)
    pipeline.add_pass(DCEPass())             # HIR -> HIR (pre-lowering cleanup)
    pipeline.add_pass(HIRToLIRPass())        # HIR -> LIR
    pipeline.add_pass(CopyPropagationPass()) # LIR -> LIR (propagate COPY sources)
    pipeline.add_pass(LIRDCEPass())          # LIR -> LIR (remove dead COPYs)
    pipeline.add_pass(SimplifyCFGPass())     # LIR -> LIR (CFG cleanup after DCE)
    pipeline.add_pass(PhiEliminationPass())  # LIR -> LIR
    pipeline.add_pass(LIRToMIRPass())        # LIR -> MIR (with instruction scheduling)
    pipeline.add_pass(MIRRegisterAllocationPass())  # MIR -> MIR
    pipeline.add_pass(MIRToVLIWPass())       # MIR -> VLIW

    # Run the full pipeline
    return pipeline.run(hir)
