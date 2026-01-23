"""
Main Compilation Entry Point

Provides the compile_hir_to_vliw function that orchestrates the full
compilation pipeline from HIR to VLIW bundles using the CompilerPipeline.
"""

from typing import Optional

from .hir import HIRFunction
from .pass_manager import CompilerPipeline
from .passes import DCEPass, LoopUnrollPass, CSEPass, HIRToLIRPass, PhiEliminationPass, LIRToVLIWPass


def compile_hir_to_vliw(
    hir: HIRFunction,
    print_after_all: bool = False,
    config_path: Optional[str] = None,
    print_metrics: bool = False
) -> list[dict]:
    """
    Full compilation from HIR to VLIW with optional debug printing.

    Args:
        hir: The HIR function to compile
        print_after_all: If True, print IR after each compilation phase
        config_path: Optional path to JSON config file for pass options
        print_metrics: If True, print pass metrics and diagnostics

    Returns:
        List of VLIW instruction bundles
    """
    # Create pipeline with all passes in order
    pipeline = CompilerPipeline(
        print_after_all=print_after_all,
        print_metrics=print_metrics
    )

    # Register all passes in pipeline order
    pipeline.add_pass(DCEPass())             # HIR -> HIR (pre-unroll cleanup)
    pipeline.add_pass(LoopUnrollPass())      # HIR -> HIR
    pipeline.add_pass(CSEPass())             # HIR -> HIR
    pipeline.add_pass(DCEPass())             # HIR -> HIR (post-CSE cleanup)
    pipeline.add_pass(HIRToLIRPass())        # HIR -> LIR
    pipeline.add_pass(PhiEliminationPass())  # LIR -> LIR
    pipeline.add_pass(LIRToVLIWPass())       # LIR -> VLIW

    # Load config if provided
    if config_path:
        pipeline.load_config(config_path)

    # Run the full pipeline
    return pipeline.run(hir)
