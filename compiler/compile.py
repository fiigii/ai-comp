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
    SLPVectorizationPass, MADSynthesisPass, LoadElimPass, DSEPass,
    TreeLevelCachePass,
    LIRToMIRPass, InstSchedulingPass, MIRRegPressureProfilerPass,
    MIRRegisterAllocationPass, MIRToVLIWPass
)

PASS_REGISTRY = {
    "dce": DCEPass,
    "loop-unroll": LoopUnrollPass,
    "simplify": SimplifyPass,
    "cse": CSEPass,
    "load-elim": LoadElimPass,
    "dse": DSEPass,
    "tree-level-cache": TreeLevelCachePass,
    "slp-vectorization": SLPVectorizationPass,
    "mad-synthesis": MADSynthesisPass,
    "hir-to-lir": HIRToLIRPass,
    "copy-propagation": CopyPropagationPass,
    "lir-dce": LIRDCEPass,
    "simplify-cfg": SimplifyCFGPass,
    "phi-elimination": PhiEliminationPass,
    "inst-scheduling": InstSchedulingPass,
    "lir-to-mir": LIRToMIRPass,
    "mir-reg-pressure-profiler": MIRRegPressureProfilerPass,
    "mir-register-allocation": MIRRegisterAllocationPass,
    "mir-to-vliw": MIRToVLIWPass,
}


def compile_hir_to_vliw(
    hir: HIRFunction,
    print_after_all: bool = False,
    print_metrics: bool = False,
    print_ddg_after_all: bool = False,
    profile_reg_pressure: bool = False,
) -> list[dict]:
    """
    Full compilation from HIR to VLIW with optional debug printing.

    Args:
        hir: The HIR function to compile
        print_after_all: If True, print IR after each compilation phase
        print_metrics: If True, print pass metrics and diagnostics
        print_ddg_after_all: If True, print DDGs after each compilation pass
        profile_reg_pressure: If True, run register pressure profiler and emit HTML chart

    Returns:
        List of VLIW instruction bundles
    """
    # Load config from compiler/pass_config.json
    config_path = os.path.join(os.path.dirname(__file__), "pass_config.json")
    with open(config_path) as f:
        config_data = json.load(f)

    passes_cfg = config_data.get("passes", {})
    inst_sched_enabled = passes_cfg.get("inst-scheduling", {}).get("enabled", False)

    # Enforce mutual exclusion: exactly one LIR -> MIR lowering pass enabled
    passes_cfg.setdefault("lir-to-mir", {})
    passes_cfg["lir-to-mir"]["enabled"] = not inst_sched_enabled

    # Control mir-reg-pressure-profiler via the profile_reg_pressure flag
    passes_cfg.setdefault("mir-reg-pressure-profiler", {})
    passes_cfg["mir-reg-pressure-profiler"]["enabled"] = profile_reg_pressure

    # Create pipeline with all passes in order
    pipeline = CompilerPipeline(
        print_after_all=print_after_all,
        print_metrics=print_metrics,
        print_ddg_after_all=print_ddg_after_all
    )
    pipeline.set_config(config_data)

    # Build pipeline from config-driven pass list
    pipeline_order = config_data.get("pipeline")
    if pipeline_order is None:
        raise ValueError("pass_config.json missing required 'pipeline' key")

    for pass_name in pipeline_order:
        pass_cls = PASS_REGISTRY.get(pass_name)
        if pass_cls is None:
            raise ValueError(
                f"Unknown pass '{pass_name}' in pipeline config. "
                f"Known passes: {', '.join(sorted(PASS_REGISTRY))}"
            )
        pipeline.add_pass(pass_cls())

    # Run the full pipeline
    return pipeline.run(hir)
