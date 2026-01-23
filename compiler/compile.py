"""
Main Compilation Entry Point

Provides the compile_hir_to_vliw function that orchestrates the full
compilation pipeline from HIR to VLIW bundles.
"""

from typing import Optional

from problem import SCRATCH_SIZE

from .hir import HIRFunction
from .pass_manager import PassManager
from .passes.loop_unroll import LoopUnrollPass
from .lowering import lower_to_lir
from .phi_elimination import eliminate_phis
from .codegen import compile_to_vliw
from .printing import print_hir, print_lir, print_vliw


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
    """
    if print_after_all:
        print("\n" + "=" * 60)
        print("COMPILATION START")
        print("=" * 60)
        print_hir(hir)

    # Phase 0: Run HIR optimization passes
    pm = PassManager(print_after_all=print_after_all, print_metrics=print_metrics)
    pm.add_pass(LoopUnrollPass())

    if config_path:
        pm.load_config(config_path)

    hir = pm.run(hir)

    if print_after_all and pm.passes:
        print("-" * 60)
        print("After HIR passes:")
        print("-" * 60)
        print_hir(hir)

    # Phase 1: Lower HIR to LIR
    lir = lower_to_lir(hir)
    if print_after_all:
        print("-" * 60)
        print("After HIR -> LIR lowering:")
        print("-" * 60)
        print_lir(lir)

    # Phase 2: Eliminate phis
    # Use scratch slot after all allocated ones for phi temp (avoids collision)
    temp_scratch = lir.max_scratch_used + 1 if lir.max_scratch_used >= 0 else 0
    assert temp_scratch < SCRATCH_SIZE, "No room for phi temp scratch"
    eliminate_phis(lir, temp_scratch=temp_scratch)
    if print_after_all:
        print("-" * 60)
        print("After phi elimination:")
        print("-" * 60)
        print_lir(lir)

    # Phase 3: Compile to VLIW
    bundles = compile_to_vliw(lir)
    if print_after_all:
        print("-" * 60)
        print("After LIR -> VLIW codegen:")
        print("-" * 60)
        print_vliw(bundles)
        print("=" * 60)
        print("COMPILATION END")
        print("=" * 60 + "\n")

    return bundles
