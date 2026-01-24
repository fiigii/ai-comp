"""
Compiler Passes

This module contains passes for the compilation pipeline:
- HIR optimization passes (e.g., loop unrolling)
- Lowering pass (HIR -> LIR)
- LIR transformation passes (e.g., phi elimination)
- Codegen pass (LIR -> VLIW)
"""

from .loop_unroll import LoopUnrollPass
from .dce import DCEPass
from .cse import CSEPass
from .simplify import SimplifyPass
from .lowering import HIRToLIRPass
from .simplify_cfg import SimplifyCFGPass
from .copy_prop import CopyPropagationPass
from .lir_dce import LIRDCEPass
from .phi_elimination import PhiEliminationPass
from .codegen import LIRToVLIWPass
from .vectorize import VectorizePass

__all__ = [
    'LoopUnrollPass',
    'DCEPass',
    'CSEPass',
    'VectorizePass',
    'SimplifyPass',
    'HIRToLIRPass',
    'SimplifyCFGPass',
    'CopyPropagationPass',
    'LIRDCEPass',
    'PhiEliminationPass',
    'LIRToVLIWPass',
]
