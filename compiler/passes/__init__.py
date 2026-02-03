"""
Compiler Passes

This module contains passes for the compilation pipeline:
- HIR optimization passes (e.g., loop unrolling)
- Lowering pass (HIR -> LIR)
- LIR transformation passes (e.g., phi elimination)
- MIR passes (instruction scheduling, register allocation)
- Codegen pass (MIR -> VLIW)
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
from .register_allocation import RegisterAllocationPass
from .codegen import LIRToVLIWPass
from .slp import SLPVectorizationPass
from .mad_synthesis import MADSynthesisPass
from .lir_to_mir import LIRToMIRPass
from .mir_register_allocation import MIRRegisterAllocationPass
from .mir_codegen import MIRToVLIWPass

__all__ = [
    'LoopUnrollPass',
    'DCEPass',
    'CSEPass',
    'SimplifyPass',
    'HIRToLIRPass',
    'SimplifyCFGPass',
    'CopyPropagationPass',
    'LIRDCEPass',
    'PhiEliminationPass',
    'RegisterAllocationPass',
    'LIRToVLIWPass',
    'SLPVectorizationPass',
    'MADSynthesisPass',
    'LIRToMIRPass',
    'MIRRegisterAllocationPass',
    'MIRToVLIWPass',
]
