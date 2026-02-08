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
from .slp import SLPVectorizationPass
from .mad_synthesis import MADSynthesisPass
from .load_elim import LoadElimPass
from .dse import DSEPass
from .tree_level_cache import TreeLevelCachePass
from .lir_to_mir import LIRToMIRPass
from .inst_scheduling import InstSchedulingPass
from .mir_reg_pressure_profiler import MIRRegPressureProfilerPass
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
    'SLPVectorizationPass',
    'MADSynthesisPass',
    'LoadElimPass',
    'DSEPass',
    'TreeLevelCachePass',
    'LIRToMIRPass',
    'InstSchedulingPass',
    'MIRRegPressureProfilerPass',
    'MIRRegisterAllocationPass',
    'MIRToVLIWPass',
]
