"""
IR Compiler for VLIW SIMD Virtual Machine

Two-level IR:
- HIR (High-Level IR): SSA form with explicit loops and branches
- LIR (Low-Level IR): Basic blocks with jumps, close to machine code

Compilation pipeline: HIR -> LIR -> VLIW assembly
"""

# HIR types
from .hir import (
    SSAValue,
    Const,
    Operand,
    Op,
    Halt,
    Pause,
    ForLoop,
    If,
    Statement,
    HIRFunction,
)

# HIR builder
from .hir_builder import HIRBuilder

# LIR types
from .lir import (
    LIROpcode,
    LIRInst,
    Phi,
    BasicBlock,
    LIRFunction,
)

# Pass infrastructure
from .pass_manager import (
    PassConfig,
    PassMetrics,
    CompilerPass,
    Pass,
    HIRPass,
    LoweringPass,
    LIRPass,
    CodegenPass,
    PassManager,
    CompilerPipeline,
    count_statements,
    count_lir_instructions,
    count_lir_phis,
)

# SSA utilities
from .ssa_context import SSARenumberContext

# Compilation stages
from .lowering import lower_to_lir
from .phi_elimination import eliminate_phis
from .codegen import compile_to_vliw

# Main entry point
from .compile import compile_hir_to_vliw

# Printing utilities
from .printing import print_hir, print_lir, print_vliw

# Passes
from .passes import DCEPass, LoopUnrollPass, CSEPass, VectorizePass, SimplifyPass, HIRToLIRPass, SimplifyCFGPass, CopyPropagationPass, LIRDCEPass, PhiEliminationPass, LIRToVLIWPass

__all__ = [
    # HIR
    'SSAValue', 'Const', 'Operand', 'Op', 'Halt', 'Pause', 'ForLoop', 'If',
    'Statement', 'HIRFunction',
    # Builder
    'HIRBuilder',
    # LIR
    'LIROpcode', 'LIRInst', 'Phi', 'BasicBlock', 'LIRFunction',
    # Pass infrastructure
    'PassConfig', 'PassMetrics', 'CompilerPass', 'Pass', 'HIRPass',
    'LoweringPass', 'LIRPass', 'CodegenPass', 'PassManager', 'CompilerPipeline',
    'count_statements', 'count_lir_instructions', 'count_lir_phis',
    # SSA utilities
    'SSARenumberContext',
    # Compilation
    'lower_to_lir', 'eliminate_phis', 'compile_to_vliw', 'compile_hir_to_vliw',
    # Printing
    'print_hir', 'print_lir', 'print_vliw',
    # Passes
    'DCEPass', 'LoopUnrollPass', 'CSEPass', 'VectorizePass', 'SimplifyPass', 'HIRToLIRPass', 'SimplifyCFGPass', 'CopyPropagationPass', 'LIRDCEPass', 'PhiEliminationPass', 'LIRToVLIWPass',
]
