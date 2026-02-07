"""
IR Compiler for VLIW SIMD Virtual Machine

Three-level IR:
- HIR (High-Level IR): SSA form with explicit loops and branches
- LIR (Low-Level IR): Basic blocks with jumps, close to machine code
- MIR (Machine IR): Bundled VLIW instructions after scheduling

Compilation pipeline: HIR -> LIR -> MIR -> VLIW assembly
"""

# HIR types
from .hir import (
    SSAValue,
    Const,
    VectorConst,
    Value,
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

# Use-def chain infrastructure
from .use_def import UseDefContext, DefLocation, UseLocation

# Alias analysis
from .alias_analysis import AliasAnalysis, AliasResult, AddrKey

# Data Dependency Graph
from .ddg import (
    DDGNode,
    DataDependencyDAG,
    BlockDDGs,
    DDGBuilder,
    HIRDDGBuilder,
    LIRDDGBuilder,
    get_dag_depth,
    get_dag_width,
    find_independent_nodes,
    print_dag,
    print_dag_tree,
    print_block_ddgs,
    print_dag_dot,
)

# Compilation stages
from .lowering import lower_to_lir
from .passes.phi_elimination import eliminate_phis

# Main entry point
from .compile import compile_hir_to_vliw

# Printing utilities
from .printing import print_hir, print_lir, print_vliw

# Passes
from .passes import DCEPass, LoopUnrollPass, CSEPass, SimplifyPass, HIRToLIRPass, SimplifyCFGPass, CopyPropagationPass, LIRDCEPass, PhiEliminationPass, LoadElimPass, DSEPass, TreeLevelCachePass


# Public API
def compile(hir, **kwargs):
    """Compile HIR program to VLIW instructions."""
    return compile_hir_to_vliw(hir, **kwargs)


def execute(instrs, mem, debug_info=None, n_cores=None, trace=False, **kwargs):
    """Execute VLIW instructions on the VM."""
    from vm import Machine, DebugInfo, N_CORES as DEFAULT_CORES
    if debug_info is None:
        debug_info = DebugInfo(scratch_map={})
    if n_cores is None:
        n_cores = DEFAULT_CORES
    machine = Machine(mem, instrs, debug_info, n_cores=n_cores, trace=trace, **kwargs)
    machine.run()
    return machine


__all__ = [
    # HIR
    'SSAValue', 'Const', 'VectorConst', 'Value', 'Op', 'Halt', 'Pause', 'ForLoop', 'If',
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
    # Use-def chain infrastructure
    'UseDefContext', 'DefLocation', 'UseLocation',
    # Alias analysis
    'AliasAnalysis', 'AliasResult', 'AddrKey',
    # Data Dependency Graph
    'DDGNode', 'DataDependencyDAG', 'BlockDDGs', 'DDGBuilder',
    'HIRDDGBuilder', 'LIRDDGBuilder',
    'get_dag_depth', 'get_dag_width', 'find_independent_nodes',
    'print_dag', 'print_dag_tree', 'print_block_ddgs', 'print_dag_dot',
    # Compilation
    'lower_to_lir', 'eliminate_phis', 'compile_hir_to_vliw',
    # Printing
    'print_hir', 'print_lir', 'print_vliw',
    # Passes
    'DCEPass', 'LoopUnrollPass', 'CSEPass', 'SimplifyPass', 'HIRToLIRPass', 'SimplifyCFGPass', 'CopyPropagationPass', 'LIRDCEPass', 'PhiEliminationPass', 'LoadElimPass', 'DSEPass', 'TreeLevelCachePass',
    # Public API
    'compile', 'execute',
]
