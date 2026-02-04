"""Shared fixtures and imports for compiler tests."""

import os
import sys

# Add parent directories to path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_this_dir))
sys.path.insert(0, _repo_root)

import unittest
import random

from problem import (
    Machine,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    N_CORES,
    DebugInfo,
)
from perf_takehome import KernelBuilder
from compiler import (
    HIRBuilder,
    LIRFunction,
    BasicBlock,
    LIRInst,
    LIROpcode,
    Phi,
    compile_hir_to_vliw,
    lower_to_lir,
    eliminate_phis,
)
from compiler.passes import InstSchedulingPass, MIRRegisterAllocationPass, MIRToVLIWPass
from compiler.pass_manager import PassConfig


def _cfg(name, **opts):
    """Helper to create PassConfig."""
    return PassConfig(name=name, enabled=True, options=opts)


def compile_to_vliw(lir: LIRFunction) -> list[dict]:
    """Compile LIR to VLIW through the MIR pipeline.

    Note: Phis must be eliminated before calling this function.
    """
    mir = InstSchedulingPass().run(lir, _cfg('inst-scheduling'))
    mir = MIRRegisterAllocationPass().run(mir, _cfg('mir-regalloc'))
    bundles = MIRToVLIWPass().run(mir, _cfg('mir-codegen'))
    return bundles


def run_program(instrs, mem):
    """Helper to run a compiled program."""
    machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine
