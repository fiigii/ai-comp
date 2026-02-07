"""
Programs for the VLIW SIMD Virtual Machine

This module contains program implementations that compile and run on the VM.
It also provides shared CLI flag handling so every program supports the same
compiler/VM diagnostic flags.
"""

import argparse
import json
import os
import sys

# Ensure project root is on the path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .tree_hash import build_tree_hash_kernel

_ENGINE_PRINT_ORDER = ("load", "alu", "valu", "store", "flow")


def _ordered_vliw_bundle(bundle: dict) -> dict:
    ordered: dict = {}
    for engine in _ENGINE_PRINT_ORDER:
        if engine in bundle:
            ordered[engine] = bundle[engine]
    for engine in sorted(bundle.keys()):
        if engine not in ordered:
            ordered[engine] = bundle[engine]
    return ordered


def add_compiler_flags(parser: argparse.ArgumentParser) -> None:
    """Add common compiler diagnostic flags to an argument parser."""
    parser.add_argument("--print-vliw", action="store_true",
                        help="Print the final VLIW instructions")
    parser.add_argument("--print-after-all", action="store_true",
                        help="Print IR after each compilation pass")
    parser.add_argument("--print-metrics", action="store_true",
                        help="Print pass metrics and diagnostics")
    parser.add_argument("--print-ddg-after-all", action="store_true",
                        help="Print DDGs after each compilation pass")
    parser.add_argument("--trace", action="store_true",
                        help="Enable VM execution trace")


# The set of flag names added by add_compiler_flags, for has_custom_flag checks.
COMPILER_FLAGS = {
    '--print-vliw', '--print-after-all', '--print-metrics',
    '--print-ddg-after-all', '--trace',
}


def compiler_kwargs(args: argparse.Namespace) -> dict:
    """Extract compiler keyword args from parsed CLI args."""
    return {
        'print_after_all': args.print_after_all,
        'print_metrics': args.print_metrics,
        'print_ddg_after_all': args.print_ddg_after_all,
    }


def print_vliw(instrs: list[dict]) -> None:
    """Pretty-print VLIW instruction bundles."""
    for i, instr in enumerate(instrs):
        print(f"[{i:4d}] {json.dumps(_ordered_vliw_bundle(instr))}")


__all__ = [
    'build_tree_hash_kernel',
    'add_compiler_flags',
    'COMPILER_FLAGS',
    'compiler_kwargs',
    'print_vliw',
]
