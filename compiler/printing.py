"""
IR Printing Utilities

Pretty-printing functions for HIR, LIR, and VLIW instructions.
"""

from .hir import HIRFunction, Statement, Op, ForLoop, If, Halt, Pause
from .lir import LIRFunction

_ENGINE_PRINT_ORDER = ("load", "alu", "valu", "store", "flow")


def _engine_sort_key(engine: str) -> tuple[int, str]:
    """Sort known engines by canonical order, then unknown engines by name."""
    try:
        return (_ENGINE_PRINT_ORDER.index(engine), engine)
    except ValueError:
        return (len(_ENGINE_PRINT_ORDER), engine)


def _ordered_vliw_bundle(bundle: dict) -> dict:
    """Return bundle with deterministic engine-key ordering."""
    ordered: dict = {}
    for engine in _ENGINE_PRINT_ORDER:
        if engine in bundle:
            ordered[engine] = bundle[engine]
    for engine in sorted(bundle.keys()):
        if engine not in ordered:
            ordered[engine] = bundle[engine]
    return ordered


def print_hir(hir: HIRFunction, indent: int = 0):
    """Pretty-print HIR function."""
    prefix = "  " * indent
    print(f"{prefix}=== HIR: {hir.name} ({hir.num_ssa_values} SSA values) ===")
    for stmt in hir.body:
        _print_hir_stmt(stmt, indent)
    print()


def _print_hir_stmt(stmt: Statement, indent: int = 0):
    """Pretty-print a single HIR statement."""
    prefix = "  " * indent
    if isinstance(stmt, Op):
        print(f"{prefix}{stmt}")
    elif isinstance(stmt, ForLoop):
        print(f"{prefix}for {stmt.counter} in range({stmt.start}, {stmt.end}):")
        if stmt.iter_args:
            print(f"{prefix}  iter_args: {stmt.iter_args}")
            print(f"{prefix}  body_params: {stmt.body_params}")
        for s in stmt.body:
            _print_hir_stmt(s, indent + 1)
        if stmt.yields:
            print(f"{prefix}  yields: {stmt.yields}")
        if stmt.results:
            print(f"{prefix}  results: {stmt.results}")
    elif isinstance(stmt, If):
        print(f"{prefix}if {stmt.cond}:")
        for s in stmt.then_body:
            _print_hir_stmt(s, indent + 1)
        if stmt.then_yields:
            print(f"{prefix}  then_yields: {stmt.then_yields}")
        if stmt.else_body:
            print(f"{prefix}else:")
            for s in stmt.else_body:
                _print_hir_stmt(s, indent + 1)
            if stmt.else_yields:
                print(f"{prefix}  else_yields: {stmt.else_yields}")
        if stmt.results:
            print(f"{prefix}results: {stmt.results}")
    elif isinstance(stmt, Halt):
        print(f"{prefix}halt")
    elif isinstance(stmt, Pause):
        print(f"{prefix}pause")
    else:
        print(f"{prefix}{stmt}")


def print_lir(lir: LIRFunction):
    """Pretty-print LIR function."""
    print(f"=== LIR (entry: {lir.entry}) ===")
    for name, block in lir.blocks.items():
        print(f"\n{name}:")
        if block.phis:
            for phi in block.phis:
                print(f"  {phi}")
        for inst in block.instructions:
            print(f"  {inst}")
        if block.terminator:
            print(f"  {block.terminator}")
    print()


def print_vliw(bundles: list[dict]):
    """Pretty-print VLIW bundles."""
    print(f"=== VLIW ({len(bundles)} bundles) ===")
    for i, bundle in enumerate(bundles):
        print(f"[{i:4d}] {_ordered_vliw_bundle(bundle)}")
    print()


def print_mir(mfunc):
    """Pretty-print MachineFunction.

    Format:
    [bundle_num] { (0) inst1,
                   (1) inst2,
                   ...
                 }
    """
    from .mir import MachineFunction

    print(f"=== MIR (entry: {mfunc.entry}, {mfunc.total_bundles()} bundles, {mfunc.total_instructions()} insts) ===")

    for name in mfunc.get_block_order():
        block = mfunc.blocks[name]
        pred_str = f" <- {', '.join(block.predecessors)}" if block.predecessors else ""
        succ_str = f" -> {', '.join(block.successors)}" if block.successors else ""
        print(f"\n{name}:{pred_str}{succ_str}")

        for bundle_idx, bundle in enumerate(block.bundles):
            sorted_insts = [
                inst
                for _, inst in sorted(
                    enumerate(bundle.instructions),
                    key=lambda pair: (_engine_sort_key(pair[1].engine), pair[0]),
                )
            ]
            num_insts = len(sorted_insts)
            bundle_prefix = f"  [{bundle_idx:4d}]"

            if num_insts == 0:
                print(f"{bundle_prefix} {{ <empty> }}")
            elif num_insts == 1:
                # Single instruction - compact format
                inst = sorted_insts[0]
                print(f"{bundle_prefix} {{ (0) {inst} }}")
            else:
                # Multiple instructions - multi-line format
                print(f"{bundle_prefix} {{ (0) {sorted_insts[0]},")
                indent = " " * (len(bundle_prefix) + 3)  # Align with first instruction
                for inst_idx in range(1, num_insts):
                    inst = sorted_insts[inst_idx]
                    if inst_idx == num_insts - 1:
                        # Last instruction - no comma, close brace
                        print(f"{indent}({inst_idx}) {inst} }}")
                    else:
                        print(f"{indent}({inst_idx}) {inst},")

    print()
