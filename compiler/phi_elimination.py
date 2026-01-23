"""
Phi Elimination

Replaces phi nodes with copies at the end of predecessor blocks.
Uses a parallel copy algorithm to handle cycles (e.g., swaps) correctly.
"""

from typing import Optional

from problem import SCRATCH_SIZE

from .lir import LIROpcode, LIRInst, LIRFunction


def _compute_parallel_copy_order(copies: list[tuple[int, int]], temp_scratch: int) -> list[tuple[int, int]]:
    """
    Compute a valid order for parallel copies that preserves semantics.

    Args:
        copies: List of (dest, src) pairs representing parallel copies
        temp_scratch: Scratch location to use for breaking cycles

    Returns:
        List of (dest, src) pairs in a valid sequential order
    """
    if not copies:
        return []

    # Build maps for analysis
    dest_to_src = {dest: src for dest, src in copies}
    src_to_dests = {}
    for dest, src in copies:
        if src not in src_to_dests:
            src_to_dests[src] = []
        src_to_dests[src].append(dest)

    result = []
    remaining = set(dest_to_src.keys())

    # Keep emitting safe copies until we can't anymore
    while remaining:
        # Find a copy whose dest is not a source for any remaining copy
        safe_dest = None
        for dest in remaining:
            # Check if this dest is used as a source by another remaining copy
            is_needed_as_source = False
            for other_dest in remaining:
                if other_dest != dest and dest_to_src[other_dest] == dest:
                    is_needed_as_source = True
                    break
            if not is_needed_as_source:
                safe_dest = dest
                break

        if safe_dest is not None:
            # Emit this copy
            result.append((safe_dest, dest_to_src[safe_dest]))
            remaining.remove(safe_dest)
        else:
            # All remaining copies form cycles - break one using temp
            # Pick any dest from remaining
            cycle_dest = next(iter(remaining))
            cycle_src = dest_to_src[cycle_dest]

            # Save the dest value (about to be overwritten) to temp first
            # This preserves the value for any other copy that reads from cycle_dest
            result.append((temp_scratch, cycle_dest))

            # Now we can safely overwrite cycle_dest
            result.append((cycle_dest, cycle_src))
            remaining.remove(cycle_dest)

            # Update any copy that used cycle_dest as source to use temp instead
            for dest in list(remaining):
                if dest_to_src[dest] == cycle_dest:
                    dest_to_src[dest] = temp_scratch

    return result


def eliminate_phis(lir: LIRFunction, temp_scratch: Optional[int] = None):
    """
    Replace phi nodes with copies at the end of predecessor blocks.

    Uses a parallel copy algorithm to handle cycles (e.g., swaps) correctly.
    This must be done before linearization since the machine doesn't have phi.

    Args:
        lir: The LIR function to transform
        temp_scratch: Scratch location for breaking cycles. If None, uses SCRATCH_SIZE-1.
    """
    if temp_scratch is None:
        temp_scratch = SCRATCH_SIZE - 1  # Reserve last scratch for temp

    # Group phis by predecessor block
    for block in lir.blocks.values():
        if not block.phis:
            continue

        # Collect copies per predecessor
        pred_copies: dict[str, list[tuple[int, int]]] = {}  # pred_name -> [(dest, src)]

        for phi in block.phis:
            for pred_name, src_scratch in phi.incoming.items():
                if pred_name not in pred_copies:
                    pred_copies[pred_name] = []
                pred_copies[pred_name].append((phi.dest, src_scratch))

        # For each predecessor, compute safe copy order and emit
        for pred_name, copies in pred_copies.items():
            pred_block = lir.blocks[pred_name]
            ordered_copies = _compute_parallel_copy_order(copies, temp_scratch)

            for dest, src in ordered_copies:
                if dest != src:  # Skip no-op copies
                    copy_inst = LIRInst(LIROpcode.COPY, dest, [src], "alu")
                    pred_block.instructions.append(copy_inst)

        block.phis = []
