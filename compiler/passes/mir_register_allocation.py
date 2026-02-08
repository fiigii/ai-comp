"""
MIR Register Allocation Pass

Implements linear scan register allocation on MIR with bundle-aware liveness.
Each bundle gets a unique index since bundles are atomic scheduling units.
Supports register spilling to memory when scratch registers are exhausted.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Optional

from vm import SCRATCH_SIZE, VLEN

from ..pass_manager import MIRPass, PassConfig
from ..lir import LIROpcode
from ..mir import MachineFunction, MachineBasicBlock, MBundle, MachineInst


@dataclass
class LivenessInfo:
    """Liveness information for a basic block (bundle-based)."""
    live_in: set[int] = field(default_factory=set)
    live_out: set[int] = field(default_factory=set)
    gen: set[int] = field(default_factory=set)
    kill: set[int] = field(default_factory=set)


@dataclass
class LiveInterval:
    """Live interval for a virtual register (bundle-point indexed)."""
    vreg: int              # Virtual register (scratch address)
    start: int             # First point where live
    end: int               # Last point where live
    is_vector: bool        # Whether this is a vector register (8 contiguous)


@dataclass
class SpillInfo:
    """Information about a spilled interval."""
    vreg: int              # Virtual register that was spilled
    mem_offset: int        # Offset from spill base address in memory
    is_vector: bool        # Whether this is a vector spill
    interval: LiveInterval # The original live interval


def _is_contiguous_vector_list(op: list) -> bool:
    """Check if a list operand is a contiguous vector (base, base+1, ..., base+7)."""
    if not op or len(op) != VLEN:
        return False
    if not isinstance(op[0], int):
        return False
    base = op[0]
    return all(isinstance(op[i], int) and op[i] == base + i for i in range(VLEN))


def _collect_bundle_uses(bundle: MBundle) -> set[int]:
    """Collect all scratch addresses used by a bundle."""
    uses: set[int] = set()
    for inst in bundle.instructions:
        uses.update(inst.get_uses())
    return uses


def _collect_bundle_defs(bundle: MBundle) -> set[int]:
    """Collect all scratch addresses defined by a bundle."""
    defs: set[int] = set()
    for inst in bundle.instructions:
        defs.update(inst.get_defs())
    return defs


def _detect_vector_bases(mfunc: MachineFunction) -> tuple[set[int], set[int]]:
    """Detect all vector base addresses and the full set of vector addresses.

    Returns:
        (vector_bases, vector_addrs) where:
        - vector_bases: set of base addresses for vectors
        - vector_addrs: set of all addresses that are part of a vector
    """
    vector_bases: set[int] = set()
    vector_addrs: set[int] = set()

    # First pass: detect vector destinations
    for block in mfunc.blocks.values():
        for bundle in block.bundles:
            for inst in bundle.instructions:
                if inst.dest is not None and isinstance(inst.dest, list) and inst.dest:
                    base = inst.dest[0]
                    if isinstance(base, int):
                        vector_bases.add(base)
                        for i in range(len(inst.dest)):
                            vector_addrs.add(base + i)

    # Detect LOAD_OFFSET patterns (vgather)
    load_offset_dests: dict[int, set[int]] = {}
    load_offset_addrs: dict[int, set[int]] = {}
    for block in mfunc.blocks.values():
        for bundle in block.bundles:
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.LOAD_OFFSET:
                    dest = inst.dest
                    offset = inst.operands[1] if len(inst.operands) > 1 else 0
                    if isinstance(dest, int) and isinstance(offset, int):
                        if dest not in load_offset_dests:
                            load_offset_dests[dest] = set()
                        load_offset_dests[dest].add(offset)

                        # Also track address vectors
                        addr = inst.operands[0]
                        if isinstance(addr, int):
                            if addr not in load_offset_addrs:
                                load_offset_addrs[addr] = set()
                            load_offset_addrs[addr].add(offset)

    for dest, offsets in load_offset_dests.items():
        if offsets == set(range(VLEN)):
            vector_bases.add(dest)
            for i in range(VLEN):
                vector_addrs.add(dest + i)

    for addr, offsets in load_offset_addrs.items():
        if offsets == set(range(VLEN)) and addr not in vector_bases:
            vector_bases.add(addr)
            for i in range(VLEN):
                vector_addrs.add(addr + i)

    # Detect constructed vectors in operands
    for block in mfunc.blocks.values():
        for bundle in block.bundles:
            for inst in bundle.instructions:
                for op in inst.operands:
                    if isinstance(op, list) and _is_contiguous_vector_list(op):
                        base = op[0]
                        if base not in vector_bases:
                            vector_bases.add(base)
                            for i in range(VLEN):
                                vector_addrs.add(base + i)

    return vector_bases, vector_addrs


def _compute_liveness(mfunc: MachineFunction, vector_bases: set[int],
                      vector_addrs: set[int]) -> dict[str, LivenessInfo]:
    """Compute bundle-based liveness for all blocks."""
    # Build addr_to_base map for efficient base lookup
    addr_to_base: dict[int, int] = {}
    for base in vector_bases:
        for i in range(VLEN):
            addr_to_base[base + i] = base

    def normalize_addr(addr: int) -> int:
        """Normalize a scratch address to its base (for vectors)."""
        if addr in vector_addrs and addr not in vector_bases:
            return addr_to_base.get(addr, addr)
        return addr

    liveness: dict[str, LivenessInfo] = {}

    # Initialize liveness info for each block
    for name, block in mfunc.blocks.items():
        info = LivenessInfo()

        # Process bundles in forward order
        for bundle in block.bundles:
            uses = _collect_bundle_uses(bundle)
            defs = _collect_bundle_defs(bundle)

            # Normalize uses/defs to handle vector elements
            normalized_uses: set[int] = set()
            for u in uses:
                nu = normalize_addr(u)
                if nu not in info.kill:
                    normalized_uses.add(nu)
            info.gen.update(normalized_uses)

            for d in defs:
                info.kill.add(normalize_addr(d))

        liveness[name] = info

    # Iterate until fixed point
    block_order = mfunc.get_block_order()
    changed = True
    while changed:
        changed = False

        for name in reversed(block_order):
            block = mfunc.blocks[name]
            info = liveness[name]

            # live_out = union of live_in of all successors
            new_live_out: set[int] = set()
            for succ_name in block.successors:
                if succ_name in liveness:
                    new_live_out |= liveness[succ_name].live_in

            # live_in = gen | (live_out - kill)
            new_live_in = info.gen | (new_live_out - info.kill)

            if new_live_in != info.live_in or new_live_out != info.live_out:
                changed = True
                info.live_in = new_live_in
                info.live_out = new_live_out

    return liveness


def _build_live_intervals(mfunc: MachineFunction, liveness: dict[str, LivenessInfo],
                          vector_bases: set[int], vector_addrs: set[int]) -> list[LiveInterval]:
    """Build live intervals for all virtual registers using bundle points.

    Each bundle contributes two timeline points:
    - use point: reads happen from pre-bundle state
    - def point: writes commit at bundle end

    This allows an interval that ends on a bundle use point to free before
    another interval that starts at the same bundle's def point.
    """
    # Build addr_to_base map
    addr_to_base: dict[int, int] = {}
    for base in vector_bases:
        for i in range(VLEN):
            addr_to_base[base + i] = base

    # Linearize blocks and number bundles globally
    block_order = mfunc.get_block_order()

    # Map block name to (start_idx, end_idx) in global numbering
    block_ranges: dict[str, tuple[int, int]] = {}
    bundle_idx = 0

    for name in block_order:
        block = mfunc.blocks[name]
        start = bundle_idx
        count = len(block.bundles)
        if count == 0:
            count = 1  # Empty block still occupies one position
        end = bundle_idx + count - 1
        block_ranges[name] = (start, end)
        bundle_idx = end + 1

    # Track live ranges: vreg -> (start, end, is_vector)
    intervals: dict[int, tuple[int, int, bool]] = {}

    def update_interval(vreg: int, idx: int, is_vector: bool):
        if vreg in intervals:
            old_start, old_end, old_is_vec = intervals[vreg]
            intervals[vreg] = (min(old_start, idx), max(old_end, idx), old_is_vec or is_vector)
        else:
            intervals[vreg] = (idx, idx, is_vector)

    def use_point(bundle_idx: int) -> int:
        return bundle_idx * 2

    def def_point(bundle_idx: int) -> int:
        return bundle_idx * 2 + 1

    # Process each block
    for name in block_order:
        block = mfunc.blocks[name]
        block_start, block_end = block_ranges[name]
        info = liveness[name]

        # Variables live at block entry extend from block start
        for vreg in info.live_in:
            is_vec = vreg in vector_bases
            update_interval(vreg, use_point(block_start), is_vec)

        # Variables live at block exit extend to block end
        for vreg in info.live_out:
            is_vec = vreg in vector_bases
            update_interval(vreg, def_point(block_end), is_vec)

        # Process bundles
        idx = block_start
        for bundle in block.bundles:
            # Process uses first (reads pre-bundle state)
            for inst in bundle.instructions:
                uses = inst.get_uses()
                for u in uses:
                    if u in vector_addrs:
                        if u in vector_bases:
                            update_interval(u, use_point(idx), True)
                        else:
                            # Map to base
                            base = addr_to_base.get(u)
                            if base is not None:
                                update_interval(base, use_point(idx), True)
                    else:
                        update_interval(u, use_point(idx), False)

            # Process defs second (writes at bundle end)
            for inst in bundle.instructions:
                defs = inst.get_defs()
                for d in defs:
                    if d in vector_addrs:
                        if d in vector_bases:
                            update_interval(d, def_point(idx), True)
                        # Skip non-base vector elements
                    else:
                        update_interval(d, def_point(idx), False)

            idx += 1

    # Convert to LiveInterval objects
    result = []
    for vreg, (start, end, is_vector) in intervals.items():
        result.append(LiveInterval(vreg, start, end, is_vector))

    # Sort by start position
    result.sort(key=lambda x: (x.start, -x.end))

    return result


def _linear_scan(intervals: list[LiveInterval]) -> tuple[dict[int, int], int]:
    """Linear scan register allocation (no spilling).

    Args:
        intervals: List of live intervals sorted by start position

    Returns:
        (allocation, max_preg_used) where:
        - allocation: maps virtual register to physical register
        - max_preg_used: highest physical register used
    """
    allocation, max_preg_used, _ = _linear_scan_with_spill(intervals, reg_limit=-1)
    return allocation, max_preg_used


def _linear_scan_with_spill(
    intervals: list[LiveInterval],
    reg_limit: int = -1,
    use_counts: dict[int, int] | None = None,
) -> tuple[dict[int, int], int, list[SpillInfo]]:
    """Linear scan register allocation with spill support.

    When reg_limit > 0 and allocation would exceed it, evicts the active
    interval with the fewest uses (lowest reload cost) and marks it as spilled.
    The caller must insert spill store/load code around def/use sites.

    Args:
        intervals: List of live intervals sorted by start position
        reg_limit: Max physical registers to use (-1 = unlimited)
        use_counts: Optional map of vreg -> total use count for spill cost estimation

    Returns:
        (allocation, max_preg_used, spills) where:
        - allocation: maps virtual register to physical register
        - max_preg_used: highest physical register used
        - spills: list of SpillInfo for spilled intervals
    """
    allocation: dict[int, int] = {}
    free_ranges: list[tuple[int, int]] = []
    next_preg = 0
    active: list[LiveInterval] = []
    spills: list[SpillInfo] = []
    spill_offset = 0

    def _spill_cost(iv: LiveInterval) -> int:
        """Estimate the cost of spilling this interval (lower = better to spill)."""
        if use_counts is None:
            # Fall back to span-based heuristic (prefer spilling long-lived)
            return -(iv.end - iv.start)
        # Each use requires ~3 bundles for reload. Prefer spilling fewer-use values.
        return use_counts.get(iv.vreg, 0)

    for interval in intervals:
        # Expire old intervals
        new_active = []
        for a in active:
            if a.end < interval.start:
                if a.vreg in allocation:
                    preg = allocation[a.vreg]
                    size = VLEN if a.is_vector else 1
                    _add_free_range(free_ranges, preg, preg + size - 1)
            else:
                new_active.append(a)
        active = new_active

        size = VLEN if interval.is_vector else 1
        preg = _alloc_from_free(free_ranges, size)

        if preg is None:
            if reg_limit > 0 and next_preg + size > reg_limit:
                # Would exceed limit. Must spill to free a register.
                # Pick the active interval with the LOWEST spill cost (fewest uses).
                best_vi = None
                best_cost = None
                for vi in range(len(active)):
                    candidate = active[vi]
                    if candidate.end <= interval.end:
                        continue  # Current interval lives longer
                    cand_size = VLEN if candidate.is_vector else 1
                    if cand_size < size:
                        continue  # Too small
                    cost = _spill_cost(candidate)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_vi = vi

                if best_vi is not None:
                    victim = active.pop(best_vi)
                    victim_preg = allocation[victim.vreg]
                    del allocation[victim.vreg]
                    cand_size = VLEN if victim.is_vector else 1

                    spills.append(SpillInfo(
                        vreg=victim.vreg,
                        mem_offset=spill_offset,
                        is_vector=victim.is_vector,
                        interval=victim,
                    ))
                    spill_offset += VLEN if victim.is_vector else 1

                    _add_free_range(free_ranges, victim_preg, victim_preg + cand_size - 1)
                    preg = _alloc_from_free(free_ranges, size)
                    if preg is None:
                        preg = next_preg
                        next_preg += size
                else:
                    # No suitable victim found — spill current interval
                    spills.append(SpillInfo(
                        vreg=interval.vreg,
                        mem_offset=spill_offset,
                        is_vector=interval.is_vector,
                        interval=interval,
                    ))
                    spill_offset += VLEN if interval.is_vector else 1
                    continue
            else:
                preg = next_preg
                next_preg += size

        allocation[interval.vreg] = preg
        bisect.insort_right(active, interval, key=lambda x: x.end)

    max_preg_used = next_preg - 1 if next_preg > 0 else 0
    return allocation, max_preg_used, spills


def _add_free_range(free_ranges: list[tuple[int, int]], start: int, end: int) -> None:
    """Add a free range and merge overlaps/adjacent ranges."""
    if start > end:
        return
    free_ranges.append((start, end))
    free_ranges.sort()
    merged: list[tuple[int, int]] = []
    for s, e in free_ranges:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    free_ranges[:] = merged


def _alloc_from_free(free_ranges: list[tuple[int, int]], size: int) -> int | None:
    """Allocate a contiguous range of the given size from free_ranges."""

    def pick_best(indices: list[int]) -> int | None:
        best_idx = None
        best_size = None
        for i in indices:
            s, e = free_ranges[i]
            avail = e - s + 1
            if avail < size:
                continue
            if best_size is None or avail < best_size:
                best_idx = i
                best_size = avail
        return best_idx

    idx = None
    if size == 1:
        small = [i for i, (s, e) in enumerate(free_ranges) if (e - s + 1) < VLEN]
        idx = pick_best(small)
        if idx is None:
            larger = [i for i, (s, e) in enumerate(free_ranges) if (e - s + 1) > VLEN]
            idx = pick_best(larger)
        if idx is None:
            exact = [i for i, (s, e) in enumerate(free_ranges) if (e - s + 1) == VLEN]
            idx = pick_best(exact)
    else:
        idx = pick_best(list(range(len(free_ranges))))

    if idx is None:
        return None

    s, e = free_ranges[idx]
    alloc_start = s
    alloc_end = s + size - 1
    if alloc_end == e:
        free_ranges.pop(idx)
    else:
        free_ranges[idx] = (alloc_end + 1, e)
    return alloc_start


def _rewrite_mir(mfunc: MachineFunction, allocation: dict[int, int],
                 vector_bases: set[int], vector_addrs: set[int]) -> None:
    """Rewrite all virtual register references with physical registers."""
    # Build complete allocation map including vector element addresses
    full_allocation = dict(allocation)

    # Add mappings for vector elements
    for base in vector_bases:
        if base in allocation:
            new_base = allocation[base]
            for i in range(VLEN):
                if base + i not in full_allocation:
                    full_allocation[base + i] = new_base + i

    def rewrite_scratch(val):
        if isinstance(val, int) and val in full_allocation:
            return full_allocation[val]
        return val

    def rewrite_operand(op, inst: MachineInst, op_idx: int):
        if isinstance(op, list):
            if op and isinstance(op[0], int):
                base = op[0]
                if base in vector_bases:
                    if base in full_allocation:
                        new_base = full_allocation[base]
                        return [new_base + i for i in range(len(op))]
                else:
                    return [rewrite_scratch(elem) if isinstance(elem, int) else elem
                            for elem in op]
            return op
        elif isinstance(op, int):
            # Check context for immediates
            if inst.opcode == LIROpcode.CONST:
                return op
            if inst.opcode == LIROpcode.LOAD_OFFSET and op_idx == 1:
                return op
            if inst.opcode in (LIROpcode.JUMP, LIROpcode.COND_JUMP):
                if inst.opcode == LIROpcode.COND_JUMP and op_idx == 0:
                    return rewrite_scratch(op)
                return op
            return rewrite_scratch(op)
        else:
            return op

    for block in mfunc.blocks.values():
        for bundle in block.bundles:
            for inst in bundle.instructions:
                # Rewrite destination
                if inst.dest is not None:
                    if inst.opcode == LIROpcode.LOAD_OFFSET:
                        if inst.dest in vector_bases:
                            if inst.dest in full_allocation:
                                inst.dest = full_allocation[inst.dest]
                        else:
                            offset = inst.operands[1] if len(inst.operands) > 1 else 0
                            if isinstance(offset, int):
                                actual_dest = inst.dest + offset
                                if actual_dest in full_allocation:
                                    inst.dest = full_allocation[actual_dest] - offset
                    elif isinstance(inst.dest, list):
                        if inst.dest and inst.dest[0] in full_allocation:
                            new_base = full_allocation[inst.dest[0]]
                            inst.dest = [new_base + i for i in range(len(inst.dest))]
                    elif inst.dest in full_allocation:
                        inst.dest = full_allocation[inst.dest]

                # Rewrite operands
                new_operands = []
                for op_idx, op in enumerate(inst.operands):
                    new_operands.append(rewrite_operand(op, inst, op_idx))
                inst.operands = new_operands


def _materialize_zero_for_copies(mfunc: MachineFunction, max_preg_used: int) -> int:
    """Materialize zero constant for COPY operations.

    After register allocation, we need a fresh scratch address for const 0
    to expand COPY pseudo-ops to ADD with zero.

    Returns updated max_preg_used.
    """
    # Check if there are any COPY instructions
    has_copy = False
    for block in mfunc.blocks.values():
        for bundle in block.bundles:
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.COPY:
                    has_copy = True
                    break
            if has_copy:
                break
        if has_copy:
            break

    if not has_copy:
        return max_preg_used

    # Allocate a fresh scratch for zero constant
    zero_scratch = max_preg_used + 1

    # Add const 0 at the beginning of entry block
    entry_block = mfunc.blocks[mfunc.entry]
    zero_inst = MachineInst(LIROpcode.CONST, zero_scratch, [0], "load")

    # Insert at the beginning of first bundle or create new bundle
    if entry_block.bundles:
        first_bundle = entry_block.bundles[0]
        if first_bundle.has_slot_available("load"):
            first_bundle.instructions.insert(0, zero_inst)
        else:
            # Create new bundle for zero const
            new_bundle = MBundle()
            new_bundle.add_instruction(zero_inst)
            entry_block.bundles.insert(0, new_bundle)
    else:
        new_bundle = MBundle()
        new_bundle.add_instruction(zero_inst)
        entry_block.bundles.insert(0, new_bundle)

    # Convert all COPY instructions to ADD with zero
    for block in mfunc.blocks.values():
        for bundle in block.bundles:
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.COPY:
                    inst.opcode = LIROpcode.ADD
                    inst.operands = [inst.operands[0], zero_scratch]
                    inst.engine = "alu"

    return zero_scratch


def _split_live_ranges(
    mfunc: MachineFunction,
    spills: list[SpillInfo],
    vector_bases: set[int],
    vector_addrs: set[int],
) -> None:
    """Split long live ranges by inserting store/load to memory.

    For each spilled interval, inserts:
    - vstore/store after each def bundle (save to memory)
    - vload/load into a FRESH vreg before each use bundle, then rewrites the
      using instruction to reference the fresh vreg.

    This breaks long live intervals into short def→store and load→use pairs.
    Each reload gets its own vreg so the original vreg is dead after the spill
    store, and reloaded copies are only live for 1-2 bundles.
    The spill area is at mem[inp_values_p + batch_size] = mem[mem[6] + mem[2]].
    """
    if not spills:
        return

    # Find max virtual register in use
    max_vreg = 0
    for block in mfunc.blocks.values():
        for bundle in block.bundles:
            for inst in bundle.instructions:
                for d in inst.get_defs():
                    if isinstance(d, int):
                        max_vreg = max(max_vreg, d)
                for u in inst.get_uses():
                    if isinstance(u, int):
                        max_vreg = max(max_vreg, u)

    next_fresh = max_vreg + 1

    def alloc_fresh(size: int = 1) -> int:
        nonlocal next_fresh
        base = next_fresh
        next_fresh += size
        return base

    # Allocate fresh virtual registers for spill infrastructure
    spill_base_vreg = alloc_fresh()       # Holds spill base memory address
    spill_addr_vreg = alloc_fresh()       # Temp for address computation (reused)
    spill_const_vreg = alloc_fresh()      # Temp for constants (reused)

    # Build addr_to_base map
    addr_to_base: dict[int, int] = {}
    for base in vector_bases:
        for i in range(VLEN):
            addr_to_base[base + i] = base

    # Build spill lookup
    spilled_vregs: dict[int, SpillInfo] = {s.vreg: s for s in spills}

    # Insert spill base address computation at entry block
    entry_block = mfunc.blocks[mfunc.entry]
    b1 = MBundle()
    b1.add_instruction(MachineInst(LIROpcode.CONST, spill_const_vreg, [6], "load"))
    b1.add_instruction(MachineInst(LIROpcode.CONST, spill_addr_vreg, [2], "load"))
    b2 = MBundle()
    b2.add_instruction(MachineInst(LIROpcode.LOAD, spill_const_vreg, [spill_const_vreg], "load"))
    b2.add_instruction(MachineInst(LIROpcode.LOAD, spill_addr_vreg, [spill_addr_vreg], "load"))
    b3 = MBundle()
    b3.add_instruction(MachineInst(LIROpcode.ADD, spill_base_vreg,
                                    [spill_const_vreg, spill_addr_vreg], "alu"))
    entry_block.bundles.insert(0, b1)
    entry_block.bundles.insert(1, b2)
    entry_block.bundles.insert(2, b3)

    # Walk all blocks and insert store/load around spilled vreg defs/uses.
    # Each reload goes into a FRESH vreg and we rewrite the using instructions.
    block_order = mfunc.get_block_order()
    for block_name in block_order:
        block = mfunc.blocks[block_name]
        new_bundles: list[MBundle] = []

        for bundle in block.bundles:
            # Identify spilled vregs used/defined by this bundle
            bundle_uses: set[int] = set()
            bundle_defs: set[int] = set()

            for inst in bundle.instructions:
                for u in inst.get_uses():
                    vreg = u
                    if u in vector_addrs and u not in vector_bases:
                        vreg = addr_to_base.get(u, u)
                    if vreg in spilled_vregs:
                        bundle_uses.add(vreg)

                for d in inst.get_defs():
                    vreg = d
                    if d in vector_addrs and d not in vector_bases:
                        vreg = addr_to_base.get(d, d)
                    if vreg in spilled_vregs:
                        bundle_defs.add(vreg)

            # For each spilled vreg used: allocate fresh vreg, reload into it,
            # and rewrite the using instruction's operands.
            rewrite_map: dict[int, int] = {}  # old_addr -> new_addr
            for vreg in bundle_uses:
                sp = spilled_vregs[vreg]
                if sp.is_vector:
                    fresh_base = alloc_fresh(VLEN)
                    # Record rewrite for all lanes
                    for i in range(VLEN):
                        rewrite_map[vreg + i] = fresh_base + i
                    # Also mark fresh_base as vector
                    vector_bases.add(fresh_base)
                    for i in range(VLEN):
                        vector_addrs.add(fresh_base + i)
                        addr_to_base[fresh_base + i] = fresh_base
                else:
                    fresh = alloc_fresh()
                    rewrite_map[vreg] = fresh

                # Emit reload into fresh vreg
                addr_b1 = MBundle()
                addr_b1.add_instruction(
                    MachineInst(LIROpcode.CONST, spill_addr_vreg, [sp.mem_offset], "load")
                )
                addr_b2 = MBundle()
                addr_b2.add_instruction(
                    MachineInst(LIROpcode.ADD, spill_addr_vreg,
                                [spill_base_vreg, spill_addr_vreg], "alu")
                )
                load_b = MBundle()
                if sp.is_vector:
                    load_b.add_instruction(
                        MachineInst(LIROpcode.VLOAD,
                                    list(range(fresh_base, fresh_base + VLEN)),
                                    [spill_addr_vreg], "load")
                    )
                else:
                    load_b.add_instruction(
                        MachineInst(LIROpcode.LOAD, fresh, [spill_addr_vreg], "load")
                    )
                new_bundles.extend([addr_b1, addr_b2, load_b])

            # Rewrite the using instructions' operands
            if rewrite_map:
                for inst in bundle.instructions:
                    new_operands = []
                    for op in inst.operands:
                        if isinstance(op, list):
                            new_operands.append([rewrite_map.get(x, x) if isinstance(x, int) else x for x in op])
                        elif isinstance(op, int) and op in rewrite_map:
                            new_operands.append(rewrite_map[op])
                        else:
                            new_operands.append(op)
                    inst.operands = new_operands

            new_bundles.append(bundle)

            # Insert store after bundle for each spilled vreg defined
            for vreg in bundle_defs:
                sp = spilled_vregs[vreg]
                addr_b1 = MBundle()
                addr_b1.add_instruction(
                    MachineInst(LIROpcode.CONST, spill_addr_vreg, [sp.mem_offset], "load")
                )
                addr_b2 = MBundle()
                addr_b2.add_instruction(
                    MachineInst(LIROpcode.ADD, spill_addr_vreg,
                                [spill_base_vreg, spill_addr_vreg], "alu")
                )
                store_b = MBundle()
                if sp.is_vector:
                    store_b.add_instruction(
                        MachineInst(LIROpcode.VSTORE, None,
                                    [spill_addr_vreg,
                                     list(range(vreg, vreg + VLEN))], "store")
                    )
                else:
                    store_b.add_instruction(
                        MachineInst(LIROpcode.STORE, None,
                                    [spill_addr_vreg, vreg], "store")
                    )
                new_bundles.extend([addr_b1, addr_b2, store_b])

        block.bundles = new_bundles


class MIRRegisterAllocationPass(MIRPass):
    """
    Register allocation pass for MIR.

    Uses linear scan algorithm with bundle-aware liveness analysis.
    Each bundle gets a unique index since bundles are atomic scheduling units.
    Supports spilling to memory when register pressure exceeds SCRATCH_SIZE.
    """

    @property
    def name(self) -> str:
        return "mir-register-allocation"

    def run(self, mir: MachineFunction, config: PassConfig) -> MachineFunction:
        """Run register allocation on MIR."""
        self._init_metrics()

        # Read config options
        enable_spilling = bool(config.options.get("enable_spilling", True))
        spill_infra_size = int(config.options.get("spill_infra_size", 35))
        debug_spills = bool(config.options.get("debug_spills", False))

        # Step 1: Detect vector bases
        vector_bases, vector_addrs = _detect_vector_bases(mir)

        # Step 2: Compute bundle-based liveness
        liveness = _compute_liveness(mir, vector_bases, vector_addrs)

        # Step 3: Build live intervals
        intervals = _build_live_intervals(mir, liveness, vector_bases, vector_addrs)

        if not intervals:
            if self._metrics:
                self._metrics.custom = {
                    "virtual_regs": 0,
                    "physical_regs_used": 0,
                    "scalars_allocated": 0,
                    "vectors_allocated": 0,
                }
            return mir

        # Count scalars and vectors
        n_scalars = sum(1 for i in intervals if not i.is_vector)
        n_vectors = sum(1 for i in intervals if i.is_vector)

        # Step 4: Linear scan allocation
        allocation, max_preg_used = _linear_scan(intervals)
        n_spills = 0

        if max_preg_used < SCRATCH_SIZE:
            # Fits without spilling
            _rewrite_mir(mir, allocation, vector_bases, vector_addrs)
            max_preg_used = _materialize_zero_for_copies(mir, max_preg_used)

            if max_preg_used >= SCRATCH_SIZE:
                raise RuntimeError(
                    f"Register allocation failed: need {max_preg_used + 1} registers "
                    f"but only {SCRATCH_SIZE} available"
                )
        else:
            # Need spilling: identify intervals to spill, insert store/load,
            # then re-run allocation on the modified MIR.
            import sys

            if not enable_spilling:
                raise RuntimeError(
                    f"Register allocation failed: need {max_preg_used + 1} registers "
                    f"but only {SCRATCH_SIZE} available (spilling disabled by config)"
                )

            reg_limit = SCRATCH_SIZE - spill_infra_size

            # Compute use counts per vreg (normalized to vector bases)
            # to avoid spilling high-use broadcast constants
            _addr_to_base: dict[int, int] = {}
            for base in vector_bases:
                for i in range(VLEN):
                    _addr_to_base[base + i] = base

            use_counts: dict[int, int] = {}
            for block in mir.blocks.values():
                for bundle in block.bundles:
                    for inst in bundle.instructions:
                        for u in inst.get_uses():
                            vreg = _addr_to_base.get(u, u)
                            use_counts[vreg] = use_counts.get(vreg, 0) + 1

            _, _, spills = _linear_scan_with_spill(intervals, reg_limit, use_counts)
            if not spills:
                raise RuntimeError(
                    f"Register allocation failed: need {max_preg_used + 1} registers "
                    f"but only {SCRATCH_SIZE} available (no spill candidates)"
                )
            n_spills = len(spills)
            if debug_spills:
                print(f"[regalloc] Spilling {n_spills} intervals to memory", file=sys.stderr)
                for sp in spills:
                    uc = use_counts.get(sp.vreg, 0)
                    span = sp.interval.end - sp.interval.start
                    print(f"  spill vreg={sp.vreg} uses={uc} span={span} vec={sp.is_vector}", file=sys.stderr)

            # Insert store/load instructions with fresh vregs per reload
            _split_live_ranges(mir, spills, vector_bases, vector_addrs)

            # Re-detect vectors (split_live_ranges adds new vector bases)
            vector_bases2, vector_addrs2 = _detect_vector_bases(mir)
            liveness2 = _compute_liveness(mir, vector_bases2, vector_addrs2)
            intervals2 = _build_live_intervals(mir, liveness2, vector_bases2, vector_addrs2)
            allocation2, max_preg_used = _linear_scan(intervals2)

            if debug_spills:
                print(f"[regalloc] After spill: {len(intervals2)} intervals, "
                      f"max_preg={max_preg_used}", file=sys.stderr)

            if max_preg_used >= SCRATCH_SIZE:
                raise RuntimeError(
                    f"Register allocation failed after spilling: need {max_preg_used + 1} "
                    f"registers but only {SCRATCH_SIZE} available "
                    f"({n_spills} intervals spilled)"
                )

            _rewrite_mir(mir, allocation2, vector_bases2, vector_addrs2)
            max_preg_used = _materialize_zero_for_copies(mir, max_preg_used)

            if max_preg_used >= SCRATCH_SIZE:
                raise RuntimeError(
                    f"Register allocation failed: need {max_preg_used + 1} registers "
                    f"but only {SCRATCH_SIZE} available"
                )

        mir.max_scratch_used = max_preg_used

        if self._metrics:
            self._metrics.custom = {
                "virtual_regs": len(intervals),
                "physical_regs_used": max_preg_used + 1,
                "scalars_allocated": n_scalars,
                "vectors_allocated": n_vectors,
                "spills": n_spills,
            }

        return mir
