"""
Register Allocation Pass

Implements linear scan register allocation on LIR to reuse scratch addresses
when variables become dead. This enables full loop unrolling without exceeding
SCRATCH_SIZE (1536).

The pass runs after phi elimination and before codegen.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from problem import SCRATCH_SIZE, VLEN

from ..pass_manager import LIRPass, PassConfig
from ..lir import LIRFunction, LIROpcode, LIRInst, BasicBlock


@dataclass
class LivenessInfo:
    """Liveness information for a basic block."""
    live_in: set[int] = field(default_factory=set)   # Variables live at block entry
    live_out: set[int] = field(default_factory=set)  # Variables live at block exit
    gen: set[int] = field(default_factory=set)       # Uses before def in block
    kill: set[int] = field(default_factory=set)      # Definitions in block


@dataclass
class LiveInterval:
    """Live interval for a virtual register."""
    vreg: int              # Virtual register (scratch address)
    start: int             # First instruction index where live
    end: int               # Last instruction index where live
    is_vector: bool        # Whether this is a vector register (8 contiguous)


def _get_block_order(lir: LIRFunction) -> list[str]:
    """Get blocks in reverse postorder for liveness analysis."""
    visited = set()
    postorder = []

    def dfs(name: str):
        if name in visited:
            return
        visited.add(name)
        block = lir.blocks[name]
        if block.terminator:
            if block.terminator.opcode == LIROpcode.JUMP:
                dfs(block.terminator.operands[0])
            elif block.terminator.opcode == LIROpcode.COND_JUMP:
                dfs(block.terminator.operands[1])
                dfs(block.terminator.operands[2])
        postorder.append(name)

    dfs(lir.entry)
    return list(reversed(postorder))


def _get_successors(block: BasicBlock) -> list[str]:
    """Get successor block names."""
    if block.terminator is None:
        return []
    if block.terminator.opcode == LIROpcode.JUMP:
        return [block.terminator.operands[0]]
    if block.terminator.opcode == LIROpcode.COND_JUMP:
        return [block.terminator.operands[1], block.terminator.operands[2]]
    return []


def _collect_uses(inst: LIRInst, uses: set[int]) -> None:
    """Collect all scratch addresses used by an instruction (for liveness).

    This adds ALL scratch addresses, including all 8 elements of vectors.
    Used for liveness analysis where we need to track all addresses.
    """
    # CONST operands are immediate values, not scratch references
    if inst.opcode == LIROpcode.CONST:
        return

    # LOAD_OFFSET has [addr_scratch, offset_immediate]
    # The actual scratch read is scratch[addr + offset], not just scratch[addr]
    # This is important for vgather patterns where addr is a vector base
    if inst.opcode == LIROpcode.LOAD_OFFSET and len(inst.operands) == 2:
        addr = inst.operands[0]
        offset = inst.operands[1]
        if isinstance(addr, int) and isinstance(offset, int):
            uses.add(addr + offset)
        elif isinstance(addr, int):
            uses.add(addr)  # Fallback if offset isn't an int
        return

    # For JUMP, operands are labels, not scratch
    if inst.opcode == LIROpcode.JUMP:
        return

    # For COND_JUMP, first operand is condition scratch, rest are labels
    if inst.opcode == LIROpcode.COND_JUMP:
        cond = inst.operands[0]
        if isinstance(cond, int):
            uses.add(cond)
        return

    for op in inst.operands:
        if isinstance(op, int):
            uses.add(op)
        elif isinstance(op, list):
            for s in op:
                if isinstance(s, int):
                    uses.add(s)


def _collect_uses_with_vectors(inst: LIRInst, vector_bases: set[int]) -> tuple[set[int], set[int]]:
    """Collect scratch addresses used by an instruction, separating scalars and vectors.

    Args:
        inst: The instruction
        vector_bases: Set of addresses that are vector bases (from vector defs)

    Returns:
        (scalar_uses, vector_base_uses) where vector_base_uses contains the base
        address of each true vector operand.
    """
    scalar_uses: set[int] = set()
    vector_uses: set[int] = set()

    # CONST operands are immediate values, not scratch references
    if inst.opcode == LIROpcode.CONST:
        return scalar_uses, vector_uses

    # LOAD_OFFSET has [addr_scratch, offset_immediate]
    # The actual scratch read is scratch[addr + offset] (a lane), not scratch[addr].
    #
    # Treat this as a scalar use of the actual lane address. If the lane belongs to a
    # vector (constructed or real), interval building will map it back to its base.
    if inst.opcode == LIROpcode.LOAD_OFFSET and len(inst.operands) == 2:
        addr = inst.operands[0]
        offset = inst.operands[1]
        if isinstance(addr, int) and isinstance(offset, int):
            scalar_uses.add(addr + offset)
        elif isinstance(addr, int):
            scalar_uses.add(addr)  # Fallback if offset isn't an int
        return scalar_uses, vector_uses

    # For JUMP, operands are labels, not scratch
    if inst.opcode == LIROpcode.JUMP:
        return scalar_uses, vector_uses

    # For COND_JUMP, first operand is condition scratch, rest are labels
    if inst.opcode == LIROpcode.COND_JUMP:
        cond = inst.operands[0]
        if isinstance(cond, int):
            scalar_uses.add(cond)
        return scalar_uses, vector_uses

    for op in inst.operands:
        if isinstance(op, int):
            scalar_uses.add(op)
        elif isinstance(op, list) and op:
            # Check if this is a true vector (base is a vector_base) or a constructed list
            if isinstance(op[0], int):
                if op[0] in vector_bases:
                    # True vector - track only base
                    vector_uses.add(op[0])
                else:
                    # Constructed list - track all elements as scalars
                    for elem in op:
                        if isinstance(elem, int):
                            scalar_uses.add(elem)

    return scalar_uses, vector_uses


def _collect_defs(inst: LIRInst, defs: set[int]) -> None:
    """Collect all scratch addresses defined by an instruction (for liveness).

    For vectors, adds all 8 addresses to properly track kills.
    For LOAD_OFFSET, the actual write is to dest + offset.
    """
    if inst.dest is None:
        return

    # LOAD_OFFSET writes to dest + offset, not dest
    if inst.opcode == LIROpcode.LOAD_OFFSET:
        offset = inst.operands[1] if len(inst.operands) > 1 else 0
        if isinstance(offset, int):
            defs.add(inst.dest + offset)
        else:
            defs.add(inst.dest)  # Fallback if offset isn't an int
        return

    if isinstance(inst.dest, list):
        for d in inst.dest:
            defs.add(d)
    else:
        defs.add(inst.dest)


def _is_contiguous_vector_list(op: list) -> bool:
    """Check if a list operand is a contiguous vector (base, base+1, ..., base+7)."""
    if not op or len(op) != VLEN:
        return False
    if not isinstance(op[0], int):
        return False
    base = op[0]
    return all(isinstance(op[i], int) and op[i] == base + i for i in range(VLEN))


def _detect_constructed_vectors(lir: LIRFunction, vector_bases: set[int], vector_addrs: set[int]) -> None:
    """
    Detect "constructed vectors" - contiguous scalar lists used as vector operands.

    These can appear in ANY vector operation (VSTORE, VADD, VSELECT, etc.), not just VSTORE.
    If a contiguous list [base, base+1, ..., base+7] appears as an operand to a vector op,
    it must be treated as a vector to maintain contiguity after allocation.

    Also detects LOAD_OFFSET address vectors - when multiple LOAD_OFFSET instructions
    read from scratch[addr + offset] with the same base addr and offsets 0..VLEN-1,
    the address range [addr, addr+VLEN-1] forms a constructed vector.
    """
    for block in lir.blocks.values():
        for inst in block.instructions:
            # Check all operands that are lists
            for op in inst.operands:
                if isinstance(op, list) and _is_contiguous_vector_list(op):
                    base = op[0]
                    if base not in vector_bases:
                        vector_bases.add(base)
                        for i in range(VLEN):
                            vector_addrs.add(base + i)

    # Detect LOAD_OFFSET address vectors:
    # If LOAD_OFFSET instructions use scratch[addr + offset] with the same base addr
    # and offsets 0..VLEN-1, then [addr, addr+1, ..., addr+VLEN-1] is a constructed vector.
    # We need to group by the base address (addr) and check if all VLEN offsets are present.
    addr_offsets: dict[int, set[int]] = {}  # base_addr -> set of offsets seen
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.opcode == LIROpcode.LOAD_OFFSET and len(inst.operands) >= 2:
                addr = inst.operands[0]
                offset = inst.operands[1]
                if isinstance(addr, int) and isinstance(offset, int):
                    if addr not in addr_offsets:
                        addr_offsets[addr] = set()
                    addr_offsets[addr].add(offset)

    # If a base addr has all VLEN offsets (0..VLEN-1), it's a vector pattern
    for addr, offsets in addr_offsets.items():
        if offsets == set(range(VLEN)) and addr not in vector_bases:
            vector_bases.add(addr)
            for i in range(VLEN):
                vector_addrs.add(addr + i)


def _compute_liveness(lir: LIRFunction) -> dict[str, LivenessInfo]:
    """Compute liveness information for all blocks using backward dataflow."""
    liveness: dict[str, LivenessInfo] = {}

    # Initialize liveness info for each block
    for name, block in lir.blocks.items():
        info = LivenessInfo()

        # Compute gen and kill sets
        # Process instructions in forward order
        for inst in block.instructions:
            # Uses before def contribute to gen
            uses: set[int] = set()
            _collect_uses(inst, uses)
            for u in uses:
                if u not in info.kill:
                    info.gen.add(u)

            # Defs contribute to kill
            _collect_defs(inst, info.kill)

        # Process terminator
        if block.terminator:
            uses = set()
            _collect_uses(block.terminator, uses)
            for u in uses:
                if u not in info.kill:
                    info.gen.add(u)

        liveness[name] = info

    # Iterate until fixed point
    changed = True
    while changed:
        changed = False

        # Process blocks in reverse order for backward analysis
        for name in reversed(_get_block_order(lir)):
            block = lir.blocks[name]
            info = liveness[name]

            # live_out = union of live_in of all successors
            new_live_out: set[int] = set()
            for succ_name in _get_successors(block):
                if succ_name in liveness:
                    new_live_out |= liveness[succ_name].live_in

            # live_in = gen | (live_out - kill)
            new_live_in = info.gen | (new_live_out - info.kill)

            if new_live_in != info.live_in or new_live_out != info.live_out:
                changed = True
                info.live_in = new_live_in
                info.live_out = new_live_out

    return liveness


def _build_live_intervals(lir: LIRFunction, liveness: dict[str, LivenessInfo]) -> list[LiveInterval]:
    """Build live intervals for all virtual registers."""
    # First pass: identify all vector base addresses and their ranges
    # This allows us to skip non-base vector addresses in liveness sets
    vector_bases: set[int] = set()
    vector_addrs: set[int] = set()  # All addresses that are part of a vector

    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.dest is not None and isinstance(inst.dest, list) and inst.dest:
                base = inst.dest[0]
                if isinstance(base, int):
                    vector_bases.add(base)
                    for i in range(len(inst.dest)):
                        vector_addrs.add(base + i)

    # Detect LOAD_OFFSET instructions that write to vectors (from vgather)
    # These have the same dest (vector base) but offsets 0-7
    load_offset_dests: dict[int, set[int]] = {}  # dest -> set of offsets
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.opcode == LIROpcode.LOAD_OFFSET:
                dest = inst.dest
                offset = inst.operands[1] if len(inst.operands) > 1 else 0
                if isinstance(dest, int) and isinstance(offset, int):
                    if dest not in load_offset_dests:
                        load_offset_dests[dest] = set()
                    load_offset_dests[dest].add(offset)

    # If a dest has LOAD_OFFSET with offsets 0..VLEN-1, treat it as a vector base
    for dest, offsets in load_offset_dests.items():
        if offsets == set(range(VLEN)):  # Has all offsets 0..VLEN-1
            vector_bases.add(dest)
            for i in range(VLEN):
                vector_addrs.add(dest + i)

    # Detect "constructed vectors" - contiguous scalar lists used in ANY vector operation
    # This handles vinsert chains used in VADD, VSELECT, VSTORE, etc.
    _detect_constructed_vectors(lir, vector_bases, vector_addrs)

    # Build O(1) lookup from vector address to its base (for efficient base finding)
    addr_to_base: dict[int, int] = {}
    for base in vector_bases:
        for i in range(VLEN):
            addr_to_base[base + i] = base

    # Linearize blocks and number instructions globally
    block_order = _get_block_order(lir)

    # Map block name to (start_idx, end_idx) in global numbering
    block_ranges: dict[str, tuple[int, int]] = {}
    inst_idx = 0

    for name in block_order:
        block = lir.blocks[name]
        start = inst_idx
        # Count instructions + terminator
        count = len(block.instructions) + (1 if block.terminator else 0)
        if count == 0:
            count = 1  # Empty block still occupies one position
        end = inst_idx + count - 1
        block_ranges[name] = (start, end)
        inst_idx = end + 1

    # Track live ranges for each virtual register
    # vreg -> (start, end, is_vector)
    intervals: dict[int, tuple[int, int, bool]] = {}

    def update_interval(vreg: int, idx: int, is_vector: bool):
        if vreg in intervals:
            old_start, old_end, old_is_vec = intervals[vreg]
            intervals[vreg] = (min(old_start, idx), max(old_end, idx), old_is_vec or is_vector)
        else:
            intervals[vreg] = (idx, idx, is_vector)

    # Process each block
    for name in block_order:
        block = lir.blocks[name]
        block_start, block_end = block_ranges[name]
        info = liveness[name]

        # Variables live at block entry extend from block start
        for vreg in info.live_in:
            # Skip non-base vector addresses
            if vreg in vector_addrs and vreg not in vector_bases:
                continue
            is_vec = vreg in vector_bases
            update_interval(vreg, block_start, is_vec)

        # Variables live at block exit extend to block end
        for vreg in info.live_out:
            # Skip non-base vector addresses
            if vreg in vector_addrs and vreg not in vector_bases:
                continue
            is_vec = vreg in vector_bases
            update_interval(vreg, block_end, is_vec)

        # Process instructions
        idx = block_start
        for inst in block.instructions:
            # Defs start here
            if inst.dest is not None:
                # LOAD_OFFSET writes to dest + offset
                actual_dest = inst.dest
                if inst.opcode == LIROpcode.LOAD_OFFSET:
                    offset = inst.operands[1] if len(inst.operands) > 1 else 0
                    if isinstance(offset, int):
                        actual_dest = inst.dest + offset

                is_vec = isinstance(actual_dest, list)
                if is_vec:
                    # Vector: track base only
                    update_interval(actual_dest[0], idx, True)
                elif actual_dest in vector_addrs:
                    # Scalar that's part of a constructed vector
                    if actual_dest in vector_bases:
                        # This is the base of the vector
                        update_interval(actual_dest, idx, True)
                    # else: skip non-base elements, they'll share the base's interval
                else:
                    update_interval(actual_dest, idx, False)

            # Uses extend interval to here
            scalar_uses, vector_uses = _collect_uses_with_vectors(inst, vector_bases)
            for vreg in scalar_uses:
                # Skip if this is part of a vector (use the base instead)
                if vreg in vector_addrs and vreg not in vector_bases:
                    # Find the base using the lookup map
                    base = addr_to_base.get(vreg)
                    if base is not None:
                        update_interval(base, idx, True)
                else:
                    update_interval(vreg, idx, vreg in vector_bases)
            for vreg in vector_uses:
                update_interval(vreg, idx, True)

            idx += 1

        # Process terminator
        if block.terminator:
            scalar_uses, vector_uses = _collect_uses_with_vectors(block.terminator, vector_bases)
            for vreg in scalar_uses:
                # Skip if this is part of a vector (use the base instead)
                if vreg in vector_addrs and vreg not in vector_bases:
                    # Find the base using the lookup map
                    base = addr_to_base.get(vreg)
                    if base is not None:
                        update_interval(base, idx, True)
                else:
                    update_interval(vreg, idx, vreg in vector_bases)
            for vreg in vector_uses:
                update_interval(vreg, idx, True)

    # Convert to LiveInterval objects
    result = []
    for vreg, (start, end, is_vector) in intervals.items():
        result.append(LiveInterval(vreg, start, end, is_vector))

    # Sort by start position
    result.sort(key=lambda x: (x.start, -x.end))

    return result


def _linear_scan(intervals: list[LiveInterval]) -> tuple[dict[int, int], int]:
    """
    Linear scan register allocation.

    Args:
        intervals: List of live intervals sorted by start position

    Returns:
        (allocation, max_preg_used) where:
        - allocation: maps virtual register to physical register
        - max_preg_used: highest physical register used
    """
    allocation: dict[int, int] = {}

    # Separate free lists for scalars and vectors
    # Vectors need VLEN contiguous slots, so we track base addresses
    scalar_free: list[int] = []
    vector_free: list[int] = []  # Base addresses of free VLEN-slot blocks

    # Next available physical register
    next_preg = 0

    # Active intervals sorted by end position
    active: list[LiveInterval] = []

    for interval in intervals:
        # Expire old intervals
        new_active = []
        for a in active:
            if a.end < interval.start:
                # This interval has expired, free its register
                preg = allocation[a.vreg]
                if a.is_vector:
                    vector_free.append(preg)
                else:
                    scalar_free.append(preg)
            else:
                new_active.append(a)
        active = new_active

        # Allocate register for current interval
        if interval.is_vector:
            # Need VLEN contiguous slots
            if vector_free:
                preg = vector_free.pop()
            else:
                # Allocate new VLEN-slot block
                preg = next_preg
                next_preg += VLEN
        else:
            # Need 1 slot
            if scalar_free:
                preg = scalar_free.pop()
            else:
                preg = next_preg
                next_preg += 1

        allocation[interval.vreg] = preg

        # Add to active list, keep sorted by end (use bisect for O(log n) insertion)
        # We use a key list to maintain sort order by end position
        bisect.insort_right(active, interval, key=lambda x: x.end)

    max_preg_used = next_preg - 1 if next_preg > 0 else 0
    return allocation, max_preg_used


def _rewrite_lir(lir: LIRFunction, allocation: dict[int, int]) -> None:
    """Rewrite all virtual register references with physical registers."""
    # Build complete allocation map including vector element addresses
    # For vector bases in allocation, add mappings for all 8 elements
    full_allocation = dict(allocation)

    # Find all vector definitions (before rewriting) and track vector bases
    vector_bases: set[int] = set()
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.dest is not None and isinstance(inst.dest, list) and inst.dest:
                old_base = inst.dest[0]
                if isinstance(old_base, int):
                    vector_bases.add(old_base)
                    if old_base in allocation:
                        new_base = allocation[old_base]
                        # Map each element of the old vector to the new vector
                        for i in range(len(inst.dest)):
                            if old_base + i not in full_allocation:
                                full_allocation[old_base + i] = new_base + i

    # Detect LOAD_OFFSET instructions that write to vectors (from vgather)
    # These have the same dest (vector base) but offsets 0..VLEN-1
    load_offset_dests: dict[int, set[int]] = {}
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.opcode == LIROpcode.LOAD_OFFSET:
                dest = inst.dest
                offset = inst.operands[1] if len(inst.operands) > 1 else 0
                if isinstance(dest, int) and isinstance(offset, int):
                    if dest not in load_offset_dests:
                        load_offset_dests[dest] = set()
                    load_offset_dests[dest].add(offset)

    for dest, offsets in load_offset_dests.items():
        if offsets == set(range(VLEN)) and dest not in vector_bases:
            vector_bases.add(dest)
            if dest in allocation:
                new_base = allocation[dest]
                for i in range(VLEN):
                    if dest + i not in full_allocation:
                        full_allocation[dest + i] = new_base + i

    # Detect LOAD_OFFSET address vectors:
    # A vgather-style expansion reads scratch[addr + offset] for offsets 0..VLEN-1.
    # If we see all offsets for the same addr base, then [addr, addr+1, ..., addr+VLEN-1]
    # is a constructed address vector that must remain contiguous after allocation.
    load_offset_addrs: dict[int, set[int]] = {}  # addr_base -> set of offsets seen
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.opcode == LIROpcode.LOAD_OFFSET and len(inst.operands) >= 2:
                addr = inst.operands[0]
                offset = inst.operands[1]
                if isinstance(addr, int) and isinstance(offset, int):
                    if addr not in load_offset_addrs:
                        load_offset_addrs[addr] = set()
                    load_offset_addrs[addr].add(offset)

    for addr, offsets in load_offset_addrs.items():
        if offsets == set(range(VLEN)):
            vector_bases.add(addr)
            if addr in allocation:
                new_base = allocation[addr]
                for i in range(VLEN):
                    if addr + i not in full_allocation:
                        full_allocation[addr + i] = new_base + i

    # Detect "constructed vectors" - contiguous scalar lists in ANY vector operand
    # This handles vinsert chains used in VADD, VSELECT, VSTORE, etc.
    for block in lir.blocks.values():
        for inst in block.instructions:
            for op in inst.operands:
                if isinstance(op, list) and _is_contiguous_vector_list(op):
                    base = op[0]
                    if base not in vector_bases and base in allocation:
                        vector_bases.add(base)
                        new_base = allocation[base]
                        for i in range(VLEN):
                            if base + i not in full_allocation:
                                full_allocation[base + i] = new_base + i

    def rewrite_scratch(val):
        """Rewrite a single scratch reference."""
        if isinstance(val, int) and val in full_allocation:
            return full_allocation[val]
        return val

    def rewrite_operand(op, inst: LIRInst, op_idx: int):
        """Rewrite an operand.

        Args:
            op: The operand value
            inst: The instruction containing this operand
            op_idx: The index of this operand in inst.operands
        """
        if isinstance(op, list):
            # List of scratch addresses - could be a true vector or a constructed list
            if op and isinstance(op[0], int):
                base = op[0]
                # If base is a vector base, use contiguous mapping
                if base in vector_bases:
                    if base in full_allocation:
                        new_base = full_allocation[base]
                        return [new_base + i for i in range(len(op))]
                else:
                    # Not a vector base - rewrite each element individually
                    return [rewrite_scratch(elem) if isinstance(elem, int) else elem
                            for elem in op]
            return op
        elif isinstance(op, int):
            # Could be scratch or immediate - check context
            # CONST operands are immediates
            if inst.opcode == LIROpcode.CONST:
                return op  # Immediate
            # LOAD_OFFSET second operand is immediate offset
            if inst.opcode == LIROpcode.LOAD_OFFSET and op_idx == 1:
                return op  # Immediate offset
            # Jump targets are labels/addresses, not scratch
            if inst.opcode in (LIROpcode.JUMP, LIROpcode.COND_JUMP):
                # First operand of COND_JUMP is scratch, rest are labels
                if inst.opcode == LIROpcode.COND_JUMP:
                    if op_idx == 0:
                        return rewrite_scratch(op)
                return op  # Label/address
            return rewrite_scratch(op)
        else:
            return op  # Label or other non-scratch value

    for block in lir.blocks.values():
        for inst in block.instructions:
            # Rewrite destination
            if inst.dest is not None:
                # LOAD_OFFSET: if dest is a vector base, just map the base
                # Otherwise, compute actual_dest = dest + offset and map that
                if inst.opcode == LIROpcode.LOAD_OFFSET:
                    if inst.dest in vector_bases:
                        # dest is a vector base, map it directly (offset stays same)
                        if inst.dest in full_allocation:
                            inst.dest = full_allocation[inst.dest]
                    else:
                        # dest is a scalar, compute actual_dest
                        offset = inst.operands[1] if len(inst.operands) > 1 else 0
                        if isinstance(offset, int):
                            actual_dest = inst.dest + offset
                            if actual_dest in full_allocation:
                                # new_dest + offset should equal the allocated address
                                inst.dest = full_allocation[actual_dest] - offset
                elif isinstance(inst.dest, list):
                    # Vector dest - rewrite base and regenerate
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

        # Rewrite terminator
        if block.terminator:
            term = block.terminator
            new_operands = []
            for op_idx, op in enumerate(term.operands):
                new_operands.append(rewrite_operand(op, term, op_idx))
            term.operands = new_operands


def _materialize_zero_for_copies(lir: LIRFunction, max_preg_used: int) -> int:
    """
    Materialize a zero constant for COPY operations and convert COPYs to ADDs.

    After register allocation, the scratch address that held const 0 may be
    reused for other values. This function creates a dedicated zero constant
    at a fresh scratch address and converts all COPY pseudo-ops to explicit
    ADD instructions.

    Args:
        lir: The LIR function
        max_preg_used: The maximum physical register used so far

    Returns:
        Updated max_preg_used (may be +1 if we added a zero constant)
    """
    # First check if there are any COPY instructions
    has_copy = False
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.opcode == LIROpcode.COPY:
                has_copy = True
                break
        if has_copy:
            break

    if not has_copy:
        return max_preg_used

    # Allocate a fresh scratch for zero constant
    zero_scratch = max_preg_used + 1

    # Add const 0 at the beginning of entry block
    entry_block = lir.blocks[lir.entry]
    zero_inst = LIRInst(LIROpcode.CONST, zero_scratch, [0], "load")
    entry_block.instructions.insert(0, zero_inst)

    # Convert all COPY instructions to ADD with zero
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.opcode == LIROpcode.COPY:
                # COPY dest, src -> ADD dest, src, zero
                inst.opcode = LIROpcode.ADD
                inst.operands = [inst.operands[0], zero_scratch]

    return zero_scratch


class RegisterAllocationPass(LIRPass):
    """
    Register allocation pass for LIR.

    Uses linear scan algorithm to reuse scratch addresses when variables
    become dead. This enables full loop unrolling without exceeding
    SCRATCH_SIZE (1536).

    The pass:
    1. Computes liveness using backward dataflow analysis
    2. Builds live intervals for all virtual registers
    3. Allocates physical registers using linear scan
    4. Rewrites all references to use physical registers
    5. Materializes zero constant for COPY operations
    """

    @property
    def name(self) -> str:
        return "register-allocation"

    def run(self, lir: LIRFunction, config: PassConfig) -> LIRFunction:
        """Run register allocation on LIR."""
        self._init_metrics()

        # Step 1: Compute liveness
        liveness = _compute_liveness(lir)

        # Step 2: Build live intervals
        intervals = _build_live_intervals(lir, liveness)

        if not intervals:
            # No registers to allocate
            if self._metrics:
                self._metrics.custom = {
                    "virtual_regs": 0,
                    "physical_regs_used": 0,
                    "scalars_allocated": 0,
                    "vectors_allocated": 0,
                }
            return lir

        # Count scalars and vectors
        n_scalars = sum(1 for i in intervals if not i.is_vector)
        n_vectors = sum(1 for i in intervals if i.is_vector)

        # Step 3: Linear scan allocation
        allocation, max_preg_used = _linear_scan(intervals)

        # Check if we exceeded SCRATCH_SIZE
        if max_preg_used >= SCRATCH_SIZE:
            raise RuntimeError(
                f"Register allocation failed: need {max_preg_used + 1} registers "
                f"but only {SCRATCH_SIZE} available"
            )

        # Step 4: Rewrite LIR
        _rewrite_lir(lir, allocation)

        # Step 5: Materialize zero constant for COPY operations
        max_preg_used = _materialize_zero_for_copies(lir, max_preg_used)

        # Check again after materializing zero (adds at most 1 register)
        if max_preg_used >= SCRATCH_SIZE:
            raise RuntimeError(
                f"Register allocation failed: need {max_preg_used + 1} registers "
                f"but only {SCRATCH_SIZE} available"
            )

        # Update max_scratch_used
        lir.max_scratch_used = max_preg_used

        # Record metrics
        if self._metrics:
            self._metrics.custom = {
                "virtual_regs": len(intervals),
                "physical_regs_used": max_preg_used + 1,
                "scalars_allocated": n_scalars,
                "vectors_allocated": n_vectors,
            }

        return lir
