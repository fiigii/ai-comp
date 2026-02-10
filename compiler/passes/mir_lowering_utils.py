"""
Shared LIR -> MIR lowering utilities.

Functions used by both LIRToMIRPass (lir_to_mir.py) and
InstSchedulingPass (inst_scheduling.py).
"""

from __future__ import annotations

from ..lir import LIRFunction, LIROpcode, LIRInst, BasicBlock
from ..mir import MachineInst


def lir_inst_to_machine_inst(inst: LIRInst) -> MachineInst:
    """Convert a LIRInst to a MachineInst."""
    return MachineInst(
        opcode=inst.opcode,
        dest=inst.dest,
        operands=list(inst.operands),
        engine=inst.engine,
    )


def get_successors(block: BasicBlock) -> list[str]:
    """Get successor block names from a LIR basic block."""
    if block.terminator is None:
        return []
    if block.terminator.opcode == LIROpcode.JUMP:
        return [block.terminator.operands[0]]
    if block.terminator.opcode == LIROpcode.COND_JUMP:
        return [block.terminator.operands[1], block.terminator.operands[2]]
    return []


def get_block_order(lir: LIRFunction) -> list[str]:
    """Get blocks in reverse postorder for scheduling."""
    visited: set[str] = set()
    postorder: list[str] = []

    def dfs(name: str):
        if name in visited:
            return
        visited.add(name)
        block = lir.blocks.get(name)
        if block:
            for succ_name in get_successors(block):
                dfs(succ_name)
            postorder.append(name)

    dfs(lir.entry)
    return list(reversed(postorder))
