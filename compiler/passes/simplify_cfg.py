"""
SimplifyCFG Pass

LIR → LIR pass that reduces basic blocks and unconditional jumps by:
1. Removing unreachable blocks
2. Jump threading (trampoline elimination)
3. Block merging
4. Redundant conditional branch cleanup
"""

from ..pass_manager import LIRPass, PassConfig
from ..lir import LIRFunction, BasicBlock, LIROpcode, LIRInst


def _get_successors(block: BasicBlock) -> list[str]:
    """Get successor block names from terminator."""
    if block.terminator is None:
        return []

    opcode = block.terminator.opcode
    if opcode == LIROpcode.JUMP:
        return [block.terminator.operands[0]]
    elif opcode == LIROpcode.COND_JUMP:
        # operands: [cond, true_target, false_target]
        return [block.terminator.operands[1], block.terminator.operands[2]]
    else:
        # halt, pause - no successors
        return []


def _build_pred_map(blocks: dict[str, BasicBlock]) -> dict[str, set[str]]:
    """Build predecessor map for all blocks."""
    preds: dict[str, set[str]] = {name: set() for name in blocks}
    for name, block in blocks.items():
        for succ in _get_successors(block):
            if succ in preds:
                preds[succ].add(name)
    return preds


def _compute_reachable(entry: str, blocks: dict[str, BasicBlock]) -> set[str]:
    """Compute all blocks reachable from entry."""
    reachable = set()
    worklist = [entry]
    while worklist:
        name = worklist.pop()
        if name in reachable:
            continue
        if name not in blocks:
            continue
        reachable.add(name)
        for succ in _get_successors(blocks[name]):
            if succ not in reachable:
                worklist.append(succ)
    return reachable


def _remove_unreachable(lir: LIRFunction) -> int:
    """Remove blocks not reachable from entry. Returns count removed."""
    reachable = _compute_reachable(lir.entry, lir.blocks)
    unreachable = set(lir.blocks.keys()) - reachable
    for name in unreachable:
        del lir.blocks[name]
    return len(unreachable)


def _update_phi_incoming(block: BasicBlock, old_pred: str, new_pred: str):
    """Rename phi incoming key from old_pred to new_pred."""
    for phi in block.phis:
        if old_pred in phi.incoming:
            val = phi.incoming.pop(old_pred)
            phi.incoming[new_pred] = val


def _is_trampoline(block: BasicBlock) -> bool:
    """Check if block is a trampoline (no phis, no instructions, just jump)."""
    return (
        len(block.phis) == 0 and
        len(block.instructions) == 0 and
        block.terminator is not None and
        block.terminator.opcode == LIROpcode.JUMP
    )


def _try_thread_jump(
    trampoline_name: str,
    blocks: dict[str, BasicBlock],
    preds: dict[str, set[str]]
) -> bool:
    """
    Try to thread through a trampoline block. Returns True if changed.

    Pattern: Block T has no phis, no instructions, terminates with jump(U),
    and has exactly one predecessor P.

    Rewrite: Redirect P's edge from T to U, update U's phi incoming keys,
    delete T.

    Safety: Do not thread if U has phis that already have P as an incoming key,
    as this would create a conflict (two different values from the same pred).
    """
    trampoline = blocks[trampoline_name]

    # Must be a trampoline
    if not _is_trampoline(trampoline):
        return False

    # Must have exactly one predecessor
    pred_set = preds.get(trampoline_name, set())
    if len(pred_set) != 1:
        return False

    pred_name = next(iter(pred_set))
    if pred_name not in blocks:
        return False

    pred = blocks[pred_name]
    target = trampoline.terminator.operands[0]  # The jump target U

    # Check if target block exists
    if target not in blocks:
        return False

    target_block = blocks[target]

    # Safety check: if target has phis with incoming key T, we would rename
    # T → P. But if the phi already has P as a key, we'd create a conflict.
    # This happens when P has a cond_jump to two trampolines that both target U.
    for phi in target_block.phis:
        if trampoline_name in phi.incoming and pred_name in phi.incoming:
            # Would create conflict - cannot thread
            return False

    # Redirect P's terminator edge(s) from T to U
    if pred.terminator is None:
        return False

    if pred.terminator.opcode == LIROpcode.JUMP:
        if pred.terminator.operands[0] == trampoline_name:
            pred.terminator.operands[0] = target
    elif pred.terminator.opcode == LIROpcode.COND_JUMP:
        # operands: [cond, true_target, false_target]
        if pred.terminator.operands[1] == trampoline_name:
            pred.terminator.operands[1] = target
        if pred.terminator.operands[2] == trampoline_name:
            pred.terminator.operands[2] = target
    else:
        return False

    # Update U's phi incoming keys: T → P
    _update_phi_incoming(target_block, trampoline_name, pred_name)

    # Delete T
    del blocks[trampoline_name]
    return True


def _try_merge_block(
    pred_name: str,
    blocks: dict[str, BasicBlock],
    preds: dict[str, set[str]]
) -> bool:
    """
    Try to merge a block into its single predecessor. Returns True if changed.

    Pattern: P terminates with jump(B), B has exactly one predecessor (P),
    B has no phis.

    Rewrite: Append B's instructions to P, replace P's terminator with B's
    terminator, update successors' phi incoming keys, delete B.
    """
    pred = blocks[pred_name]

    # P must terminate with unconditional jump
    if pred.terminator is None or pred.terminator.opcode != LIROpcode.JUMP:
        return False

    succ_name = pred.terminator.operands[0]
    if succ_name not in blocks:
        return False

    succ = blocks[succ_name]

    # B must have exactly one predecessor (P) and no phis
    succ_preds = preds.get(succ_name, set())
    if len(succ_preds) != 1 or pred_name not in succ_preds:
        return False
    if len(succ.phis) > 0:
        return False

    # Don't merge entry block away (if succ is entry, something is odd)
    # but we can merge into the entry block, which is fine

    # Append B's instructions to P
    pred.instructions.extend(succ.instructions)

    # Replace P's terminator with B's terminator
    pred.terminator = succ.terminator

    # Update successors' phi incoming keys: B → P
    for new_succ_name in _get_successors(pred):
        if new_succ_name in blocks:
            _update_phi_incoming(blocks[new_succ_name], succ_name, pred_name)

    # Delete B
    del blocks[succ_name]
    return True


def _simplify_cond_jump(block: BasicBlock) -> bool:
    """
    Convert cond_jump(c, X, X) to jump(X). Returns True if changed.
    """
    if block.terminator is None:
        return False
    if block.terminator.opcode != LIROpcode.COND_JUMP:
        return False

    # operands: [cond, true_target, false_target]
    true_target = block.terminator.operands[1]
    false_target = block.terminator.operands[2]

    if true_target == false_target:
        block.terminator = LIRInst(
            opcode=LIROpcode.JUMP,
            dest=None,
            operands=[true_target],
            engine="flow"
        )
        return True
    return False


class SimplifyCFGPass(LIRPass):
    """
    LIR → LIR pass that simplifies the control flow graph.

    Transformations:
    1. Remove unreachable blocks
    2. Jump threading (single-predecessor trampoline elimination)
    3. Block merging (single-predecessor, no-phi blocks)
    4. Redundant conditional branch cleanup
    """

    @property
    def name(self) -> str:
        return "simplify-cfg"

    def run(self, lir: LIRFunction, config: PassConfig) -> LIRFunction:
        """Simplify the LIR control flow graph."""
        self._init_metrics()

        blocks_before = len(lir.blocks)
        unreachable_removed = 0
        trampolines_threaded = 0
        blocks_merged = 0
        cond_jumps_simplified = 0

        # Fixed-point loop
        changed = True
        while changed:
            changed = False

            # 1. Remove unreachable blocks
            removed = _remove_unreachable(lir)
            if removed > 0:
                unreachable_removed += removed
                changed = True

            # Rebuild predecessor map after removals
            preds = _build_pred_map(lir.blocks)

            # 2. Apply jump threading to trampolines
            # Iterate over a copy since we modify blocks
            for name in list(lir.blocks.keys()):
                if name not in lir.blocks:
                    continue
                if _try_thread_jump(name, lir.blocks, preds):
                    trampolines_threaded += 1
                    changed = True
                    # Rebuild preds after structural change
                    preds = _build_pred_map(lir.blocks)

            # 3. Apply block merging
            for name in list(lir.blocks.keys()):
                if name not in lir.blocks:
                    continue
                if _try_merge_block(name, lir.blocks, preds):
                    blocks_merged += 1
                    changed = True
                    # Rebuild preds after structural change
                    preds = _build_pred_map(lir.blocks)

            # 4. Apply cond_jump simplification
            for block in lir.blocks.values():
                if _simplify_cond_jump(block):
                    cond_jumps_simplified += 1
                    changed = True

        # Record metrics
        if self._metrics:
            self._metrics.custom = {
                "blocks_before": blocks_before,
                "blocks_after": len(lir.blocks),
                "unreachable_removed": unreachable_removed,
                "trampolines_threaded": trampolines_threaded,
                "blocks_merged": blocks_merged,
                "cond_jumps_simplified": cond_jumps_simplified,
            }

        return lir
