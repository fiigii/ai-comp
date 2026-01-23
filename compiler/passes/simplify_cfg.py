"""
SimplifyCFG Pass (LIR)

Performs basic CFG cleanups after lowering:
- remove unreachable blocks
- thread single-predecessor jump-only blocks
- merge single-predecessor jump-target blocks
- simplify cond_jump with identical targets
"""

from __future__ import annotations

from typing import Optional

from ..pass_manager import LIRPass, PassConfig
from ..lir import LIRFunction, BasicBlock, LIROpcode, LIRInst


def _successors(block: BasicBlock) -> list[str]:
    if block.terminator is None:
        return []
    if block.terminator.opcode == LIROpcode.JUMP:
        return [block.terminator.operands[0]]
    if block.terminator.opcode == LIROpcode.COND_JUMP:
        return [block.terminator.operands[1], block.terminator.operands[2]]
    return []


def _compute_preds(lir: LIRFunction) -> dict[str, set[str]]:
    preds: dict[str, set[str]] = {name: set() for name in lir.blocks}
    for name, block in lir.blocks.items():
        for succ in _successors(block):
            if succ in preds:
                preds[succ].add(name)
    return preds


def _reachable_blocks(lir: LIRFunction) -> set[str]:
    seen: set[str] = set()
    work = [lir.entry]
    while work:
        name = work.pop()
        if name in seen or name not in lir.blocks:
            continue
        seen.add(name)
        for succ in _successors(lir.blocks[name]):
            if succ not in seen:
                work.append(succ)
    return seen


def _can_redirect_phi(block: BasicBlock, old_pred: str, new_pred: str) -> bool:
    if not block.phis:
        return True
    for phi in block.phis:
        if old_pred not in phi.incoming:
            continue
        if new_pred in phi.incoming and phi.incoming[new_pred] != phi.incoming[old_pred]:
            return False
    return True


def _redirect_phi(block: BasicBlock, old_pred: str, new_pred: str) -> None:
    if not block.phis:
        return
    for phi in block.phis:
        if old_pred not in phi.incoming:
            continue
        if new_pred in phi.incoming:
            # If same value, just drop the old pred entry
            if phi.incoming[new_pred] == phi.incoming[old_pred]:
                del phi.incoming[old_pred]
        else:
            phi.incoming[new_pred] = phi.incoming.pop(old_pred)


def _prune_phi_incomings(lir: LIRFunction) -> None:
    existing = set(lir.blocks.keys())
    for block in lir.blocks.values():
        if not block.phis:
            continue
        for phi in block.phis:
            for pred in list(phi.incoming.keys()):
                if pred not in existing:
                    del phi.incoming[pred]


class SimplifyCFGPass(LIRPass):
    """
    LIR CFG cleanup pass.

    Options (all default to True):
      - remove_unreachable
      - thread_trampolines
      - merge_blocks
      - simplify_cond_jump
    """

    @property
    def name(self) -> str:
        return "simplify-cfg"

    def run(self, lir: LIRFunction, config: PassConfig) -> LIRFunction:
        self._init_metrics()

        before_blocks = len(lir.blocks)

        opts = {
            "remove_unreachable": config.options.get("remove_unreachable", True),
            "thread_trampolines": config.options.get("thread_trampolines", True),
            "merge_blocks": config.options.get("merge_blocks", True),
            "simplify_cond_jump": config.options.get("simplify_cond_jump", True),
        }

        removed_unreachable = 0
        trampolines_threaded = 0
        blocks_merged = 0
        cond_simplified = 0

        changed = True
        while changed:
            changed = False

            # Remove unreachable blocks
            if opts["remove_unreachable"]:
                reachable = _reachable_blocks(lir)
                for name in list(lir.blocks.keys()):
                    if name not in reachable and name != lir.entry:
                        del lir.blocks[name]
                        removed_unreachable += 1
                        changed = True
                if changed:
                    _prune_phi_incomings(lir)
                if changed:
                    continue

            preds = _compute_preds(lir)

            # Simplify cond_jump with identical targets
            if opts["simplify_cond_jump"]:
                for block in lir.blocks.values():
                    term = block.terminator
                    if term and term.opcode == LIROpcode.COND_JUMP:
                        _, t, f = term.operands
                        if t == f:
                            block.terminator = LIRInst(LIROpcode.JUMP, None, [t], "flow")
                            cond_simplified += 1
                            changed = True
                if changed:
                    continue

            # Thread jump-only blocks with a single predecessor
            if opts["thread_trampolines"]:
                for name, block in list(lir.blocks.items()):
                    if name == lir.entry:
                        continue
                    if block.phis or block.instructions:
                        continue
                    term = block.terminator
                    if term is None or term.opcode != LIROpcode.JUMP:
                        continue
                    pred_set = preds.get(name, set())
                    if len(pred_set) != 1:
                        continue
                    pred = next(iter(pred_set))
                    target = term.operands[0]
                    if target not in lir.blocks:
                        continue
                    # Ensure phi rewrite is safe
                    if not _can_redirect_phi(lir.blocks[target], name, pred):
                        continue

                    pred_term = lir.blocks[pred].terminator
                    if pred_term is None:
                        continue
                    if pred_term.opcode == LIROpcode.JUMP and pred_term.operands[0] == name:
                        pred_term.operands[0] = target
                    elif pred_term.opcode == LIROpcode.COND_JUMP:
                        if pred_term.operands[1] == name:
                            pred_term.operands[1] = target
                        if pred_term.operands[2] == name:
                            pred_term.operands[2] = target
                    else:
                        continue

                    _redirect_phi(lir.blocks[target], name, pred)
                    del lir.blocks[name]
                    trampolines_threaded += 1
                    changed = True
                    break
                if changed:
                    continue

            # Merge blocks with single predecessor and no phis
            if opts["merge_blocks"]:
                preds = _compute_preds(lir)
                for name, block in list(lir.blocks.items()):
                    if name == lir.entry:
                        continue
                    pred_set = preds.get(name, set())
                    if len(pred_set) != 1:
                        continue
                    pred = next(iter(pred_set))
                    pred_block = lir.blocks[pred]
                    pred_term = pred_block.terminator
                    if pred_term is None or pred_term.opcode != LIROpcode.JUMP:
                        continue
                    if pred_term.operands[0] != name:
                        continue
                    if block.phis:
                        continue

                    succs = _successors(block)
                    safe = True
                    for succ in succs:
                        if succ not in lir.blocks:
                            continue
                        if not _can_redirect_phi(lir.blocks[succ], name, pred):
                            safe = False
                            break
                    if not safe:
                        continue

                    pred_block.instructions.extend(block.instructions)
                    pred_block.terminator = block.terminator
                    for succ in succs:
                        if succ in lir.blocks:
                            _redirect_phi(lir.blocks[succ], name, pred)
                    del lir.blocks[name]
                    blocks_merged += 1
                    changed = True
                    break

        if self._metrics:
            self._metrics.custom = {
                "blocks_before": before_blocks,
                "blocks_after": len(lir.blocks),
                "unreachable_removed": removed_unreachable,
                "trampolines_threaded": trampolines_threaded,
                "blocks_merged": blocks_merged,
                "cond_jumps_simplified": cond_simplified,
            }

        return lir
