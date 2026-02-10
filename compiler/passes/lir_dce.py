"""
LIR Dead Code Elimination Pass

Removes LIR instructions whose results are never used.
Runs after copy propagation to eliminate dead COPY instructions.
"""

from __future__ import annotations

from ..pass_manager import LIRPass, PassConfig
from ..lir import LIRFunction, LIROpcode, LIRInst


# Instructions with side effects that should never be eliminated
SIDE_EFFECT_OPCODES = {
    LIROpcode.STORE,
    LIROpcode.VSTORE,
    LIROpcode.HALT,
    LIROpcode.PAUSE,
}


class LIRDCEPass(LIRPass):
    """
    Dead Code Elimination for LIR.

    Algorithm:
    1. Mark all uses (operands referenced by live instructions)
    2. Instructions are live if:
       - They have side effects (store, vstore, halt, pause)
       - Their result is used by a live instruction
    3. Remove dead instructions
    """

    @property
    def name(self) -> str:
        return "lir-dce"

    def run(self, lir: LIRFunction, config: PassConfig) -> LIRFunction:
        self._init_metrics()

        instructions_before = sum(len(b.instructions) for b in lir.blocks.values())
        instructions_removed = 0

        # Iterate until no changes (for transitive dead code)
        changed = True
        while changed:
            changed = False

            # Pass 1: Compute used scratches
            used: set[int] = set()

            for block in lir.blocks.values():
                # Collect uses from terminator
                if block.terminator:
                    self._collect_uses(block.terminator, used)

                # Collect uses from instructions (backward, only from live ones)
                # First pass: mark all side-effect instructions as live
                for inst in block.instructions:
                    if self._has_side_effects(inst):
                        self._collect_uses(inst, used)

                # Collect uses from phis
                for phi in block.phis:
                    # Phi results might be used
                    for src in phi.incoming.values():
                        if isinstance(src, int):
                            used.add(src)

            # Iteratively mark instructions as live if their result is used
            prev_used_size = -1
            while len(used) != prev_used_size:
                prev_used_size = len(used)
                for block in lir.blocks.values():
                    for inst in block.instructions:
                        defs = inst.get_defs()
                        if defs and defs.intersection(used):
                            self._collect_uses(inst, used)

            # Pass 2: Remove dead instructions
            for block in lir.blocks.values():
                new_instructions = []
                for inst in block.instructions:
                    if self._is_live(inst, used):
                        new_instructions.append(inst)
                    else:
                        instructions_removed += 1
                        changed = True

                block.instructions = new_instructions

        if self._metrics:
            self._metrics.custom = {
                "instructions_before": instructions_before,
                "instructions_removed": instructions_removed,
            }

        return lir

    def _has_side_effects(self, inst: LIRInst) -> bool:
        """Check if instruction has side effects."""
        return inst.opcode in SIDE_EFFECT_OPCODES

    def _is_live(self, inst: LIRInst, used: set[int]) -> bool:
        """Check if instruction is live."""
        # Side effects are always live
        if self._has_side_effects(inst):
            return True

        defs = inst.get_defs()
        if not defs:
            return False
        return bool(defs.intersection(used))

    def _collect_uses(self, inst: LIRInst, used: set[int]) -> None:
        """Collect all scratch addresses used by an instruction.

        Note: CONST operands are immediate values, not scratch references.
        """
        used.update(inst.get_uses())
