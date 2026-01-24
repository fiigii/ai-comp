"""
Copy Propagation Pass (LIR)

Eliminates unnecessary COPY instructions in SSA form by propagating
the source scratch address directly to all uses of the copy destination.

COPY is a pseudo-op that becomes `ADD dest, src, zero` in codegen,
costing one ALU slot per copy. This pass propagates copy sources to
uses, making the COPY instructions dead (removed later by DCE).

This pass runs on SSA form (before phi elimination), where each scratch
address is defined exactly once. This simplifies the algorithm to a
simple global copy map with transitive closure.
"""

from __future__ import annotations

from ..pass_manager import LIRPass, PassConfig
from ..lir import LIRFunction, LIROpcode


class CopyPropagationPass(LIRPass):
    """
    Copy propagation pass for LIR in SSA form.

    Replaces uses of COPY destinations with the original source,
    making the COPY instructions dead for subsequent DCE.

    Algorithm:
    1. Build global copy map (dest -> src, transitively resolved)
    2. Rewrite all operands using the copy map
    3. Rewrite phi incoming values using the copy map
    """

    @property
    def name(self) -> str:
        return "copy-propagation"

    def run(self, lir: LIRFunction, config: PassConfig) -> LIRFunction:
        self._init_metrics()
        self._copies_found = 0
        self._operands_propagated = 0

        # Pass 1: Build global copy map (SSA guarantees single definition)
        copy_map: dict[int, int] = {}
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.opcode == LIROpcode.COPY:
                    self._copies_found += 1
                    src = inst.operands[0]
                    # Resolve transitively
                    while src in copy_map:
                        src = copy_map[src]
                    copy_map[inst.dest] = src

        # Pass 2: Rewrite all operands
        # Note: Skip instructions with immediate operands (not scratch references)
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.opcode == LIROpcode.CONST:
                    continue
                if inst.opcode == LIROpcode.LOAD_OFFSET and len(inst.operands) == 2:
                    rewritten = self._rewrite_operands([inst.operands[0]], copy_map)
                    inst.operands = [rewritten[0], inst.operands[1]]
                    continue
                inst.operands = self._rewrite_operands(inst.operands, copy_map)
            if block.terminator:
                block.terminator.operands = self._rewrite_operands(
                    block.terminator.operands, copy_map
                )
            # Rewrite phi incoming values
            for phi in block.phis:
                for pred, src in list(phi.incoming.items()):
                    if src in copy_map:
                        phi.incoming[pred] = copy_map[src]
                        self._operands_propagated += 1

        if self._metrics:
            self._metrics.custom = {
                "copies_found": self._copies_found,
                "operands_propagated": self._operands_propagated,
            }

        return lir

    def _rewrite_operands(self, operands: list, copy_map: dict[int, int]) -> list:
        """Rewrite operands using the copy map.

        Handles scalar operands (int). Vector operands (list[int]) are only
        rewritten if ALL elements map to a contiguous range of sources -
        the machine requires contiguous scratch addresses for vector operations.

        Non-integer operands (labels, immediates) are left unchanged.
        """
        result = []
        for op in operands:
            if isinstance(op, int) and op in copy_map:
                result.append(copy_map[op])
                self._operands_propagated += 1
            elif isinstance(op, list):
                # Vector operand: only rewrite if sources form contiguous range
                rewritten = self._try_rewrite_vector_operand(op, copy_map)
                result.append(rewritten)
            else:
                result.append(op)
        return result

    def _try_rewrite_vector_operand(self, vec: list[int], copy_map: dict[int, int]) -> list[int]:
        """Try to rewrite a vector operand if sources are contiguous.

        If all elements are COPY dests and their sources form a contiguous
        range [base, base+1, ..., base+n-1], rewrite to use sources directly.
        Otherwise return the original vector unchanged.
        """
        # Check if all elements have mappings
        sources = []
        for s in vec:
            if s in copy_map:
                sources.append(copy_map[s])
            else:
                # Not all elements are COPY dests, can't rewrite
                return vec

        # Check if sources form a contiguous range
        if len(sources) == 0:
            return vec

        base = sources[0]
        for i, src in enumerate(sources):
            if src != base + i:
                # Not contiguous, can't rewrite
                return vec

        # Sources are contiguous! Rewrite the vector operand
        self._operands_propagated += len(sources)
        return sources
