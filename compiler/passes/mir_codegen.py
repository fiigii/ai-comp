"""
MIR to VLIW Codegen Pass

Converts MIR (after register allocation) to VLIW bundle format.
"""

from __future__ import annotations

from typing import Optional

from ..pass_manager import MIRCodegenPass, PassConfig
from ..lir import LIROpcode
from ..mir import MachineFunction, MachineBasicBlock, MBundle, MachineInst


def _inst_to_slot(inst: MachineInst, zero_scratch: int) -> Optional[tuple]:
    """Convert a MachineInst to a machine slot tuple."""
    match inst.opcode:
        # ALU operations
        case LIROpcode.ADD | LIROpcode.SUB | LIROpcode.MUL | LIROpcode.DIV | \
             LIROpcode.MOD | LIROpcode.XOR | LIROpcode.AND | LIROpcode.OR | \
             LIROpcode.SHL | LIROpcode.SHR | LIROpcode.LT | LIROpcode.EQ:
            return (inst.opcode.value, inst.dest, inst.operands[0], inst.operands[1])

        # Load operations
        case LIROpcode.CONST:
            return ("const", inst.dest, inst.operands[0])
        case LIROpcode.LOAD:
            return ("load", inst.dest, inst.operands[0])
        case LIROpcode.LOAD_OFFSET:
            return ("load_offset", inst.dest, inst.operands[0], inst.operands[1])

        # Store operations
        case LIROpcode.STORE:
            return ("store", inst.operands[0], inst.operands[1])

        # Flow operations
        case LIROpcode.SELECT:
            return ("select", inst.dest, inst.operands[0], inst.operands[1], inst.operands[2])
        case LIROpcode.JUMP:
            return ("jump", inst.operands[0])
        case LIROpcode.COND_JUMP:
            # Handled specially in codegen
            return None
        case LIROpcode.HALT:
            return ("halt",)
        case LIROpcode.PAUSE:
            return ("pause",)

        # Pseudo-ops
        case LIROpcode.COPY:
            # Should have been converted to ADD by register allocation
            return ("+", inst.dest, inst.operands[0], zero_scratch)

        # Vector ALU operations
        case LIROpcode.VADD | LIROpcode.VSUB | LIROpcode.VMUL | LIROpcode.VDIV | \
             LIROpcode.VMOD | LIROpcode.VXOR | LIROpcode.VAND | LIROpcode.VOR | \
             LIROpcode.VSHL | LIROpcode.VSHR | LIROpcode.VLT | LIROpcode.VEQ:
            scalar_op = inst.opcode.value[1:]  # "v+" -> "+"
            dest_base = inst.dest[0]
            a_base = inst.operands[0][0]
            b_base = inst.operands[1][0]
            return (scalar_op, dest_base, a_base, b_base)

        case LIROpcode.VBROADCAST:
            dest_base = inst.dest[0]
            scalar = inst.operands[0]
            return ("vbroadcast", dest_base, scalar)

        case LIROpcode.MULTIPLY_ADD:
            dest_base = inst.dest[0]
            a_base = inst.operands[0][0]
            b_base = inst.operands[1][0]
            c_base = inst.operands[2][0]
            return ("multiply_add", dest_base, a_base, b_base, c_base)

        case LIROpcode.VLOAD:
            dest_base = inst.dest[0]
            addr = inst.operands[0]
            return ("vload", dest_base, addr)

        case LIROpcode.VSTORE:
            addr = inst.operands[0]
            src_base = inst.operands[1][0]
            return ("vstore", addr, src_base)

        case LIROpcode.VSELECT:
            dest_base = inst.dest[0]
            cond_base = inst.operands[0][0]
            a_base = inst.operands[1][0]
            b_base = inst.operands[2][0]
            return ("vselect", dest_base, cond_base, a_base, b_base)

        case _:
            raise NotImplementedError(f"Codegen for {inst.opcode}")


def _find_zero_scratch(mir: MachineFunction) -> int:
    """Find or determine zero scratch address for COPY expansion."""
    # Look for an existing const 0
    for block in mir.blocks.values():
        for bundle in block.bundles:
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.CONST and inst.operands[0] == 0:
                    return inst.dest

    # If not found, use max_scratch_used + 1 (but this shouldn't happen
    # since MIRRegisterAllocationPass should have added one)
    return mir.max_scratch_used + 1


class MIRToVLIWPass(MIRCodegenPass):
    """
    Pass that generates VLIW bundles from MIR.

    Linearizes the MIR control flow graph and converts MBundles to VLIW
    instruction bundles (dict format).
    """

    @property
    def name(self) -> str:
        return "mir-to-vliw"

    def run(self, mir: MachineFunction, config: PassConfig) -> list[dict]:
        """Generate VLIW bundles from MIR."""
        if not mir.phi_eliminated:
            raise RuntimeError(
                "mir-to-vliw: MIR was produced without phi-elimination. "
                "Run phi-elimination before lir-to-mir lowering."
            )
        self._init_metrics()

        # Find zero constant for COPY expansion
        zero_scratch = _find_zero_scratch(mir)

        # Linearize blocks
        block_order = mir.get_block_order()

        # Generate bundles and track block positions for label resolution
        bundles: list[dict] = []
        bundle_positions: dict[str, int] = {}  # Block name -> bundle index

        for i, block_name in enumerate(block_order):
            block = mir.blocks[block_name]
            bundle_positions[block_name] = len(bundles)

            for j, mbundle in enumerate(block.bundles):
                # Check for terminator
                term = mbundle.get_terminator()

                if term and term.opcode == LIROpcode.COND_JUMP:
                    # Handle COND_JUMP expansion specially
                    # First emit non-terminator instructions in this bundle
                    vliw_bundle: dict = {}
                    for inst in mbundle.instructions:
                        if not inst.is_terminator():
                            slot = _inst_to_slot(inst, zero_scratch)
                            if slot:
                                engine = inst.engine
                                if engine not in vliw_bundle:
                                    vliw_bundle[engine] = []
                                vliw_bundle[engine].append(slot)

                    # Now emit cond_jump (labels resolved in final pass)
                    cond = term.operands[0]
                    true_target = term.operands[1]
                    false_target = term.operands[2]

                    cond_slot = ("cond_jump", cond, true_target)
                    if "flow" not in vliw_bundle:
                        vliw_bundle["flow"] = []
                    vliw_bundle["flow"].append(cond_slot)

                    if vliw_bundle:
                        bundles.append(vliw_bundle)

                    # Check if we need to emit jump for false target
                    # False target is fallthrough if it's the next block
                    next_block = block_order[i + 1] if i + 1 < len(block_order) else None
                    if false_target != next_block:
                        jump_bundle = {"flow": [("jump", false_target)]}
                        bundles.append(jump_bundle)
                else:
                    # Normal bundle
                    vliw_bundle = {}

                    # Check if this bundle has a JUMP to the next block (fallthrough)
                    # If so, skip the JUMP instruction
                    next_block = block_order[i + 1] if i + 1 < len(block_order) else None
                    skip_jump_to = None
                    term = mbundle.get_terminator()
                    if term and term.opcode == LIROpcode.JUMP:
                        if term.operands[0] == next_block:
                            skip_jump_to = next_block

                    for inst in mbundle.instructions:
                        # Skip JUMP to fallthrough block
                        if (skip_jump_to and inst.opcode == LIROpcode.JUMP and
                            inst.operands[0] == skip_jump_to):
                            continue

                        slot = _inst_to_slot(inst, zero_scratch)
                        if slot:
                            engine = inst.engine
                            if engine not in vliw_bundle:
                                vliw_bundle[engine] = []
                            vliw_bundle[engine].append(slot)

                    if vliw_bundle:
                        bundles.append(vliw_bundle)

        # Resolve label references in jumps using bundle_positions
        for bundle in bundles:
            if "flow" in bundle:
                new_flow = []
                for slot in bundle["flow"]:
                    if slot[0] == "jump":
                        target = slot[1]
                        if isinstance(target, str) and target in bundle_positions:
                            new_flow.append(("jump", bundle_positions[target]))
                        else:
                            new_flow.append(slot)
                    elif slot[0] == "cond_jump":
                        cond, target = slot[1], slot[2]
                        if isinstance(target, str) and target in bundle_positions:
                            new_flow.append(("cond_jump", cond, bundle_positions[target]))
                        else:
                            new_flow.append(slot)
                    else:
                        new_flow.append(slot)
                bundle["flow"] = new_flow

        # Record metrics
        if self._metrics:
            self._metrics.custom = {
                "bundles": len(bundles),
            }

        return bundles
