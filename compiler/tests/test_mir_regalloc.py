"""Tests for MIR register allocation."""

import json
import os
import tempfile
import unittest

from compiler.mir import MachineFunction, MachineBasicBlock, MBundle, MachineInst
from compiler.lir import LIROpcode
from compiler.passes import MIRRegisterAllocationPass
from compiler.passes.mir_register_allocation import (
    LiveInterval,
    SpillInfo,
    _operand_is_immediate,
    _split_live_ranges,
)
from compiler.pass_manager import PassConfig
from compiler import HIRBuilder, compile_hir_to_vliw
from vm import Machine, DebugInfo, N_CORES


class TestMIRRegisterAllocation(unittest.TestCase):
    def test_scalar_reuses_freed_vector_range(self):
        # Bundle 0: vector def only (no uses)
        vec_inst = MachineInst(
            LIROpcode.VBROADCAST,
            dest=list(range(8)),
            operands=["imm"],  # non-int operand avoids introducing extra uses
            engine="valu",
        )
        bundle0 = MBundle(instructions=[vec_inst])

        # Bundle 1: scalar const (no uses)
        scalar_inst = MachineInst(
            LIROpcode.CONST,
            dest=8,
            operands=[1],
            engine="load",
        )
        bundle1 = MBundle(instructions=[scalar_inst])

        block = MachineBasicBlock(
            name="entry",
            bundles=[bundle0, bundle1],
            predecessors=[],
            successors=[],
        )
        mfunc = MachineFunction(entry="entry", blocks={"entry": block})

        cfg = PassConfig(name="mir-regalloc", enabled=True, options={})
        MIRRegisterAllocationPass().run(mfunc, cfg)

        # Scalar should reuse the freed vector range (base 0).
        self.assertEqual(
            mfunc.blocks["entry"].bundles[1].instructions[0].dest,
            0,
        )
        # High-water mark should reflect only the vector allocation.
        self.assertEqual(mfunc.max_scratch_used, 7)

    def test_use_then_def_in_same_bundle_can_reuse_register(self):
        # Bundle 0: define scalar v10
        b0 = MBundle(
            instructions=[MachineInst(LIROpcode.CONST, 10, [7], "load")]
        )
        # Bundle 1: use v10, define v11 (same bundle use->def should permit reuse)
        b1 = MBundle(
            instructions=[MachineInst(LIROpcode.ADD, 11, [10, 10], "alu")]
        )
        # Bundle 2: keep v11 live
        b2 = MBundle(
            instructions=[MachineInst(LIROpcode.ADD, 12, [11, 11], "alu")]
        )

        block = MachineBasicBlock(
            name="entry",
            bundles=[b0, b1, b2],
            predecessors=[],
            successors=[],
        )
        mfunc = MachineFunction(entry="entry", blocks={"entry": block})

        cfg = PassConfig(name="mir-regalloc", enabled=True, options={})
        MIRRegisterAllocationPass().run(mfunc, cfg)

        const_dest = mfunc.blocks["entry"].bundles[0].instructions[0].dest
        add1 = mfunc.blocks["entry"].bundles[1].instructions[0]
        self.assertEqual(add1.dest, const_dest)
        self.assertEqual(add1.operands[0], const_dest)
        self.assertEqual(add1.operands[1], const_dest)


class TestOperandIsImmediate(unittest.TestCase):
    """Unit tests for _operand_is_immediate (spill-path immediate immunity)."""

    def test_const_operand_is_immediate(self):
        inst = MachineInst(LIROpcode.CONST, 1, [42], "load")
        self.assertTrue(_operand_is_immediate(inst, 0))

    def test_load_offset_offset_operand_is_immediate(self):
        inst = MachineInst(LIROpcode.LOAD_OFFSET, 8, [3, 2], "load")
        self.assertTrue(_operand_is_immediate(inst, 1))
        # Operand 0 is the address base (a scratch register).
        self.assertFalse(_operand_is_immediate(inst, 0))

    def test_add_imm_immediate_operand_is_immediate(self):
        inst = MachineInst(LIROpcode.ADD_IMM, 5, [4, 7], "alu")
        self.assertTrue(_operand_is_immediate(inst, 1))
        # Operand 0 is the scratch source register.
        self.assertFalse(_operand_is_immediate(inst, 0))

    def test_jump_operands_are_immediate(self):
        inst = MachineInst(LIROpcode.JUMP, None, ["target"], "flow")
        self.assertTrue(_operand_is_immediate(inst, 0))

    def test_cond_jump_labels_are_immediate_condition_is_not(self):
        inst = MachineInst(LIROpcode.COND_JUMP, None, [3, "then", "else"],
                           "flow")
        self.assertFalse(_operand_is_immediate(inst, 0))
        self.assertTrue(_operand_is_immediate(inst, 1))
        self.assertTrue(_operand_is_immediate(inst, 2))

    def test_add_operands_are_not_immediate(self):
        inst = MachineInst(LIROpcode.ADD, 5, [1, 2], "alu")
        self.assertFalse(_operand_is_immediate(inst, 0))
        self.assertFalse(_operand_is_immediate(inst, 1))


class TestSpillPathCorrectness(unittest.TestCase):
    """Regression tests for the register spilling fixes."""

    def _run_program(self, instrs, mem):
        m = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        m.enable_pause = False
        m.enable_debug = False
        m.run()
        return m

    def test_spilled_program_with_colliding_immediates(self):
        """End-to-end spill correctness with immediate immunity.

        With 1700 simultaneously live scalars (exceeding the 1536-word
        scratch file), the allocator must spill. Spilled vreg ids are small
        integers that numerically collide with immediate operands in the
        program (e.g. CONST values, ADD_IMM immediates, LOAD_OFFSET
        offsets). Before _operand_is_immediate was applied on the spill
        rewrite paths, those immediates were renamed to fresh reload vregs
        and the program crashed (IndexError) or miscompiled; this exercises
        that immediate-immunity fix.
        """
        N = 1700

        b = HIRBuilder()
        vals = [b.load(b.const(10 + k)) for k in range(N)]
        # Store in reversed order so all N loaded values are simultaneously
        # live across the whole store sequence.
        for k in reversed(range(N)):
            b.store(b.const(4000 + k), vals[k])
        hir = b.build()

        # Enable spilling via a config override (off by default).
        config_path = os.path.join(os.path.dirname(__file__), "..",
                                   "pass_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["passes"]["mir-register-allocation"]["options"][
            "enable_spilling"] = True
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as tf:
            json.dump(cfg, tf)
            tmp_path = tf.name
        try:
            instrs = compile_hir_to_vliw(hir, pass_config=tmp_path)
        finally:
            os.unlink(tmp_path)

        # Spill area lives at mem[mem[6] + mem[2]] = mem[6000].
        mem = [0] * 8000
        mem[2] = 100
        mem[6] = 5900
        for k in range(N):
            mem[10 + k] = k * 7 + 1

        machine = self._run_program(instrs, mem)
        for k in range(N):
            self.assertEqual(machine.mem[4000 + k], k * 7 + 1,
                             f"wrong value stored at mem[{4000 + k}]")

    def test_partial_lane_vector_spill_saves_only_written_lanes(self):
        """A spilled vector whose bundle defines only some lanes must be
        saved with scalar STOREs of exactly those lanes. Previously
        _split_live_ranges emitted a full VSTORE after any def, which
        saved not-yet-defined lanes and clobbered the spill slot.
        """
        # Bundle 0 defines ONLY lanes 0 and 3 of vector vreg base 100.
        b0 = MBundle()
        b0.add_instruction(MachineInst(LIROpcode.CONST, 100, [5], "load"))
        b0.add_instruction(MachineInst(LIROpcode.CONST, 103, [9], "load"))
        # Bundle 1 references a high vreg so the fresh vregs allocated by
        # _split_live_ranges cannot collide with the vector's lanes.
        b1 = MBundle()
        b1.add_instruction(MachineInst(LIROpcode.ADD, 300, [300, 300], "alu"))
        block = MachineBasicBlock(name="entry", bundles=[b0, b1],
                                  predecessors=[], successors=[])
        mfunc = MachineFunction(entry="entry", blocks={"entry": block})

        vector_bases = {100}
        vector_addrs = set(range(100, 108))
        mem_offset = 16
        spill = SpillInfo(
            vreg=100,
            mem_offset=mem_offset,
            is_vector=True,
            interval=LiveInterval(vreg=100, start=0, end=10, is_vector=True),
        )

        _split_live_ranges(mfunc, [spill], vector_bases, vector_addrs)

        insts = [inst
                 for bundle in mfunc.blocks["entry"].bundles
                 for inst in bundle.instructions]

        # No full-vector save is allowed for a partially defined vector.
        vstores = [i for i in insts if i.opcode == LIROpcode.VSTORE]
        self.assertEqual(vstores, [])

        # Exactly the two written lanes are saved with scalar stores.
        stores = [i for i in insts if i.opcode == LIROpcode.STORE]
        self.assertEqual(sorted(s.operands[1] for s in stores), [100, 103])

        # The store addresses are computed from mem_offset + lane.
        const_values = [i.operands[0] for i in insts
                        if i.opcode == LIROpcode.CONST
                        and isinstance(i.operands[0], int)]
        self.assertIn(mem_offset + 0, const_values)
        self.assertIn(mem_offset + 3, const_values)


if __name__ == "__main__":
    unittest.main()
