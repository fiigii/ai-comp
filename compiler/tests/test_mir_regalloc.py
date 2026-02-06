"""Tests for MIR register allocation."""

import unittest

from compiler.mir import MachineFunction, MachineBasicBlock, MBundle, MachineInst
from compiler.lir import LIROpcode
from compiler.passes import MIRRegisterAllocationPass
from compiler.pass_manager import PassConfig


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


if __name__ == "__main__":
    unittest.main()
