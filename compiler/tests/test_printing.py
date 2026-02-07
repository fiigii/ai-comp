"""Tests for MIR/VLIW printing order."""

import io
import unittest
from contextlib import redirect_stdout

from compiler.lir import LIROpcode
from compiler.mir import MachineBasicBlock, MachineFunction, MBundle, MachineInst
from compiler.printing import print_mir, print_vliw


class TestPrinting(unittest.TestCase):
    def test_print_mir_sorts_bundle_by_engine_order(self):
        bundle = MBundle(
            instructions=[
                MachineInst(LIROpcode.ADD, 1, [2, 3], "alu"),
                MachineInst(LIROpcode.STORE, None, [10, 1], "store"),
                MachineInst(LIROpcode.CONST, 2, [7], "load"),
                MachineInst(LIROpcode.VADD, list(range(8, 16)), [list(range(16, 24)), list(range(24, 32))], "valu"),
                MachineInst(LIROpcode.JUMP, None, ["exit"], "flow"),
            ]
        )
        mfunc = MachineFunction(
            entry="entry",
            blocks={
                "entry": MachineBasicBlock(
                    name="entry",
                    bundles=[bundle],
                    predecessors=[],
                    successors=[],
                )
            },
        )

        out = io.StringIO()
        with redirect_stdout(out):
            print_mir(mfunc)
        text = out.getvalue()

        load_pos = text.find("[load]")
        alu_pos = text.find("[alu]")
        valu_pos = text.find("[valu]")
        store_pos = text.find("[store]")
        flow_pos = text.find("[flow]")

        self.assertTrue(load_pos < alu_pos < valu_pos < store_pos < flow_pos)

    def test_print_vliw_sorts_bundle_keys_by_engine_order(self):
        bundles = [
            {
                "flow": [("jump", 3)],
                "store": [("store", 1, 2)],
                "alu": [("+", 0, 1, 2)],
                "load": [("const", 1, 7)],
                "valu": [("v+", list(range(8)), list(range(16)), list(range(24)))],
            }
        ]

        out = io.StringIO()
        with redirect_stdout(out):
            print_vliw(bundles)
        text = out.getvalue()

        load_pos = text.find("'load'")
        alu_pos = text.find("'alu'")
        valu_pos = text.find("'valu'")
        store_pos = text.find("'store'")
        flow_pos = text.find("'flow'")

        self.assertTrue(load_pos < alu_pos < valu_pos < store_pos < flow_pos)


if __name__ == "__main__":
    unittest.main()
