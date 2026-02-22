"""Tests for codegen CFG optimizations."""

import unittest

from compiler.tests.conftest import (
    LIRFunction,
    BasicBlock,
    LIRInst,
    LIROpcode,
    Phi,
    compile_to_vliw,
    _cfg,
)
from compiler.passes import LIRToMIRPass, InstSchedulingPass, MIRToVLIWPass
from compiler.mir import MachineFunction


class TestCodegenCFGOptimizations(unittest.TestCase):
    """Tests for codegen CFG simplifications."""

    def _has_jump(self, bundles):
        for bundle in bundles:
            slots = bundle.get("flow", [])
            for slot in slots:
                if slot[0] == "jump":
                    return True
        return False

    def test_codegen_omits_jump_to_next_block(self):
        """Unconditional jump to fallthrough block is omitted."""
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [0], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "exit": exit_block})
        bundles = compile_to_vliw(lir)
        self.assertFalse(self._has_jump(bundles))

    def test_codegen_omits_false_fallthrough_jump(self):
        """Conditional jump omits false-target jump when it falls through."""
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [0], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["header"], "flow"),
        )
        header = BasicBlock(
            name="header",
            instructions=[],
            terminator=LIRInst(
                LIROpcode.COND_JUMP,
                None,
                [0, "header", "exit"],
                "flow",
            ),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(
            entry="entry",
            blocks={"entry": entry, "header": header, "exit": exit_block},
        )
        bundles = compile_to_vliw(lir)
        self.assertFalse(self._has_jump(bundles))


class TestPhiEliminationGuard(unittest.TestCase):
    """Tests that lowering passes reject LIR with remaining phi nodes."""

    def _lir_with_phi(self):
        """Build a minimal LIR that still contains a phi node."""
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [0], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["loop"], "flow"),
        )
        loop = BasicBlock(
            name="loop",
            phis=[Phi(dest=1, incoming={"entry": 0, "loop": 1})],
            instructions=[],
            terminator=LIRInst(LIROpcode.COND_JUMP, None, [1, "loop", "exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        return LIRFunction(
            entry="entry",
            blocks={"entry": entry, "loop": loop, "exit": exit_block},
        )

    def _lir_without_phi(self):
        """Build a minimal LIR with no phi nodes."""
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [0], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        return LIRFunction(
            entry="entry",
            blocks={"entry": entry, "exit": exit_block},
        )

    def test_lir_to_mir_rejects_remaining_phis(self):
        """lir-to-mir must raise if phi nodes have not been eliminated."""
        lir = self._lir_with_phi()
        with self.assertRaises(RuntimeError) as ctx:
            LIRToMIRPass().run(lir, _cfg("lir-to-mir"))
        self.assertIn("phi", str(ctx.exception).lower())

    def test_inst_scheduling_rejects_remaining_phis(self):
        """inst-scheduling must raise if phi nodes have not been eliminated."""
        lir = self._lir_with_phi()
        with self.assertRaises(RuntimeError) as ctx:
            InstSchedulingPass().run(lir, _cfg("inst-scheduling"))
        self.assertIn("phi", str(ctx.exception).lower())

    def test_mir_to_vliw_rejects_without_phi_eliminated_flag(self):
        """mir-to-vliw must raise if MachineFunction.phi_eliminated is False."""
        mfunc = MachineFunction(entry="entry", phi_eliminated=False)
        with self.assertRaises(RuntimeError) as ctx:
            MIRToVLIWPass().run(mfunc, _cfg("mir-to-vliw"))
        self.assertIn("phi", str(ctx.exception).lower())

    def test_lir_to_mir_accepts_phi_free_lir(self):
        """lir-to-mir succeeds when no phi nodes remain."""
        lir = self._lir_without_phi()
        mir = LIRToMIRPass().run(lir, _cfg("lir-to-mir"))
        self.assertTrue(mir.phi_eliminated)

    def test_pipeline_sets_phi_eliminated_flag(self):
        """lir-to-mir sets phi_eliminated so mir-to-vliw accepts the MIR."""
        lir = self._lir_without_phi()
        mir = LIRToMIRPass().run(lir, _cfg("lir-to-mir"))
        # Should not raise
        MIRToVLIWPass().run(mir, _cfg("mir-to-vliw"))


if __name__ == "__main__":
    unittest.main()
