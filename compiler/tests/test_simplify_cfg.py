"""Tests for SimplifyCFG LIR pass."""

import unittest

from compiler.tests.conftest import (
    LIRFunction,
    BasicBlock,
    LIRInst,
    LIROpcode,
    Phi,
)
from compiler import SimplifyCFGPass, PassConfig


class TestSimplifyCFGPass(unittest.TestCase):
    """Tests for SimplifyCFG LIR pass."""

    def _run(self, lir, options=None):
        p = SimplifyCFGPass()
        cfg = PassConfig(name=p.name, options=options or {})
        return p.run(lir, cfg)

    def test_removes_unreachable_and_prunes_phi(self):
        """Unreachable blocks are removed and phi incomings pruned."""
        entry = BasicBlock(
            name="entry",
            instructions=[],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            phis=[Phi(dest=0, incoming={"entry": 1, "dead": 2})],
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        dead = BasicBlock(
            name="dead",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "exit": exit_block, "dead": dead})
        out = self._run(lir, options={"merge_blocks": False})
        self.assertNotIn("dead", out.blocks)
        self.assertEqual(set(out.blocks["exit"].phis[0].incoming.keys()), {"entry"})

    def test_simplifies_cond_jump_same_targets(self):
        """cond_jump with identical targets becomes jump."""
        entry = BasicBlock(
            name="entry",
            instructions=[],
            terminator=LIRInst(LIROpcode.COND_JUMP, None, [0, "exit", "exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "exit": exit_block})
        out = self._run(lir, options={"merge_blocks": False})
        term = out.blocks["entry"].terminator
        self.assertEqual(term.opcode, LIROpcode.JUMP)
        self.assertEqual(term.operands, ["exit"])

    def test_threads_trampoline_and_redirects_phi(self):
        """Threading jump-only block rewrites preds and phi inputs."""
        entry = BasicBlock(
            name="entry",
            instructions=[],
            terminator=LIRInst(LIROpcode.JUMP, None, ["tramp"], "flow"),
        )
        tramp = BasicBlock(
            name="tramp",
            instructions=[],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            phis=[Phi(dest=0, incoming={"tramp": 1})],
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "tramp": tramp, "exit": exit_block})
        out = self._run(lir)
        self.assertNotIn("tramp", out.blocks)
        self.assertEqual(out.blocks["entry"].terminator.operands, ["exit"])
        self.assertEqual(set(out.blocks["exit"].phis[0].incoming.keys()), {"entry"})

    def test_merges_block_with_single_pred(self):
        """Merge block with single pred and no phis."""
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [0], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["mid"], "flow"),
        )
        mid = BasicBlock(
            name="mid",
            instructions=[LIRInst(LIROpcode.CONST, 1, [1], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            phis=[Phi(dest=2, incoming={"mid": 1})],
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "mid": mid, "exit": exit_block})
        out = self._run(lir)
        self.assertNotIn("mid", out.blocks)
        self.assertEqual(len(out.blocks["entry"].instructions), 2)
        self.assertEqual(out.blocks["entry"].terminator.operands, ["exit"])
        self.assertEqual(set(out.blocks["exit"].phis[0].incoming.keys()), {"entry"})


if __name__ == "__main__":
    unittest.main()
