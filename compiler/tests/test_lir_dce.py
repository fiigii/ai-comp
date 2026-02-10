"""Tests for LIR Dead Code Elimination pass."""

import unittest

from compiler.tests.conftest import (
    LIRFunction,
    BasicBlock,
    LIRInst,
    LIROpcode,
    Phi,
)
from compiler import LIRDCEPass, PassConfig


class TestLIRDCEPass(unittest.TestCase):
    """Tests for LIR DCE pass."""

    def _run(self, lir, options=None):
        p = LIRDCEPass()
        cfg = PassConfig(name=p.name, options=options or {})
        result = p.run(lir, cfg)
        return result, p.get_metrics()

    def test_dead_const_removed(self):
        """Dead constant is removed."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # dead
                LIRInst(LIROpcode.CONST, 1, [100], "load"),  # used by store
                LIRInst(LIROpcode.STORE, None, [1, 1], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        self.assertEqual(len(out.blocks["entry"].instructions), 2)
        self.assertEqual(metrics.custom["instructions_removed"], 1)

    def test_dead_alu_removed(self):
        """Dead ALU instruction is removed."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [1], "load"),
                LIRInst(LIROpcode.CONST, 1, [2], "load"),
                LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),  # dead - result unused
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        # All instructions are dead (no side effects, no uses)
        self.assertEqual(len(out.blocks["entry"].instructions), 0)
        self.assertEqual(metrics.custom["instructions_removed"], 3)

    def test_store_kept(self):
        """Store instruction is never removed (side effect)."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [100], "load"),
                LIRInst(LIROpcode.CONST, 1, [42], "load"),
                LIRInst(LIROpcode.STORE, None, [0, 1], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        # All kept because store uses them
        self.assertEqual(len(out.blocks["entry"].instructions), 3)
        self.assertEqual(metrics.custom["instructions_removed"], 0)

    def test_vstore_kept(self):
        """Vstore instruction is never removed (side effect)."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [100], "load"),
                LIRInst(LIROpcode.VSTORE, None, [0, [1, 2, 3, 4, 5, 6, 7, 8]], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        self.assertEqual(len(out.blocks["entry"].instructions), 2)

    def test_transitive_liveness(self):
        """Instructions used by live instructions are kept."""
        # s0 = 1, s1 = 2, s2 = s0 + s1, store uses s2
        # All should be kept due to transitive liveness
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [1], "load"),
                LIRInst(LIROpcode.CONST, 1, [2], "load"),
                LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),
                LIRInst(LIROpcode.CONST, 3, [100], "load"),
                LIRInst(LIROpcode.STORE, None, [3, 2], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        self.assertEqual(len(out.blocks["entry"].instructions), 5)
        self.assertEqual(metrics.custom["instructions_removed"], 0)

    def test_dead_copy_removed(self):
        """Dead COPY instructions are removed."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),  # dead
                LIRInst(LIROpcode.COPY, 2, [1], "alu"),  # dead
                LIRInst(LIROpcode.COPY, 3, [2], "alu"),  # dead
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        # All dead - no side effects, no uses
        self.assertEqual(len(out.blocks["entry"].instructions), 0)
        self.assertEqual(metrics.custom["instructions_removed"], 4)

    def test_copy_chain_partial_live(self):
        """Only the live part of a copy chain is kept."""
        # s0 = 42, s1 = copy(s0), s2 = copy(s1), store uses s1
        # s0 and s1 kept, s2 dead
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),
                LIRInst(LIROpcode.COPY, 2, [1], "alu"),  # dead
                LIRInst(LIROpcode.CONST, 3, [100], "load"),
                LIRInst(LIROpcode.STORE, None, [3, 1], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        self.assertEqual(len(out.blocks["entry"].instructions), 4)
        self.assertEqual(metrics.custom["instructions_removed"], 1)

    def test_vector_instruction_dead(self):
        """Dead vector instructions are removed."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),
                LIRInst(LIROpcode.VBROADCAST, [1, 2, 3, 4, 5, 6, 7, 8], [0], "valu"),  # dead
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        self.assertEqual(len(out.blocks["entry"].instructions), 0)
        self.assertEqual(metrics.custom["instructions_removed"], 2)

    def test_vector_instruction_live(self):
        """Vector instructions used by vstore are kept."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),
                LIRInst(LIROpcode.VBROADCAST, [1, 2, 3, 4, 5, 6, 7, 8], [0], "valu"),
                LIRInst(LIROpcode.CONST, 9, [100], "load"),
                LIRInst(LIROpcode.VSTORE, None, [9, [1, 2, 3, 4, 5, 6, 7, 8]], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        self.assertEqual(len(out.blocks["entry"].instructions), 4)
        self.assertEqual(metrics.custom["instructions_removed"], 0)

    def test_cross_block_liveness(self):
        """Liveness propagates across blocks."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # used in block2
                LIRInst(LIROpcode.CONST, 1, [99], "load"),  # dead
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["block2"], "flow"),
        )
        block2 = BasicBlock(
            name="block2",
            instructions=[
                LIRInst(LIROpcode.CONST, 2, [100], "load"),
                LIRInst(LIROpcode.STORE, None, [2, 0], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "block2": block2})
        out, metrics = self._run(lir)

        self.assertEqual(len(out.blocks["entry"].instructions), 1)  # s0 kept
        self.assertEqual(len(out.blocks["block2"].instructions), 2)
        self.assertEqual(metrics.custom["instructions_removed"], 1)

    def test_phi_keeps_incoming_live(self):
        """Phi incoming values are considered uses."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # used by phi
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        merge = BasicBlock(
            name="merge",
            phis=[Phi(dest=1, incoming={"entry": 0})],
            instructions=[
                LIRInst(LIROpcode.CONST, 2, [100], "load"),
                LIRInst(LIROpcode.STORE, None, [2, 1], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "merge": merge})
        out, metrics = self._run(lir)

        # s0 is kept because it's used by phi, which feeds store
        self.assertEqual(len(out.blocks["entry"].instructions), 1)
        self.assertEqual(metrics.custom["instructions_removed"], 0)

    def test_cond_jump_operand_live(self):
        """Condition used by cond_jump is kept live."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [1], "load"),  # condition
                LIRInst(LIROpcode.CONST, 1, [99], "load"),  # dead
            ],
            terminator=LIRInst(LIROpcode.COND_JUMP, None, [0, "then", "else"], "flow"),
        )
        then_block = BasicBlock(
            name="then",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        else_block = BasicBlock(
            name="else",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(
            entry="entry",
            blocks={"entry": entry, "then": then_block, "else": else_block}
        )
        out, metrics = self._run(lir)

        # s0 kept (used by cond_jump), s1 removed (dead)
        self.assertEqual(len(out.blocks["entry"].instructions), 1)
        self.assertEqual(metrics.custom["instructions_removed"], 1)

    def test_metrics_tracked(self):
        """Verify metrics are tracked correctly."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [1], "load"),  # dead
                LIRInst(LIROpcode.CONST, 1, [2], "load"),  # dead
                LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),  # dead
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        self.assertEqual(metrics.custom["instructions_before"], 3)
        self.assertEqual(metrics.custom["instructions_removed"], 3)

    def test_load_offset_lane_liveness(self):
        """LOAD_OFFSET liveness should be tracked on dest+offset lane scratch."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [1000], "load"),  # base for lane-address vector
                LIRInst(LIROpcode.CONST, 5, [200], "load"),   # lane addr scratch[5]
                LIRInst(LIROpcode.LOAD_OFFSET, 1, [0, 5], "load"),  # defines scratch[6]
                LIRInst(LIROpcode.CONST, 10, [500], "load"),
                LIRInst(LIROpcode.STORE, None, [10, 6], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out, metrics = self._run(lir)

        # scratch[0] is dead; scratch[5] -> load_offset -> store chain remains live.
        self.assertEqual(len(out.blocks["entry"].instructions), 4)
        self.assertEqual(metrics.custom["instructions_removed"], 1)


if __name__ == "__main__":
    unittest.main()
