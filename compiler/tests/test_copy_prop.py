"""Tests for CopyPropagation LIR pass."""

import unittest

from compiler.tests.conftest import (
    LIRFunction,
    BasicBlock,
    LIRInst,
    LIROpcode,
    Phi,
)
from compiler import CopyPropagationPass, PassConfig


class TestCopyPropagationPass(unittest.TestCase):
    """Tests for CopyPropagation LIR pass."""

    def _run(self, lir, options=None):
        p = CopyPropagationPass()
        cfg = PassConfig(name=p.name, options=options or {})
        return p.run(lir, cfg)

    def test_basic_propagation(self):
        """COPY followed by use of dest should use source directly."""
        # s1 = COPY s0
        # s2 = s1 + s1
        # After: s2 = s0 + s0
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # s0 = 42
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),     # s1 = COPY s0
                LIRInst(LIROpcode.ADD, 2, [1, 1], "alu"),   # s2 = s1 + s1
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out = self._run(lir)

        # The ADD should now use s0 instead of s1
        add_inst = out.blocks["entry"].instructions[2]
        self.assertEqual(add_inst.operands, [0, 0])

    def test_transitive_propagation(self):
        """Chain of COPYs should use original source."""
        # s1 = COPY s0
        # s2 = COPY s1
        # s3 = COPY s2
        # s4 = s3 + s0
        # After: s4 = s0 + s0
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # s0 = 42
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),     # s1 = COPY s0
                LIRInst(LIROpcode.COPY, 2, [1], "alu"),     # s2 = COPY s1
                LIRInst(LIROpcode.COPY, 3, [2], "alu"),     # s3 = COPY s2
                LIRInst(LIROpcode.ADD, 4, [3, 0], "alu"),   # s4 = s3 + s0
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out = self._run(lir)

        # The ADD should now use s0 instead of s3
        add_inst = out.blocks["entry"].instructions[4]
        self.assertEqual(add_inst.operands, [0, 0])

    def test_cross_block_propagation(self):
        """Propagate copies across basic blocks (SSA allows this)."""
        # entry: s1 = COPY s0; jump block2
        # block2: s2 = s1 + s1
        # After: block2: s2 = s0 + s0
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # s0 = 42
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),     # s1 = COPY s0
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["block2"], "flow"),
        )
        block2 = BasicBlock(
            name="block2",
            instructions=[
                LIRInst(LIROpcode.ADD, 2, [1, 1], "alu"),   # s2 = s1 + s1
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "block2": block2})
        out = self._run(lir)

        # The ADD in block2 should now use s0 instead of s1
        add_inst = out.blocks["block2"].instructions[0]
        self.assertEqual(add_inst.operands, [0, 0])

    def test_vector_operands_non_contiguous_not_propagated(self):
        """Vector operands with non-contiguous sources are NOT propagated."""
        # s8 = COPY s0  (source s0, but vec has s8 at position 0)
        # VSTORE [s8, s1, s2, s3, s4, s5, s6, s7], addr
        # Sources would be [0, ?, ?, ?, ?, ?, ?, ?] - not contiguous, so unchanged
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # s0 = 42
                LIRInst(LIROpcode.COPY, 8, [0], "alu"),     # s8 = COPY s0
                LIRInst(LIROpcode.VSTORE, None, [[8, 1, 2, 3, 4, 5, 6, 7], 100], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out = self._run(lir)

        # Not all elements are COPY dests (s1-s7 aren't), so unchanged
        vstore_inst = out.blocks["entry"].instructions[2]
        self.assertEqual(vstore_inst.operands[0], [8, 1, 2, 3, 4, 5, 6, 7])

    def test_vector_operands_contiguous_propagated(self):
        """Vector operands with contiguous sources ARE propagated."""
        # s8 = COPY s0, s9 = COPY s1, ..., s15 = COPY s7
        # VSTORE [s8, s9, s10, s11, s12, s13, s14, s15], addr
        # Sources [0, 1, 2, 3, 4, 5, 6, 7] are contiguous!
        # After: VSTORE [s0, s1, s2, s3, s4, s5, s6, s7], addr
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.COPY, 8, [0], "alu"),
                LIRInst(LIROpcode.COPY, 9, [1], "alu"),
                LIRInst(LIROpcode.COPY, 10, [2], "alu"),
                LIRInst(LIROpcode.COPY, 11, [3], "alu"),
                LIRInst(LIROpcode.COPY, 12, [4], "alu"),
                LIRInst(LIROpcode.COPY, 13, [5], "alu"),
                LIRInst(LIROpcode.COPY, 14, [6], "alu"),
                LIRInst(LIROpcode.COPY, 15, [7], "alu"),
                LIRInst(LIROpcode.VSTORE, None, [[8, 9, 10, 11, 12, 13, 14, 15], 100], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out = self._run(lir)

        # Sources are contiguous, so vector operand gets rewritten
        vstore_inst = out.blocks["entry"].instructions[8]
        self.assertEqual(vstore_inst.operands[0], [0, 1, 2, 3, 4, 5, 6, 7])

    def test_phi_incoming_rewrite(self):
        """Phi incoming values that reference COPY dests get rewritten."""
        # entry: s1 = COPY s0; jump merge
        # other: s2 = COPY s0; jump merge
        # merge: s3 = phi(entry:s1, other:s2)
        # After: s3 = phi(entry:s0, other:s0)
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # s0 = 42
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),     # s1 = COPY s0
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        other = BasicBlock(
            name="other",
            instructions=[
                LIRInst(LIROpcode.COPY, 2, [0], "alu"),     # s2 = COPY s0
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        merge = BasicBlock(
            name="merge",
            phis=[Phi(dest=3, incoming={"entry": 1, "other": 2})],
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(
            entry="entry",
            blocks={"entry": entry, "other": other, "merge": merge}
        )
        out = self._run(lir)

        # The phi should now have s0 from both predecessors
        phi = out.blocks["merge"].phis[0]
        self.assertEqual(phi.incoming["entry"], 0)
        self.assertEqual(phi.incoming["other"], 0)

    def test_terminator_operand_propagation(self):
        """Propagate to terminator operands like cond_jump."""
        # s1 = COPY s0
        # cond_jump s1, then, else
        # After: cond_jump s0, then, else
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [1], "load"),   # s0 = 1
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),     # s1 = COPY s0
            ],
            terminator=LIRInst(LIROpcode.COND_JUMP, None, [1, "then", "else"], "flow"),
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
        out = self._run(lir)

        # The cond_jump should now use s0 instead of s1
        term = out.blocks["entry"].terminator
        self.assertEqual(term.operands[0], 0)
        # Labels should be unchanged
        self.assertEqual(term.operands[1], "then")
        self.assertEqual(term.operands[2], "else")

    def test_metrics_tracking(self):
        """Verify metrics are tracked correctly."""
        # s0 = 42
        # s1 = COPY s0
        # s2 = COPY s1  <- s1 operand gets propagated to s0 (1 propagation)
        # s3 = s1 + s2  <- both s1 and s2 get propagated to s0 (2 propagations)
        # Total: 3 propagations
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # s0 = 42
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),     # s1 = COPY s0
                LIRInst(LIROpcode.COPY, 2, [1], "alu"),     # s2 = COPY s1
                LIRInst(LIROpcode.ADD, 3, [1, 2], "alu"),   # s3 = s1 + s2
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})

        p = CopyPropagationPass()
        cfg = PassConfig(name=p.name)
        p.run(lir, cfg)

        metrics = p.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.custom["copies_found"], 2)
        self.assertEqual(metrics.custom["operands_propagated"], 3)

    def test_no_propagation_for_non_copy_dest(self):
        """Non-COPY destinations should not be in the copy map."""
        # s0 = 42 (CONST, not COPY)
        # s1 = s0 + s0
        # Should not change since s0 is not a COPY dest
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # s0 = 42
                LIRInst(LIROpcode.ADD, 1, [0, 0], "alu"),   # s1 = s0 + s0
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out = self._run(lir)

        # The ADD operands should remain unchanged
        add_inst = out.blocks["entry"].instructions[1]
        self.assertEqual(add_inst.operands, [0, 0])

    def test_copy_source_not_modified(self):
        """COPY source operand itself should be propagated if it's a copy dest."""
        # s1 = COPY s0
        # s2 = COPY s1   <- s1 is both copy dest and copy source
        # s3 = s2 + s2
        # After: s3 = s0 + s0
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [42], "load"),  # s0 = 42
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),     # s1 = COPY s0
                LIRInst(LIROpcode.COPY, 2, [1], "alu"),     # s2 = COPY s1
                LIRInst(LIROpcode.ADD, 3, [2, 2], "alu"),   # s3 = s2 + s2
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out = self._run(lir)

        # The ADD should now use s0 (transitively resolved)
        add_inst = out.blocks["entry"].instructions[3]
        self.assertEqual(add_inst.operands, [0, 0])


    def test_const_immediate_not_rewritten(self):
        """CONST immediate values must not be rewritten even if they match a copy dest.

        Regression test: if s1 = COPY s0 creates mapping 1->0, and we have
        CONST s2, [1], the immediate value 1 should NOT be rewritten to 0.
        """
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [99], "load"),  # s0 = 99
                LIRInst(LIROpcode.COPY, 1, [0], "alu"),     # s1 = COPY s0, creates 1->0
                LIRInst(LIROpcode.CONST, 2, [1], "load"),   # s2 = 1 (immediate, NOT s1!)
                LIRInst(LIROpcode.CONST, 3, [100], "load"),
                LIRInst(LIROpcode.STORE, None, [3, 2], "store"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})
        out = self._run(lir)

        # The CONST s2, [1] should still have immediate value 1, not 0
        const_inst = out.blocks["entry"].instructions[2]
        self.assertEqual(const_inst.opcode, LIROpcode.CONST)
        self.assertEqual(const_inst.operands, [1])  # immediate value unchanged


if __name__ == "__main__":
    unittest.main()
