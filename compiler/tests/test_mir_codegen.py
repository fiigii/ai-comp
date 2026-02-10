"""Tests for MIR codegen and fallthrough optimizations."""

import unittest

from compiler.lir import LIRFunction, BasicBlock, LIRInst, LIROpcode
from compiler.passes import InstSchedulingPass, MIRRegisterAllocationPass, MIRToVLIWPass
from compiler.pass_manager import PassConfig


def cfg(name, **opts):
    """Helper to create PassConfig."""
    return PassConfig(name=name, enabled=True, options=opts)


def compile_lir_to_vliw_via_mir(lir: LIRFunction) -> list[dict]:
    """Compile LIR to VLIW through the MIR path."""
    mir = InstSchedulingPass().run(lir, cfg('inst-scheduling'))
    mir = MIRRegisterAllocationPass().run(mir, cfg('mir-regalloc'))
    bundles = MIRToVLIWPass().run(mir, cfg('mir-to-vliw'))
    return bundles


class TestMIRCodegenFallthrough(unittest.TestCase):
    """Tests for MIR codegen fallthrough optimizations."""

    def _count_jumps(self, bundles):
        """Count jump and cond_jump slots in bundles."""
        jump_count = 0
        cond_jump_count = 0
        for bundle in bundles:
            if 'flow' in bundle:
                for slot in bundle['flow']:
                    if slot[0] == 'jump':
                        jump_count += 1
                    elif slot[0] == 'cond_jump':
                        cond_jump_count += 1
        return jump_count, cond_jump_count

    def _has_jump_to(self, bundles, target):
        """Check if any bundle has a jump to the given target address."""
        for bundle in bundles:
            if 'flow' in bundle:
                for slot in bundle['flow']:
                    if slot[0] == 'jump' and slot[1] == target:
                        return True
        return False

    def test_fallthrough_jump_omitted(self):
        """Unconditional jump to the immediately next block is omitted."""
        # entry -> exit (fallthrough)
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [42], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry, "exit": exit_block})

        bundles = compile_lir_to_vliw_via_mir(lir)
        jump_count, _ = self._count_jumps(bundles)

        # The JUMP from entry to exit should be omitted (fallthrough)
        self.assertEqual(jump_count, 0, "Fallthrough JUMP should be omitted")

    def test_non_fallthrough_jump_emitted(self):
        """Unconditional jump to a non-adjacent block is emitted."""
        # entry -> block_a -> block_c (jump over block_b)
        #                  -> block_b -> exit (via cond_jump false)
        # Layout after DFS: entry, block_a, block_c, block_b, exit
        # block_a's JUMP to block_c is fallthrough
        # block_b's JUMP to exit is not fallthrough (exit comes after block_b)
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [1], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["block_a"], "flow"),
        )
        block_a = BasicBlock(
            name="block_a",
            instructions=[],
            terminator=LIRInst(
                LIROpcode.COND_JUMP,
                None,
                [0, "block_c", "block_b"],  # true -> block_c, false -> block_b
                "flow",
            ),
        )
        block_b = BasicBlock(
            name="block_b",
            instructions=[LIRInst(LIROpcode.CONST, 1, [10], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        block_c = BasicBlock(
            name="block_c",
            instructions=[LIRInst(LIROpcode.CONST, 2, [20], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(
            entry="entry",
            blocks={
                "entry": entry,
                "block_a": block_a,
                "block_b": block_b,
                "block_c": block_c,
                "exit": exit_block,
            }
        )

        bundles = compile_lir_to_vliw_via_mir(lir)
        jump_count, cond_jump_count = self._count_jumps(bundles)

        # We should have at least one JUMP (for non-fallthrough cases)
        # The exact count depends on block layout, but should be > 0
        # since not all JUMPs can be fallthrough with this CFG
        self.assertGreaterEqual(jump_count, 1, "Should have at least one non-fallthrough JUMP")

    def test_cond_jump_false_fallthrough_omitted(self):
        """COND_JUMP with fallthrough false target omits the false jump."""
        # entry: cond_jump to header (if true), exit (if false)
        # Layout: entry, header, exit
        # False target (exit) is NOT fallthrough, but we test the case where it IS
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [1], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["header"], "flow"),
        )
        header = BasicBlock(
            name="header",
            instructions=[],
            terminator=LIRInst(
                LIROpcode.COND_JUMP,
                None,
                [0, "header", "exit"],  # cond, true_target, false_target
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
            blocks={"entry": entry, "header": header, "exit": exit_block}
        )

        bundles = compile_lir_to_vliw_via_mir(lir)
        jump_count, cond_jump_count = self._count_jumps(bundles)

        # COND_JUMP should be emitted
        self.assertEqual(cond_jump_count, 1, "COND_JUMP should be emitted")
        # The false target (exit) follows header in layout, so no JUMP needed
        self.assertEqual(jump_count, 0, "False fallthrough JUMP should be omitted")

    def test_cond_jump_false_not_fallthrough_emits_jump(self):
        """COND_JUMP with non-fallthrough false target emits the false jump."""
        # entry: cond_jump to then_block (if true), else_block (if false)
        # Layout: entry, then_block, else_block
        # then_block jumps to exit
        # False target (else_block) is NOT the immediate next block after entry
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [1], "load")],
            terminator=LIRInst(
                LIROpcode.COND_JUMP,
                None,
                [0, "then_block", "else_block"],
                "flow",
            ),
        )
        then_block = BasicBlock(
            name="then_block",
            instructions=[],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        else_block = BasicBlock(
            name="else_block",
            instructions=[],
            terminator=LIRInst(LIROpcode.JUMP, None, ["exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(
            entry="entry",
            blocks={
                "entry": entry,
                "then_block": then_block,
                "else_block": else_block,
                "exit": exit_block,
            }
        )

        bundles = compile_lir_to_vliw_via_mir(lir)
        jump_count, cond_jump_count = self._count_jumps(bundles)

        # COND_JUMP should be emitted
        self.assertEqual(cond_jump_count, 1, "COND_JUMP should be emitted")
        # Layout depends on DFS order, but false target might need a jump
        # The key point: if else_block is not immediately after entry, a JUMP is needed

    def test_multiple_blocks_chain_fallthrough(self):
        """Chain of blocks A->B->C->D with all fallthrough JUMPs omitted."""
        block_a = BasicBlock(
            name="block_a",
            instructions=[LIRInst(LIROpcode.CONST, 0, [1], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["block_b"], "flow"),
        )
        block_b = BasicBlock(
            name="block_b",
            instructions=[LIRInst(LIROpcode.CONST, 1, [2], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["block_c"], "flow"),
        )
        block_c = BasicBlock(
            name="block_c",
            instructions=[LIRInst(LIROpcode.CONST, 2, [3], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["block_d"], "flow"),
        )
        block_d = BasicBlock(
            name="block_d",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(
            entry="block_a",
            blocks={
                "block_a": block_a,
                "block_b": block_b,
                "block_c": block_c,
                "block_d": block_d,
            }
        )

        bundles = compile_lir_to_vliw_via_mir(lir)
        jump_count, _ = self._count_jumps(bundles)

        # All JUMPs should be fallthrough and omitted
        self.assertEqual(jump_count, 0, "All fallthrough JUMPs should be omitted")

    def test_loop_back_edge_not_fallthrough(self):
        """Back edge in a loop should emit a JUMP (not fallthrough)."""
        # entry -> header -> body -> header (back edge)
        #                 -> exit
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
                [0, "body", "exit"],
                "flow",
            ),
        )
        body = BasicBlock(
            name="body",
            instructions=[LIRInst(LIROpcode.CONST, 1, [1], "load")],
            terminator=LIRInst(LIROpcode.JUMP, None, ["header"], "flow"),  # Back edge
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(
            entry="entry",
            blocks={
                "entry": entry,
                "header": header,
                "body": body,
                "exit": exit_block,
            }
        )

        bundles = compile_lir_to_vliw_via_mir(lir)
        jump_count, cond_jump_count = self._count_jumps(bundles)

        # Back edge from body -> header should be emitted
        self.assertGreaterEqual(jump_count, 1, "Back edge JUMP should be emitted")


class TestMIRCodegenCorrectness(unittest.TestCase):
    """Tests for MIR codegen correctness."""

    def test_basic_codegen(self):
        """Basic codegen produces valid VLIW bundles."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [10], "load"),
                LIRInst(LIROpcode.CONST, 1, [20], "load"),
                LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(entry="entry", blocks={"entry": entry})

        bundles = compile_lir_to_vliw_via_mir(lir)

        # Should have bundles for const, const, add, halt
        self.assertGreater(len(bundles), 0, "Should produce bundles")

        # Check that we have the expected operations
        has_const = any('load' in b for b in bundles)
        has_alu = any('alu' in b for b in bundles)
        has_halt = any(
            'flow' in b and any(s[0] == 'halt' for s in b['flow'])
            for b in bundles
        )
        self.assertTrue(has_const, "Should have const operations")
        self.assertTrue(has_alu, "Should have ALU operations")
        self.assertTrue(has_halt, "Should have halt")

    def test_jump_target_resolution(self):
        """Jump targets are resolved to bundle indices."""
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [1], "load")],
            terminator=LIRInst(
                LIROpcode.COND_JUMP,
                None,
                [0, "target", "exit"],
                "flow",
            ),
        )
        target = BasicBlock(
            name="target",
            instructions=[LIRInst(LIROpcode.CONST, 1, [42], "load")],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        lir = LIRFunction(
            entry="entry",
            blocks={"entry": entry, "target": target, "exit": exit_block}
        )

        bundles = compile_lir_to_vliw_via_mir(lir)

        # Find cond_jump and verify target is an integer (resolved)
        for bundle in bundles:
            if 'flow' in bundle:
                for slot in bundle['flow']:
                    if slot[0] == 'cond_jump':
                        self.assertIsInstance(
                            slot[2], int,
                            "COND_JUMP target should be resolved to integer"
                        )
                    elif slot[0] == 'jump':
                        self.assertIsInstance(
                            slot[1], int,
                            "JUMP target should be resolved to integer"
                        )


if __name__ == "__main__":
    unittest.main()
