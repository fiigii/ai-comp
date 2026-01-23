"""Tests for codegen CFG optimizations."""

import unittest

from compiler.tests.conftest import (
    LIRFunction,
    BasicBlock,
    LIRInst,
    LIROpcode,
    compile_to_vliw,
)


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


if __name__ == "__main__":
    unittest.main()
