"""Tests for instruction scheduling (LIR -> MIR)."""

import unittest

from compiler.lir import LIRFunction, BasicBlock, LIRInst, LIROpcode
from compiler.passes import InstSchedulingPass
from compiler.pass_manager import PassConfig


def _cfg(name, **opts):
    return PassConfig(name=name, enabled=True, options=opts)


def _schedule_single_block(instructions, **opts):
    entry = BasicBlock(
        name="entry",
        instructions=instructions,
        terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
    )
    lir = LIRFunction(entry="entry", blocks={"entry": entry})
    mir = InstSchedulingPass().run(lir, _cfg("inst-scheduling", **opts))
    return mir.blocks["entry"].bundles


def _find_bundle_index(bundles, opcode):
    for i, bundle in enumerate(bundles):
        for inst in bundle.instructions:
            if inst.opcode == opcode:
                return i
    return None


def _bundle_signature(bundles):
    sig = []
    for bundle in bundles:
        insts = []
        for inst in bundle.instructions:
            dest = tuple(inst.dest) if isinstance(inst.dest, list) else inst.dest
            ops = []
            for op in inst.operands:
                if isinstance(op, list):
                    ops.append(tuple(op))
                else:
                    ops.append(op)
            insts.append((inst.opcode.value, dest, tuple(ops), inst.engine))
        sig.append(tuple(insts))
    return tuple(sig)


class TestInstructionScheduling(unittest.TestCase):
    """Scheduling correctness and determinism tests."""

    def test_no_same_bundle_raw(self):
        const = LIRInst(LIROpcode.CONST, 0, [1], "load")
        add = LIRInst(LIROpcode.ADD, 1, [0, 0], "alu")
        bundles = _schedule_single_block([const, add])

        const_idx = _find_bundle_index(bundles, LIROpcode.CONST)
        add_idx = _find_bundle_index(bundles, LIROpcode.ADD)

        self.assertIsNotNone(const_idx)
        self.assertIsNotNone(add_idx)
        self.assertNotEqual(const_idx, add_idx, "RAW must not co-issue in same bundle")
        self.assertLess(const_idx, add_idx, "RAW consumer must be scheduled after producer")

    def test_store_then_load_separated(self):
        const_addr = LIRInst(LIROpcode.CONST, 0, [10], "load")
        const_val = LIRInst(LIROpcode.CONST, 1, [7], "load")
        store = LIRInst(LIROpcode.STORE, None, [0, 1], "store")
        load = LIRInst(LIROpcode.LOAD, 2, [0], "load")
        bundles = _schedule_single_block([const_addr, const_val, store, load])

        store_idx = _find_bundle_index(bundles, LIROpcode.STORE)
        load_idx = _find_bundle_index(bundles, LIROpcode.LOAD)

        self.assertIsNotNone(store_idx)
        self.assertIsNotNone(load_idx)
        self.assertNotEqual(store_idx, load_idx, "Store->load must not co-issue")
        self.assertLess(store_idx, load_idx, "Load must be scheduled after prior store")

    def test_load_then_store_can_coissue(self):
        const_addr = LIRInst(LIROpcode.CONST, 0, [10], "load")
        const_val = LIRInst(LIROpcode.CONST, 1, [5], "load")
        load = LIRInst(LIROpcode.LOAD, 2, [0], "load")
        store = LIRInst(LIROpcode.STORE, None, [0, 1], "store")
        bundles = _schedule_single_block([const_addr, const_val, load, store])

        load_idx = _find_bundle_index(bundles, LIROpcode.LOAD)
        store_idx = _find_bundle_index(bundles, LIROpcode.STORE)

        self.assertIsNotNone(load_idx)
        self.assertIsNotNone(store_idx)
        self.assertEqual(load_idx, store_idx, "Load->store should be able to co-issue")

    def test_deterministic_bundles(self):
        const0 = LIRInst(LIROpcode.CONST, 0, [1], "load")
        const1 = LIRInst(LIROpcode.CONST, 1, [2], "load")
        add = LIRInst(LIROpcode.ADD, 2, [0, 1], "alu")
        mul = LIRInst(LIROpcode.MUL, 3, [0, 1], "alu")

        bundles_a = _schedule_single_block([const0, const1, add, mul])
        bundles_b = _schedule_single_block([const0, const1, add, mul])

        self.assertEqual(
            _bundle_signature(bundles_a),
            _bundle_signature(bundles_b),
            "Scheduling must be deterministic",
        )

    def test_devectorize_valu_to_alu_when_valu_saturated(self):
        def vec(base):
            return [base + i for i in range(8)]

        instructions = [
            LIRInst(LIROpcode.VADD, vec(100 + i * 8), [vec(1000), vec(2000)], "valu")
            for i in range(7)
        ]

        bundles_no_dev = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=False,
        )
        bundles_dev = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=True,
        )

        non_term_no_dev = bundles_no_dev[:-1]
        non_term_dev = bundles_dev[:-1]
        self.assertEqual(len(non_term_no_dev), 2)
        self.assertEqual(len(non_term_dev), 1)

        first_bundle = non_term_dev[0]
        valu_count = sum(1 for inst in first_bundle.instructions if inst.engine == "valu")
        alu_count = sum(1 for inst in first_bundle.instructions if inst.engine == "alu")
        self.assertEqual(valu_count, 6)
        self.assertEqual(alu_count, 8)
        self.assertTrue(
            any(inst.opcode == LIROpcode.ADD and inst.engine == "alu" for inst in first_bundle.instructions),
            "Expected devectorized scalar ALU instructions",
        )

    def test_devectorize_valu_to_alu_skips_multiply_add(self):
        def vec(base):
            return [base + i for i in range(8)]

        instructions = [
            LIRInst(LIROpcode.VADD, vec(100 + i * 8), [vec(1000), vec(2000)], "valu")
            for i in range(6)
        ]
        instructions.append(
            LIRInst(LIROpcode.MULTIPLY_ADD, vec(200), [vec(300), vec(400), vec(500)], "valu")
        )

        bundles = _schedule_single_block(
            instructions,
            devectorize_valu_to_alu=True,
        )

        non_term = bundles[:-1]
        self.assertEqual(len(non_term), 2)
        self.assertEqual(
            sum(1 for inst in non_term[0].instructions if inst.engine == "alu"),
            0,
            "multiply_add must not be devectorized",
        )
        self.assertTrue(
            any(inst.opcode == LIROpcode.MULTIPLY_ADD for inst in non_term[1].instructions),
            "multiply_add should remain a valu instruction",
        )


if __name__ == "__main__":
    unittest.main()
