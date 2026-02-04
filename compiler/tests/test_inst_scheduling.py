"""Tests for instruction scheduling (LIR -> MIR)."""

import unittest

from compiler.lir import LIRFunction, BasicBlock, LIRInst, LIROpcode
from compiler.passes import InstSchedulingPass
from compiler.pass_manager import PassConfig


def _cfg(name, **opts):
    return PassConfig(name=name, enabled=True, options=opts)


def _schedule_single_block(instructions):
    entry = BasicBlock(
        name="entry",
        instructions=instructions,
        terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
    )
    lir = LIRFunction(entry="entry", blocks={"entry": entry})
    mir = InstSchedulingPass().run(lir, _cfg("inst-scheduling"))
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


if __name__ == "__main__":
    unittest.main()
