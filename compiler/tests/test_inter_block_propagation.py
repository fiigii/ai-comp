"""Tests for inter-block value info propagation in instruction scheduling."""

import unittest

from compiler.lir import LIRFunction, BasicBlock, LIRInst, LIROpcode
from compiler.mir import MachineInst
from compiler.passes import InstSchedulingPass
from compiler.pass_manager import PassConfig
from compiler.passes.inst_scheduling import (
    _compute_exit_value_state,
    _build_dep_graph,
    AddrExpr,
)


def _cfg(name, **opts):
    return PassConfig(name=name, enabled=True, options=opts)


def _mi(opcode, dest, operands, engine):
    """Shorthand for MachineInst construction."""
    return MachineInst(opcode=opcode, dest=dest, operands=list(operands), engine=engine)


def _schedule_multi_block(blocks_dict, entry="entry"):
    """Schedule a multi-block LIR function and return the MIR MachineFunction."""
    lir = LIRFunction(entry=entry, blocks=blocks_dict)
    return InstSchedulingPass().run(lir, _cfg("inst-scheduling"))


class TestComputeExitValueState(unittest.TestCase):
    """Tests for _compute_exit_value_state helper."""

    def test_empty_instructions(self):
        const_val, addr_expr = _compute_exit_value_state([])
        self.assertEqual(const_val, {})
        self.assertEqual(addr_expr, {})

    def test_const_tracked(self):
        insts = [_mi(LIROpcode.CONST, 0, [42], "load")]
        const_val, addr_expr = _compute_exit_value_state(insts)
        self.assertEqual(const_val, {0: 42})
        self.assertEqual(addr_expr, {})

    def test_const_then_add(self):
        insts = [
            _mi(LIROpcode.CONST, 0, [10], "load"),
            _mi(LIROpcode.CONST, 1, [20], "load"),
            _mi(LIROpcode.ADD, 2, [0, 1], "alu"),
        ]
        const_val, addr_expr = _compute_exit_value_state(insts)
        self.assertEqual(const_val[0], 10)
        self.assertEqual(const_val[1], 20)
        self.assertEqual(const_val[2], 30)

    def test_header_pointer_load_creates_addr_expr(self):
        """Loading from mem[4], mem[5], mem[6] creates addr exprs (base pointers)."""
        insts = [
            _mi(LIROpcode.CONST, 0, [4], "load"),   # const s0 = 4
            _mi(LIROpcode.LOAD, 1, [0], "load"),     # s1 = load(s0) → addr(4, 0)
        ]
        const_val, addr_expr = _compute_exit_value_state(insts)
        self.assertEqual(const_val[0], 4)
        self.assertIn(1, addr_expr)
        self.assertEqual(addr_expr[1], AddrExpr(base=4, offset=0))

    def test_addr_plus_const(self):
        """addr(base, 0) + const(5) → addr(base, 5)."""
        insts = [
            _mi(LIROpcode.CONST, 0, [5], "load"),   # const s0 = 5
            _mi(LIROpcode.LOAD, 1, [0], "load"),     # s1 = load(s0) → addr(5, 0)
            _mi(LIROpcode.CONST, 2, [10], "load"),   # const s2 = 10
            _mi(LIROpcode.ADD, 3, [1, 2], "alu"),    # s3 = s1 + s2 → addr(5, 10)
        ]
        const_val, addr_expr = _compute_exit_value_state(insts)
        self.assertIn(3, addr_expr)
        self.assertEqual(addr_expr[3], AddrExpr(base=5, offset=10))

    def test_overwrite_clears_old(self):
        """Redefining a scratch reg clears its old const/addr info."""
        insts = [
            _mi(LIROpcode.CONST, 0, [42], "load"),
            _mi(LIROpcode.CONST, 0, [99], "load"),
        ]
        const_val, addr_expr = _compute_exit_value_state(insts)
        self.assertEqual(const_val[0], 99)

    def test_initial_state_propagated(self):
        """Initial const_val and addr_expr are used as starting state."""
        insts = [
            _mi(LIROpcode.ADD, 2, [0, 1], "alu"),
        ]
        initial_const = {0: 10, 1: 20}
        const_val, addr_expr = _compute_exit_value_state(insts, initial_const)
        self.assertEqual(const_val[2], 30)
        # Original entries preserved
        self.assertEqual(const_val[0], 10)
        self.assertEqual(const_val[1], 20)

    def test_initial_addr_expr_propagated(self):
        """Initial addr_expr is propagated through instructions."""
        insts = [
            _mi(LIROpcode.CONST, 1, [3], "load"),
            _mi(LIROpcode.ADD, 2, [0, 1], "alu"),  # s0(addr) + s1(const 3)
        ]
        initial_addr = {0: AddrExpr(base=5, offset=0)}
        const_val, addr_expr = _compute_exit_value_state(insts, None, initial_addr)
        self.assertIn(2, addr_expr)
        self.assertEqual(addr_expr[2], AddrExpr(base=5, offset=3))

    def test_initial_state_not_mutated(self):
        """Initial dicts must not be mutated."""
        insts = [_mi(LIROpcode.CONST, 0, [99], "load")]
        initial_const = {0: 42}
        initial_addr = {0: AddrExpr(base=5, offset=0)}
        _compute_exit_value_state(insts, initial_const, initial_addr)
        self.assertEqual(initial_const, {0: 42})
        self.assertEqual(initial_addr, {0: AddrExpr(base=5, offset=0)})

    def test_unknown_op_clears_info(self):
        """Unknown/unmodeled ops clear value info for their dests."""
        insts = [
            _mi(LIROpcode.CONST, 0, [42], "load"),
            _mi(LIROpcode.MUL, 0, [0, 0], "alu"),  # MUL not tracked
        ]
        const_val, addr_expr = _compute_exit_value_state(insts)
        self.assertNotIn(0, const_val)


class TestBuildDepGraphWithInitialState(unittest.TestCase):
    """Tests for _build_dep_graph accepting initial state."""

    def test_no_initial_state_same_as_before(self):
        """Without initial state, behavior is unchanged."""
        insts = [
            _mi(LIROpcode.CONST, 0, [10], "load"),
            _mi(LIROpcode.STORE, None, [0, 0], "store"),
            _mi(LIROpcode.LOAD, 1, [0], "load"),
        ]
        nodes_default = _build_dep_graph(insts)
        nodes_none = _build_dep_graph(insts, None, None)
        # Same dep edges
        for i in range(len(nodes_default)):
            self.assertEqual(nodes_default[i].succs, nodes_none[i].succs)

    def test_initial_addr_disambiguates_stores(self):
        """With initial addr_expr, stores to different bases are disambiguated."""
        # s0 holds addr(4, 0), s1 holds addr(5, 0) from initial state
        # store(s0, s2) and store(s1, s3) should not depend on each other
        insts = [
            _mi(LIROpcode.STORE, None, [0, 10], "store"),  # store to addr(4, 0)
            _mi(LIROpcode.STORE, None, [1, 11], "store"),  # store to addr(5, 0)
        ]
        initial_addr = {
            0: AddrExpr(base=4, offset=0),
            1: AddrExpr(base=5, offset=0),
        }

        # Without initial state: both get key=None → alias → dependency
        nodes_no_init = _build_dep_graph(insts)
        has_dep_no_init = 1 in nodes_no_init[0].succs

        # With initial state: different bases → no alias → no dependency
        nodes_with_init = _build_dep_graph(insts, None, initial_addr)
        has_dep_with_init = 1 in nodes_with_init[0].succs

        self.assertTrue(has_dep_no_init, "Without init state, stores should alias (key=None)")
        self.assertFalse(has_dep_with_init, "With init state, stores to different bases should not alias")

    def test_initial_addr_same_base_still_aliases(self):
        """Stores to same base with same offset still alias."""
        insts = [
            _mi(LIROpcode.STORE, None, [0, 10], "store"),
            _mi(LIROpcode.STORE, None, [1, 11], "store"),
        ]
        initial_addr = {
            0: AddrExpr(base=5, offset=3),
            1: AddrExpr(base=5, offset=3),
        }
        nodes = _build_dep_graph(insts, None, initial_addr)
        self.assertIn(1, nodes[0].succs, "Stores to same addr must still have a dependency")

    def test_initial_const_enables_header_load_tracking(self):
        """Initial const_val lets LOAD from mem[4..6] create addr exprs."""
        # s0 = const 5 (from initial state), s1 = load(s0) → creates addr(5, 0)
        # s2 = store to s1 → key is addr(5, 0)
        # s3 = store to s4 (unknown) → key is None → aliases
        insts = [
            _mi(LIROpcode.LOAD, 1, [0], "load"),      # load(s0=const 5) → addr(5, 0)
            _mi(LIROpcode.STORE, None, [1, 10], "store"),  # store to addr(5, 0)
            _mi(LIROpcode.STORE, None, [1, 11], "store"),  # store to addr(5, 0)
        ]
        initial_const = {0: 5}
        nodes = _build_dep_graph(insts, initial_const, None)
        # Both stores to same addr → must alias
        self.assertIn(2, nodes[1].succs)


class TestInterBlockPropagation(unittest.TestCase):
    """Tests for the full inter-block propagation in InstSchedulingPass.run()."""

    def _make_lir(self, blocks_dict, entry="entry"):
        return LIRFunction(entry=entry, blocks=blocks_dict)

    def test_single_block_unchanged(self):
        """Single block: propagation has no effect (entry state is empty)."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [10], "load"),
                LIRInst(LIROpcode.CONST, 1, [20], "load"),
                LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({"entry": entry})
        self.assertIn("entry", mir.blocks)
        self.assertTrue(len(mir.blocks["entry"].bundles) > 0)

    def test_linear_chain_propagates_const(self):
        """Constants defined in block A propagate to block B via a jump."""
        block_a = BasicBlock(
            name="block_a",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [4], "load"),
                LIRInst(LIROpcode.LOAD, 1, [0], "load"),  # s1 = addr(4, 0)
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["block_b"], "flow"),
        )
        # block_b: two stores to s1 (inherited addr(4,0)) and to different offsets
        # With propagation, scheduler knows s1's addr → can disambiguate
        block_b = BasicBlock(
            name="block_b",
            instructions=[
                LIRInst(LIROpcode.CONST, 2, [1], "load"),
                LIRInst(LIROpcode.ADD, 3, [1, 2], "alu"),    # s3 = addr(4, 1)
                LIRInst(LIROpcode.CONST, 4, [2], "load"),
                LIRInst(LIROpcode.ADD, 5, [1, 4], "alu"),    # s5 = addr(4, 2)
                LIRInst(LIROpcode.STORE, None, [3, 10], "store"),  # store @ addr(4,1)
                LIRInst(LIROpcode.STORE, None, [5, 11], "store"),  # store @ addr(4,2)
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({"block_a": block_a, "block_b": block_b}, entry="block_a")

        # Both blocks should be present
        self.assertIn("block_a", mir.blocks)
        self.assertIn("block_b", mir.blocks)

        # Verify the two stores to different offsets can be in the same bundle
        # (they're disambiguated by different offsets under the same base)
        block_b_bundles = mir.blocks["block_b"].bundles
        store_bundles = []
        for i, bundle in enumerate(block_b_bundles):
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.STORE:
                    store_bundles.append(i)
        # With propagation, stores to addr(4,1) and addr(4,2) are disambiguated
        # and can co-issue in the same bundle
        self.assertEqual(len(store_bundles), 2)
        self.assertEqual(store_bundles[0], store_bundles[1],
                         "Stores to different offsets should co-issue with propagated addr info")

    def test_without_propagation_store_load_serialized(self):
        """Without propagation, store then load with unknown addr get serialized (delay=1)."""
        # s1 is "unknown" here (no const/addr info, not defined in this block).
        # store(s1, val) then load(s1) → both get key=None → alias → load after store.
        block = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.STORE, None, [1, 10], "store"),   # store to unknown s1
                LIRInst(LIROpcode.LOAD, 2, [1], "load"),            # load from unknown s1
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({"entry": block})
        block_bundles = mir.blocks["entry"].bundles
        store_idx = None
        load_idx = None
        for i, bundle in enumerate(block_bundles):
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.STORE:
                    store_idx = i
                if inst.opcode == LIROpcode.LOAD:
                    load_idx = i
        self.assertIsNotNone(store_idx)
        self.assertIsNotNone(load_idx)
        self.assertLess(store_idx, load_idx,
                        "Without propagation, store→load with unknown addr should be serialized")

    def test_diamond_cfg_intersection(self):
        """Diamond CFG: only values agreed upon by both predecessors survive.

        entry defines s0=4 and s1=5. block_a redefines s1=6, block_b keeps s1=5.
        At merge: s0=4 (agreed) propagates, s1 is dropped (disagreed).
        We verify s0 propagates by checking that load(s0) creates addr(4,0),
        and that a subsequent store→load to that addr has a real dependency.
        We verify s1 is dropped: since s1 disagrees, load(s1) should NOT create
        an addr expr, so store/load through s1 base aliases everything.
        """
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [4], "load"),
                LIRInst(LIROpcode.CONST, 1, [5], "load"),
                LIRInst(LIROpcode.CONST, 99, [1], "load"),  # condition
            ],
            terminator=LIRInst(LIROpcode.COND_JUMP, None, [99, "block_a", "block_b"], "flow"),
        )
        block_a = BasicBlock(
            name="block_a",
            instructions=[
                LIRInst(LIROpcode.CONST, 1, [6], "load"),  # s1 = 6 (disagrees)
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        block_b = BasicBlock(
            name="block_b",
            instructions=[],  # s1 stays 5
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        merge = BasicBlock(
            name="merge",
            instructions=[
                # s0=4 is agreed → load(s0) creates addr(4,0)
                LIRInst(LIROpcode.LOAD, 2, [0], "load"),  # s2 = addr(4, 0)
                # s1 is disagreed → load(s1) has unknown addr
                LIRInst(LIROpcode.LOAD, 3, [1], "load"),  # s3 = unknown addr
                # Store via unknown s3, then load via known s2
                # Since s3 is unknown (key=None), the store aliases everything,
                # so the subsequent load MUST be serialized after it.
                LIRInst(LIROpcode.STORE, None, [3, 10], "store"),
                LIRInst(LIROpcode.LOAD, 4, [2], "load"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({
            "entry": entry,
            "block_a": block_a,
            "block_b": block_b,
            "merge": merge,
        })
        self.assertIn("merge", mir.blocks)

        merge_bundles = mir.blocks["merge"].bundles
        store_idx = None
        load4_idx = None
        for i, bundle in enumerate(merge_bundles):
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.STORE:
                    store_idx = i
                if inst.opcode == LIROpcode.LOAD and inst.dest == 4:
                    load4_idx = i
        self.assertIsNotNone(store_idx)
        self.assertIsNotNone(load4_idx)
        # Store through unknown addr must serialize before subsequent load
        self.assertLess(store_idx, load4_idx,
                        "Store through disagreed (unknown) addr must serialize before load")

    def test_diamond_agreed_values_propagate(self):
        """Diamond CFG: values that agree across both arms propagate to merge."""
        entry = BasicBlock(
            name="entry",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [4], "load"),   # s0 = 4
                LIRInst(LIROpcode.CONST, 1, [5], "load"),   # s1 = 5
                LIRInst(LIROpcode.CONST, 99, [1], "load"),
            ],
            terminator=LIRInst(LIROpcode.COND_JUMP, None, [99, "left", "right"], "flow"),
        )
        left = BasicBlock(
            name="left",
            instructions=[],  # No modifications
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        right = BasicBlock(
            name="right",
            instructions=[],  # No modifications
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        # In merge: s0=4, s1=5 from both paths → load(s0) gives addr(4,0), load(s1) gives addr(5,0)
        merge = BasicBlock(
            name="merge",
            instructions=[
                LIRInst(LIROpcode.LOAD, 2, [0], "load"),  # addr(4, 0)
                LIRInst(LIROpcode.LOAD, 3, [1], "load"),  # addr(5, 0)
                LIRInst(LIROpcode.STORE, None, [2, 10], "store"),  # store @ base 4
                LIRInst(LIROpcode.STORE, None, [3, 11], "store"),  # store @ base 5
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({
            "entry": entry,
            "left": left,
            "right": right,
            "merge": merge,
        })
        merge_bundles = mir.blocks["merge"].bundles
        store_bundles = []
        for i, bundle in enumerate(merge_bundles):
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.STORE:
                    store_bundles.append(i)
        self.assertEqual(len(store_bundles), 2)
        # Stores to different bases (4 vs 5) → disambiguated → can co-issue
        self.assertEqual(store_bundles[0], store_bundles[1],
                         "Stores to different bases should co-issue when values propagate through diamond")

    def test_back_edge_skipped(self):
        """Back edges (from loop body to header) are skipped — not processed yet in RPO."""
        # for_init → for_body → for_body (back edge to self via loop)
        for_init = BasicBlock(
            name="for_init",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [5], "load"),
                LIRInst(LIROpcode.LOAD, 1, [0], "load"),  # s1 = addr(5, 0)
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["for_body"], "flow"),
        )
        for_body = BasicBlock(
            name="for_body",
            instructions=[
                LIRInst(LIROpcode.CONST, 2, [1], "load"),
                LIRInst(LIROpcode.ADD, 3, [1, 2], "alu"),  # s3 = addr(5, 1) if s1 propagated
                LIRInst(LIROpcode.CONST, 4, [2], "load"),
                LIRInst(LIROpcode.ADD, 5, [1, 4], "alu"),  # s5 = addr(5, 2) if s1 propagated
                LIRInst(LIROpcode.STORE, None, [3, 10], "store"),
                LIRInst(LIROpcode.STORE, None, [5, 11], "store"),
                LIRInst(LIROpcode.CONST, 98, [0], "load"),
            ],
            terminator=LIRInst(LIROpcode.COND_JUMP, None, [98, "for_body", "exit"], "flow"),
        )
        exit_block = BasicBlock(
            name="exit",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({
            "for_init": for_init,
            "for_body": for_body,
            "exit": exit_block,
        }, entry="for_init")

        # The key test: for_body has two predecessors (for_init and itself via back edge).
        # The back edge from for_body→for_body is unprocessed in RPO, so only for_init's
        # exit state is used. This means s1=addr(5,0) propagates correctly.
        body_bundles = mir.blocks["for_body"].bundles
        store_bundles = []
        for i, bundle in enumerate(body_bundles):
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.STORE:
                    store_bundles.append(i)
        self.assertEqual(len(store_bundles), 2)
        # Stores to addr(5,1) and addr(5,2) → different offsets → co-issue
        self.assertEqual(store_bundles[0], store_bundles[1],
                         "Back edge should be skipped; for_init's state should propagate to for_body")

    def test_predecessors_computed_correctly(self):
        """Verify predecessors are properly set on MachineBasicBlocks."""
        block_a = BasicBlock(
            name="a",
            instructions=[],
            terminator=LIRInst(LIROpcode.JUMP, None, ["b"], "flow"),
        )
        block_b = BasicBlock(
            name="b",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({"a": block_a, "b": block_b}, entry="a")
        self.assertEqual(mir.blocks["a"].predecessors, [])
        self.assertEqual(mir.blocks["b"].predecessors, ["a"])

    def test_predecessors_diamond(self):
        """Predecessor lists for diamond CFG."""
        entry = BasicBlock(
            name="entry",
            instructions=[LIRInst(LIROpcode.CONST, 0, [1], "load")],
            terminator=LIRInst(LIROpcode.COND_JUMP, None, [0, "left", "right"], "flow"),
        )
        left = BasicBlock(
            name="left",
            instructions=[],
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        right = BasicBlock(
            name="right",
            instructions=[],
            terminator=LIRInst(LIROpcode.JUMP, None, ["merge"], "flow"),
        )
        merge = BasicBlock(
            name="merge",
            instructions=[],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({
            "entry": entry,
            "left": left,
            "right": right,
            "merge": merge,
        })
        self.assertEqual(sorted(mir.blocks["merge"].predecessors), ["left", "right"])
        self.assertEqual(mir.blocks["entry"].predecessors, [])

    def test_no_predecessors_entry_block(self):
        """Entry block starts with empty value state (no predecessors).

        Store then load through undefined s0 → key=None → alias → must serialize.
        """
        entry = BasicBlock(
            name="entry",
            instructions=[
                # s0 is undefined / not tracked
                LIRInst(LIROpcode.STORE, None, [0, 10], "store"),
                LIRInst(LIROpcode.LOAD, 1, [0], "load"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({"entry": entry})
        bundles = mir.blocks["entry"].bundles
        store_idx = None
        load_idx = None
        for i, bundle in enumerate(bundles):
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.STORE:
                    store_idx = i
                if inst.opcode == LIROpcode.LOAD:
                    load_idx = i
        self.assertIsNotNone(store_idx)
        self.assertIsNotNone(load_idx)
        # Store then load to unknown addr → delay=1 → serialized
        self.assertLess(store_idx, load_idx)

    def test_load_store_different_bases_disambiguated_cross_block(self):
        """Load and store to different propagated bases don't create false deps."""
        init = BasicBlock(
            name="init",
            instructions=[
                LIRInst(LIROpcode.CONST, 0, [4], "load"),  # forest_values_p
                LIRInst(LIROpcode.LOAD, 1, [0], "load"),   # s1 = addr(4, 0)
                LIRInst(LIROpcode.CONST, 2, [5], "load"),  # inp_indices_p
                LIRInst(LIROpcode.LOAD, 3, [2], "load"),   # s3 = addr(5, 0)
            ],
            terminator=LIRInst(LIROpcode.JUMP, None, ["body"], "flow"),
        )
        body = BasicBlock(
            name="body",
            instructions=[
                # Store to inp_indices base
                LIRInst(LIROpcode.STORE, None, [3, 10], "store"),
                # Load from forest_values base (different base → no dependency)
                LIRInst(LIROpcode.LOAD, 4, [1], "load"),
            ],
            terminator=LIRInst(LIROpcode.HALT, None, [], "flow"),
        )
        mir = _schedule_multi_block({"init": init, "body": body}, entry="init")
        body_bundles = mir.blocks["body"].bundles

        store_idx = None
        load_idx = None
        for i, bundle in enumerate(body_bundles):
            for inst in bundle.instructions:
                if inst.opcode == LIROpcode.STORE:
                    store_idx = i
                if inst.opcode == LIROpcode.LOAD:
                    load_idx = i
        self.assertIsNotNone(store_idx)
        self.assertIsNotNone(load_idx)
        # Different bases (4 vs 5) → no alias dependency → can co-issue
        self.assertEqual(store_idx, load_idx,
                         "Store and load to different bases should co-issue with cross-block propagation")


class TestComputeExitValueStateVectorOps(unittest.TestCase):
    """Test _compute_exit_value_state with vector operations."""

    def test_vbroadcast_propagates_const(self):
        """VBROADCAST of a known const propagates to all lanes."""
        insts = [
            _mi(LIROpcode.CONST, 0, [42], "load"),
            _mi(LIROpcode.VBROADCAST, [10, 11, 12, 13, 14, 15, 16, 17], [0], "valu"),
        ]
        const_val, addr_expr = _compute_exit_value_state(insts)
        for lane in range(10, 18):
            self.assertEqual(const_val[lane], 42)

    def test_vbroadcast_propagates_addr(self):
        """VBROADCAST of a known addr propagates to all lanes."""
        insts = [
            _mi(LIROpcode.CONST, 0, [5], "load"),
            _mi(LIROpcode.LOAD, 1, [0], "load"),  # addr(5, 0)
            _mi(LIROpcode.VBROADCAST, [10, 11, 12, 13, 14, 15, 16, 17], [1], "valu"),
        ]
        const_val, addr_expr = _compute_exit_value_state(insts)
        for lane in range(10, 18):
            self.assertEqual(addr_expr[lane], AddrExpr(base=5, offset=0))

    def test_vadd_const_lanes(self):
        """VADD of known-const lanes produces known-const results."""
        insts = [
            _mi(LIROpcode.CONST, 0, [10], "load"),
            _mi(LIROpcode.VBROADCAST, [100, 101], [0], "valu"),  # 2-lane for simplicity
            _mi(LIROpcode.CONST, 1, [5], "load"),
            _mi(LIROpcode.VBROADCAST, [200, 201], [1], "valu"),
            _mi(LIROpcode.VADD, [300, 301], [[100, 101], [200, 201]], "valu"),
        ]
        const_val, _ = _compute_exit_value_state(insts)
        self.assertEqual(const_val.get(300), 15)
        self.assertEqual(const_val.get(301), 15)

    def test_copy_propagates_const(self):
        """COPY of a known const propagates."""
        insts = [
            _mi(LIROpcode.CONST, 0, [42], "load"),
            _mi(LIROpcode.COPY, 1, [0], "alu"),
        ]
        const_val, _ = _compute_exit_value_state(insts)
        self.assertEqual(const_val.get(1), 42)

    def test_copy_propagates_addr(self):
        """COPY of a known addr propagates."""
        insts = [
            _mi(LIROpcode.CONST, 0, [5], "load"),
            _mi(LIROpcode.LOAD, 1, [0], "load"),
            _mi(LIROpcode.COPY, 2, [1], "alu"),
        ]
        _, addr_expr = _compute_exit_value_state(insts)
        self.assertEqual(addr_expr.get(2), AddrExpr(base=5, offset=0))


if __name__ == "__main__":
    unittest.main()
