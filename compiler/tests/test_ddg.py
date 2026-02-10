"""Tests for Data Dependency Graph."""

import pytest
from compiler.ddg import (
    DDGNode,
    DataDependencyDAG,
    BlockDDGs,
    HIRDDGBuilder,
    LIRDDGBuilder,
    get_dag_depth,
    get_dag_width,
    find_independent_nodes,
    print_dag,
    print_dag_tree,
    print_block_ddgs,
    print_dag_dot,
)
from compiler.hir import Op, SSAValue, VectorSSAValue, Const
from compiler.lir import LIRInst, LIROpcode


class TestHIRDDG:

    def test_simple_chain(self):
        """Test linear dependency: a = const; b = a + 1; store b"""
        a = SSAValue(0, "a")
        b = SSAValue(1, "b")
        addr = SSAValue(2, "addr")

        ops = [
            Op("const", a, [Const(5)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("store", None, [addr, b], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        assert len(ddgs.dags) == 1  # One root (store)
        dag = ddgs.dags[0]
        assert dag.root.instruction.opcode == "store"
        assert len(dag.nodes) == 4  # All 4 ops in the DAG

    def test_multiple_roots(self):
        """Test multiple stores -> multiple DAGs."""
        a = SSAValue(0)
        b = SSAValue(1)
        addr1 = SSAValue(2)
        addr2 = SSAValue(3)

        ops = [
            Op("const", a, [Const(1)], "load"),
            Op("const", b, [Const(2)], "load"),
            Op("const", addr1, [Const(100)], "load"),
            Op("const", addr2, [Const(200)], "load"),
            Op("store", None, [addr1, a], "store"),
            Op("store", None, [addr2, b], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        assert len(ddgs.dags) == 2  # Two stores = two roots

    def test_shared_dependency(self):
        """Test when two roots share a dependency."""
        a = SSAValue(0)
        b = SSAValue(1)
        c = SSAValue(2)
        addr1 = SSAValue(3)
        addr2 = SSAValue(4)

        ops = [
            Op("const", a, [Const(5)], "load"),
            Op("const", addr1, [Const(100)], "load"),
            Op("const", addr2, [Const(200)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("+", c, [a, Const(2)], "alu"),
            Op("store", None, [addr1, b], "store"),
            Op("store", None, [addr2, c], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        assert len(ddgs.dags) == 2
        # Both DAGs should include node for 'a'
        # def_map uses SSAValue objects directly as keys
        assert a in ddgs.def_map
        a_node = ddgs.def_map[a]
        # Check that 'a' appears in both DAGs
        assert a_node in ddgs.dags[0].nodes
        assert a_node in ddgs.dags[1].nodes

    def test_unused_result_is_root(self):
        """Test that unused results become roots."""
        a = SSAValue(0)
        b = SSAValue(1)  # Unused

        ops = [
            Op("const", a, [Const(5)], "load"),
            Op("+", b, [a, Const(1)], "alu"),  # Result unused
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        assert len(ddgs.dags) == 1
        assert ddgs.dags[0].root.is_root
        assert ddgs.dags[0].root.instruction.opcode == "+"

    def test_vector_operations(self):
        """Test DDG with vector SSA values."""
        v0 = VectorSSAValue(0, "v0")
        v1 = VectorSSAValue(1, "v1")
        v2 = VectorSSAValue(2, "v2")
        addr = SSAValue(0, "addr")

        ops = [
            Op("const", addr, [Const(100)], "load"),
            Op("vload", v0, [addr], "load"),
            Op("vbroadcast", v1, [Const(2)], "valu"),
            Op("v*", v2, [v0, v1], "valu"),
            Op("vstore", None, [addr, v2], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        assert len(ddgs.dags) == 1
        dag = ddgs.dags[0]
        assert dag.root.instruction.opcode == "vstore"
        assert len(dag.nodes) == 5

    def test_diamond_dependency(self):
        """Test diamond-shaped dependency graph."""
        a = SSAValue(0)
        b = SSAValue(1)
        c = SSAValue(2)
        d = SSAValue(3)
        addr = SSAValue(4)

        #     a
        #    / \
        #   b   c
        #    \ /
        #     d
        #     |
        #   store
        ops = [
            Op("const", a, [Const(10)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("+", c, [a, Const(2)], "alu"),
            Op("+", d, [b, c], "alu"),
            Op("const", addr, [Const(100)], "load"),
            Op("store", None, [addr, d], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        assert len(ddgs.dags) == 1
        dag = ddgs.dags[0]
        assert len(dag.nodes) == 6

        # Check edge structure
        store_node = dag.root
        assert len(store_node.operand_nodes) == 2  # addr and d
        d_node = ddgs.def_map[d]
        assert len(d_node.operand_nodes) == 2  # b and c
        assert len(d_node.user_nodes) == 1  # store

    def test_topological_iteration(self):
        """Test that topological iteration works correctly."""
        a = SSAValue(0)
        b = SSAValue(1)
        c = SSAValue(2)
        addr = SSAValue(3)

        ops = [
            Op("const", a, [Const(5)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("+", c, [b, Const(2)], "alu"),
            Op("store", None, [addr, c], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        # Topological order: leaves first
        topo_order = list(dag.iter_topological())
        # Root should be last
        assert topo_order[-1] == dag.root

        # Reverse topological: root first
        rev_topo = list(dag.iter_reverse_topological())
        assert rev_topo[0] == dag.root

    def test_def_map_and_inst_map(self):
        """Test lookup maps in BlockDDGs."""
        a = SSAValue(0)
        b = SSAValue(1)

        ops = [
            Op("const", a, [Const(5)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        # def_map uses SSAValue objects directly as keys
        assert a in ddgs.def_map
        assert b in ddgs.def_map
        assert ddgs.def_map[a].instruction.opcode == "const"
        assert ddgs.def_map[b].instruction.opcode == "+"

        # inst_map should contain both instructions
        assert id(ops[0]) in ddgs.inst_map
        assert id(ops[1]) in ddgs.inst_map


class TestLIRDDG:

    def test_scratch_dependencies(self):
        """Test LIR with scratch address dependencies."""
        insts = [
            LIRInst(LIROpcode.CONST, 0, [42], "load"),       # scratch[0] = 42
            LIRInst(LIROpcode.CONST, 1, [100], "load"),      # scratch[1] = 100
            LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),        # scratch[2] = scratch[0] + scratch[1]
            LIRInst(LIROpcode.STORE, None, [1, 2], "store"), # mem[scratch[1]] = scratch[2]
        ]

        builder = LIRDDGBuilder()
        ddgs = builder.build(insts)

        assert len(ddgs.dags) == 1
        dag = ddgs.dags[0]
        assert dag.root.instruction.opcode == LIROpcode.STORE
        assert len(dag.nodes) == 4

    def test_lir_multiple_roots(self):
        """Test LIR with multiple stores."""
        insts = [
            LIRInst(LIROpcode.CONST, 0, [10], "load"),
            LIRInst(LIROpcode.CONST, 1, [20], "load"),
            LIRInst(LIROpcode.CONST, 2, [100], "load"),  # addr1
            LIRInst(LIROpcode.CONST, 3, [200], "load"),  # addr2
            LIRInst(LIROpcode.STORE, None, [2, 0], "store"),
            LIRInst(LIROpcode.STORE, None, [3, 1], "store"),
        ]

        builder = LIRDDGBuilder()
        ddgs = builder.build(insts)

        assert len(ddgs.dags) == 2

    def test_lir_vector_dependencies(self):
        """Test LIR with vector scratch addresses."""
        # Vector dest is list of 8 scratch addresses
        vec_dest = list(range(0, 8))
        vec_dest2 = list(range(8, 16))

        insts = [
            LIRInst(LIROpcode.CONST, 100, [0], "load"),  # addr
            LIRInst(LIROpcode.VLOAD, vec_dest, [100], "load"),
            LIRInst(LIROpcode.VBROADCAST, vec_dest2, [42], "valu"),
            LIRInst(LIROpcode.VMUL, list(range(16, 24)), [vec_dest, vec_dest2], "valu"),
            LIRInst(LIROpcode.VSTORE, None, [100, list(range(16, 24))], "store"),
        ]

        builder = LIRDDGBuilder()
        ddgs = builder.build(insts)

        assert len(ddgs.dags) == 1
        assert len(ddgs.dags[0].nodes) == 5

    def test_lir_jump_is_root(self):
        """Test that jump instructions are treated as roots."""
        insts = [
            LIRInst(LIROpcode.CONST, 0, [1], "load"),
            LIRInst(LIROpcode.COND_JUMP, None, [0, "then", "else"], "flow"),
        ]

        builder = LIRDDGBuilder()
        ddgs = builder.build(insts)

        assert len(ddgs.dags) == 1
        assert ddgs.dags[0].root.instruction.opcode == LIROpcode.COND_JUMP

    def test_const_has_no_operand_dependencies(self):
        """Test that CONST instructions have no operand dependencies.

        CONST has [immediate_value] which is NOT a scratch address.
        This is a regression test for a bug where the immediate value
        was incorrectly treated as a scratch address dependency.
        """
        insts = [
            LIRInst(LIROpcode.CONST, 0, [42], "load"),    # scratch[0] = 42
            LIRInst(LIROpcode.CONST, 1, [100], "load"),   # scratch[1] = 100
            LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),     # scratch[2] = scratch[0] + scratch[1]
            LIRInst(LIROpcode.STORE, None, [1, 2], "store"),
        ]

        builder = LIRDDGBuilder()
        ddgs = builder.build(insts)

        # Check that CONST instructions have no operand dependencies
        const0_node = ddgs.def_map[0]
        const1_node = ddgs.def_map[1]

        # CONST should have empty operand_nodes (no dependencies)
        assert len(const0_node.operand_nodes) == 0, \
            f"CONST should have no dependencies, got {const0_node.operand_nodes}"
        assert len(const1_node.operand_nodes) == 0, \
            f"CONST should have no dependencies, got {const1_node.operand_nodes}"

        # ADD should have 2 dependencies (scratch[0] and scratch[1])
        add_node = ddgs.def_map[2]
        assert len(add_node.operand_nodes) == 2
        # The operand_nodes should be the CONST nodes (not None)
        assert const0_node in add_node.operand_nodes
        assert const1_node in add_node.operand_nodes

    def test_load_offset_depends_on_lane_address(self):
        """Test that LOAD_OFFSET depends on lane address (base + offset)."""
        insts = [
            LIRInst(LIROpcode.CONST, 0, [1000], "load"),   # scratch[0] = 1000 (base addr)
            LIRInst(LIROpcode.CONST, 5, [1234], "load"),   # scratch[5] = lane address
            # load_offset: scratch[1] = mem[scratch[0] + 5]
            # operands are [base_scratch=0, offset_immediate=5]
            LIRInst(LIROpcode.LOAD_OFFSET, 1, [0, 5], "load"),
            # load_offset writes to dest+offset => scratch[6]
            LIRInst(LIROpcode.STORE, None, [0, 6], "store"),
        ]

        builder = LIRDDGBuilder()
        ddgs = builder.build(insts)

        # Get nodes
        lane_addr_node = ddgs.def_map[5]
        load_offset_node = ddgs.def_map[6]

        # LOAD_OFFSET should depend on the lane address scratch (base + offset).
        assert len(load_offset_node.operand_nodes) == 1, \
            f"LOAD_OFFSET should have 1 dependency (lane addr), got {len(load_offset_node.operand_nodes)}"
        assert load_offset_node.operand_nodes[0] == lane_addr_node, \
            "LOAD_OFFSET should depend on the CONST that defines the lane address scratch"

    def test_load_offset_not_root_when_vector_consumed(self):
        """LOAD_OFFSET lane defs should not be roots when a vector op consumes them."""
        lane_addrs = list(range(100, 108))
        gather_lanes = list(range(200, 208))
        out_lanes = list(range(300, 308))

        insts = []
        for lane in lane_addrs:
            insts.append(LIRInst(LIROpcode.CONST, lane, [1000 + lane], "load"))

        # Lowered vgather form: each LOAD_OFFSET writes one lane (dest+offset).
        for i in range(8):
            insts.append(LIRInst(LIROpcode.LOAD_OFFSET, 200, [100, i], "load"))

        # Consume gathered vector lanes as a tuple operand.
        insts.append(LIRInst(LIROpcode.VADD, out_lanes, [gather_lanes, gather_lanes], "valu"))

        builder = LIRDDGBuilder()
        ddgs = builder.build(insts)

        # All gathered lane defs should exist in def_map.
        for lane in gather_lanes:
            assert lane in ddgs.def_map

        # LOAD_OFFSET nodes should not be marked as roots (their results are used).
        roots = [dag.root.instruction.opcode for dag in ddgs.dags]
        assert LIROpcode.LOAD_OFFSET not in roots

    def test_vector_results_have_tuple_and_lane_keys(self):
        """Vector defs should be accessible by both tuple key and per-lane keys."""
        vec = list(range(10, 18))
        out = list(range(30, 38))
        insts = [
            LIRInst(LIROpcode.VBROADCAST, vec, [5], "valu"),
            LIRInst(LIROpcode.VADD, out, [vec, vec], "valu"),
        ]

        builder = LIRDDGBuilder()
        ddgs = builder.build(insts)

        assert tuple(vec) in ddgs.def_map
        for lane in vec:
            assert lane in ddgs.def_map

    def test_vector_operand_falls_back_to_lane_dependencies(self):
        """If vector tuple producer is missing, DDG should connect per-lane defs."""
        gather_lanes = list(range(200, 208))
        out_lanes = list(range(300, 308))
        insts = []

        for lane in range(100, 108):
            insts.append(LIRInst(LIROpcode.CONST, lane, [1000 + lane], "load"))
        for i in range(8):
            insts.append(LIRInst(LIROpcode.LOAD_OFFSET, 200, [100, i], "load"))
        insts.append(LIRInst(LIROpcode.VADD, out_lanes, [gather_lanes, gather_lanes], "valu"))

        ddgs = LIRDDGBuilder().build(insts)
        vadd_node = ddgs.def_map[tuple(out_lanes)]

        # Two vector operands, each should fall back to 8 lane dependencies.
        lane_dep_nodes = [n for n in vadd_node.operand_nodes if n is not None and n.instruction.opcode == LIROpcode.LOAD_OFFSET]
        assert len(lane_dep_nodes) == 16


class TestDAGUtilities:

    def test_get_dag_depth_linear(self):
        """Test depth computation on linear chain."""
        a = SSAValue(0)
        b = SSAValue(1)
        c = SSAValue(2)
        d = SSAValue(3)
        addr = SSAValue(4)

        # a -> b -> c -> d -> store
        # Depths: a=0, addr=0, b=1, c=2, d=3, store=4
        ops = [
            Op("const", a, [Const(1)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("+", c, [b, Const(1)], "alu"),
            Op("+", d, [c, Const(1)], "alu"),
            Op("store", None, [addr, d], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        depth = get_dag_depth(dag)
        assert depth == 4  # store(depth 4) -> d(3) -> c(2) -> b(1) -> a(0); addr(0)

    def test_get_dag_depth_parallel(self):
        """Test depth computation with parallel paths."""
        a = SSAValue(0)
        b = SSAValue(1)
        c = SSAValue(2)
        addr = SSAValue(3)

        # a and b are independent, both feed into c
        # depth is 2 (c -> a or b -> store)
        ops = [
            Op("const", a, [Const(1)], "load"),
            Op("const", b, [Const(2)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", c, [a, b], "alu"),
            Op("store", None, [addr, c], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        depth = get_dag_depth(dag)
        assert depth == 2  # store -> c -> (a or b)

    def test_get_dag_width(self):
        """Test width computation."""
        a = SSAValue(0)
        b = SSAValue(1)
        c = SSAValue(2)
        d = SSAValue(3)
        addr = SSAValue(4)

        # Level 0: a, b, addr (3 nodes)
        # Level 1: c, d (2 nodes)
        # Level 2: store (1 node)
        ops = [
            Op("const", a, [Const(1)], "load"),
            Op("const", b, [Const(2)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", c, [a, Const(1)], "alu"),
            Op("+", d, [b, Const(1)], "alu"),
            Op("store", None, [addr, c], "store"),  # Only uses c, not d
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        # d is unused, so it's a separate root
        assert len(ddgs.dags) == 2

    def test_find_independent_nodes(self):
        """Test grouping nodes by independence level."""
        a = SSAValue(0)
        b = SSAValue(1)
        c = SSAValue(2)
        addr = SSAValue(3)

        ops = [
            Op("const", a, [Const(1)], "load"),
            Op("const", b, [Const(2)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", c, [a, b], "alu"),
            Op("store", None, [addr, c], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        levels = find_independent_nodes(dag)

        # Level 0: a, b, addr (independent leaves)
        # Level 1: c (depends on a, b)
        # Level 2: store (depends on addr, c)
        assert len(levels) == 3
        assert len(levels[0]) == 3  # a, b, addr
        assert len(levels[1]) == 1  # c
        assert len(levels[2]) == 1  # store


class TestPrettyPrint:

    def test_print_dag(self):
        """Test basic DAG pretty printing."""
        a = SSAValue(0, "a")
        b = SSAValue(1, "b")
        addr = SSAValue(2, "addr")

        ops = [
            Op("const", a, [Const(5)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("store", None, [addr, b], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        output = print_dag(dag)

        # Check that output contains expected elements
        assert "DAG" in output
        assert "Level 0" in output
        assert "Level 1" in output
        assert "[ROOT]" in output
        assert "depends on:" in output

    def test_print_block_ddgs(self):
        """Test BlockDDGs pretty printing."""
        a = SSAValue(0)
        b = SSAValue(1)
        addr1 = SSAValue(2)
        addr2 = SSAValue(3)

        ops = [
            Op("const", a, [Const(1)], "load"),
            Op("const", b, [Const(2)], "load"),
            Op("const", addr1, [Const(100)], "load"),
            Op("const", addr2, [Const(200)], "load"),
            Op("store", None, [addr1, a], "store"),
            Op("store", None, [addr2, b], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        output = print_block_ddgs(ddgs, "test_block")

        assert "test_block" in output
        assert "Total DAGs: 2" in output
        assert "DAG 0" in output
        assert "DAG 1" in output

    def test_print_dag_dot(self):
        """Test DOT format output."""
        a = SSAValue(0)
        b = SSAValue(1)
        addr = SSAValue(2)

        ops = [
            Op("const", a, [Const(5)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("store", None, [addr, b], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        output = print_dag_dot(dag, "test_dag")

        # Check DOT structure
        assert "digraph test_dag {" in output
        assert "rankdir=BT" in output
        assert "n0" in output  # Node 0
        assert "->" in output  # Edges

    def test_print_dag_tree(self):
        """Test tree-format DAG pretty printing."""
        a = SSAValue(0, "a")
        b = SSAValue(1, "b")
        addr = SSAValue(2, "addr")

        ops = [
            Op("const", a, [Const(5)], "load"),
            Op("const", addr, [Const(100)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("store", None, [addr, b], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        output = print_dag_tree(dag)

        # Check that output contains tree structure elements
        assert "[ROOT]" in output
        assert "[3]" in output  # Root node id
        assert "├──" in output or "└──" in output  # Tree connectors
        assert "store" in output

    def test_print_dag_tree_diamond(self):
        """Test tree printing with shared dependencies (diamond shape)."""
        a = SSAValue(0, "a")
        b = SSAValue(1, "b")
        c = SSAValue(2, "c")
        d = SSAValue(3, "d")
        addr = SSAValue(4, "addr")

        #     a
        #    / \
        #   b   c
        #    \ /
        #     d
        #     |
        #   store
        ops = [
            Op("const", a, [Const(10)], "load"),
            Op("+", b, [a, Const(1)], "alu"),
            Op("+", c, [a, Const(2)], "alu"),
            Op("+", d, [b, c], "alu"),
            Op("const", addr, [Const(100)], "load"),
            Op("store", None, [addr, d], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        output = print_dag_tree(dag)

        # Check that shared node 'a' is marked as duplicate
        assert "[*]" in output  # Should have duplicate marker
        assert "[ROOT]" in output

    def test_print_dag_tree_single_node(self):
        """Test tree printing with a single unused node."""
        a = SSAValue(0, "a")

        ops = [
            Op("const", a, [Const(42)], "load"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)
        dag = ddgs.dags[0]

        output = print_dag_tree(dag)

        # Single node should be marked as root with no children
        assert "[ROOT]" in output
        assert "[0]" in output
        assert "├──" not in output  # No children
        assert "└──" not in output


class TestExternalOperands:
    """Tests for DDG handling of external operands (defined outside the block)."""

    def test_external_operand_placeholder(self):
        """Test that external operands get None placeholders in operand_nodes."""
        # Simulate a scenario where 'base' is defined outside the block
        # and used in stores within the block
        base = SSAValue(0, "base")  # External - not defined in ops list
        val1 = SSAValue(1, "val1")
        val2 = SSAValue(2, "val2")

        # Only val1 and val2 are defined in this block
        # base is external (not in ops list)
        ops = [
            Op("const", val1, [Const(42)], "load"),
            Op("const", val2, [Const(43)], "load"),
            # store uses external 'base' as address, internal 'val1' as value
            Op("store", None, [base, val1], "store"),
            Op("store", None, [base, val2], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        # Get store nodes
        store1_node = ddgs.inst_map[id(ops[2])]
        store2_node = ddgs.inst_map[id(ops[3])]

        # operand_nodes should maintain position correspondence:
        # operand_nodes[0] = None (external base)
        # operand_nodes[1] = node for val1/val2
        assert len(store1_node.operand_nodes) == 2
        assert store1_node.operand_nodes[0] is None  # External operand
        assert store1_node.operand_nodes[1] is not None  # Internal val1
        assert store1_node.operand_nodes[1].instruction.result == val1

        assert len(store2_node.operand_nodes) == 2
        assert store2_node.operand_nodes[0] is None  # External operand
        assert store2_node.operand_nodes[1] is not None  # Internal val2
        assert store2_node.operand_nodes[1].instruction.result == val2

    def test_mixed_internal_external_operands(self):
        """Test ops with mix of internal and external operands."""
        ext1 = SSAValue(0, "ext1")  # External
        ext2 = SSAValue(1, "ext2")  # External
        int1 = SSAValue(2, "int1")  # Internal

        ops = [
            Op("const", int1, [Const(10)], "load"),
            # add uses two external operands
            Op("+", SSAValue(3, "sum1"), [ext1, ext2], "alu"),
            # add uses one external, one internal
            Op("+", SSAValue(4, "sum2"), [ext1, int1], "alu"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        # sum1 = ext1 + ext2: both operands external
        sum1_node = ddgs.inst_map[id(ops[1])]
        assert len(sum1_node.operand_nodes) == 2
        assert sum1_node.operand_nodes[0] is None  # ext1 external
        assert sum1_node.operand_nodes[1] is None  # ext2 external

        # sum2 = ext1 + int1: first external, second internal
        sum2_node = ddgs.inst_map[id(ops[2])]
        assert len(sum2_node.operand_nodes) == 2
        assert sum2_node.operand_nodes[0] is None  # ext1 external
        assert sum2_node.operand_nodes[1] is not None  # int1 internal
        assert sum2_node.operand_nodes[1].instruction.result == int1

    def test_dag_traversal_skips_none(self):
        """Test that DAG traversal correctly skips None (external) operand nodes."""
        ext_base = SSAValue(0, "base")  # External
        val = SSAValue(1, "val")

        ops = [
            Op("const", val, [Const(42)], "load"),
            Op("store", None, [ext_base, val], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        # Should have one DAG rooted at store
        assert len(ddgs.dags) == 1
        dag = ddgs.dags[0]

        # DAG should include store and val's const, but not crash on None
        assert dag.root.instruction.opcode == "store"
        # Both ops should be in the DAG
        assert len(dag.nodes) == 2

    def test_slp_pattern_simplified_address(self):
        """
        Test the SLP pattern where simplify pass converts +(base, #0) to base.

        This is the key pattern that was failing:
        - Iteration 0: store(base, v1) - base used directly
        - Iteration 1: store(+(base, #1), v2) - base used in add

        Both should have operand_nodes with proper position correspondence.
        """
        # External base (simulating inp_indices_p defined outside loop body)
        base = SSAValue(0, "inp_indices_p")

        # Internal values and addresses
        val0 = SSAValue(1, "val0")
        val1 = SSAValue(2, "val1")
        addr1 = SSAValue(3, "addr1")

        ops = [
            # Compute values
            Op("const", val0, [Const(100)], "load"),
            Op("const", val1, [Const(200)], "load"),
            # Compute address for iteration 1: base + 1
            Op("+", addr1, [base, Const(1)], "alu"),
            # Iteration 0 store: uses base directly (after simplify removed +(base, #0))
            Op("store", None, [base, val0], "store"),
            # Iteration 1 store: uses addr1 = +(base, #1)
            Op("store", None, [addr1, val1], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        # Get store nodes
        store0 = ddgs.inst_map[id(ops[3])]  # store(base, val0)
        store1 = ddgs.inst_map[id(ops[4])]  # store(addr1, val1)

        # Store 0: operands are [base (ext), val0 (int)]
        assert len(store0.operand_nodes) == 2
        assert store0.operand_nodes[0] is None  # base is external
        assert store0.operand_nodes[1] is not None  # val0 is internal
        assert store0.operand_nodes[1].instruction.result == val0

        # Store 1: operands are [addr1 (int), val1 (int)]
        assert len(store1.operand_nodes) == 2
        assert store1.operand_nodes[0] is not None  # addr1 is internal
        assert store1.operand_nodes[0].instruction.result == addr1
        assert store1.operand_nodes[1] is not None  # val1 is internal
        assert store1.operand_nodes[1].instruction.result == val1

        # addr1's operands: [base (ext), Const(1)]
        # Note: Const operands don't have DDG entries
        addr1_node = ddgs.def_map[addr1]
        assert len(addr1_node.operand_nodes) == 1  # Only base (Const is skipped)
        assert addr1_node.operand_nodes[0] is None  # base is external


class TestEdgeCases:

    def test_empty_block(self):
        """Test empty instruction list."""
        builder = HIRDDGBuilder()
        ddgs = builder.build([])

        assert len(ddgs.dags) == 0
        assert len(ddgs.def_map) == 0
        assert len(ddgs.inst_map) == 0

    def test_no_dependencies(self):
        """Test instructions with no dependencies between them."""
        a = SSAValue(0)
        b = SSAValue(1)

        ops = [
            Op("const", a, [Const(1)], "load"),
            Op("const", b, [Const(2)], "load"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        # Both are unused, so both are roots
        assert len(ddgs.dags) == 2

    def test_single_store(self):
        """Test single store with immediate operands."""
        addr = SSAValue(0)

        ops = [
            Op("const", addr, [Const(100)], "load"),
            Op("store", None, [addr, Const(42)], "store"),
        ]

        builder = HIRDDGBuilder()
        ddgs = builder.build(ops)

        assert len(ddgs.dags) == 1
        assert len(ddgs.dags[0].nodes) == 2

    def test_node_equality_and_hash(self):
        """Test DDGNode __eq__ and __hash__."""
        a = SSAValue(0)
        op = Op("const", a, [Const(1)], "load")

        node1 = DDGNode(id=0, instruction=op)
        node2 = DDGNode(id=0, instruction=op)
        node3 = DDGNode(id=1, instruction=op)

        assert node1 == node2
        assert node1 != node3
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)

        # Test in set
        s = {node1}
        assert node2 in s
        assert node3 not in s
