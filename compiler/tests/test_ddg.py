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
