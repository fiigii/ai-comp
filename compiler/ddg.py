"""
Data Dependency Graph (DDG)

Represents instruction data dependencies within a basic block.
Works with both HIR (SSA values) and LIR (scratch addresses).

A DDG is a DAG where:
- Nodes are instructions
- Edges represent data dependencies (use-def chains)
- Roots are stores or instructions whose results are not used in the block
- A block can have multiple DAGs (one per root)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar, Generic, Iterator

from .hir import Op, SSAValue, VectorSSAValue, Const, Halt, Pause
from .lir import LIRInst, LIROpcode

# Instruction type (HIR Op or LIR LIRInst)
T = TypeVar('T')


@dataclass
class DDGNode(Generic[T]):
    """A node in the data dependency graph."""
    id: int
    instruction: T
    # Dependencies: nodes whose results this instruction uses
    operand_nodes: list['DDGNode[T]'] = field(default_factory=list)
    # Reverse edges: nodes that use this instruction's result
    user_nodes: list['DDGNode[T]'] = field(default_factory=list)
    is_root: bool = False

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, DDGNode) and self.id == other.id

    def __repr__(self):
        return f"DDGNode({self.id}, {self.instruction})"


@dataclass
class DataDependencyDAG(Generic[T]):
    """A single DAG rooted at a store or unused-result instruction."""
    root: DDGNode[T]
    # All nodes reachable from root, in reverse topological order
    # (root first, leaves last - for backward traversal)
    nodes: list[DDGNode[T]] = field(default_factory=list)

    def iter_topological(self) -> Iterator[DDGNode[T]]:
        """Iterate nodes in topological order (leaves first, root last)."""
        return reversed(self.nodes)

    def iter_reverse_topological(self) -> Iterator[DDGNode[T]]:
        """Iterate in reverse topological order (root first)."""
        return iter(self.nodes)


@dataclass
class BlockDDGs(Generic[T]):
    """All DAGs for a basic block."""
    dags: list[DataDependencyDAG[T]]
    # Lookup: result_id -> node that produces it
    def_map: dict[Any, DDGNode[T]] = field(default_factory=dict)
    # Lookup: id(instruction) -> its node
    inst_map: dict[int, DDGNode[T]] = field(default_factory=dict)


class DDGBuilder(ABC, Generic[T]):
    """Abstract builder for constructing DDGs from a basic block."""

    @abstractmethod
    def get_result_id(self, inst: T) -> Optional[Any]:
        """Get the result identifier (SSA id for HIR, scratch addr for LIR).
        Returns None if instruction has no result (e.g., store)."""
        pass

    @abstractmethod
    def get_operand_ids(self, inst: T) -> list[Any]:
        """Get the identifiers of values this instruction uses."""
        pass

    @abstractmethod
    def is_side_effect(self, inst: T) -> bool:
        """Check if instruction has side effects (store, halt, etc.)."""
        pass

    def build(self, instructions: list[T], *, build_dags: bool = True) -> BlockDDGs[T]:
        """Build DDGs for a list of instructions (basic block).

        Most users only need the use/def graph (def_map, inst_map, operand_nodes,
        user_nodes). Building per-root DAGs (ddgs.dags) is much more expensive on
        large blocks because it repeats traversal for each root.

        build_dags=True keeps the previous behavior.
        build_dags=False returns an empty dags list but still populates def_map
        and inst_map and wires operand/user edges.
        """
        # Step 1: Create nodes for all instructions
        nodes: list[DDGNode[T]] = []
        def_map: dict[Any, DDGNode[T]] = {}  # result_id -> defining node
        inst_map: dict[int, DDGNode[T]] = {}  # id(inst) -> node

        for i, inst in enumerate(instructions):
            node = DDGNode(id=i, instruction=inst)
            nodes.append(node)
            inst_map[id(inst)] = node

            result_id = self.get_result_id(inst)
            if result_id is not None:
                def_map[result_id] = node

        # Step 2: Build edges (operand dependencies)
        # Note: operand_nodes maintains position correspondence with operands.
        # If an operand is external (not defined in this block), we add None
        # as a placeholder to keep indices aligned.
        used_results: Optional[set[Any]] = set() if build_dags else None

        for node in nodes:
            operand_ids = self.get_operand_ids(node.instruction)
            for op_id in operand_ids:
                if used_results is not None:
                    used_results.add(op_id)
                if op_id in def_map:
                    dep_node = def_map[op_id]
                    node.operand_nodes.append(dep_node)
                    dep_node.user_nodes.append(node)
                else:
                    # External operand - add None placeholder to maintain position
                    node.operand_nodes.append(None)

        if not build_dags:
            return BlockDDGs(dags=[], def_map=def_map, inst_map=inst_map)

        # Step 3: Identify roots (stores or unused results)
        roots: list[DDGNode[T]] = []
        for node in nodes:
            result_id = self.get_result_id(node.instruction)
            # used_results is non-None when build_dags=True.
            is_unused = result_id is not None and result_id not in used_results
            is_side_effect = self.is_side_effect(node.instruction)

            if is_side_effect or is_unused:
                node.is_root = True
                roots.append(node)

        # Step 4: Build DAGs by backward traversal from each root
        dags: list[DataDependencyDAG[T]] = []

        for root in roots:
            dag_nodes: list[DDGNode[T]] = []
            visited: set[int] = set()

            def visit(n: DDGNode[T]):
                if n is None:
                    return  # Skip None (external operands)
                if n.id in visited:
                    return
                visited.add(n.id)
                dag_nodes.append(n)
                for dep in n.operand_nodes:
                    visit(dep)

            visit(root)
            dags.append(DataDependencyDAG(root=root, nodes=dag_nodes))

        return BlockDDGs(dags=dags, def_map=def_map, inst_map=inst_map)


class HIRDDGBuilder(DDGBuilder[Op]):
    """Builder for HIR Op lists (within a single scope/region).

    Uses SSAValue/VectorSSAValue objects directly as keys since they are
    frozen dataclasses and thus hashable.
    """

    def get_result_id(self, inst: Op) -> Optional[SSAValue | VectorSSAValue]:
        if inst.result is None:
            return None
        if isinstance(inst.result, (SSAValue, VectorSSAValue)):
            return inst.result
        return None

    def get_operand_ids(self, inst: Op) -> list[SSAValue | VectorSSAValue]:
        ids = []
        for op in inst.operands:
            if isinstance(op, (SSAValue, VectorSSAValue)):
                ids.append(op)
            # Skip Const - no data dependency
        return ids

    def is_side_effect(self, inst: Op) -> bool:
        return inst.opcode in ("store", "vstore", "halt", "pause")


class LIRDDGBuilder(DDGBuilder[LIRInst]):
    """Builder for LIR basic blocks."""

    def get_result_id(self, inst: LIRInst) -> Optional[int | tuple]:
        if inst.dest is None:
            return None
        if isinstance(inst.dest, list):
            # Vector result - use tuple as key
            return tuple(inst.dest)
        return inst.dest  # Scalar scratch address

    def get_operand_ids(self, inst: LIRInst) -> list[int | tuple]:
        # Handle instructions with immediate operands (not scratch dependencies)
        if inst.opcode == LIROpcode.CONST:
            # CONST has [immediate_value] - no scratch dependencies
            return []

        if inst.opcode == LIROpcode.LOAD_OFFSET:
            # LOAD_OFFSET has [base_scratch, offset_immediate]
            # Only base_scratch is a dependency
            base = inst.operands[0]
            if isinstance(base, int):
                return [base]
            return []

        # Default: all int operands are scratch addresses
        ids = []
        for op in inst.operands:
            if isinstance(op, int):
                ids.append(op)  # Scratch address
            elif isinstance(op, list):
                ids.append(tuple(op))  # Vector scratch addresses
            # Skip strings (labels)
        return ids

    def is_side_effect(self, inst: LIRInst) -> bool:
        return inst.opcode in (
            LIROpcode.STORE, LIROpcode.VSTORE,
            LIROpcode.HALT, LIROpcode.PAUSE,
            LIROpcode.JUMP, LIROpcode.COND_JUMP
        )


# === Utility functions ===

def get_dag_depth(dag: DataDependencyDAG) -> int:
    """Compute the critical path length of a DAG."""
    depths: dict[int, int] = {}

    def compute_depth(node: DDGNode) -> int:
        if node.id in depths:
            return depths[node.id]
        children = [d for d in node.operand_nodes if d is not None]
        if not children:
            depths[node.id] = 0
        else:
            depths[node.id] = 1 + max(compute_depth(d) for d in children)
        return depths[node.id]

    return compute_depth(dag.root)


def get_dag_width(dag: DataDependencyDAG) -> dict[int, int]:
    """Compute width at each depth level (for parallelism analysis).

    Returns a dict mapping depth -> number of nodes at that depth.
    """
    depths: dict[int, int] = {}
    width_at_depth: dict[int, int] = {}

    def compute_depth(node: DDGNode) -> int:
        if node.id in depths:
            return depths[node.id]
        children = [d for d in node.operand_nodes if d is not None]
        if not children:
            depths[node.id] = 0
        else:
            depths[node.id] = 1 + max(compute_depth(d) for d in children)
        return depths[node.id]

    for node in dag.nodes:
        d = compute_depth(node)
        width_at_depth[d] = width_at_depth.get(d, 0) + 1

    return width_at_depth


def find_independent_nodes(dag: DataDependencyDAG) -> list[set[DDGNode]]:
    """Group nodes into independent sets (same depth = can execute in parallel)."""
    depths: dict[int, int] = {}
    levels: dict[int, set[DDGNode]] = {}

    def compute_depth(node: DDGNode) -> int:
        if node.id in depths:
            return depths[node.id]
        children = [d for d in node.operand_nodes if d is not None]
        if not children:
            depths[node.id] = 0
        else:
            depths[node.id] = 1 + max(compute_depth(d) for d in children)
        return depths[node.id]

    for node in dag.nodes:
        d = compute_depth(node)
        if d not in levels:
            levels[d] = set()
        levels[d].add(node)

    return [levels[d] for d in sorted(levels.keys())]


# === Pretty printing ===

def print_dag(dag: DataDependencyDAG, indent: str = "") -> str:
    """Pretty print a single DAG.

    Shows the DAG structure with nodes grouped by depth level.
    """
    lines = []
    levels = find_independent_nodes(dag)
    depth = get_dag_depth(dag)

    lines.append(f"{indent}DAG (root: node {dag.root.id}, depth: {depth})")
    lines.append(f"{indent}{'=' * 50}")

    for level_idx, level_nodes in enumerate(levels):
        lines.append(f"{indent}Level {level_idx}:")
        for node in sorted(level_nodes, key=lambda n: n.id):
            root_marker = " [ROOT]" if node.is_root else ""
            deps = ", ".join(str(d.id) for d in node.operand_nodes)
            users = ", ".join(str(u.id) for u in node.user_nodes)
            lines.append(f"{indent}  [{node.id}]{root_marker} {node.instruction}")
            if deps:
                lines.append(f"{indent}       depends on: [{deps}]")
            if users:
                lines.append(f"{indent}       used by: [{users}]")

    return "\n".join(lines)


def print_block_ddgs(ddgs: BlockDDGs, block_name: str = "block") -> str:
    """Pretty print all DAGs for a basic block.

    Args:
        ddgs: The BlockDDGs to print
        block_name: Optional name for the block (for labeling)

    Returns:
        A formatted string representation of the DDGs
    """
    lines = []
    lines.append(f"Data Dependency Graphs for {block_name}")
    lines.append(f"{'#' * 60}")
    lines.append(f"Total DAGs: {len(ddgs.dags)}")
    lines.append(f"Total definitions: {len(ddgs.def_map)}")
    lines.append("")

    for i, dag in enumerate(ddgs.dags):
        lines.append(f"--- DAG {i} ---")
        lines.append(print_dag(dag))
        lines.append("")

    return "\n".join(lines)


def print_dag_tree(dag: DataDependencyDAG) -> str:
    """Pretty print a DAG as a tree structure.

    Shows the DAG with root at top and dependencies indented below.
    Uses tree-drawing characters for visual structure.
    Handles shared nodes by marking duplicates with [*].

    Example output:
        [5] store(addr, val) [ROOT]
        ├── [4] addr = const(100)
        └── [3] val = +(a, b)
            ├── [0] a = const(5)
            └── [1] b = const(10)
    """
    lines = []
    visited: set[int] = set()

    def format_node(node: DDGNode) -> str:
        """Format a node's instruction for display."""
        inst = node.instruction
        # Handle HIR Op
        if hasattr(inst, 'result') and hasattr(inst, 'opcode') and hasattr(inst, 'operands'):
            operands_str = ", ".join(str(op) for op in inst.operands)
            if inst.result is not None:
                return f"{inst.result} = {inst.opcode}({operands_str})"
            return f"{inst.opcode}({operands_str})"
        # Fallback for other instruction types
        return str(inst)

    def print_tree(node: DDGNode, prefix: str, is_last: bool, is_root: bool):
        """Recursively print the tree structure."""
        # Connector characters
        connector = "" if is_root else ("└── " if is_last else "├── ")
        root_marker = " [ROOT]" if node.is_root else ""
        dup_marker = " [*]" if node.id in visited else ""

        lines.append(f"{prefix}{connector}[{node.id}] {format_node(node)}{root_marker}{dup_marker}")

        if node.id in visited:
            return  # Don't recurse into already-visited nodes
        visited.add(node.id)

        # New prefix for children
        if is_root:
            child_prefix = prefix
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")

        children = [c for c in node.operand_nodes if c is not None]
        for i, child in enumerate(children):
            print_tree(child, child_prefix, i == len(children) - 1, False)

    print_tree(dag.root, "", True, True)
    return "\n".join(lines)


def print_dag_dot(dag: DataDependencyDAG, name: str = "dag") -> str:
    """Generate DOT (Graphviz) representation of a DAG.

    Args:
        dag: The DAG to convert
        name: Name for the graph

    Returns:
        DOT format string (can be rendered with graphviz)
    """
    lines = []
    lines.append(f"digraph {name} {{")
    lines.append("  rankdir=BT;")  # Bottom to top (leaves at bottom, root at top)
    lines.append("  node [shape=box];")

    # Add nodes
    for node in dag.nodes:
        label = str(node.instruction).replace('"', '\\"')
        style = "bold" if node.is_root else "solid"
        lines.append(f'  n{node.id} [label="{node.id}: {label}" style="{style}"];')

    # Add edges (from operand to user)
    for node in dag.nodes:
        for dep in node.operand_nodes:
            if dep is not None:
                lines.append(f"  n{dep.id} -> n{node.id};")

    lines.append("}")
    return "\n".join(lines)
