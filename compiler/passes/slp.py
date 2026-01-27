"""
SLP (Superword Level Parallelism) Vectorization Pass

Converts groups of 8 isomorphic scalar operations into single vector operations.
Operates on HIR after loop unrolling and CSE.

Uses DDG (Data Dependency Graph) for:
- Finding seeds (store roots)
- Extending packs (following operand edges)
- Checking legality (no internal dependencies)
"""

from dataclasses import dataclass, field
from typing import Optional

from ..hir import (
    SSAValue, VectorSSAValue, Variable, Const, VectorConst, Value, Op, ForLoop, If,
    Halt, Pause, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..ddg import HIRDDGBuilder, BlockDDGs, DDGNode

# Vector length for this architecture
VLEN = 8


def _make_cache_key(value: Value) -> tuple:
    """Create a stable cache key for a value.

    This is needed because using id() for cache keys can cause issues:
    - Two Const(5) objects have different id()s but represent the same value
    - This leads to cache misses for semantically identical values

    For Const: Use ("const", const.value) as cache key
    For SSAValue/VectorSSAValue: Use ("ssa", ssa.id) where .id is the semantic SSA identifier
    """
    if isinstance(value, Const):
        return ("const", value.value)
    elif isinstance(value, (SSAValue, VectorSSAValue)):
        return ("ssa", value.id)
    else:
        # Fallback for other value types
        return ("id", id(value))

# Scalar ALU ops that have vector equivalents
VECTORIZABLE_ALU_OPS = {
    "+", "-", "*", "//", "%", "^", "&", "|", "<<", ">>", "<", "=="
}

# Commutative operations (operands can be reordered)
COMMUTATIVE_OPS = {"+", "*", "^", "&", "|", "=="}

# Scalar to vector opcode mapping
SCALAR_TO_VECTOR_OP = {
    "+": "v+", "-": "v-", "*": "v*", "//": "v//", "%": "v%",
    "^": "v^", "&": "v&", "|": "v|", "<<": "v<<", ">>": "v>>",
    "<": "v<", "==": "v==",
    "select": "vselect",
    "load": "vload",
    "store": "vstore",
}


@dataclass
class Pack:
    """
    Group of VLEN isomorphic scalar instructions to vectorize.

    Elements are ordered by their lane index (0..VLEN-1).
    All elements must have the same opcode and be in the same basic block.
    """
    elements: list[Op]  # Exactly VLEN instructions
    opcode: str         # Common opcode
    is_memory: bool     # True for load/store packs

    def __post_init__(self):
        assert len(self.elements) == VLEN, f"Pack must have {VLEN} elements"
        assert all(op.opcode == self.opcode for op in self.elements)

    def __hash__(self):
        return hash(tuple(id(op) for op in self.elements))

    def __eq__(self, other):
        if not isinstance(other, Pack):
            return False
        return all(id(a) == id(b) for a, b in zip(self.elements, other.elements))


@dataclass
class SLPContext:
    """Context for SLP vectorization within a basic block."""
    # Data dependency graph for the block
    ddg: BlockDDGs[Op]
    # Maps scalar SSA -> (vector SSA, lane index)
    scalar_to_vector: dict[Variable, tuple[VectorSSAValue, int]] = field(default_factory=dict)
    # Maps cache_key(scalar_value) to its broadcast vector (for uniform scalars)
    # Uses stable cache keys via _make_cache_key() to avoid id() misclassification
    broadcast_cache: dict[tuple, VectorSSAValue] = field(default_factory=dict)
    # All discovered packs
    packs: list[Pack] = field(default_factory=list)
    # Set of ops that are part of a pack (for deduplication)
    packed_ops: set[int] = field(default_factory=set)  # id(Op) -> bool
    # Counter for new vector SSA values
    next_vec_ssa_id: int = 0
    # Counter for new scalar SSA values (for extracts)
    next_ssa_id: int = 0
    # Pending operations to emit (like broadcasts)
    pending_ops: list[Op] = field(default_factory=list)
    # Deferred broadcasts: maps cache_key(scalar_operand) -> broadcast op
    # Used to place broadcasts after their defining instruction
    deferred_broadcasts: dict[tuple, Op] = field(default_factory=dict)
    # Maps cache_key(result) -> op for ops that define values in this block
    # Used to determine if a value is externally defined
    result_to_op: dict[tuple, Op] = field(default_factory=dict)

    def get_node(self, op: Op) -> Optional[DDGNode[Op]]:
        """Get DDG node for an op."""
        op_id = id(op)
        if op_id in self.ddg.inst_map:
            return self.ddg.inst_map[op_id]
        return None

    def get_def_node(self, ssa: SSAValue) -> Optional[DDGNode[Op]]:
        """Get DDG node that defines an SSA value."""
        # def_map keys are SSAValue objects; try direct lookup first
        if ssa in self.ddg.def_map:
            return self.ddg.def_map[ssa]
        # Fall back to searching by SSA id field (for cases with different Python objects)
        for key, node in self.ddg.def_map.items():
            if isinstance(key, SSAValue) and key.id == ssa.id:
                return node
        return None

    def all_nodes(self) -> list[DDGNode[Op]]:
        """Get all DDG nodes."""
        return list(self.ddg.inst_map.values())


class SLPVectorizationPass(Pass):
    """
    Superword Level Parallelism vectorization pass.

    Converts groups of 8 isomorphic scalar operations into vector operations.

    Algorithm:
    1. Build DDG for the block
    2. Find seeds: consecutive store operations (DDG roots)
    3. Extend packs bottom-up along DDG operand edges
    4. Check legality (no internal deps)
    5. Generate vector code
    """

    def __init__(self):
        super().__init__()
        self._seeds_found = 0
        self._packs_created = 0
        self._ops_vectorized = 0
        # Broadcasts to hoist to function entry (for values defined outside processed blocks)
        self._entry_broadcasts: list[Op] = []
        # Cache of already-emitted entry broadcasts by stable cache key (via _make_cache_key())
        self._entry_broadcast_cache: dict[tuple, VectorSSAValue] = {}

    @property
    def name(self) -> str:
        return "slp-vectorization"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        self._seeds_found = 0
        self._packs_created = 0
        self._ops_vectorized = 0
        self._entry_broadcasts = []
        self._entry_broadcast_cache = {}

        if not config.enabled:
            return hir

        # Transform the function body
        new_body = self._transform_statements(hir.body, hir)

        # Insert hoisted broadcasts at function entry (after existing entry ops)
        if self._entry_broadcasts:
            # Find the insertion point: after initial loads/consts, before loops
            insert_idx = 0
            for i, stmt in enumerate(new_body):
                if isinstance(stmt, Op) and stmt.opcode in ("load", "const"):
                    insert_idx = i + 1
                elif isinstance(stmt, (ForLoop, If)):
                    break
                elif isinstance(stmt, Op):
                    # Non-load/const op - insert broadcasts before it
                    break
            # Insert broadcasts at the found position
            new_body = new_body[:insert_idx] + self._entry_broadcasts + new_body[insert_idx:]

        # Record metrics
        if self._metrics:
            self._metrics.custom = {
                "seeds_found": self._seeds_found,
                "packs_created": self._packs_created,
                "ops_vectorized": self._ops_vectorized,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=hir.num_ssa_values,
            num_vec_ssa_values=hir.num_vec_ssa_values + self._ops_vectorized
        )

    def _transform_statements(
        self,
        stmts: list[Statement],
        hir: HIRFunction
    ) -> list[Statement]:
        """Transform a list of statements, vectorizing where possible."""
        result = []

        for stmt in stmts:
            if isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt, hir))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, hir))
            elif isinstance(stmt, Op):
                result.append(stmt)
            else:
                # Halt, Pause
                result.append(stmt)

        # Try to vectorize flat op sequences in this block
        ops_only = [s for s in result if isinstance(s, Op)]
        if len(ops_only) >= VLEN:
            vectorized = self._vectorize_block(ops_only, hir)
            if vectorized is not None:
                # Replace ops with vectorized version
                return vectorized

        return result

    def _transform_for_loop(self, loop: ForLoop, hir: HIRFunction) -> ForLoop:
        """Transform a ForLoop, vectorizing its body."""
        new_body = self._transform_statements(loop.body, hir)

        return ForLoop(
            counter=loop.counter,
            start=loop.start,
            end=loop.end,
            iter_args=loop.iter_args,
            body_params=loop.body_params,
            body=new_body,
            yields=loop.yields,
            results=loop.results,
            pragma_unroll=loop.pragma_unroll
        )

    def _transform_if(self, if_stmt: If, hir: HIRFunction) -> If:
        """Transform an If statement, vectorizing its branches."""
        new_then = self._transform_statements(if_stmt.then_body, hir)
        new_else = self._transform_statements(if_stmt.else_body, hir)

        return If(
            cond=if_stmt.cond,
            then_body=new_then,
            then_yields=if_stmt.then_yields,
            else_body=new_else,
            else_yields=if_stmt.else_yields,
            results=if_stmt.results
        )

    def _vectorize_block(
        self,
        ops: list[Op],
        hir: HIRFunction
    ) -> Optional[list[Statement]]:
        """
        Try to vectorize a flat block of operations using DDG.

        Returns the vectorized statements or None if no vectorization possible.
        """
        if len(ops) < VLEN:
            return None

        # Build DDG for this block
        builder = HIRDDGBuilder()
        ddg = builder.build(ops)

        # Build map from cache_key(result) -> op for ops that define values in this block
        result_to_op: dict[tuple, Op] = {}
        for op in ops:
            if op.result:
                result_to_op[_make_cache_key(op.result)] = op

        # Create SLP context
        ctx = SLPContext(
            ddg=ddg,
            next_vec_ssa_id=hir.num_vec_ssa_values,
            next_ssa_id=hir.num_ssa_values,
            result_to_op=result_to_op
        )

        # Phase 1: Find seeds from DDG store roots
        seeds = self._find_seeds_from_ddg(ops, ctx)
        self._seeds_found += len(seeds)

        if not seeds:
            return None

        # Phase 2: Extend packs along DDG operand edges
        for seed in seeds:
            self._extend_pack_via_ddg(seed, ctx)

        self._packs_created = len(ctx.packs)

        if not ctx.packs:
            return None

        # Phase 3: Check legality
        legal_packs = [p for p in ctx.packs if self._is_legal_pack(p, ctx)]

        if not legal_packs:
            return None

        # Phase 4: Generate vector code
        return self._generate_vector_code(ops, legal_packs, ctx, hir)

    def _find_seeds_from_ddg(self, ops: list[Op], ctx: SLPContext) -> list[Pack]:
        """
        Find seed packs from DDG roots (stores with no users).

        Only uses stores as seeds to ensure we work backwards from data sinks.
        """
        seeds = []

        # Find all store operations (DDG roots - no users)
        stores = [op for op in ops if op.opcode == "store"]

        if len(stores) < VLEN:
            return seeds

        # Group stores by base address pattern and find consecutive groups
        store_seeds = self._find_consecutive_ops(stores, ctx)
        seeds.extend(store_seeds)

        return seeds

    def _find_consecutive_ops(
        self,
        ops: list[Op],
        ctx: SLPContext
    ) -> list[Pack]:
        """Find groups of VLEN operations with consecutive addresses."""
        packs = []

        if len(ops) < VLEN:
            return packs

        # Group by base address pattern
        addr_groups: dict[tuple, list[tuple[Op, int]]] = {}  # base -> [(op, offset)]

        for op in ops:
            addr = op.operands[0]
            base, offset = self._analyze_address(addr, ctx)
            if base is not None:
                if base not in addr_groups:
                    addr_groups[base] = []
                addr_groups[base].append((op, offset))

        # Find complete groups of VLEN consecutive offsets
        for base_key, group in addr_groups.items():
            if len(group) < VLEN:
                continue

            # Sort by offset
            group.sort(key=lambda x: x[1])

            # Find runs of consecutive offsets
            i = 0
            while i <= len(group) - VLEN:
                start_offset = group[i][1]
                is_consecutive = all(
                    group[i + j][1] == start_offset + j
                    for j in range(VLEN)
                )

                if is_consecutive:
                    pack_ops = [group[i + j][0] for j in range(VLEN)]

                    # Check not already packed
                    if not any(id(op) in ctx.packed_ops for op in pack_ops):
                        opcode = pack_ops[0].opcode
                        pack = Pack(elements=pack_ops, opcode=opcode, is_memory=True)
                        packs.append(pack)
                        for op in pack_ops:
                            ctx.packed_ops.add(id(op))
                        ctx.packs.append(pack)

                    i += VLEN
                else:
                    i += 1

        return packs

    def _analyze_address(
        self,
        addr: Value,
        ctx: SLPContext
    ) -> tuple[Optional[tuple], int]:
        """
        Analyze an address to extract base + offset pattern.

        Returns (base_pattern, offset) where base_pattern is a hashable tuple.
        """
        if isinstance(addr, Const):
            return (("const",), addr.value)

        if isinstance(addr, SSAValue):
            # Look up the defining op in DDG
            for node in ctx.all_nodes():
                if node.instruction.result and id(node.instruction.result) == id(addr):
                    def_op = node.instruction

                    if def_op.opcode == "+":
                        left, right = def_op.operands[0], def_op.operands[1]

                        # Check if right is a constant
                        if isinstance(right, Const):
                            left_base, left_offset = self._analyze_address(left, ctx)
                            if left_base is not None:
                                return (left_base, left_offset + right.value)

                        # Check if left is a constant (commutative)
                        if isinstance(left, Const):
                            right_base, right_offset = self._analyze_address(right, ctx)
                            if right_base is not None:
                                return (right_base, right_offset + left.value)

                        # Both non-constant: combine bases
                        left_base, left_offset = self._analyze_address(left, ctx)
                        right_base, right_offset = self._analyze_address(right, ctx)

                        if left_base is not None and right_base is not None:
                            combined_base = (left_base, right_base)
                            return (combined_base, left_offset + right_offset)

                    # Use the op as base
                    return ((id(def_op),), 0)

            # Unknown def - use the SSA value itself as base
            return ((id(addr),), 0)

        return (None, 0)

    def _extend_pack_via_ddg(self, seed: Pack, ctx: SLPContext) -> None:
        """
        Extend pack along DDG operand edges (bottom-up).

        Starting from a seed, examine operands via DDG and form new packs
        if they're isomorphic.
        """
        worklist = [seed]

        while worklist:
            pack = worklist.pop()

            # Get DDG nodes for pack elements
            pack_nodes = [ctx.get_node(op) for op in pack.elements]

            # Skip if any node not found
            if None in pack_nodes:
                continue

            # For each operand position
            num_operands = len(pack.elements[0].operands)

            for operand_idx in range(num_operands):
                # Skip address operand for memory ops
                if pack.is_memory and operand_idx == 0:
                    continue

                # Collect operand definitions via DDG
                operand_ops = []
                for node in pack_nodes:
                    # Get operand nodes at this position
                    if operand_idx < len(node.operand_nodes):
                        dep_node = node.operand_nodes[operand_idx]
                        if dep_node is not None:
                            operand_ops.append(dep_node.instruction)
                        else:
                            operand_ops.append(None)
                    else:
                        operand_ops.append(None)

                # Check if all operands have definitions
                if None in operand_ops:
                    continue

                # Check if they can form a pack
                if self._can_form_pack(operand_ops, ctx):
                    new_pack = self._try_create_pack(operand_ops, ctx)
                    if new_pack and new_pack not in ctx.packs:
                        ctx.packs.append(new_pack)
                        worklist.append(new_pack)

    def _can_form_pack(self, ops: list[Optional[Op]], ctx: SLPContext) -> bool:
        """Check if a list of ops can form a valid pack."""
        if len(ops) != VLEN:
            return False

        if any(op is None for op in ops):
            return False

        # All ops must be distinct
        if len(set(id(op) for op in ops)) != VLEN:
            return False

        # Same opcode
        opcodes = set(op.opcode for op in ops)
        if len(opcodes) != 1:
            return False

        opcode = ops[0].opcode

        # Opcode is vectorizable
        if opcode not in VECTORIZABLE_ALU_OPS and opcode not in ("select", "load"):
            return False

        # Not already packed
        if any(id(op) in ctx.packed_ops for op in ops):
            return False

        # For loads, check consecutive addresses
        if opcode == "load":
            if not self._are_consecutive_loads(ops, ctx):
                return False

        return True

    def _are_consecutive_loads(self, ops: list[Op], ctx: SLPContext) -> bool:
        """Check if load operations have consecutive addresses."""
        base_offsets = []
        for op in ops:
            addr = op.operands[0]
            base, offset = self._analyze_address(addr, ctx)
            if base is None:
                return False
            base_offsets.append((base, offset))

        # Same base
        first_base = base_offsets[0][0]
        if not all(b == first_base for b, _ in base_offsets):
            return False

        # Consecutive offsets
        offsets = sorted(o for _, o in base_offsets)
        for i in range(len(offsets) - 1):
            if offsets[i + 1] - offsets[i] != 1:
                return False

        return True

    def _try_create_pack(self, ops: list[Op], ctx: SLPContext) -> Optional[Pack]:
        """Try to create a pack from the given ops."""
        if not self._can_form_pack(ops, ctx):
            return None

        opcode = ops[0].opcode
        pack = Pack(elements=list(ops), opcode=opcode, is_memory=False)

        for op in ops:
            ctx.packed_ops.add(id(op))

        return pack

    def _is_legal_pack(self, pack: Pack, ctx: SLPContext) -> bool:
        """
        Check if a pack is legal to vectorize.

        No internal dependencies within the pack.
        """
        pack_node_ids = set()
        for op in pack.elements:
            node = ctx.get_node(op)
            if node:
                pack_node_ids.add(id(node))

        # Check for internal dependencies
        for op in pack.elements:
            node = ctx.get_node(op)
            if node:
                for dep in node.operand_nodes:
                    if dep and id(dep) in pack_node_ids:
                        return False

        return True

    def _generate_vector_code(
        self,
        original_ops: list[Op],
        packs: list[Pack],
        ctx: SLPContext,
        hir: HIRFunction
    ) -> list[Statement]:
        """
        Generate vectorized code from the packs.

        Strategy:
        1. Walk through original ops in order
        2. When we see the LAST element of a pack, emit vector op
        3. Skip other elements of vectorized packs
        4. Defer non-pack ops that depend on vectorized values

        Broadcast placement:
        - Constants: hoisted to function entry (always safe, handled by _get_or_create_broadcast)
        - SSA defined in this block: placed immediately after the defining instruction
        - SSA defined outside this block: placed at block start (not function entry, dominance)
        """
        pack_results: dict[int, VectorSSAValue] = {}

        # Check which packs can be fully vectorized
        vectorizable_packs = [p for p in packs if self._can_fully_vectorize(p, ctx, packs)]

        if not vectorizable_packs:
            return original_ops

        # Collect pack element results
        pack_element_results: set[int] = set()
        for pack in vectorizable_packs:
            for elem in pack.elements:
                if elem.result:
                    pack_element_results.add(id(elem.result))

        # Map last element -> pack
        last_element_to_pack: dict[int, Pack] = {}
        pack_elements: set[int] = set()
        for pack in vectorizable_packs:
            last_idx = -1
            last_op = None
            for elem in pack.elements:
                for i, op in enumerate(original_ops):
                    if id(op) == id(elem) and i > last_idx:
                        last_idx = i
                        last_op = elem
            if last_op:
                last_element_to_pack[id(last_op)] = pack
            for op in pack.elements:
                pack_elements.add(id(op))

        # Generate code incrementally, tracking broadcasts for later placement
        result: list[Statement] = []
        emitted_packs: set[int] = set()
        defined_values: set[int] = set()
        deferred_results: set[int] = set()
        deferred_ops: list[Op] = []
        emitted_broadcasts: set[tuple] = set()  # Track which broadcasts have been emitted (by cache key)

        def emit_deferred_ops():
            nonlocal deferred_ops
            changed = True
            while changed:
                changed = False
                still_deferred = []
                for deferred_op in deferred_ops:
                    can_emit = True
                    for operand in deferred_op.operands:
                        if isinstance(operand, SSAValue):
                            op_id = id(operand)
                            if op_id in pack_element_results and op_id not in defined_values:
                                can_emit = False
                                break
                            if op_id in deferred_results and op_id not in defined_values:
                                can_emit = False
                                break
                    if can_emit:
                        result.append(deferred_op)
                        if deferred_op.result:
                            defined_values.add(id(deferred_op.result))
                            deferred_results.discard(id(deferred_op.result))
                        changed = True
                    else:
                        still_deferred.append(deferred_op)
                deferred_ops = still_deferred

        def emit_broadcasts_for_value(value: Value):
            """Emit any broadcasts that use this value as their scalar operand."""
            cache_key = _make_cache_key(value)
            if cache_key in ctx.deferred_broadcasts and cache_key not in emitted_broadcasts:
                broadcast_op = ctx.deferred_broadcasts[cache_key]
                result.append(broadcast_op)
                emitted_broadcasts.add(cache_key)

        for op in original_ops:
            op_id = id(op)

            if op_id in pack_elements:
                if op_id in last_element_to_pack and hash(last_element_to_pack[op_id]) not in emitted_packs:
                    pack = last_element_to_pack[op_id]
                    ctx.pending_ops.clear()

                    vec_op = self._generate_pack_code(pack, ctx, pack_results, hir)
                    if vec_op:
                        # Emit pending ops (non-broadcast ops like vextract, vadd, vgather)
                        while ctx.pending_ops:
                            result.append(ctx.pending_ops.pop(0))

                        result.append(vec_op)
                        self._ops_vectorized += VLEN
                        emitted_packs.add(hash(pack))

                        # Emit vextracts for scalar uses
                        if pack.opcode in ("load",) + tuple(VECTORIZABLE_ALU_OPS):
                            for lane, pack_op in enumerate(pack.elements):
                                if pack_op.result:
                                    has_scalar_use = self._has_scalar_use(pack_op, pack_elements, ctx)
                                    if has_scalar_use:
                                        vec_ssa = ctx.scalar_to_vector.get(pack_op.result)
                                        if vec_ssa:
                                            vec_val, _ = vec_ssa
                                            extract_op = Op(
                                                opcode="vextract",
                                                result=pack_op.result,
                                                operands=[vec_val, Const(lane)],
                                                engine="alu"
                                            )
                                            result.append(extract_op)
                                            defined_values.add(id(pack_op.result))

                        emit_deferred_ops()
                    else:
                        ctx.pending_ops.clear()
                        emitted_packs.add(hash(pack))
                        # Emit scalar ops
                        elem_positions = []
                        for elem in pack.elements:
                            for i, orig_op in enumerate(original_ops):
                                if id(orig_op) == id(elem):
                                    elem_positions.append((i, elem))
                                    break
                        elem_positions.sort(key=lambda x: x[0])
                        for _, elem in elem_positions:
                            result.append(elem)
                            if elem.result:
                                defined_values.add(id(elem.result))
                                # Emit broadcasts for this newly defined value
                                emit_broadcasts_for_value(elem.result)
            else:
                # Non-pack op
                needs_defer = False
                for operand in op.operands:
                    if isinstance(operand, SSAValue):
                        operand_id = id(operand)
                        if operand_id in pack_element_results and operand_id not in defined_values:
                            needs_defer = True
                            break
                        if operand_id in deferred_results and operand_id not in defined_values:
                            needs_defer = True
                            break

                if needs_defer:
                    deferred_ops.append(op)
                    if op.result:
                        deferred_results.add(id(op.result))
                else:
                    result.append(op)
                    if op.result:
                        defined_values.add(id(op.result))
                        # Emit broadcasts for this newly defined value
                        emit_broadcasts_for_value(op.result)

        # Emit remaining deferred ops
        while deferred_ops:
            prev_len = len(deferred_ops)
            emit_deferred_ops()
            if len(deferred_ops) == prev_len:
                for deferred_op in deferred_ops:
                    result.append(deferred_op)
                break

        # Emit any remaining deferred broadcasts that weren't emitted
        # (these are for values defined in this block but processed before
        # the broadcast was created during pack code generation)
        #
        # Pre-build an index for O(1) lookup (Issue #5: O(n*m) late insertion performance)
        result_index: dict[int, int] = {id(stmt): i for i, stmt in enumerate(result) if isinstance(stmt, Op)}

        # Track insertions to adjust positions
        insertions: list[tuple[int, Op]] = []

        for cache_key, broadcast_op in ctx.deferred_broadcasts.items():
            if cache_key not in emitted_broadcasts:
                defining_op = ctx.result_to_op.get(cache_key)
                if defining_op and id(defining_op) in result_index:
                    # Value defined in this block - insert after defining op
                    insert_pos = result_index[id(defining_op)] + 1
                    insertions.append((insert_pos, broadcast_op))
                    emitted_broadcasts.add(cache_key)
                # Note: External values (not in result_to_op) are hoisted to function
                # entry in _get_or_create_broadcast, so they won't be in deferred_broadcasts

        # Sort insertions by position in descending order to maintain correct indices
        insertions.sort(key=lambda x: x[0], reverse=True)
        for insert_pos, broadcast_op in insertions:
            result.insert(insert_pos, broadcast_op)

        return result if result else original_ops

    def _has_scalar_use(self, op: Op, pack_elements: set[int], ctx: SLPContext) -> bool:
        """Check if an op's result has any scalar (non-vectorized) users."""
        if not op.result:
            return False

        # Check all nodes that use this result
        op_node = ctx.get_node(op)
        if op_node:
            for user_node in op_node.user_nodes:
                if id(user_node.instruction) not in pack_elements:
                    return True
        return False

    def _can_fully_vectorize(
        self,
        pack: Pack,
        ctx: SLPContext,
        all_packs: list[Pack]
    ) -> bool:
        """Check if a pack can be fully vectorized."""
        if pack.opcode == "store":
            operands = [pack.elements[lane].operands[1] for lane in range(VLEN)]

            # All same (uniform)
            if all(self._values_equal(operands[i], operands[0]) for i in range(VLEN)):
                return True

            # All from same vector
            if all(isinstance(op, SSAValue) for op in operands):
                first_vec = ctx.scalar_to_vector.get(operands[0])
                if first_vec:
                    vec_ssa, _ = first_vec
                    if all(
                        ctx.scalar_to_vector.get(op) == (vec_ssa, lane)
                        for lane, op in enumerate(operands)
                    ):
                        return True

            # Operands from other packs
            for op in operands:
                if isinstance(op, SSAValue):
                    op_node = None
                    for node in ctx.all_nodes():
                        if node.instruction.result and id(node.instruction.result) == id(op):
                            op_node = node
                            break
                    if op_node:
                        in_pack = any(op_node.instruction in p.elements for p in all_packs if p != pack)
                        if not in_pack:
                            return False

            return True

        return True

    def _generate_pack_code(
        self,
        pack: Pack,
        ctx: SLPContext,
        pack_results: dict[int, VectorSSAValue],
        hir: HIRFunction
    ) -> Optional[Op]:
        """Generate a vector op for a pack."""
        opcode = pack.opcode

        if opcode == "store":
            return self._generate_vstore(pack, ctx, pack_results, hir)
        elif opcode == "load":
            return self._generate_vload(pack, ctx, pack_results, hir)
        elif opcode in VECTORIZABLE_ALU_OPS:
            return self._generate_valu_op(pack, ctx, pack_results, hir)
        elif opcode == "select":
            return self._generate_vselect(pack, ctx, pack_results, hir)

        return None

    def _generate_vload(
        self,
        pack: Pack,
        ctx: SLPContext,
        pack_results: dict[int, VectorSSAValue],
        hir: HIRFunction
    ) -> Op:
        """Generate a vload for a load pack."""
        base_addr = pack.elements[0].operands[0]

        # Handle vectorized addresses
        if isinstance(base_addr, SSAValue):
            vec_mapping = ctx.scalar_to_vector.get(base_addr)
            if vec_mapping:
                vec_addr, lane = vec_mapping
                extracted_addr = SSAValue(id=ctx.next_ssa_id, name="vload_base_addr")
                ctx.next_ssa_id += 1
                extract_op = Op(
                    opcode="vextract",
                    result=extracted_addr,
                    operands=[vec_addr, Const(lane)],
                    engine="alu"
                )
                ctx.pending_ops.append(extract_op)
                base_addr = extracted_addr

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vload_result")
        ctx.next_vec_ssa_id += 1

        for lane, op in enumerate(pack.elements):
            if op.result:
                ctx.scalar_to_vector[op.result] = (vec_result, lane)

        pack_results[hash(pack)] = vec_result

        return Op(
            opcode="vload",
            result=vec_result,
            operands=[base_addr],
            engine="load"
        )

    def _generate_vstore(
        self,
        pack: Pack,
        ctx: SLPContext,
        pack_results: dict[int, VectorSSAValue],
        hir: HIRFunction
    ) -> Optional[Op]:
        """Generate a vstore for a store pack."""
        base_addr = pack.elements[0].operands[0]

        # Handle vectorized addresses
        if isinstance(base_addr, SSAValue):
            vec_mapping = ctx.scalar_to_vector.get(base_addr)
            if vec_mapping:
                vec_addr, lane = vec_mapping
                extracted_addr = SSAValue(id=ctx.next_ssa_id, name="vstore_base_addr")
                ctx.next_ssa_id += 1
                extract_op = Op(
                    opcode="vextract",
                    result=extracted_addr,
                    operands=[vec_addr, Const(lane)],
                    engine="alu"
                )
                ctx.pending_ops.append(extract_op)
                base_addr = extracted_addr

        vec_value = self._get_vector_operand(pack, 1, ctx, pack_results, hir)
        if vec_value is None:
            return None

        return Op(
            opcode="vstore",
            result=None,
            operands=[base_addr, vec_value],
            engine="store"
        )

    def _generate_valu_op(
        self,
        pack: Pack,
        ctx: SLPContext,
        pack_results: dict[int, VectorSSAValue],
        hir: HIRFunction
    ) -> Optional[Op]:
        """Generate a vector ALU op for an ALU pack."""
        vec_opcode = SCALAR_TO_VECTOR_OP[pack.opcode]

        vec_operands = []
        for i in range(len(pack.elements[0].operands)):
            vec_op = self._get_vector_operand(pack, i, ctx, pack_results, hir)
            if vec_op is None:
                return None
            vec_operands.append(vec_op)

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name=f"v{pack.opcode}_result")
        ctx.next_vec_ssa_id += 1

        for lane, op in enumerate(pack.elements):
            if op.result:
                ctx.scalar_to_vector[op.result] = (vec_result, lane)

        pack_results[hash(pack)] = vec_result

        return Op(
            opcode=vec_opcode,
            result=vec_result,
            operands=vec_operands,
            engine="valu"
        )

    def _generate_vselect(
        self,
        pack: Pack,
        ctx: SLPContext,
        pack_results: dict[int, VectorSSAValue],
        hir: HIRFunction
    ) -> Optional[Op]:
        """Generate a vselect for a select pack."""
        vec_cond = self._get_vector_operand(pack, 0, ctx, pack_results, hir)
        vec_true = self._get_vector_operand(pack, 1, ctx, pack_results, hir)
        vec_false = self._get_vector_operand(pack, 2, ctx, pack_results, hir)

        if vec_cond is None or vec_true is None or vec_false is None:
            return None

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vselect_result")
        ctx.next_vec_ssa_id += 1

        for lane, op in enumerate(pack.elements):
            if op.result:
                ctx.scalar_to_vector[op.result] = (vec_result, lane)

        pack_results[hash(pack)] = vec_result

        return Op(
            opcode="vselect",
            result=vec_result,
            operands=[vec_cond, vec_true, vec_false],
            engine="flow"
        )

    def _get_vector_operand(
        self,
        pack: Pack,
        operand_idx: int,
        ctx: SLPContext,
        pack_results: dict[int, VectorSSAValue],
        hir: HIRFunction
    ) -> Optional[VectorSSAValue]:
        """
        Get or create a vector operand for a pack.

        Cases:
        1. All same scalar -> vbroadcast
        2. From another pack -> use that pack's vector result
        3. Different SSA values -> build vector with vinsert
        """
        operands = [pack.elements[lane].operands[operand_idx] for lane in range(VLEN)]

        # Check if from same vector pack
        if all(isinstance(op, (SSAValue, VectorSSAValue)) for op in operands):
            first_vec = ctx.scalar_to_vector.get(operands[0])
            if first_vec:
                vec_ssa, _ = first_vec
                all_from_same = all(
                    ctx.scalar_to_vector.get(op) == (vec_ssa, lane)
                    for lane, op in enumerate(operands)
                )
                if all_from_same:
                    return vec_ssa

        # All same (uniform)
        if all(self._values_equal(operands[i], operands[0]) for i in range(VLEN)):
            return self._get_or_create_broadcast(operands[0], ctx)

        # All same constants
        if all(isinstance(op, Const) for op in operands):
            if all(op.value == operands[0].value for op in operands):
                return self._get_or_create_broadcast(operands[0], ctx)

        # Build vector from different SSA values
        if all(isinstance(op, SSAValue) for op in operands):
            return self._build_vector_from_scalars(operands, ctx)

        return None

    def _build_vector_from_scalars(
        self,
        scalars: list[SSAValue],
        ctx: SLPContext
    ) -> VectorSSAValue:
        """Build a vector from VLEN different scalar SSA values."""
        assert len(scalars) == VLEN

        cache_key = tuple(id(s) for s in scalars)
        if cache_key in ctx.broadcast_cache:
            return ctx.broadcast_cache[cache_key]

        # Try vgather pattern (loads with indexed addresses)
        gather_result = self._try_generate_vgather(scalars, ctx)
        if gather_result is not None:
            ctx.broadcast_cache[cache_key] = gather_result
            return gather_result

        # Try to detect consecutive offset pattern: [base, base+1, base+2, ..., base+7]
        consecutive = self._try_vectorize_consecutive_offsets(scalars, ctx)
        if consecutive is not None:
            ctx.broadcast_cache[cache_key] = consecutive
            return consecutive

        # Fall back to vinsert chain
        return self._build_vector_via_vinsert(scalars, ctx, cache_key)

    def _try_vectorize_consecutive_offsets(
        self,
        scalars: list[SSAValue],
        ctx: SLPContext
    ) -> Optional[VectorSSAValue]:
        """
        Detect pattern: [base, base+1, base+2, ..., base+7]

        This pattern appears when loop counters are unrolled:
        - Lane 0: base value (or base+0 simplified to base)
        - Lane 1-7: base + N (where N is the lane index)

        Generate: v+(vbroadcast(base), const_vec[0,1,2,3,4,5,6,7])
        """
        # Get defining ops for each scalar
        def_ops = []
        for scalar in scalars:
            def_node = ctx.get_def_node(scalar)
            def_ops.append(def_node.instruction if def_node else None)

        # Find which lanes have + ops with constant second operand
        base_value = None
        offsets = [None] * VLEN

        for lane, (scalar, def_op) in enumerate(zip(scalars, def_ops)):
            if def_op is not None and def_op.opcode == "+":
                # Check if second operand is a constant
                if len(def_op.operands) == 2 and isinstance(def_op.operands[1], Const):
                    offset = def_op.operands[1].value
                    potential_base = def_op.operands[0]

                    if base_value is None:
                        base_value = potential_base
                    elif not self._values_equal(potential_base, base_value):
                        # Different bases, check if potential_base IS base_value
                        # (for case where lane 0 uses base directly)
                        continue

                    offsets[lane] = offset
            elif def_op is None:
                # No defining op in this block - this might be the base value
                # (defined outside or is an input)
                if base_value is None:
                    base_value = scalar
                    offsets[lane] = 0  # Treat as base + 0
                elif self._values_equal(scalar, base_value):
                    offsets[lane] = 0

        # If we haven't found a base yet, check lane 0
        if base_value is None and scalars[0] is not None:
            # Check if other lanes use scalars[0] as base in their + ops
            for lane in range(1, VLEN):
                def_op = def_ops[lane]
                if def_op is not None and def_op.opcode == "+" and len(def_op.operands) == 2:
                    if isinstance(def_op.operands[1], Const):
                        potential_base = def_op.operands[0]
                        if self._values_equal(potential_base, scalars[0]):
                            base_value = scalars[0]
                            offsets[0] = 0
                            offsets[lane] = def_op.operands[1].value

        # Re-scan with known base to fill in missing offsets
        if base_value is not None:
            for lane, (scalar, def_op) in enumerate(zip(scalars, def_ops)):
                if offsets[lane] is None:
                    if self._values_equal(scalar, base_value):
                        offsets[lane] = 0
                    elif def_op is not None and def_op.opcode == "+":
                        if len(def_op.operands) == 2 and isinstance(def_op.operands[1], Const):
                            if self._values_equal(def_op.operands[0], base_value):
                                offsets[lane] = def_op.operands[1].value

        # Check we have all offsets
        if None in offsets:
            return None

        # Verify consecutive pattern [start, start+1, start+2, ..., start+7]
        start_offset = offsets[0]
        expected = list(range(start_offset, start_offset + VLEN))
        if offsets != expected:
            return None

        # Mark the defining + ops as consumed (they'll be replaced by vector code)
        for def_op in def_ops:
            if def_op is not None and def_op.opcode == "+":
                ctx.packed_ops.add(id(def_op))

        # Generate vectorized code: v+(vbroadcast(base), const_vec[start, start+1, ...])
        vec_base = self._get_or_create_broadcast(base_value, ctx)

        # Build constant vector [start, start+1, ..., start+7]
        const_vec = self._build_const_vector([Const(start_offset + i) for i in range(VLEN)], ctx)

        # Generate v+ operation
        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vconsec_result")
        ctx.next_vec_ssa_id += 1

        vadd_op = Op(
            opcode="v+",
            result=vec_result,
            operands=[vec_base, const_vec],
            engine="valu"
        )
        ctx.pending_ops.append(vadd_op)

        # Map scalar results to vector lanes
        for lane, scalar in enumerate(scalars):
            ctx.scalar_to_vector[scalar] = (vec_result, lane)

        return vec_result

    def _vectorize_operands(
        self,
        operands: list[Value],
        ctx: SLPContext
    ) -> Optional[VectorSSAValue]:
        """
        Vectorize a list of operands (one per lane).

        Handles:
        - Uniform values (all same) -> vbroadcast
        - From same vector -> use that vector
        - All constants -> build constant vector
        - All SSA values -> recursively vectorize
        """
        # Check if from same vector pack
        if all(isinstance(op, SSAValue) for op in operands):
            first_vec = ctx.scalar_to_vector.get(operands[0])
            if first_vec:
                vec_ssa, _ = first_vec
                all_from_same = all(
                    ctx.scalar_to_vector.get(op) == (vec_ssa, lane)
                    for lane, op in enumerate(operands)
                )
                if all_from_same:
                    return vec_ssa

        # All same value -> broadcast
        if all(self._values_equal(operands[i], operands[0]) for i in range(VLEN)):
            return self._get_or_create_broadcast(operands[0], ctx)

        # All constants -> build constant vector
        if all(isinstance(op, Const) for op in operands):
            return self._build_const_vector(operands, ctx)

        # All SSA values -> try recursive vectorization
        if all(isinstance(op, SSAValue) for op in operands):
            return self._build_vector_from_scalars(operands, ctx)

        return None

    def _build_const_vector(
        self,
        consts: list[Const],
        ctx: SLPContext
    ) -> VectorConst:
        """Build a constant vector from constant values.

        Returns a VectorConst which is a compile-time constant that can be
        used directly as an operand without generating vbroadcast/vinsert ops.
        """
        # If all same, just broadcast (more efficient for uniform values)
        if all(c.value == consts[0].value for c in consts):
            return self._get_or_create_broadcast(consts[0], ctx)

        # Create VectorConst directly - no ops needed!
        values = tuple(c.value for c in consts)
        return VectorConst(values=values)

    def _try_generate_vgather(
        self,
        scalars: list[SSAValue],
        ctx: SLPContext
    ) -> Optional[VectorSSAValue]:
        """
        Try to generate vgather for gather load pattern.

        Pattern: scalars come from loads with addresses base + offset[i]
        where offsets come from lanes of the same vector.
        """
        # Check if all scalars come from loads
        load_ops = []
        for scalar in scalars:
            load_op = None
            for node in ctx.all_nodes():
                if node.instruction.result and id(node.instruction.result) == id(scalar):
                    if node.instruction.opcode == "load":
                        load_op = node.instruction
                    break
            if load_op is None:
                return None
            load_ops.append(load_op)

        # Analyze addresses for gather pattern
        base_value = None
        offset_vector = None
        offset_lanes = []

        for i, load_op in enumerate(load_ops):
            addr = load_op.operands[0]

            if not isinstance(addr, SSAValue):
                return None

            # Find defining op
            addr_op = None
            for node in ctx.all_nodes():
                if node.instruction.result and id(node.instruction.result) == id(addr):
                    addr_op = node.instruction
                    break

            if addr_op is None or addr_op.opcode != "+":
                return None

            left, right = addr_op.operands[0], addr_op.operands[1]

            # Try both orderings
            found_pattern = False
            for base_candidate, offset_candidate in [(left, right), (right, left)]:
                if not isinstance(offset_candidate, SSAValue):
                    continue

                vec_mapping = ctx.scalar_to_vector.get(offset_candidate)
                if vec_mapping is None:
                    continue

                vec_src, lane = vec_mapping

                if i == 0:
                    base_value = base_candidate
                    offset_vector = vec_src
                    offset_lanes.append(lane)
                    found_pattern = True
                    break
                else:
                    if self._values_equal(base_candidate, base_value) and vec_src == offset_vector:
                        offset_lanes.append(lane)
                        found_pattern = True
                        break

            if not found_pattern:
                return None

        # Verify lanes are 0..VLEN-1
        if offset_lanes != list(range(VLEN)):
            return None

        # Mark the scalar load ops and their address ops as consumed
        for load_op in load_ops:
            ctx.packed_ops.add(id(load_op))
            # Also mark the address calculation op
            addr = load_op.operands[0]
            if isinstance(addr, SSAValue):
                for node in ctx.all_nodes():
                    if node.instruction.result and id(node.instruction.result) == id(addr):
                        if node.instruction.opcode == "+":
                            ctx.packed_ops.add(id(node.instruction))
                        break

        # Generate vgather
        vec_base = self._get_or_create_broadcast(base_value, ctx)

        vec_addrs = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vgather_addrs")
        ctx.next_vec_ssa_id += 1

        vadd_op = Op(
            opcode="v+",
            result=vec_addrs,
            operands=[vec_base, offset_vector],
            engine="valu"
        )
        ctx.pending_ops.append(vadd_op)

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vgather_result")
        ctx.next_vec_ssa_id += 1

        vgather_op = Op(
            opcode="vgather",
            result=vec_result,
            operands=[vec_addrs],
            engine="load"
        )
        ctx.pending_ops.append(vgather_op)

        for lane, scalar in enumerate(scalars):
            ctx.scalar_to_vector[scalar] = (vec_result, lane)

        return vec_result

    def _build_vector_via_vinsert(
        self,
        scalars: list[SSAValue],
        ctx: SLPContext,
        cache_key: tuple
    ) -> VectorSSAValue:
        """Build a vector from scalars using vbroadcast + vinsert chain.

        Note: The broadcast added to pending_ops here is intentionally NOT routed
        through the deferred_broadcasts mechanism. This is because:
        1. It's part of a vinsert chain, not a uniform broadcast
        2. It needs to be emitted immediately before the vinsert sequence
        3. The scalar operand is already available at this point in code generation
        """
        # Handle scalars from vectorized packs
        actual_scalars = []
        for i, scalar in enumerate(scalars):
            vec_mapping = ctx.scalar_to_vector.get(scalar)
            if vec_mapping:
                vec_val, lane = vec_mapping
                extracted = SSAValue(id=ctx.next_ssa_id, name=f"vinsert_extract_{i}")
                ctx.next_ssa_id += 1
                extract_op = Op(
                    opcode="vextract",
                    result=extracted,
                    operands=[vec_val, Const(lane)],
                    engine="alu"
                )
                ctx.pending_ops.append(extract_op)
                actual_scalars.append(extracted)
            else:
                actual_scalars.append(scalar)

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vinsert_result")
        ctx.next_vec_ssa_id += 1

        first_broadcast = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vinsert_base")
        ctx.next_vec_ssa_id += 1

        broadcast_op = Op(
            opcode="vbroadcast",
            result=first_broadcast,
            operands=[actual_scalars[0]],
            engine="valu"
        )
        ctx.pending_ops.append(broadcast_op)

        current_vec = first_broadcast
        for lane in range(1, VLEN):
            if lane == VLEN - 1:
                insert_result = vec_result
            else:
                insert_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name=f"vinsert_tmp_{lane}")
                ctx.next_vec_ssa_id += 1

            insert_op = Op(
                opcode="vinsert",
                result=insert_result,
                operands=[current_vec, actual_scalars[lane], Const(lane)],
                engine="valu"
            )
            ctx.pending_ops.append(insert_op)
            current_vec = insert_result

        ctx.broadcast_cache[cache_key] = vec_result
        return vec_result

    def _get_or_create_broadcast(
        self,
        scalar_val: Value,
        ctx: SLPContext
    ) -> VectorSSAValue:
        """Get or create a vbroadcast for a scalar value.

        The broadcast op placement depends on where the scalar is defined:
        - If defined in the current block: placed after the defining instruction
        - If defined outside the block (external): hoisted to function entry
          (This includes Const values and SSA values defined at function level)

        Uses stable cache keys via _make_cache_key() to ensure semantically identical
        values (e.g., two Const(5) objects) share the same broadcast.
        """
        cache_key = _make_cache_key(scalar_val)

        # Check instance-level entry broadcast cache first (for externally-defined values)
        if cache_key in self._entry_broadcast_cache:
            return self._entry_broadcast_cache[cache_key]

        # Check block-level cache
        if cache_key in ctx.broadcast_cache:
            return ctx.broadcast_cache[cache_key]

        # Determine if value is external (not defined in current block)
        # External values include: Const, function-level SSA values, loop parameters
        is_external = cache_key not in ctx.result_to_op

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vbroadcast_result")
        ctx.next_vec_ssa_id += 1

        broadcast_op = Op(
            opcode="vbroadcast",
            result=vec_result,
            operands=[scalar_val],
            engine="valu"
        )

        if is_external:
            # Hoist to function entry
            self._entry_broadcasts.append(broadcast_op)
            self._entry_broadcast_cache[cache_key] = vec_result
        else:
            # Store in deferred_broadcasts for later placement after the defining op
            ctx.deferred_broadcasts[cache_key] = broadcast_op

        ctx.broadcast_cache[cache_key] = vec_result

        return vec_result

    def _values_equal(self, a: Value, b: Value) -> bool:
        """Check if two values are equal."""
        if isinstance(a, SSAValue) and isinstance(b, SSAValue):
            return id(a) == id(b)
        if isinstance(a, Const) and isinstance(b, Const):
            return a.value == b.value
        return False
