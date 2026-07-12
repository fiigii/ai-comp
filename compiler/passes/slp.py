"""
SLP (Superword Level Parallelism) Vectorization Pass

Converts groups of 8 isomorphic scalar operations into single vector operations.
Operates on HIR after loop unrolling and CSE.

Uses DDG (Data Dependency Graph) for:
- Finding seeds (store roots)
- Extending packs (following operand edges)
- Checking legality (no internal dependencies)
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Optional

from ..hir import (
    SSAValue, VectorSSAValue, Const, VectorConst, Value, Op, ForLoop, If,
    Statement, HIRFunction
)
from ..alias_analysis import AliasAnalysis, AliasResult
from ..pass_manager import Pass, PassConfig
from ..ddg import HIRDDGBuilder, BlockDDGs, DDGNode
from ..use_def import UseDefContext

# Vector length for this architecture
VLEN = 8


def _make_cache_key(value: Value) -> tuple:
    """Create a stable cache key for a value.

    This is needed because using id() for cache keys can cause issues:
    - Two Const(5) objects have different id()s but represent the same value
    - This leads to cache misses for semantically identical values

    Scalar and vector SSA values use separate tags because their numeric ID
    spaces are independent.
    """
    if isinstance(value, Const):
        return ("const", value.value)
    if isinstance(value, SSAValue):
        return ("scalar_ssa", value.id)
    if isinstance(value, VectorSSAValue):
        return ("vector_ssa", value.id)
    if isinstance(value, VectorConst):
        return ("vector_const", value.values)
    return ("id", id(value))

# Scalar ALU ops that have vector equivalents
VECTORIZABLE_ALU_OPS = {
    "+", "-", "*", "//", "%", "^", "&", "|", "<<", ">>", "<", "=="
}

# Scalar to vector opcode mapping
SCALAR_TO_VECTOR_OP = {
    "+": "v+", "-": "v-", "*": "v*", "//": "v//", "%": "v%",
    "^": "v^", "&": "v&", "|": "v|", "<<": "v<<", ">>": "v>>",
    "<": "v<", "==": "v==",
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
    is_gather: bool = False  # True for non-consecutive load packs (vgather)

    def __post_init__(self):
        assert len(self.elements) == VLEN, f"Pack must have {VLEN} elements"
        assert all(op.opcode == self.opcode for op in self.elements)


@dataclass
class SLPContext:
    """Context for SLP vectorization within a basic block."""
    # Data dependency graph for the block
    ddg: BlockDDGs[Op]
    # Maps semantic scalar SSA keys -> (vector SSA, lane index).
    scalar_to_vector: dict[tuple, tuple[VectorSSAValue, int]] = field(default_factory=dict)
    # Maps a materialization key to a vector already emitted in this op run.
    materialized_vector_cache: dict[tuple, VectorSSAValue] = field(default_factory=dict)
    # All discovered packs
    packs: list[Pack] = field(default_factory=list)
    # Set of ops that are part of a pack (for deduplication)
    packed_ops: set[int] = field(default_factory=set)  # id(Op) -> bool
    # Counter for new vector SSA values
    next_vec_ssa_id: int = 0
    # Counter for new scalar SSA values (for extracts)
    next_ssa_id: int = 0
    # Operations generated immediately before the current vector pack.
    pending_ops: list[Op] = field(default_factory=list)
    # Non-constant broadcasts are placed after their final scalar definition
    # once scalar ownership decisions have been applied.
    ssa_broadcasts: dict[tuple, Op] = field(default_factory=dict)
    # Cache for address base/offset analysis (keyed by _make_cache_key(value))
    addr_analysis_cache: dict[tuple, tuple[Optional[tuple], int]] = field(default_factory=dict)
    # Program-order position of each op in the current block (id(op) -> idx)
    op_pos: dict[int, int] = field(default_factory=dict)
    # Memory ops of the block in program order, with their positions
    # (parallel lists, positions ascending; used for pack span checks)
    mem_op_positions: list[int] = field(default_factory=list)
    mem_ops_in_order: list[Op] = field(default_factory=list)

    def get_node(self, op: Op) -> Optional[DDGNode[Op]]:
        """Get DDG node for an op."""
        op_id = id(op)
        if op_id in self.ddg.inst_map:
            return self.ddg.inst_map[op_id]
        return None

    def get_vector_mapping(
        self, value: Value
    ) -> Optional[tuple[VectorSSAValue, int]]:
        return self.scalar_to_vector.get(_make_cache_key(value))

    def set_vector_mapping(
        self, value: SSAValue, vector: VectorSSAValue, lane: int
    ) -> None:
        self.scalar_to_vector[_make_cache_key(value)] = (vector, lane)

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
        self._packs_emitted = 0
        self._packs_rejected_materialization = 0
        self._scalar_packs_replaced = 0
        self._scalar_packs_kept_for_early_use = 0
        self._extracts_emitted = 0
        self._packs_pruned_for_pressure = 0
        self._dual_representation_lanes_before_pruning = 0
        self._ops_vectorized = 0
        self._alias: Optional[AliasAnalysis] = None
        self._use_def: Optional[UseDefContext] = None
        # Global SSA id cursors for new values created during this pass.
        # These must be monotonically increasing across all vectorized blocks
        # in the function to avoid SSA id collisions.
        self._next_vec_ssa_id = 0
        self._next_ssa_id = 0
        # Constant broadcasts are safe to hoist to function entry.
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
        self._packs_emitted = 0
        self._packs_rejected_materialization = 0
        self._scalar_packs_replaced = 0
        self._scalar_packs_kept_for_early_use = 0
        self._extracts_emitted = 0
        self._packs_pruned_for_pressure = 0
        self._dual_representation_lanes_before_pruning = 0
        self._ops_vectorized = 0
        self._next_vec_ssa_id = hir.num_vec_ssa_values
        self._next_ssa_id = hir.num_ssa_values
        self._entry_broadcasts = []
        self._entry_broadcast_cache = {}
        self._use_def = UseDefContext(hir)
        self._alias = AliasAnalysis(
            self._use_def,
            restrict_ptr=config.options.get("restrict_ptr", False),
        )
        self._vectorize_memory = config.options.get("vectorize_memory", True)
        self._gather = config.options.get("gather", True)
        self._vectorize_alu = config.options.get("vectorize_alu", True)
        # A varying VectorConst costs eight entry loads and eight long-lived
        # scratch words. Keep it opt-in until a target cost model can prove
        # that the load/register cost is amortized across enough packs.
        self._vectorize_varying_constants = config.options.get(
            "vectorize_varying_constants", False)
        self._dual_representation_prune_threshold = int(config.options.get(
            "dual_representation_prune_threshold", 512))
        if self._dual_representation_prune_threshold < 0:
            raise ValueError(
                "dual_representation_prune_threshold must be >= 0")

        if not config.enabled:
            return hir

        # Transform the function body
        new_body = self._transform_statements(hir.body)

        # Insert hoisted constant broadcasts after existing entry ops.
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
                "packs_emitted": self._packs_emitted,
                "packs_rejected_materialization": self._packs_rejected_materialization,
                "scalar_packs_replaced": self._scalar_packs_replaced,
                "scalar_packs_kept_for_early_use": self._scalar_packs_kept_for_early_use,
                "extracts_emitted": self._extracts_emitted,
                "packs_pruned_for_pressure": self._packs_pruned_for_pressure,
                "dual_representation_lanes_before_pruning": (
                    self._dual_representation_lanes_before_pruning),
                "ops_vectorized": self._ops_vectorized,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=max(hir.num_ssa_values, self._next_ssa_id),
            num_vec_ssa_values=max(hir.num_vec_ssa_values, self._next_vec_ssa_id),
        )

    def _transform_statements(
        self,
        stmts: list[Statement],
    ) -> list[Statement]:
        """Transform a list of statements, vectorizing straight-line Op segments.

        Important: Preserve non-Op statements (Pause/Halt/ForLoop/If). These are
        control-flow / synchronization markers and must not be dropped or moved.
        """
        transformed: list[Statement] = []

        for stmt in stmts:
            if isinstance(stmt, ForLoop):
                transformed.append(self._transform_for_loop(stmt))
            elif isinstance(stmt, If):
                transformed.append(self._transform_if(stmt))
            else:
                # Op, Halt, Pause
                transformed.append(stmt)

        # Vectorize contiguous Op runs only (basic-block like regions).
        out: list[Statement] = []
        i = 0
        while i < len(transformed):
            stmt = transformed[i]
            if not isinstance(stmt, Op):
                out.append(stmt)
                i += 1
                continue

            j = i
            while j < len(transformed) and isinstance(transformed[j], Op):
                j += 1

            op_run = [s for s in transformed[i:j] if isinstance(s, Op)]
            if len(op_run) >= VLEN:
                vectorized = self._vectorize_block(op_run)
                if vectorized is not None:
                    out.extend(vectorized)
                else:
                    out.extend(op_run)
            else:
                out.extend(op_run)

            i = j

        return out

    def _transform_for_loop(self, loop: ForLoop) -> ForLoop:
        """Transform a ForLoop, vectorizing its body."""
        new_body = self._transform_statements(loop.body)

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

    def _transform_if(self, if_stmt: If) -> If:
        """Transform an If statement, vectorizing its branches."""
        new_then = self._transform_statements(if_stmt.then_body)
        new_else = self._transform_statements(if_stmt.else_body)

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
    ) -> Optional[list[Statement]]:
        """
        Try to vectorize a flat block of operations using DDG.

        Returns the vectorized statements or None if no vectorization possible.
        """
        if len(ops) < VLEN:
            return None

        # Build DDG for this block. We only need the use/def graph; building
        # per-root DAGs is extremely expensive on large unrolled blocks.
        builder = HIRDDGBuilder()
        ddg = builder.build(ops, build_dags=False)

        # Create SLP context
        ctx = SLPContext(
            ddg=ddg,
            next_vec_ssa_id=self._next_vec_ssa_id,
            next_ssa_id=self._next_ssa_id,
            op_pos={id(op): i for i, op in enumerate(ops)}
        )
        for i, op in enumerate(ops):
            if op.opcode in ("load", "vload", "store", "vstore", "vgather"):
                ctx.mem_op_positions.append(i)
                ctx.mem_ops_in_order.append(op)

        # Phase 1: Find seeds from DDG store roots
        seeds = self._find_seeds_from_ddg(ops, ctx)
        self._seeds_found += len(seeds)

        if not seeds:
            return None

        # Phase 2: Extend packs along DDG operand edges
        for seed in seeds:
            self._extend_pack_via_ddg(seed, ctx)

        self._packs_created += len(ctx.packs)

        if not ctx.packs:
            return None

        # Pack registration has already established legality.
        legal_packs = list(ctx.packs)

        # Phase 3: Filter registered packs by config knobs.
        if not self._vectorize_memory:
            legal_packs = [p for p in legal_packs if p.opcode not in ("store", "load")]

        if not legal_packs:
            return None

        # Phase 4: Select profitable packs and generate vector code.
        vectorized = self._generate_vector_code(ops, legal_packs, ctx)

        # Advance global SSA cursors to avoid collisions across blocks.
        self._next_vec_ssa_id = max(self._next_vec_ssa_id, ctx.next_vec_ssa_id)
        self._next_ssa_id = max(self._next_ssa_id, ctx.next_ssa_id)

        return vectorized

    def _find_seeds_from_ddg(self, ops: list[Op], ctx: SLPContext) -> list[Pack]:
        """
        Find seed packs from DDG roots (stores with no users) and from
        gather-load groups.
        """
        seeds = []

        # Find all store operations (DDG roots - no users)
        stores = [op for op in ops if op.opcode == "store"]

        if len(stores) >= VLEN:
            # Group stores by base address pattern and find consecutive groups
            store_seeds = self._find_consecutive_store_packs(stores, ctx)
            seeds.extend(store_seeds)

        # Gather-load seeds: groups of VLEN loads from the same base with
        # varying (non-constant) offsets. These cannot be reached bottom-up
        # from store seeds when the index chain feeding the addresses has no
        # other vectorized consumer, so seed them directly. The address chain
        # then vectorizes via normal pack extension and codegen emits vgather.
        if self._gather and self._vectorize_memory:
            seeds.extend(self._find_gather_seeds(ops, ctx))

        return seeds

    def _find_gather_seeds(self, ops: list[Op], ctx: SLPContext) -> list[Pack]:
        """Find gather-load seed packs.

        A gather load has address `+(base, offset)` where both operands are
        SSA values. The base is disambiguated by frequency: the operand that
        repeats across many loads is the base, the per-lane varying operand
        is the offset. Loads are grouped VLEN at a time in program order,
        which matches the lane grouping of batch-order unrolled code.

        Legality: the fused vgather is emitted at the last pack element's
        position, which moves earlier lane loads past everything in between.
        A pack is rejected when
        - its element span contains a store the alias analysis cannot prove
          disjoint from the loads (a may-aliasing store must not be
          crossed), or
        - any element's address depends (transitively) on another element's
          loaded value (e.g. pointer chasing): the fused gather reads all
          lanes simultaneously, which would break that dependency.
        """
        def addr_depends_on_pack(pack_ops: list[Op]) -> bool:
            """True if any element's address chain reaches another element's
            loaded value. Walks def chains, pruned at ops defined before the
            pack's first element (those cannot depend on pack results)."""
            results = {op.result for op in pack_ops if op.result is not None}
            first_pos = min(ctx.op_pos[id(op)] for op in pack_ops)
            seen: set = set()
            stack = [op.operands[0] for op in pack_ops]
            while stack:
                v = stack.pop()
                if not isinstance(v, SSAValue) or v in seen:
                    continue
                seen.add(v)
                if v in results:
                    return True
                def_node = ctx.get_def_node(v)
                if def_node is None:
                    continue
                d = def_node.instruction
                pos = ctx.op_pos.get(id(d))
                if pos is None or pos < first_pos:
                    continue
                stack.extend(o for o in d.operands if isinstance(o, SSAValue))
            return False

        candidates: list[tuple[Op, Op]] = []
        operand_freq: dict[tuple, int] = {}
        for op in ops:
            if op.opcode != "load" or op.result is None:
                continue
            if id(op) in ctx.packed_ops:
                continue
            addr = op.operands[0]
            if not isinstance(addr, SSAValue):
                continue
            addr_node = ctx.get_def_node(addr)
            if addr_node is None or addr_node.instruction.opcode != "+":
                continue
            addr_op = addr_node.instruction
            if len(addr_op.operands) != 2:
                continue
            a, b = addr_op.operands
            if not (isinstance(a, SSAValue) and isinstance(b, SSAValue)):
                # Const offsets are consecutive-load / vload territory
                continue
            candidates.append((op, addr_op))
            for operand in (a, b):
                key = _make_cache_key(operand)
                operand_freq[key] = operand_freq.get(key, 0) + 1

        if len(candidates) < VLEN:
            return []

        # Group loads by their (frequency-chosen) base, in program order.
        # When neither operand repeats across loads (e.g. strength-reduced
        # address chains where every operand is per-lane unique), fall back
        # to grouping consecutive gather-shaped loads in program order,
        # which matches the lane order of batch-unrolled code.
        groups: dict[tuple, list[Op]] = {}
        group_order: list[tuple] = []
        for op, addr_op in candidates:
            a, b = addr_op.operands
            fa = operand_freq[_make_cache_key(a)]
            fb = operand_freq[_make_cache_key(b)]
            if fa != fb and operand_freq[_make_cache_key(a if fa > fb else b)] >= VLEN:
                key = _make_cache_key(a if fa > fb else b)
            else:
                key = ("__seq__",)
            if key not in groups:
                groups[key] = []
                group_order.append(key)
            groups[key].append(op)

        packs: list[Pack] = []
        for group_key in group_order:
            group = groups[group_key]
            start = 0
            while start + VLEN <= len(group):
                pack_ops = group[start:start + VLEN]
                if any(id(op) in ctx.packed_ops for op in pack_ops):
                    start += 1
                    continue
                if addr_depends_on_pack(pack_ops):
                    start += 1
                    continue
                pack = Pack(elements=pack_ops, opcode="load", is_gather=True)
                if self._register_pack(pack, ctx):
                    packs.append(pack)
                    start += VLEN
                else:
                    start += 1
        return packs

    def _find_consecutive_store_packs(
        self,
        ops: list[Op],
        ctx: SLPContext
    ) -> list[Pack]:
        """Find program-order store groups with consecutive addresses.

        Pairing the k-th occurrence of each offset can mix different unrolled
        iterations after an earlier DSE removes an uneven subset of stores.
        Instead, scan each normalized-base stream in program order and only
        register an actual adjacent run of increasing offsets.
        """
        packs = []

        if len(ops) < VLEN:
            return packs

        addr_groups: dict[tuple, list[tuple[int, Op]]] = {}

        for op in ops:
            addr = op.operands[0]
            base, offset = self._analyze_address(addr, ctx)
            if base is None:
                continue
            addr_groups.setdefault(base, []).append((offset, op))

        for stream in addr_groups.values():
            i = 0
            while i + VLEN <= len(stream):
                window = stream[i:i + VLEN]
                start_offset = window[0][0]
                if [offset for offset, _ in window] != list(
                        range(start_offset, start_offset + VLEN)):
                    i += 1
                    continue

                pack_ops = [op for _, op in window]
                pack = Pack(elements=pack_ops, opcode=pack_ops[0].opcode)
                if self._register_pack(pack, ctx):
                    packs.append(pack)
                    i += VLEN
                else:
                    # A rejected window must not hide a legal overlapping run.
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
        cache_key = _make_cache_key(addr)
        cached = ctx.addr_analysis_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._alias is None:
            result = (None, 0)
            ctx.addr_analysis_cache[cache_key] = result
            return result

        addr_key = self._alias.normalize(addr)
        if addr_key is None:
            result = (None, 0)
        else:
            result = (addr_key.base, addr_key.offset)

        ctx.addr_analysis_cache[cache_key] = result
        return result

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
                # Skip address operand for memory ops (except gathers, whose
                # per-lane address chain is exactly what we want to vectorize)
                if (pack.opcode in ("load", "store")
                        and operand_idx == 0 and not pack.is_gather):
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

                new_pack = self._try_create_pack(operand_ops, ctx)
                if new_pack and self._register_pack(new_pack, ctx):
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

        # Gate load packing on vectorize_memory knob
        if opcode == "load" and not self._vectorize_memory:
            return False

        # Gate ALU packing on vectorize_alu knob (select is unaffected)
        if opcode in VECTORIZABLE_ALU_OPS and not self._vectorize_alu:
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

        # Lane order matters: vload uses lane 0's address as its base and
        # returns increasing addresses in lanes 0..VLEN-1. Sorting here would
        # incorrectly accept a reversed or permuted producer pack.
        offsets = [offset for _, offset in base_offsets]
        return offsets == list(range(offsets[0], offsets[0] + VLEN))

    def _try_create_pack(self, ops: list[Op], ctx: SLPContext) -> Optional[Pack]:
        """Try to create a pack from the given ops."""
        if not self._can_form_pack(ops, ctx):
            return None

        if self._pack_elements_interdependent(ops, ctx):
            return None

        opcode = ops[0].opcode
        return Pack(elements=list(ops), opcode=opcode)

    def _register_pack(self, pack: Pack, ctx: SLPContext) -> bool:
        """Register a fully legal, non-overlapping pack.

        Keeping validation before the packed_ops update prevents a rejected
        candidate from hiding a later legal grouping of the same operations.
        """
        if any(id(op) in ctx.packed_ops for op in pack.elements):
            return False
        if not self._is_legal_pack(pack, ctx):
            return False
        if not self._can_materialize_pack(pack):
            self._packs_rejected_materialization += 1
            return False
        ctx.packs.append(pack)
        ctx.packed_ops.update(id(op) for op in pack.elements)
        return True

    def _pack_elements_interdependent(self, ops: list[Op], ctx: SLPContext) -> bool:
        """True if any element depends (transitively) on another element's
        result. A vector op executes all lanes simultaneously, so such a
        pack is illegal (e.g. chained recurrences x2 = f(x1) grouped by
        isomorphism across loop rounds).

        The def-chain walk prunes at ops positioned before the earliest
        pack element: they cannot depend on any element's result.
        """
        results = {op.result for op in ops if op.result is not None}
        positions = [ctx.op_pos.get(id(op)) for op in ops]
        if any(p is None for p in positions):
            return True  # unknown positions: be conservative
        min_pos = min(positions)
        seen: set = set()
        stack = [o for op in ops for o in op.operands if isinstance(o, SSAValue)]
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            if v in results:
                return True
            def_node = ctx.get_def_node(v)
            if def_node is None:
                continue
            d = def_node.instruction
            pos = ctx.op_pos.get(id(d))
            if pos is None or pos < min_pos:
                continue
            stack.extend(o for o in d.operands if isinstance(o, SSAValue))
        return False

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

        # Memory packs move earlier elements down to the last element's
        # position, so validate their full memory span before registration.
        if pack.opcode in ("load", "store"):
            if not self._mem_pack_span_is_legal(pack.elements, ctx):
                return False

        return True

    def _mem_pack_span_is_legal(self, pack_ops: list[Op], ctx: SLPContext) -> bool:
        """Check that fusing memory ops at the last element's position is legal.

        Codegen emits one vector memory op at the LAST element's position, so
        every earlier element is moved down across the ops in between.
        Crossing an intervening memory op is illegal when it may touch a
        moved element's address:
        - a store pack must not cross may-aliasing loads or stores,
        - a load pack must not cross may-aliasing stores (loads commute).
        """
        assert self._alias is not None
        positions = [ctx.op_pos.get(id(op)) for op in pack_ops]
        if any(p is None for p in positions):
            return False
        lo, hi = min(positions), max(positions)
        if hi - lo < 2:
            return True

        member_ids = {id(op) for op in pack_ops}
        elem_widths = []
        elem_keys = []
        for op in pack_ops:
            key = self._alias.normalize(op.operands[0])
            if key is None:
                return False
            elem_keys.append(key)
            elem_widths.append(VLEN if op.opcode in ("vload", "vstore") else 1)
        is_store_pack = pack_ops[0].opcode in ("store", "vstore")
        moved = sorted(zip(positions, elem_keys, elem_widths))

        start = bisect_right(ctx.mem_op_positions, lo)
        end = bisect_left(ctx.mem_op_positions, hi)
        for idx in range(start, end):
            q = ctx.mem_op_positions[idx]
            m = ctx.mem_ops_in_order[idx]
            if id(m) in member_ids:
                continue
            m_is_store = m.opcode in ("store", "vstore")
            if not (m_is_store or is_store_pack):
                continue
            m_key = self._alias.normalize(m.operands[0])
            if m_key is None:
                # Unknown address (e.g. a vector-addressed access) inside
                # the span may touch anything.
                return False
            m_width = VLEN if m.opcode in ("vload", "vstore") else 1
            # Only elements positioned before q are moved across m.
            for p, key, width in moved:
                if p >= q:
                    break
                if self._alias.alias_keys(key, width, m_key,
                                          m_width) != AliasResult.NO_ALIAS:
                    return False
        return True

    def _generate_vector_code(
        self,
        original_ops: list[Op],
        packs: list[Pack],
        ctx: SLPContext,
    ) -> list[Statement]:
        """Generate committed vector packs at their last scalar element.

        Store packs replace their scalar stores. Pure/load packs transfer
        ownership to the vector result when every earlier scalar use is also
        covered lane-wise by a later vector pack; otherwise their scalar
        definitions remain in program order. Required non-vector uses receive
        extracts after the vector definition. A following DCE is useful as
        cleanup but is no longer responsible for removing whole scalar chains.

        A side-effect-free preflight rejects packs whose operand columns
        cannot be materialized. Code generation is therefore commit-only:
        once it starts, it must succeed and may safely update caches/mappings.
        """
        op_index = {id(op): i for i, op in enumerate(original_ops)}
        packs, replaceable_pack_ids, scalar_extracts = (
            self._select_packs_for_emission(packs, op_index)
        )
        if not packs:
            return original_ops

        last_element_to_pack: dict[int, Pack] = {}
        for pack in packs:
            last_element = max(
                pack.elements,
                key=lambda elem: op_index.get(id(elem), -1),
            )
            if id(last_element) in op_index:
                if id(last_element) in last_element_to_pack:
                    raise RuntimeError("overlapping SLP packs share an anchor")
                last_element_to_pack[id(last_element)] = pack

        result: list[Statement] = []
        suppressed_store_ids: set[int] = set()
        suppressed_scalar_ids: set[int] = set()

        for op in original_ops:
            result.append(op)

            pack = last_element_to_pack.get(id(op))
            if pack is None:
                continue

            ctx.pending_ops.clear()
            vec_op = self._generate_pack_code(pack, ctx)

            result.extend(ctx.pending_ops)
            ctx.pending_ops.clear()
            result.append(vec_op)
            self._packs_emitted += 1
            self._ops_vectorized += VLEN

            if pack.opcode == "store":
                suppressed_store_ids.update(id(elem) for elem in pack.elements)
            elif id(pack) in replaceable_pack_ids:
                suppressed_scalar_ids.update(id(elem) for elem in pack.elements)
                self._scalar_packs_replaced += 1
                assert self._use_def is not None
                for lane, elem in enumerate(pack.elements):
                    if elem.result is None or elem.result not in scalar_extracts:
                        continue
                    result.append(Op(
                        opcode="vextract",
                        result=elem.result,
                        operands=[vec_op.result, Const(lane)],
                        engine="alu",
                    ))
                    self._extracts_emitted += 1
            else:
                self._scalar_packs_kept_for_early_use += 1

        suppressed_ids = suppressed_store_ids | suppressed_scalar_ids
        if suppressed_ids:
            result = [
                stmt for stmt in result
                if id(stmt) not in suppressed_ids
            ]

        # Give the scheduler the full def-to-consumer interval. A replaced
        # scalar pack may move a definition to a generated vextract, so find
        # definitions in the final result rather than in the original block.
        def_index = {
            _make_cache_key(stmt.result): index
            for index, stmt in enumerate(result)
            if stmt.result is not None
        }
        prefix_ops: list[Op] = []
        broadcasts_after: dict[int, list[Op]] = {}
        for scalar_key, broadcast in ctx.ssa_broadcasts.items():
            index = def_index.get(scalar_key)
            if index is None:
                # Region inputs (for example loop counters/body parameters)
                # are available at the start of this flat op run.
                prefix_ops.append(broadcast)
            else:
                broadcasts_after.setdefault(index, []).append(broadcast)

        if broadcasts_after:
            placed: list[Statement] = []
            for index, stmt in enumerate(result):
                placed.append(stmt)
                placed.extend(broadcasts_after.get(index, ()))
            result = placed
        return [*prefix_ops, *result]

    def _select_packs_for_emission(
        self,
        packs: list[Pack],
        op_index: dict[int, int],
    ) -> tuple[list[Pack], set[int], set[SSAValue]]:
        """Apply a pressure bailout to large partial-vectorization regions.

        A pack that cannot own its scalar definitions creates a second live
        representation. Small cases are left alone because the extra vector
        work can still be profitable. Once the configured lane budget is
        exceeded, retain only closed vector components: remove those dual
        packs and every consumer that would otherwise rebuild a removed pack
        with a vinsert chain.
        """
        replaceable, extracts = self._plan_scalar_pack_replacements(
            packs, op_index)
        dual_lanes = sum(
            VLEN
            for pack in packs
            if pack.opcode != "store" and id(pack) not in replaceable
        )
        self._dual_representation_lanes_before_pruning += dual_lanes
        if dual_lanes <= self._dual_representation_prune_threshold:
            return packs, replaceable, extracts

        original_packs = list(packs)
        result_owner = {
            elem.result: pack
            for pack in original_packs
            if pack.opcode != "store"
            for elem in pack.elements
            if elem.result is not None
        }
        selected = list(original_packs)

        while True:
            replaceable, _ = self._plan_scalar_pack_replacements(
                selected, op_index)
            selected_ids = {id(pack) for pack in selected}
            remove_ids = {
                id(pack)
                for pack in selected
                if pack.opcode != "store" and id(pack) not in replaceable
            }

            for pack in selected:
                operand_indices = self._vector_operand_indices(pack) or ()
                for operand_idx in operand_indices:
                    column = [
                        elem.operands[operand_idx]
                        for elem in pack.elements
                    ]
                    owner = result_owner.get(column[0])
                    if owner is None or id(owner) in selected_ids:
                        continue
                    if all(
                        self._values_equal(
                            value, owner.elements[lane].result)
                        for lane, value in enumerate(column)
                    ):
                        remove_ids.add(id(pack))
                        break

            if not remove_ids:
                break
            selected = [
                pack for pack in selected if id(pack) not in remove_ids
            ]

        self._packs_pruned_for_pressure += len(original_packs) - len(selected)
        replaceable, extracts = self._plan_scalar_pack_replacements(
            selected, op_index)
        return selected, replaceable, extracts

    def _plan_scalar_pack_replacements(
        self,
        packs: list[Pack],
        op_index: dict[int, int],
    ) -> tuple[set[int], set[SSAValue]]:
        """Find packs whose vector result may own the scalar definitions.

        A use at or after the producer anchor can be preserved by an extract.
        An earlier use is also safe when it is a lane-wise operand of another
        pack whose scalar elements will themselves be removed. Compute the
        greatest fixed point of those mutually vector-owned packs. Uniform or
        partially matched early consumers keep the producer scalar pack alive.
        """
        assert self._use_def is not None
        op_to_pack = {
            id(elem): pack
            for pack in packs
            for elem in pack.elements
        }
        anchors = {
            id(pack): max(op_index[id(elem)] for elem in pack.elements)
            for pack in packs
        }
        replaceable = {id(pack) for pack in packs if pack.opcode != "store"}

        def use_is_vector_owned(
            producer: Pack,
            use_statement: Op,
            operand_idx: int,
        ) -> bool:
            consumer = op_to_pack.get(id(use_statement))
            if consumer is None:
                return False
            if (consumer.opcode != "store"
                    and id(consumer) not in replaceable):
                return False
            if anchors[id(consumer)] <= anchors[id(producer)]:
                return False
            if operand_idx < 0:
                return False
            producer_results = [elem.result for elem in producer.elements]
            consumer_column = [
                elem.operands[operand_idx]
                for elem in consumer.elements
                if operand_idx < len(elem.operands)
            ]
            return (
                len(consumer_column) == VLEN
                and all(
                    self._values_equal(value, producer_results[lane])
                    for lane, value in enumerate(consumer_column)
                )
            )

        changed = True
        while changed:
            changed = False
            for pack in packs:
                pack_id = id(pack)
                if pack_id not in replaceable:
                    continue
                anchor_pos = anchors[pack_id]
                keep_scalars = False
                for elem in pack.elements:
                    if elem.result is None:
                        continue
                    for use in self._use_def.get_uses(elem.result):
                        use_pos = op_index.get(id(use.statement))
                        if use_pos is None or use_pos >= anchor_pos:
                            continue
                        if (not isinstance(use.statement, Op)
                                or not use_is_vector_owned(
                                    pack, use.statement, use.operand_index)):
                            keep_scalars = True
                            break
                    if keep_scalars:
                        break
                if keep_scalars:
                    replaceable.remove(pack_id)
                    changed = True

        extracts: set[SSAValue] = set()
        for pack in packs:
            if id(pack) not in replaceable:
                continue
            for elem in pack.elements:
                if elem.result is None:
                    continue
                if any(
                    not isinstance(use.statement, Op)
                    or not use_is_vector_owned(
                        pack, use.statement, use.operand_index)
                    for use in self._use_def.get_uses(elem.result)
                ):
                    extracts.add(elem.result)

        return replaceable, extracts

    def _can_materialize_pack(self, pack: Pack) -> bool:
        """Return whether every vector operand can be built without effects."""
        operand_indices = self._vector_operand_indices(pack)
        return operand_indices is not None and all(
            self._can_materialize_operand(pack, operand_idx)
            for operand_idx in operand_indices
        )

    def _vector_operand_indices(self, pack: Pack) -> Optional[tuple[int, ...]]:
        """Operand columns materialized as vectors by this pack."""
        if pack.opcode == "load":
            return (0,) if pack.is_gather else ()
        if pack.opcode == "store":
            return (1,)
        if pack.opcode in VECTORIZABLE_ALU_OPS:
            return tuple(range(len(pack.elements[0].operands)))
        if pack.opcode == "select":
            return (0, 1, 2)
        return None

    def _can_materialize_operand(self, pack: Pack, operand_idx: int) -> bool:
        """Keep this predicate and emission synchronized through one classifier."""
        kind = self._classify_operand_column(pack, operand_idx)
        return kind in ("uniform_scalar", "scalar_vector") or (
            kind == "const_vector" and self._allows_const_vector(pack))

    def _allows_const_vector(self, pack: Pack) -> bool:
        """Whether a varying constant vector is profitable for this pack."""
        # Scalar stores already need all lane constants, so vstore saves store
        # slots without adding constant materialization. ALU packs remain
        # opt-in because they trade abundant ALU slots for scarce load slots.
        return pack.opcode == "store" or self._vectorize_varying_constants

    def _classify_operand_column(self, pack: Pack, operand_idx: int) -> Optional[str]:
        """Classify a lane column without mutating materialization state."""
        operands = [element.operands[operand_idx] for element in pack.elements]
        if all(self._values_equal(value, operands[0]) for value in operands):
            return "uniform_scalar"
        if all(isinstance(value, Const) for value in operands):
            return "const_vector"
        if all(isinstance(value, SSAValue) for value in operands):
            return "scalar_vector"
        return None

    def _generate_pack_code(
        self,
        pack: Pack,
        ctx: SLPContext,
    ) -> Op:
        """Generate a vector op for a pack."""
        opcode = pack.opcode

        if opcode == "store":
            return self._generate_vstore(pack, ctx)
        elif opcode == "load":
            if pack.is_gather:
                return self._generate_vgather_pack(pack, ctx)
            return self._generate_vload(pack, ctx)
        elif opcode in VECTORIZABLE_ALU_OPS:
            return self._generate_valu_op(pack, ctx)
        elif opcode == "select":
            return self._generate_vselect(pack, ctx)

        raise RuntimeError(f"unsupported registered SLP pack: {opcode}")

    def _generate_vload(
        self,
        pack: Pack,
        ctx: SLPContext,
    ) -> Op:
        """Generate a vload for a load pack."""
        base_addr = pack.elements[0].operands[0]

        # Scalar ownership may remove the original address definition. Recover
        # lane 0 from its vector owner when this pack uses it as a scalar base.
        if isinstance(base_addr, SSAValue):
            vec_mapping = ctx.get_vector_mapping(base_addr)
            if vec_mapping:
                vec_addr, lane = vec_mapping
                extracted_addr = SSAValue(
                    id=ctx.next_ssa_id, name="vload_base_addr")
                ctx.next_ssa_id += 1
                ctx.pending_ops.append(Op(
                    opcode="vextract",
                    result=extracted_addr,
                    operands=[vec_addr, Const(lane)],
                    engine="alu",
                ))
                base_addr = extracted_addr

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vload_result")
        ctx.next_vec_ssa_id += 1

        for lane, op in enumerate(pack.elements):
            if op.result:
                ctx.set_vector_mapping(op.result, vec_result, lane)

        return Op(
            opcode="vload",
            result=vec_result,
            operands=[base_addr],
            engine="load"
        )

    def _generate_vgather_pack(
        self,
        pack: Pack,
        ctx: SLPContext,
    ) -> Op:
        """Generate a vgather for a gather-load pack.

        The per-lane addresses resolve to a vector (usually the result of a
        vectorized address-add pack; vinsert fallback otherwise), and the
        gather lowers to VLEN load_offset slots.
        """
        vec_addrs = self._get_vector_operand(pack, 0, ctx)

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vgather_result")
        ctx.next_vec_ssa_id += 1

        for lane, op in enumerate(pack.elements):
            if op.result:
                ctx.set_vector_mapping(op.result, vec_result, lane)

        return Op(
            opcode="vgather",
            result=vec_result,
            operands=[vec_addrs],
            engine="load"
        )

    def _generate_vstore(
        self,
        pack: Pack,
        ctx: SLPContext,
    ) -> Op:
        """Generate a vstore for a store pack."""
        base_addr = pack.elements[0].operands[0]

        # See _generate_vload: the scalar base may now be vector-owned.
        if isinstance(base_addr, SSAValue):
            vec_mapping = ctx.get_vector_mapping(base_addr)
            if vec_mapping:
                vec_addr, lane = vec_mapping
                extracted_addr = SSAValue(
                    id=ctx.next_ssa_id, name="vstore_base_addr")
                ctx.next_ssa_id += 1
                ctx.pending_ops.append(Op(
                    opcode="vextract",
                    result=extracted_addr,
                    operands=[vec_addr, Const(lane)],
                    engine="alu",
                ))
                base_addr = extracted_addr

        vec_value = self._get_vector_operand(pack, 1, ctx)

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
    ) -> Op:
        """Generate a vector ALU op for an ALU pack."""
        vec_opcode = SCALAR_TO_VECTOR_OP[pack.opcode]

        vec_operands = []
        for i in range(len(pack.elements[0].operands)):
            vec_operands.append(self._get_vector_operand(pack, i, ctx))

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name=f"v{pack.opcode}_result")
        ctx.next_vec_ssa_id += 1

        for lane, op in enumerate(pack.elements):
            if op.result:
                ctx.set_vector_mapping(op.result, vec_result, lane)

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
    ) -> Op:
        """Generate a vselect for a select pack."""
        vec_cond = self._get_vector_operand(pack, 0, ctx)
        vec_true = self._get_vector_operand(pack, 1, ctx)
        vec_false = self._get_vector_operand(pack, 2, ctx)

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vselect_result")
        ctx.next_vec_ssa_id += 1

        for lane, op in enumerate(pack.elements):
            if op.result:
                ctx.set_vector_mapping(op.result, vec_result, lane)

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
    ) -> Value:
        """
        Get or create a vector operand for a pack.

        Cases:
        1. All same scalar -> vbroadcast
        2. From another pack -> use that pack's vector result
        3. Different SSA values -> build vector with vinsert
        """
        operands = [pack.elements[lane].operands[operand_idx] for lane in range(VLEN)]
        kind = self._classify_operand_column(pack, operand_idx)
        if kind is None or (kind == "const_vector"
                            and not self._allows_const_vector(pack)):
            raise RuntimeError("registered SLP operand cannot be materialized")

        # Check if from same vector pack
        if kind == "scalar_vector":
            first_vec = ctx.get_vector_mapping(operands[0])
            if first_vec:
                vec_ssa, _ = first_vec
                all_from_same = all(
                    ctx.get_vector_mapping(op) == (vec_ssa, lane)
                    for lane, op in enumerate(operands)
                )
                if all_from_same:
                    return vec_ssa

        # All same (uniform)
        if kind == "uniform_scalar":
            return self._get_or_create_broadcast(operands[0], ctx)

        if kind == "const_vector":
            return self._build_const_vector(operands)

        if kind == "scalar_vector":
            return self._build_vector_from_scalars(operands, ctx)

        raise RuntimeError(f"unknown SLP operand materialization kind: {kind}")

    def _build_vector_from_scalars(
        self,
        scalars: list[SSAValue],
        ctx: SLPContext
    ) -> VectorSSAValue:
        """Build a vector from VLEN different scalar SSA values."""
        assert len(scalars) == VLEN

        cache_key = tuple(_make_cache_key(s) for s in scalars)
        if cache_key in ctx.materialized_vector_cache:
            return ctx.materialized_vector_cache[cache_key]

        # Try to detect consecutive offset pattern: [base, base+1, base+2, ..., base+7]
        consecutive = self._try_vectorize_consecutive_offsets(scalars, ctx)
        if consecutive is not None:
            ctx.materialized_vector_cache[cache_key] = consecutive
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

        # Generate vectorized code: v+(vbroadcast(base), const_vec[start, start+1, ...])
        vec_base = self._get_or_create_broadcast(base_value, ctx)

        # Build constant vector [start, start+1, ..., start+7]
        const_vec = self._build_const_vector(
            [Const(start_offset + i) for i in range(VLEN)])

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
            ctx.set_vector_mapping(scalar, vec_result, lane)

        return vec_result

    def _build_const_vector(
        self,
        consts: list[Const],
    ) -> VectorConst:
        """Build a constant vector from constant values.

        Returns a VectorConst which is a compile-time constant that can be
        used directly as an operand without generating vbroadcast/vinsert ops.
        """
        values = tuple(c.value for c in consts)
        return VectorConst(values=values)

    def _build_vector_via_vinsert(
        self,
        scalars: list[SSAValue],
        ctx: SLPContext,
        cache_key: tuple
    ) -> VectorSSAValue:
        """Build a vector from scalars using vbroadcast + vinsert chain.

        The seed broadcast and inserts are emitted together immediately before
        the consuming pack.
        """
        # Handle scalars from vectorized packs
        actual_scalars = []
        for i, scalar in enumerate(scalars):
            vec_mapping = ctx.get_vector_mapping(scalar)
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

        ctx.materialized_vector_cache[cache_key] = vec_result
        return vec_result

    def _get_or_create_broadcast(
        self,
        scalar_val: Value,
        ctx: SLPContext
    ) -> VectorSSAValue:
        """Get or create a vbroadcast for a scalar value.

        Constants are cached at function entry. SSA broadcasts are recorded by
        semantic scalar key and later placed immediately after the final
        scalar definition, or at the current op-run entry for region inputs.

        Uses stable cache keys via _make_cache_key() to ensure semantically identical
        values (e.g., two Const(5) objects) share the same broadcast.
        """
        cache_key = _make_cache_key(scalar_val)

        # Only constants are safe to cache and hoist across every region.
        if (isinstance(scalar_val, Const)
                and cache_key in self._entry_broadcast_cache):
            return self._entry_broadcast_cache[cache_key]

        # Check block-level cache
        if cache_key in ctx.materialized_vector_cache:
            return ctx.materialized_vector_cache[cache_key]

        vec_result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="vbroadcast_result")
        ctx.next_vec_ssa_id += 1

        broadcast_op = Op(
            opcode="vbroadcast",
            result=vec_result,
            operands=[scalar_val],
            engine="valu"
        )

        if isinstance(scalar_val, Const):
            self._entry_broadcasts.append(broadcast_op)
            self._entry_broadcast_cache[cache_key] = vec_result
        else:
            assert isinstance(scalar_val, SSAValue)
            ctx.ssa_broadcasts[cache_key] = broadcast_op

        ctx.materialized_vector_cache[cache_key] = vec_result

        return vec_result

    def _values_equal(self, a: Value, b: Value) -> bool:
        """Check if two values are equal."""
        if isinstance(a, SSAValue) and isinstance(b, SSAValue):
            return a.id == b.id
        if isinstance(a, Const) and isinstance(b, Const):
            return a.value == b.value
        return False
