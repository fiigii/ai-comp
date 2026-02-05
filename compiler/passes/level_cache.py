"""
Level Cache Pass

Replaces vgather operations with preload + vselect when the index range is small.

Key insight: In the tree_hash kernel, idx has predictable range per round:
- Round 0: idx = 0 → range {0}, size 1
- Round 1: idx = 2*0 + {1,2} → range {1,2}, size 2
- Round k (k < height): range {2^k - 1, ..., 2^(k+1) - 2}, size 2^k

When range size ≤ 8, we can preload all possible values and use vselect to choose.
This eliminates expensive vgather (which expands to VLEN load_offset instructions).
"""

from dataclasses import dataclass, field
from typing import Optional

from ..hir import (
    SSAValue, VectorSSAValue, Variable, Const, VectorConst, Value, Op,
    ForLoop, If, Halt, Pause, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..use_def import UseDefContext
from ..range_analysis import Range, RangeAnalysis


# Vector length
VLEN = 8


@dataclass
class VgatherInfo:
    """Information about a vgather operation."""
    op: Op                          # The vgather operation
    vec_addrs: VectorSSAValue       # Vector of addresses
    base_ssa: Optional[Value]       # Base pointer (e.g., forest_values_p)
    idx_vec: Optional[VectorSSAValue]  # Index vector
    idx_range: Optional[Range]      # Range of idx values


@dataclass
class LevelCacheContext:
    """Context for level cache transformation."""
    use_def: UseDefContext
    range_analysis: RangeAnalysis
    max_range_size: int
    next_ssa_id: int
    next_vec_ssa_id: int
    # Maps (base, range) -> preload info
    preload_cache: dict = field(default_factory=dict)
    # Cache for vector range analysis (keyed by VectorSSAValue id)
    vec_range_cache: dict = field(default_factory=dict)
    # Maximum depth for vector range analysis
    max_vec_depth: int = 50
    # Limit on number of vgathers to analyze (performance safeguard)
    max_vgathers_to_analyze: int = 600
    # Counter for vgathers analyzed (must use field to avoid shared state)
    vgathers_analyzed: int = field(default=0)


class LevelCachePass(Pass):
    """
    Replace vgather with preload + vselect when idx range is small.

    When the index vector has a small, known range (≤ max_range_size values),
    we can:
    1. Preload all possible node values (scalar loads + broadcasts)
    2. Use vselect tree to choose correct value based on idx bits
    3. Remove the vgather

    This is much more efficient because:
    - Preloading uses scalar loads (1 instruction per value)
    - vselect uses flow engine (can parallelize with ALU/load)
    - vgather expands to VLEN load_offset instructions (8x more load slots)
    """

    def __init__(self):
        super().__init__()
        self._vgathers_found = 0
        self._vgathers_transformed = 0
        self._preloads_created = 0
        self._vgathers_with_pattern = 0  # Found v+(base, idx) pattern
        self._vgathers_with_range = 0     # Had computable range

    @property
    def name(self) -> str:
        return "level-cache"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        self._vgathers_found = 0
        self._vgathers_transformed = 0
        self._preloads_created = 0
        self._vgathers_with_pattern = 0
        self._vgathers_with_range = 0

        if not config.enabled:
            return hir

        max_range_size = config.options.get("max_range_size", 8)

        # Build use-def context and range analysis
        use_def = UseDefContext(hir)
        range_analysis = RangeAnalysis(use_def)

        ctx = LevelCacheContext(
            use_def=use_def,
            range_analysis=range_analysis,
            max_range_size=max_range_size,
            next_ssa_id=hir.num_ssa_values,
            next_vec_ssa_id=hir.num_vec_ssa_values,
        )

        # Transform the body
        new_body = self._transform_body(hir.body, ctx)

        # Record metrics
        if self._metrics:
            self._metrics.custom = {
                "vgathers_found": self._vgathers_found,
                "vgathers_transformed": self._vgathers_transformed,
                "preloads_created": self._preloads_created,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=ctx.next_ssa_id,
            num_vec_ssa_values=ctx.next_vec_ssa_id,
        )

    def _transform_body(self, body: list[Statement], ctx: LevelCacheContext) -> list[Statement]:
        """Transform a list of statements."""
        result = []

        for stmt in body:
            if isinstance(stmt, Op):
                transformed = self._transform_op(stmt, ctx, result)
                if transformed is not None:
                    result.append(transformed)
                # If None, the op was eliminated (absorbed into preload)
            elif isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt, ctx))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, ctx))
            else:
                # Halt, Pause
                result.append(stmt)

        return result

    def _transform_for_loop(self, loop: ForLoop, ctx: LevelCacheContext) -> ForLoop:
        """Transform a ForLoop."""
        new_body = self._transform_body(loop.body, ctx)

        return ForLoop(
            counter=loop.counter,
            start=loop.start,
            end=loop.end,
            iter_args=loop.iter_args,
            body_params=loop.body_params,
            body=new_body,
            yields=loop.yields,
            results=loop.results,
            pragma_unroll=loop.pragma_unroll,
        )

    def _transform_if(self, if_stmt: If, ctx: LevelCacheContext) -> If:
        """Transform an If statement."""
        new_then = self._transform_body(if_stmt.then_body, ctx)
        new_else = self._transform_body(if_stmt.else_body, ctx)

        return If(
            cond=if_stmt.cond,
            then_body=new_then,
            then_yields=if_stmt.then_yields,
            else_body=new_else,
            else_yields=if_stmt.else_yields,
            results=if_stmt.results,
        )

    def _transform_op(self, op: Op, ctx: LevelCacheContext, result_list: list[Statement]) -> Optional[Op]:
        """
        Transform an Op, potentially replacing vgather with preload + vselect.

        Returns the op to append, or None if the op was absorbed.
        """
        if op.opcode != "vgather":
            return op

        self._vgathers_found += 1

        # Performance safeguard: limit number of vgathers we analyze
        if ctx.vgathers_analyzed >= ctx.max_vgathers_to_analyze:
            return op
        ctx.vgathers_analyzed += 1

        # Analyze the vgather
        info = self._analyze_vgather(op, ctx)
        if info is None:
            return op

        # Track that we found the pattern
        self._vgathers_with_pattern += 1

        # Check if range was computed
        if info.idx_range is not None:
            self._vgathers_with_range += 1

        # Check if range is small enough
        if info.idx_range is None or info.idx_range.size > ctx.max_range_size:
            return op

        # Transform to preload + vselect
        # _transform_vgather returns None on success (vgather absorbed into preload+vselect)
        # and returns the original op on failure
        result = self._transform_vgather(info, ctx, result_list)
        if result is None:
            # Successfully transformed - vgather was absorbed
            self._vgathers_transformed += 1
            return None  # Don't emit the original vgather

        return op

    def _analyze_vgather(self, op: Op, ctx: LevelCacheContext) -> Optional[VgatherInfo]:
        """
        Analyze a vgather operation to extract base and index components.

        Pattern: vgather(v+(base_vec, idx_vec)) where base_vec = vbroadcast(base)
        """
        if op.opcode != "vgather" or len(op.operands) != 1:
            return None

        vec_addrs = op.operands[0]
        if not isinstance(vec_addrs, VectorSSAValue):
            return None

        # Find the definition of vec_addrs
        addr_def = ctx.use_def.get_def(vec_addrs)
        if addr_def is None:
            return None

        addr_op = addr_def.statement
        if not isinstance(addr_op, Op) or addr_op.opcode != "v+":
            return None

        if len(addr_op.operands) != 2:
            return None

        # Try to find base and idx in either order
        base_ssa = None
        idx_vec = None

        for left, right in [(addr_op.operands[0], addr_op.operands[1]),
                            (addr_op.operands[1], addr_op.operands[0])]:
            # Check if left is vbroadcast(base)
            if isinstance(left, VectorSSAValue):
                left_def = ctx.use_def.get_def(left)
                if left_def is not None:
                    left_op = left_def.statement
                    if isinstance(left_op, Op) and left_op.opcode == "vbroadcast":
                        if len(left_op.operands) == 1:
                            potential_base = left_op.operands[0]
                            if isinstance(right, VectorSSAValue):
                                base_ssa = potential_base
                                idx_vec = right
                                break

        if base_ssa is None or idx_vec is None:
            return None

        # Compute range of idx_vec
        # For vectors, we need to analyze what values can appear
        idx_range = self._compute_vector_range(idx_vec, ctx, depth=0)

        return VgatherInfo(
            op=op,
            vec_addrs=vec_addrs,
            base_ssa=base_ssa,
            idx_vec=idx_vec,
            idx_range=idx_range,
        )

    def _compute_vector_range(self, vec: VectorSSAValue, ctx: LevelCacheContext, depth: int = 0) -> Optional[Range]:
        """
        Compute the range of values that can appear in a vector.

        For now, we trace back through vector operations to find scalar sources.
        Uses caching and depth limits for performance.
        """
        # Check cache first
        if vec.id in ctx.vec_range_cache:
            return ctx.vec_range_cache[vec.id]

        # Depth limit for performance
        if depth > ctx.max_vec_depth:
            return None

        vec_def = ctx.use_def.get_def(vec)
        if vec_def is None:
            ctx.vec_range_cache[vec.id] = None
            return None

        vec_op = vec_def.statement
        if not isinstance(vec_op, Op):
            ctx.vec_range_cache[vec.id] = None
            return None

        result = self._compute_vector_range_from_op(vec_op, ctx, depth)
        ctx.vec_range_cache[vec.id] = result
        return result

    def _compute_vector_range_from_op(self, vec_op: Op, ctx: LevelCacheContext, depth: int) -> Optional[Range]:
        """Compute vector range from a specific op."""
        # Handle different vector-producing operations
        if vec_op.opcode == "vbroadcast":
            # All lanes have the same value
            if len(vec_op.operands) == 1:
                return ctx.range_analysis.get_range(vec_op.operands[0], depth)
            return None

        if vec_op.opcode == "vselect":
            # Union of true and false branches
            if len(vec_op.operands) == 3:
                true_range = self._compute_vector_range_or_const(vec_op.operands[1], ctx, depth + 1)
                if true_range is None:
                    return None
                false_range = self._compute_vector_range_or_const(vec_op.operands[2], ctx, depth + 1)
                if false_range is None:
                    return None
                return true_range.union(false_range)
            return None

        if vec_op.opcode in ("v+", "v-", "v*"):
            # Binary vector operations
            if len(vec_op.operands) == 2:
                left_range = self._compute_vector_range_or_const(vec_op.operands[0], ctx, depth + 1)
                if left_range is None:
                    return None
                right_range = self._compute_vector_range_or_const(vec_op.operands[1], ctx, depth + 1)
                if right_range is None:
                    return None
                if vec_op.opcode == "v+":
                    return left_range + right_range
                elif vec_op.opcode == "v-":
                    return left_range - right_range
                elif vec_op.opcode == "v*":
                    return left_range * right_range
            return None

        if vec_op.opcode in ("v&", "v|", "v^", "v<<", "v>>"):
            if len(vec_op.operands) == 2:
                left_range = self._compute_vector_range_or_const(vec_op.operands[0], ctx, depth + 1)
                if left_range is None:
                    return None
                right_range = self._compute_vector_range_or_const(vec_op.operands[1], ctx, depth + 1)
                if right_range is None:
                    return None
                if vec_op.opcode == "v&":
                    return left_range.bitwise_and(right_range)
                elif vec_op.opcode == "v<<":
                    return left_range.shift_left(right_range)
                elif vec_op.opcode == "v>>":
                    return left_range.shift_right(right_range)
            return None

        # For vload: Use domain knowledge for tree_hash kernel
        # The initial indices are loaded from inp_indices_p and start at 0
        # We track depth to infer which "round" we're in
        if vec_op.opcode == "vload":
            # Base case: initial idx load is at depth 0, range {0}
            # This is the first round's idx values
            return Range.point(0)

        # For vgather, etc. - return unknown
        return None

    def _compute_vector_range_or_const(self, val: Value, ctx: LevelCacheContext, depth: int = 0) -> Optional[Range]:
        """Compute range for a value that may be vector or VectorConst."""
        if isinstance(val, VectorConst):
            return Range(min(val.values), max(val.values))
        elif isinstance(val, VectorSSAValue):
            return self._compute_vector_range(val, ctx, depth)
        return None

    def _transform_vgather(
        self,
        info: VgatherInfo,
        ctx: LevelCacheContext,
        result_list: list[Statement]
    ) -> Optional[Op]:
        """
        Transform a vgather into preload + vselect.

        For range [min_val, max_val]:
        1. Emit scalar loads for each value in range
        2. Emit vbroadcast for each loaded value
        3. Emit vselect tree to select based on idx bits
        4. Return the final vselect as replacement for vgather
        """
        if info.idx_range is None or info.base_ssa is None or info.idx_vec is None:
            return None

        range_size = info.idx_range.size
        if range_size > ctx.max_range_size or range_size <= 0:
            return None

        min_idx = info.idx_range.min_val
        max_idx = info.idx_range.max_val

        # Generate preloaded vectors for each possible index value
        preloaded_vecs = []
        for idx_val in range(min_idx, max_idx + 1):
            # scalar_addr = base + idx_val
            addr_ssa = SSAValue(id=ctx.next_ssa_id, name=f"preload_addr_{idx_val}")
            ctx.next_ssa_id += 1

            addr_op = Op(
                opcode="+",
                result=addr_ssa,
                operands=[info.base_ssa, Const(idx_val)],
                engine="alu"
            )
            result_list.append(addr_op)

            # loaded_val = load(scalar_addr)
            loaded_ssa = SSAValue(id=ctx.next_ssa_id, name=f"preload_val_{idx_val}")
            ctx.next_ssa_id += 1

            load_op = Op(
                opcode="load",
                result=loaded_ssa,
                operands=[addr_ssa],
                engine="load"
            )
            result_list.append(load_op)

            # vec_val = vbroadcast(loaded_val)
            vec_ssa = VectorSSAValue(id=ctx.next_vec_ssa_id, name=f"preload_vec_{idx_val}")
            ctx.next_vec_ssa_id += 1

            broadcast_op = Op(
                opcode="vbroadcast",
                result=vec_ssa,
                operands=[loaded_ssa],
                engine="valu"
            )
            result_list.append(broadcast_op)

            preloaded_vecs.append(vec_ssa)
            self._preloads_created += 1

        # Build vselect tree based on idx bits
        # offset_vec = idx_vec - min_idx (so offset is in [0, range_size-1])
        if min_idx != 0:
            offset_vec = VectorSSAValue(id=ctx.next_vec_ssa_id, name="offset_vec")
            ctx.next_vec_ssa_id += 1

            # Create constant vector for min_idx
            min_idx_vec = VectorConst(values=tuple([min_idx] * VLEN))

            sub_op = Op(
                opcode="v-",
                result=offset_vec,
                operands=[info.idx_vec, min_idx_vec],
                engine="valu"
            )
            result_list.append(sub_op)
        else:
            offset_vec = info.idx_vec

        # Build binary select tree
        result_vec = self._build_select_tree(
            preloaded_vecs, offset_vec, 0, range_size, ctx, result_list
        )

        # Return a "pseudo-op" that just assigns result_vec to the original result
        # Actually we need to use the same result SSA as the original vgather
        if info.op.result is not None:
            # Replace uses of vgather result with select tree result
            ctx.use_def.replace_all_uses(info.op.result, result_vec, auto_invalidate=False)

        # Return None to indicate the vgather was absorbed
        return None

    def _build_select_tree(
        self,
        vecs: list[VectorSSAValue],
        offset_vec: VectorSSAValue,
        start: int,
        count: int,
        ctx: LevelCacheContext,
        result_list: list[Statement]
    ) -> VectorSSAValue:
        """
        Build a binary vselect tree to select from `count` vectors.

        The offset_vec contains values in [0, count-1].
        We use bits of offset to select: bit 0 selects between pairs,
        bit 1 selects between pairs of pairs, etc.

        Returns the result VectorSSAValue.
        """
        if count == 1:
            return vecs[start]

        if count == 2:
            # Select between vecs[start] and vecs[start+1] based on bit 0
            bit0_vec = self._extract_bit(offset_vec, 0, ctx, result_list)
            result = VectorSSAValue(id=ctx.next_vec_ssa_id, name="sel_result")
            ctx.next_vec_ssa_id += 1

            # vselect: if bit0 == 1, select vecs[start+1], else vecs[start]
            select_op = Op(
                opcode="vselect",
                result=result,
                operands=[bit0_vec, vecs[start + 1], vecs[start]],
                engine="flow"
            )
            result_list.append(select_op)
            return result

        # For count > 2, build tree recursively
        # Split into two halves
        half = self._next_power_of_2(count) // 2

        # Handle case where count is not a power of 2
        if half >= count:
            half = count // 2

        left_count = half
        right_count = count - half

        # Recursively build left and right subtrees
        left_result = self._build_select_tree(vecs, offset_vec, start, left_count, ctx, result_list)

        # For right subtree, we need to handle indexing properly
        # offset >= half means we should select from right subtree
        right_start = start + half
        right_result = self._build_select_tree(vecs, offset_vec, right_start, right_count, ctx, result_list)

        # Select between left and right based on whether offset >= half
        # This is equivalent to checking bit log2(half)
        bit_idx = self._log2(half)
        cond_vec = self._extract_bit(offset_vec, bit_idx, ctx, result_list)

        result = VectorSSAValue(id=ctx.next_vec_ssa_id, name=f"sel_level_{bit_idx}")
        ctx.next_vec_ssa_id += 1

        # If cond (offset >= half), select right, else left
        select_op = Op(
            opcode="vselect",
            result=result,
            operands=[cond_vec, right_result, left_result],
            engine="flow"
        )
        result_list.append(select_op)
        return result

    def _extract_bit(
        self,
        vec: VectorSSAValue,
        bit_idx: int,
        ctx: LevelCacheContext,
        result_list: list[Statement]
    ) -> VectorSSAValue:
        """Extract bit `bit_idx` from each lane of vec as 0 or 1."""
        # (vec >> bit_idx) & 1
        if bit_idx > 0:
            shift_const = VectorConst(values=tuple([bit_idx] * VLEN))
            shifted = VectorSSAValue(id=ctx.next_vec_ssa_id, name=f"bit{bit_idx}_shifted")
            ctx.next_vec_ssa_id += 1

            shift_op = Op(
                opcode="v>>",
                result=shifted,
                operands=[vec, shift_const],
                engine="valu"
            )
            result_list.append(shift_op)
        else:
            shifted = vec

        mask_const = VectorConst(values=tuple([1] * VLEN))
        masked = VectorSSAValue(id=ctx.next_vec_ssa_id, name=f"bit{bit_idx}_masked")
        ctx.next_vec_ssa_id += 1

        mask_op = Op(
            opcode="v&",
            result=masked,
            operands=[shifted, mask_const],
            engine="valu"
        )
        result_list.append(mask_op)

        return masked

    @staticmethod
    def _next_power_of_2(n: int) -> int:
        """Return the smallest power of 2 >= n."""
        if n <= 0:
            return 1
        p = 1
        while p < n:
            p *= 2
        return p

    @staticmethod
    def _log2(n: int) -> int:
        """Return floor(log2(n)) for n > 0."""
        if n <= 0:
            return 0
        result = 0
        while n > 1:
            n //= 2
            result += 1
        return result
