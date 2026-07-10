"""
Tree Level Cache (HIR) Pass

Replaces early-round node loads with preloaded values and select trees.
This tree_hash-specific optimization consumes a zero-root SSA recurrence;
generic local-memory promotion is responsible for exposing that recurrence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..hir import SSAValue, Const, Op, ForLoop, If, Statement, HIRFunction, Value
from ..alias_analysis import AliasAnalysis, AliasResult
from ..pass_manager import Pass, PassConfig
from ..use_def import UseDefContext
from vm import VLEN


@dataclass(frozen=True)
class _IndexUpdate:
    previous: Value
    branch: SSAValue
    # Constant bounds of the wrap checks peeled off this update (empty when
    # the matched chain contains no wrap). The wrap period is derived from
    # these verified bounds, never from unrelated comparisons in the body.
    wrap_bounds: tuple[int, ...]


@dataclass(frozen=True)
class _NodeAccess:
    stmt_index: int
    index: Value
    load_result: SSAValue
    update: Optional[_IndexUpdate]


@dataclass(frozen=True)
class _Replacement:
    level: int
    load_result: SSAValue
    round_index: int
    branch_bits: tuple[SSAValue, ...]


class TreeLevelCachePass(Pass):
    """Replace early `load(forest_values_p + idx)` with cached top-level nodes."""

    def __init__(self):
        super().__init__()
        self._next_ssa_id: int = 0
        self._node_loads_seen: int = 0
        self._node_loads_replaced: int = 0
        self._preloads_inserted: int = 0

    @property
    def name(self) -> str:
        return "tree-level-cache"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        self._node_loads_seen = 0
        self._node_loads_replaced = 0
        self._preloads_inserted = 0

        if not config.enabled:
            return hir

        levels = int(config.options.get("levels", 4))
        estimate_ops = bool(config.options.get("estimate_ops", True))
        # post_wrap_levels: separate depth limit for rounds after the first
        # wrap (round >= wrap_period). These sit near the program tail where
        # register pressure is low, so deeper caching is affordable there
        # even when it is not for the pre-wrap rounds.
        post_wrap_levels = int(config.options.get("post_wrap_levels", levels))
        # base_levels: preloaded once at start; higher levels get phased preloads
        base_levels = min(int(config.options.get("base_levels", 3)), levels)
        if levels <= 0:
            self._add_metric_message("levels <= 0, skipping")
            return hir
        if levels > 4:
            self._add_metric_message("levels > 4 not supported, clamping to 4")
            levels = 4
        if post_wrap_levels > 4:
            post_wrap_levels = 4

        if any(isinstance(s, (ForLoop, If)) for s in hir.body):
            self._add_metric_message("non-flat HIR detected, skipping")
            return hir

        use_def = UseDefContext(hir)
        forest_values_p = self._find_header_load(hir.body, 4)
        if forest_values_p is None:
            self._add_metric_message("missing forest_values_p, skipping")
            return hir
        accesses, update_cache = self._collect_node_accesses(
            hir.body, use_def, forest_values_p
        )
        self._node_loads_seen = len(accesses)
        wrap_period, wrap_consistent = self._derive_wrap_period(update_cache)
        if not wrap_consistent:
            self._add_metric_message(
                "inconsistent wrap-check bounds across node-index chains, "
                "skipping"
            )
            return hir

        # LocalMem2Reg exposes one leading Const(0) node index per lane. Infer
        # that lane count from the collected records, then validate every later
        # record against the cached one-step recurrence for the same lane.
        batch_size, replacements = self._analyze_node_accesses(
            accesses,
            update_cache,
            levels,
            wrap_period,
            post_wrap_levels,
        )
        if batch_size <= 0:
            self._add_metric_message(
                "unable to infer a promoted zero-root batch, skipping"
            )
            return hir
        if not replacements:
            self._add_metric_message("no eligible node loads found")
            return hir

        # Preloaded values are only valid while the cached forest prefix is
        # unchanged. AliasAnalysis canonicalizes repeated loads from header
        # slot 4 to the same root, unlike exact SSA provenance.
        max_cached_level = max(repl.level for repl in replacements.values())
        cached_words = (1 << (max_cached_level + 1)) - 1
        # The tree-hash ABI gives its header regions distinct pointer roots.
        # Reloads of one header slot still canonicalize to the same root.
        alias = AliasAnalysis(use_def, restrict_ptr=True)
        if self._forest_may_be_modified(
            hir.body, forest_values_p, cached_words, alias
        ):
            self._add_metric_message("forest may be modified, skipping")
            return hir

        base_levels = min(base_levels, max_cached_level + 1)

        self._next_ssa_id = hir.num_ssa_values

        # Emit base preloads (levels 0..base_levels-1) once
        base_preload_ops, base_node_vals = self._emit_preloads(
            forest_values_p, base_levels
        )
        self._preloads_inserted = len(base_node_vals)

        # Group replacements by round to identify where phased preloads are needed
        # round_idx -> list of (stmt_idx, replacement)
        round_replacements: dict[
            int, list[tuple[int, _Replacement]]
        ] = {}
        for stmt_idx, repl in replacements.items():
            round_idx = repl.round_index
            if round_idx not in round_replacements:
                round_replacements[round_idx] = []
            round_replacements[round_idx].append((stmt_idx, repl))

        # Identify which rounds need higher-level preloads
        # For each such round, we emit fresh preloads right before the first replacement
        higher_level_rounds: set[int] = set()
        for round_idx, repls in round_replacements.items():
            if any(repl.level >= base_levels for _, repl in repls):
                higher_level_rounds.add(round_idx)

        # Build per-round phased node_vals (fresh preloads for each round needing level >= base_levels)
        # Maps round_idx -> (preload_ops, node_vals covering ALL levels)
        phased_preloads: dict[int, tuple[list[Op], list[SSAValue]]] = {}
        for round_idx in higher_level_rounds:
            # Emit fresh preloads covering the deepest level this round uses
            round_max_level = max(
                repl.level for _, repl in round_replacements[round_idx]
            )
            extra_ops, extra_vals = self._emit_preloads_range(
                forest_values_p, base_levels, round_max_level + 1
            )
            # Combine with base node vals for a complete set
            full_vals = list(base_node_vals) + list(extra_vals)
            phased_preloads[round_idx] = (extra_ops, full_vals)
            self._preloads_inserted += len(extra_vals)

        # Build set of stmt indices that are first-in-round for higher-level rounds
        first_in_round: dict[int, int] = {}  # stmt_idx -> round_idx
        for round_idx in higher_level_rounds:
            repls = round_replacements[round_idx]
            first_stmt = min(si for si, _ in repls)
            first_in_round[first_stmt] = round_idx

        # Place base preloads immediately before their first real use.
        first_base_use_idx = min(replacements.keys())

        # Replace loads with select trees
        new_body: list[Statement] = []
        for idx, stmt in enumerate(hir.body):
            if idx == first_base_use_idx:
                new_body.extend(base_preload_ops)

            # Insert phased preloads before first replacement in a higher-level round
            if idx in first_in_round:
                round_idx = first_in_round[idx]
                extra_ops, _ = phased_preloads[round_idx]
                new_body.extend(extra_ops)

            repl = replacements.get(idx)
            if repl is not None:
                # Pick the right node_vals: phased if this round has higher-level preloads
                round_idx = repl.round_index
                if round_idx in phased_preloads:
                    _, round_node_vals = phased_preloads[round_idx]
                else:
                    round_node_vals = base_node_vals
                select_ops: list[Op] = []
                replacement = self._build_select_for_level(
                    repl.level,
                    round_node_vals,
                    select_ops,
                    repl.branch_bits,
                )
                use_def.replace_all_uses(
                    repl.load_result, replacement, auto_invalidate=False
                )
                new_body.extend(select_ops)
                self._node_loads_replaced += 1
                continue

            new_body.append(stmt)

        if self._metrics:
            metrics = {
                "levels": levels,
                "base_levels": base_levels,
                "batch_size": batch_size,
                "wrap_period": wrap_period,
                "node_loads_seen": self._node_loads_seen,
                "node_loads_replaced": self._node_loads_replaced,
                "preloads_inserted": self._preloads_inserted,
                "root_indices_proven": batch_size,
                "higher_level_rounds": sorted(higher_level_rounds),
            }
            if estimate_ops:
                estimated_flow = self._estimate_flow_ops(replacements)
                metrics["estimated_flow_ops"] = estimated_flow
                metrics["estimated_loads_removed"] = self._node_loads_replaced
            self._metrics.custom = metrics

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=max(hir.num_ssa_values, self._next_ssa_id),
            num_vec_ssa_values=hir.num_vec_ssa_values,
        )

    @staticmethod
    def _find_header_load(body: list[Statement], slot: int) -> Optional[SSAValue]:
        for stmt in body:
            if isinstance(stmt, Op) and stmt.opcode == "load" and stmt.result is not None:
                addr = stmt.operands[0]
                if isinstance(addr, Const) and addr.value == slot:
                    return stmt.result
        return None

    @staticmethod
    def _forest_may_be_modified(
        body: list[Statement],
        forest_values_p: SSAValue,
        cached_words: int,
        alias: AliasAnalysis,
    ) -> bool:
        """Return whether a store may overlap the cached forest prefix."""

        forest_key = alias.normalize(forest_values_p)
        for stmt in body:
            if not isinstance(stmt, Op) or stmt.opcode not in ("store", "vstore"):
                continue
            store_key = alias.normalize(stmt.operands[0])
            store_width = 1 if stmt.opcode == "store" else VLEN
            if alias.alias_keys(
                forest_key, cached_words, store_key, store_width
            ) != AliasResult.NO_ALIAS:
                return True
        return False

    def _collect_node_accesses(
        self,
        body: list[Statement],
        use_def: UseDefContext,
        forest_values_p: SSAValue,
    ) -> tuple[
        list[_NodeAccess],
        dict[SSAValue, Optional[_IndexUpdate]],
    ]:
        """Collect forest loads and peel each distinct SSA index once."""

        updates: dict[SSAValue, Optional[_IndexUpdate]] = {}
        accesses: list[_NodeAccess] = []
        for stmt_index, stmt in enumerate(body):
            if (not isinstance(stmt, Op) or stmt.opcode != "load"
                    or not isinstance(stmt.result, SSAValue)):
                continue
            index = self._match_node_addr(
                stmt.operands[0], use_def, forest_values_p
            )
            if index is None:
                continue

            update: Optional[_IndexUpdate] = None
            if isinstance(index, SSAValue):
                if index not in updates:
                    updates[index] = self._peel_index_update(index, use_def)
                update = updates[index]
            accesses.append(_NodeAccess(
                stmt_index=stmt_index,
                index=index,
                load_result=stmt.result,
                update=update,
            ))
        return accesses, updates

    @staticmethod
    def _derive_wrap_period(
        updates: dict[SSAValue, Optional[_IndexUpdate]],
    ) -> tuple[Optional[int], bool]:
        """Derive the wrap period from bounds peeled off matched chains.

        For a complete binary tree (n_nodes == 2**h - 1), index depth
        advances one level per round and wraps to root every h rounds. The
        bound is taken only from wrap checks that are part of the matched
        recurrence, so unrelated comparisons elsewhere in the body cannot
        influence the result. Returns (wrap_period, consistent); mixed
        bounds across chains are unmodelable and reported as inconsistent.
        """
        bounds: set[int] = set()
        for update in updates.values():
            if update is not None:
                bounds.update(update.wrap_bounds)
        if not bounds:
            return None, True
        if len(bounds) > 1:
            return None, False
        n_nodes = bounds.pop()
        return (n_nodes + 1).bit_length() - 1, True

    def _analyze_node_accesses(
        self,
        accesses: list[_NodeAccess],
        updates: dict[SSAValue, Optional[_IndexUpdate]],
        levels: int,
        wrap_period: Optional[int],
        post_wrap_levels: int,
    ) -> tuple[int, dict[int, _Replacement]]:
        """Infer lane count and validate one recurrence chain per lane."""

        batch_size = 0
        for access in accesses:
            if not (isinstance(access.index, Const)
                    and access.index.value == 0):
                break
            batch_size += 1
        if batch_size == 0:
            return 0, {}

        replacements: dict[int, _Replacement] = {}
        previous_indices: dict[int, Value] = {}
        for access_index, access in enumerate(accesses):
            round_index = access_index // batch_size
            lane = access_index % batch_size
            if round_index == 0:
                if not (isinstance(access.index, Const)
                        and access.index.value == 0):
                    self._add_metric_message(
                        "node-index chain is not rooted in promoted zero for "
                        f"lane {lane}"
                    )
                    return batch_size, {}
            else:
                if not isinstance(access.index, SSAValue):
                    self._add_metric_message(
                        f"non-SSA node index in round {round_index}, skipping"
                    )
                    return batch_size, {}
                previous = (
                    access.update.previous if access.update is not None else None
                )
                if previous != previous_indices.get(lane):
                    self._add_metric_message(
                        "node-index recurrence does not match the preceding "
                        f"round for offset {lane}"
                    )
                    return batch_size, {}
                # With period k the index provably sits at depth k-1 when
                # round_index % k == 0, so that update must carry the wrap
                # check; other rounds cannot wrap and may omit it (a folded
                # check peels transparently and never fires).
                if (wrap_period is not None
                        and round_index % wrap_period == 0
                        and not access.update.wrap_bounds):
                    self._add_metric_message(
                        "missing wrap check at wrap round "
                        f"{round_index} for offset {lane}, skipping"
                    )
                    return batch_size, {}
            previous_indices[lane] = access.index

            phase = round_index
            round_levels = levels
            if wrap_period is not None and wrap_period > 0:
                phase = round_index % wrap_period
                if round_index >= wrap_period:
                    round_levels = post_wrap_levels
            if phase >= round_levels:
                continue

            branch_bits: tuple[SSAValue, ...] = ()
            if phase > 0:
                if not isinstance(access.index, SSAValue):
                    self._add_metric_message(
                        f"non-SSA cached index in round {round_index}, skipping"
                    )
                    return batch_size, {}
                bits = self._extract_branch_bits(
                    access.index, phase, updates
                )
                if bits is None:
                    self._add_metric_message(
                        "cached branch history does not match the validated "
                        f"recurrence in round {round_index}"
                    )
                    return batch_size, {}
                branch_bits = tuple(bits)
            replacements[access.stmt_index] = _Replacement(
                level=phase,
                load_result=access.load_result,
                round_index=round_index,
                branch_bits=branch_bits,
            )

        if len(accesses) % batch_size != 0:
            self._add_metric_message("incomplete batch of node loads, skipping")
            return batch_size, {}
        return batch_size, replacements

    @staticmethod
    def _peel_index_update(
        idx_ssa: SSAValue,
        use_def: UseDefContext,
    ) -> Optional[_IndexUpdate]:
        """Match one tree-index update and return its previous index and bit.

        Wrap checks (select or select-to-mul form, bounded by a constant
        n_nodes == 2**h - 1) are peeled transparently; their bounds are
        recorded on the returned update so the caller can derive the wrap
        period and enforce a consistent wrap schedule.
        """

        def get_def(value) -> Optional[Op]:
            if not isinstance(value, SSAValue):
                return None
            loc = use_def.get_def(value)
            if loc is None or not isinstance(loc.statement, Op):
                return None
            return loc.statement

        def bound_of_check(condition, next_index: SSAValue) -> Optional[int]:
            definition = get_def(condition)
            if (definition is None or definition.opcode != "<"
                    or len(definition.operands) != 2):
                return None
            left, right = definition.operands
            if left != next_index or not isinstance(right, Const):
                return None
            full = right.value + 1
            if right.value > 0 and full & (full - 1) == 0:
                return right.value
            return None

        def is_branch_bit(value) -> bool:
            definition = get_def(value)
            if (definition is None or definition.opcode != "&"
                    or len(definition.operands) != 2):
                return False
            return any(isinstance(operand, Const) and operand.value == 1
                       for operand in definition.operands)

        current = idx_ssa
        wrap_bounds: list[int] = []
        for _ in range(4):
            definition = get_def(current)
            if definition is None:
                return None
            if definition.opcode == "select" and len(definition.operands) == 3:
                condition, true_value, false_value = definition.operands
                if isinstance(false_value, Const) and false_value.value == 0:
                    if not isinstance(true_value, SSAValue):
                        return None
                    bound = bound_of_check(condition, true_value)
                    if bound is None:
                        return None
                    wrap_bounds.append(bound)
                    current = true_value
                    continue
                return None
            if definition.opcode == "*" and len(definition.operands) == 2:
                left, right = definition.operands
                left_def, right_def = get_def(left), get_def(right)
                if left_def is not None and left_def.opcode == "<":
                    if not isinstance(right, SSAValue):
                        return None
                    bound = bound_of_check(left, right)
                    if bound is None:
                        return None
                    wrap_bounds.append(bound)
                    current = right
                    continue
                if right_def is not None and right_def.opcode == "<":
                    if not isinstance(left, SSAValue):
                        return None
                    bound = bound_of_check(right, left)
                    if bound is None:
                        return None
                    wrap_bounds.append(bound)
                    current = left
                    continue
            break

        definition = get_def(current)
        if (definition is None or definition.opcode != "+"
                or len(definition.operands) != 2):
            return None
        for offset_value, doubled_value in (
            (definition.operands[0], definition.operands[1]),
            (definition.operands[1], definition.operands[0]),
        ):
            offset_def = get_def(offset_value)
            doubled_def = get_def(doubled_value)
            if (offset_def is None or offset_def.opcode != "+"
                    or len(offset_def.operands) != 2
                    or doubled_def is None or doubled_def.opcode != "*"
                    or len(doubled_def.operands) != 2):
                continue
            offset_left, offset_right = offset_def.operands
            if isinstance(offset_left, Const) and offset_left.value == 1:
                branch = offset_right
            elif isinstance(offset_right, Const) and offset_right.value == 1:
                branch = offset_left
            else:
                continue
            if not is_branch_bit(branch):
                continue
            doubled_left, doubled_right = doubled_def.operands
            if isinstance(doubled_left, Const) and doubled_left.value == 2:
                previous = doubled_right
            elif isinstance(doubled_right, Const) and doubled_right.value == 2:
                previous = doubled_left
            else:
                continue
            if not isinstance(previous, (SSAValue, Const)):
                return None
            if not isinstance(branch, SSAValue):
                return None
            return _IndexUpdate(previous, branch, tuple(wrap_bounds))
        return None

    @staticmethod
    def _match_node_addr(addr: Value, use_def: UseDefContext,
                         forest_values_p: SSAValue) -> Optional[Value]:
        if addr == forest_values_p:
            return Const(0)
        if not isinstance(addr, SSAValue):
            return None
        def_loc = use_def.get_def(addr)
        if def_loc is None or not isinstance(def_loc.statement, Op):
            return None
        op = def_loc.statement
        if op.opcode != "+" or len(op.operands) != 2:
            return None
        a, b = op.operands
        if a == forest_values_p and isinstance(b, (SSAValue, Const)):
            return b
        if b == forest_values_p and isinstance(a, (SSAValue, Const)):
            return a
        return None

    def _new_ssa(self, name: Optional[str] = None) -> SSAValue:
        v = SSAValue(self._next_ssa_id, name)
        self._next_ssa_id += 1
        return v

    def _emit_preloads(self, forest_values_p: SSAValue, levels: int) -> tuple[list[Op], list[SSAValue]]:
        """Emit preloads for all nodes in levels 0..levels-1."""
        return self._emit_preloads_range(forest_values_p, 0, levels)

    def _emit_preloads_range(
        self, forest_values_p: SSAValue, from_level: int, to_level: int
    ) -> tuple[list[Op], list[SSAValue]]:
        """Emit preloads for nodes in levels from_level..to_level-1 only.

        Returns ops and node_vals for nodes at indices
        (2^from_level - 1) .. (2^to_level - 2).
        """
        start_node = (1 << from_level) - 1
        end_node = (1 << to_level) - 1
        ops: list[Op] = []
        node_vals: list[SSAValue] = []
        for n in range(start_node, end_node):
            addr = self._new_ssa(f"tree_cache_addr_{n}")
            ops.append(Op("+", addr, [forest_values_p, Const(n)], "alu"))
            val = self._new_ssa(f"tree_cache_val_{n}")
            ops.append(Op("load", val, [addr], "load"))
            node_vals.append(val)
        return ops, node_vals

    def _emit_select(self, ops: list[Op], cond, a, b) -> SSAValue:
        res = self._new_ssa()
        ops.append(Op("select", res, [cond, a, b], "flow"))
        return res

    def _build_select_for_level(
        self,
        level: int,
        node_vals: list[SSAValue],
        ops: list[Op],
        branch_bits: tuple[SSAValue, ...],
    ) -> SSAValue:
        if level == 0:
            return node_vals[0]
        if len(branch_bits) != level:
            raise ValueError("cached branch history does not match tree level")

        base_idx = (1 << level) - 1
        current = [node_vals[base_idx + i] for i in range(1 << level)]
        for bit in branch_bits:
            next_level: list[SSAValue] = []
            for i in range(0, len(current), 2):
                next_level.append(self._emit_select(ops, bit, current[i + 1], current[i]))
            current = next_level
        return current[0]

    @staticmethod
    def _extract_branch_bits(
        idx_ssa: SSAValue,
        level: int,
        updates: dict[SSAValue, Optional[_IndexUpdate]],
    ) -> Optional[list[SSAValue]]:
        """Recover cached branch bits from previously peeled updates.

        Since idx = 2^d - 1 + o with o < 2^d, bit i of o is exactly the
        branch bit from (level - i) rounds ago (LSB = most recent). Returns
        bits LSB-first, or None if the cached chain is incomplete.
        """
        bits: list[SSAValue] = []
        current: Value = idx_ssa
        for _ in range(level):
            if not isinstance(current, SSAValue):
                return None
            update = updates.get(current)
            if update is None:
                return None
            bits.append(update.branch)
            current = update.previous
        return bits

    @staticmethod
    def _estimate_flow_ops(
        replacements: dict[int, _Replacement]
    ) -> int:
        """Estimate flow ops introduced by select trees from exact levels."""
        total = 0
        for replacement in replacements.values():
            if replacement.level > 0:
                total += (1 << replacement.level) - 1
        return total
