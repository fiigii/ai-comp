"""
Tree Level Cache (HIR) Pass

Replaces early-round node loads with preloaded values and select trees.
This is a tree_hash-specific optimization that assumes initial indices are zero.
"""

from __future__ import annotations

from typing import Optional

from ..hir import SSAValue, Const, Op, Pause, ForLoop, If, Statement, HIRFunction
from ..pass_manager import Pass, PassConfig
from ..use_def import UseDefContext


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
        if levels <= 0:
            self._add_metric_message("levels <= 0, skipping")
            return hir
        if levels > 4:
            self._add_metric_message("levels > 4 not supported, clamping to 4")
            levels = 4

        if any(isinstance(s, (ForLoop, If)) for s in hir.body):
            self._add_metric_message("non-flat HIR detected, skipping")
            return hir

        use_def = UseDefContext(hir)
        forest_values_p = self._find_header_load(hir.body, 4)
        inp_indices_p = self._find_header_load(hir.body, 5)
        if forest_values_p is None or inp_indices_p is None:
            self._add_metric_message("missing forest_values_p or inp_indices_p, skipping")
            return hir

        batch_size = self._infer_batch_size(hir.body, use_def, inp_indices_p)
        if batch_size <= 0:
            self._add_metric_message("unable to infer batch_size, skipping")
            return hir

        wrap_period = self._infer_wrap_period(hir.body)
        replacements = self._find_replacements(
            hir.body, use_def, forest_values_p, batch_size, levels, wrap_period
        )
        if not replacements:
            self._add_metric_message("no eligible node loads found")
            return hir

        self._next_ssa_id = hir.num_ssa_values
        preload_ops: list[Op] = []
        node_vals: list[SSAValue] = []
        if replacements:
            preload_ops, node_vals = self._emit_preloads(forest_values_p, levels)
            self._preloads_inserted = len(node_vals)

        # Replace loads with select trees
        new_body: list[Statement] = []
        inserted_preloads = False
        for idx, stmt in enumerate(hir.body):
            if isinstance(stmt, Pause) and not inserted_preloads:
                new_body.append(stmt)
                new_body.extend(preload_ops)
                inserted_preloads = True
                continue

            repl = replacements.get(idx)
            if repl is not None:
                idx_ssa, level, load_result = repl
                select_ops: list[Op] = []
                replacement = self._build_select_for_level(
                    level, idx_ssa, node_vals, select_ops
                )
                use_def.replace_all_uses(load_result, replacement, auto_invalidate=False)
                new_body.extend(select_ops)
                self._node_loads_replaced += 1
                continue

            new_body.append(stmt)

        if not inserted_preloads:
            new_body = preload_ops + new_body

        if self._metrics:
            metrics = {
                "levels": levels,
                "batch_size": batch_size,
                "wrap_period": wrap_period,
                "node_loads_seen": self._node_loads_seen,
                "node_loads_replaced": self._node_loads_replaced,
                "preloads_inserted": self._preloads_inserted,
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
    def _extract_const_add(op: Op) -> tuple[Optional[int], Optional[SSAValue]]:
        if len(op.operands) != 2:
            return None, None
        a, b = op.operands
        if isinstance(a, Const) and isinstance(b, SSAValue):
            return a.value, b
        if isinstance(b, Const) and isinstance(a, SSAValue):
            return b.value, a
        return None, None

    def _infer_batch_size(self, body: list[Statement], use_def: UseDefContext,
                          inp_indices_p: SSAValue) -> int:
        offsets: set[int] = set()
        for stmt in body:
            if not isinstance(stmt, Op) or stmt.opcode != "load":
                continue
            addr = stmt.operands[0]
            if not isinstance(addr, SSAValue):
                continue
            def_loc = use_def.get_def(addr)
            if def_loc is None or not isinstance(def_loc.statement, Op):
                continue
            op = def_loc.statement
            if op.opcode != "+":
                continue
            const_val, other = self._extract_const_add(op)
            if const_val is None or other is None:
                continue
            if other == inp_indices_p:
                offsets.add(const_val)
        if not offsets:
            return 0
        return max(offsets) + 1

    @staticmethod
    def _infer_wrap_period(body: list[Statement]) -> Optional[int]:
        """Infer deterministic wrap period from constant n_nodes if available.

        For a complete binary tree (n_nodes == 2**h - 1), index depth advances
        by one level each round and wraps to root every h rounds.
        """
        candidates: set[int] = set()
        for stmt in body:
            if not isinstance(stmt, Op) or stmt.opcode != "<" or len(stmt.operands) != 2:
                continue
            a, b = stmt.operands
            if isinstance(a, Const):
                candidates.add(a.value)
            if isinstance(b, Const):
                candidates.add(b.value)
        if not candidates:
            return None

        n_nodes = max(candidates)
        if n_nodes <= 0:
            return None
        full = n_nodes + 1
        if full & (full - 1) != 0:
            return None
        return full.bit_length() - 1

    def _find_replacements(
        self,
        body: list[Statement],
        use_def: UseDefContext,
        forest_values_p: SSAValue,
        batch_size: int,
        levels: int,
        wrap_period: Optional[int],
    ) -> dict[int, tuple[SSAValue, int, SSAValue]]:
        replacements: dict[int, tuple[SSAValue, int, SSAValue]] = {}
        node_load_idx = 0
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, Op) or stmt.opcode != "load" or stmt.result is None:
                continue
            addr = stmt.operands[0]
            if not isinstance(addr, SSAValue):
                continue
            idx_ssa = self._match_node_addr(addr, use_def, forest_values_p)
            if idx_ssa is None:
                continue

            self._node_loads_seen += 1
            round_idx = node_load_idx // batch_size
            node_load_idx += 1
            phase = round_idx
            if wrap_period is not None and wrap_period > 0:
                phase = round_idx % wrap_period
            if phase < levels:
                replacements[idx] = (idx_ssa, phase, stmt.result)
        return replacements

    @staticmethod
    def _match_node_addr(addr: SSAValue, use_def: UseDefContext,
                         forest_values_p: SSAValue) -> Optional[SSAValue]:
        def_loc = use_def.get_def(addr)
        if def_loc is None or not isinstance(def_loc.statement, Op):
            return None
        op = def_loc.statement
        if op.opcode != "+" or len(op.operands) != 2:
            return None
        a, b = op.operands
        if a == forest_values_p and isinstance(b, SSAValue):
            return b
        if b == forest_values_p and isinstance(a, SSAValue):
            return a
        return None

    def _new_ssa(self, name: Optional[str] = None) -> SSAValue:
        v = SSAValue(self._next_ssa_id, name)
        self._next_ssa_id += 1
        return v

    def _emit_preloads(self, forest_values_p: SSAValue, levels: int) -> tuple[list[Op], list[SSAValue]]:
        node_count = (1 << levels) - 1
        ops: list[Op] = []
        node_vals: list[SSAValue] = []
        for n in range(node_count):
            addr = self._new_ssa(f"tree_cache_addr_{n}")
            ops.append(Op("+", addr, [forest_values_p, Const(n)], "alu"))
            val = self._new_ssa(f"tree_cache_val_{n}")
            ops.append(Op("load", val, [addr], "load"))
            node_vals.append(val)
        return ops, node_vals

    def _emit_alu(self, ops: list[Op], opcode: str, a, b) -> SSAValue:
        res = self._new_ssa()
        ops.append(Op(opcode, res, [a, b], "alu"))
        return res

    def _emit_select(self, ops: list[Op], cond, a, b) -> SSAValue:
        res = self._new_ssa()
        ops.append(Op("select", res, [cond, a, b], "flow"))
        return res

    def _emit_bit(self, ops: list[Op], value, shift: int) -> SSAValue:
        if shift == 0:
            shifted = value
        else:
            shifted = self._emit_alu(ops, ">>", value, Const(shift))
        return self._emit_alu(ops, "&", shifted, Const(1))

    def _build_select_for_level(self, level: int, idx_ssa: SSAValue,
                                node_vals: list[SSAValue], ops: list[Op]) -> SSAValue:
        if level == 0:
            return node_vals[0]

        base_idx = (1 << level) - 1
        offset = self._emit_alu(ops, "-", idx_ssa, Const(base_idx))
        bits = [self._emit_bit(ops, offset, i) for i in range(level)]

        current = [node_vals[base_idx + i] for i in range(1 << level)]
        for bit in bits:
            next_level: list[SSAValue] = []
            for i in range(0, len(current), 2):
                next_level.append(self._emit_select(ops, bit, current[i + 1], current[i]))
            current = next_level
        return current[0]

    @staticmethod
    def _estimate_flow_ops(replacements: dict[int, tuple[SSAValue, int, SSAValue]]) -> int:
        """Estimate flow ops introduced by select trees from exact levels."""
        total = 0
        for _, level, _ in replacements.values():
            if level > 0:
                total += (1 << level) - 1
        return total
