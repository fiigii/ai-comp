"""
HIR Load Elimination Pass

Eliminates redundant loads by forwarding values from a dominating store when
no intervening store may alias the loaded address.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..hir import (
    Value, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from ..alias_analysis import AliasAnalysis, AliasResult, AddrKey
from ..pass_manager import Pass, PassConfig
from ..use_def import UseDefContext
from problem import VLEN


@dataclass
class StoreInfo:
    addr_key: Optional[AddrKey]
    value: Value
    width: int
    opcode: str


class MemoryState:
    """Tracks stores seen so far for memory dependence analysis."""

    def __init__(self, alias: AliasAnalysis):
        self._alias = alias
        self._store_history: list[StoreInfo] = []
        # Keyed by (addr_key, width, opcode)
        self._last_store: dict[tuple[AddrKey, int, str], int] = {}

    def clone(self) -> "MemoryState":
        other = MemoryState(self._alias)
        other._store_history = list(self._store_history)
        other._last_store = dict(self._last_store)
        return other

    def reset(self) -> None:
        self._store_history.clear()
        self._last_store.clear()

    def record_store(self, addr_key: Optional[AddrKey], value: Value, width: int, opcode: str) -> None:
        info = StoreInfo(addr_key, value, width, opcode)
        self._store_history.append(info)
        if addr_key is not None:
            self._last_store[(addr_key, width, opcode)] = len(self._store_history) - 1

    def find_def(self, addr_key: Optional[AddrKey], width: int, load_opcode: str) -> Optional[StoreInfo]:
        if addr_key is None:
            return None

        store_opcode = "store" if load_opcode == "load" else "vstore"
        key = (addr_key, width, store_opcode)
        idx = self._last_store.get(key)
        if idx is None:
            return None

        # Ensure no intervening store may alias
        for store in self._store_history[idx + 1:]:
            alias = self._alias.alias_keys(addr_key, width, store.addr_key, store.width)
            if alias != AliasResult.NO_ALIAS:
                return None

        return self._store_history[idx]


class LoadElimPass(Pass):
    """
    Load elimination pass: forward values from dominating stores when safe.
    """

    def __init__(self):
        super().__init__()
        self._loads_analyzed = 0
        self._loads_eliminated = 0
        self._use_def_ctx: Optional[UseDefContext] = None
        self._alias: Optional[AliasAnalysis] = None

    @property
    def name(self) -> str:
        return "load-elim"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        self._loads_analyzed = 0
        self._loads_eliminated = 0

        if not config.enabled:
            return hir

        self._use_def_ctx = UseDefContext(hir)
        self._alias = AliasAnalysis(self._use_def_ctx)

        state = MemoryState(self._alias)
        new_body = self._transform_statements(hir.body, state)

        if self._metrics:
            self._metrics.custom = {
                "loads_analyzed": self._loads_analyzed,
                "loads_eliminated": self._loads_eliminated,
                "alias_queries": self._alias.alias_queries,
                "alias_cache_hits": self._alias.alias_cache_hits,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=hir.num_ssa_values,
            num_vec_ssa_values=hir.num_vec_ssa_values,
        )

    def _transform_statements(self, stmts: list[Statement], state: MemoryState) -> list[Statement]:
        result: list[Statement] = []

        for stmt in stmts:
            if isinstance(stmt, Op):
                transformed = self._transform_op(stmt, state)
                if transformed is not None:
                    result.append(transformed)
            elif isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt, state))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, state))
            else:
                # Halt, Pause
                result.append(stmt)

        return result

    def _transform_op(self, op: Op, state: MemoryState) -> Optional[Op]:
        assert self._alias is not None

        if op.opcode in ("store", "vstore"):
            addr_key = self._alias.normalize(op.operands[0])
            width = 1 if op.opcode == "store" else VLEN
            state.record_store(addr_key, op.operands[1], width, op.opcode)
            return op

        if op.opcode in ("load", "vload"):
            self._loads_analyzed += 1
            if op.result is None:
                return op

            addr_key = self._alias.normalize(op.operands[0])
            width = 1 if op.opcode == "load" else VLEN
            store = state.find_def(addr_key, width, op.opcode)
            if store is None:
                return op

            # Forward stored value
            self._loads_eliminated += 1
            self._use_def_ctx.replace_all_uses(op.result, store.value, auto_invalidate=False)
            return None

        # Other ops
        return op

    def _transform_for_loop(self, loop: ForLoop, state: MemoryState) -> ForLoop:
        # Analyze loop body with a fresh state (no cross-iteration forwarding)
        loop_state = MemoryState(self._alias)
        new_body = self._transform_statements(loop.body, loop_state)

        # Conservative: do not forward across loops
        state.reset()

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

    def _transform_if(self, if_stmt: If, state: MemoryState) -> If:
        # Analyze each branch independently
        then_state = state.clone()
        else_state = state.clone()

        new_then_body = self._transform_statements(if_stmt.then_body, then_state)
        new_else_body = self._transform_statements(if_stmt.else_body, else_state)

        # Conservative: do not forward across control flow merges
        state.reset()

        return If(
            cond=if_stmt.cond,
            then_body=new_then_body,
            then_yields=if_stmt.then_yields,
            else_body=new_else_body,
            else_yields=if_stmt.else_yields,
            results=if_stmt.results,
        )
