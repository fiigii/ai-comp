"""
Dead Store Elimination (DSE) Pass

Eliminates stores that are overwritten by a later store to the same address
with no intervening load that may read the earlier store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..alias_analysis import AliasAnalysis, AliasResult, AddrKey
from ..hir import Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
from ..pass_manager import Pass, PassConfig
from ..use_def import UseDefContext
from problem import VLEN


@dataclass
class PendingStore:
    addr_key: Optional[AddrKey]
    width: int
    stmt_index: int
    used_by_load: bool
    unknown_epoch: int


class DSEState:
    def __init__(self, alias: AliasAnalysis):
        self._alias = alias
        self.pending: list[PendingStore] = []
        self.pending_by_key: dict[tuple[AddrKey, int, str], PendingStore] = {}
        self.pending_by_base: dict[object, list[PendingStore]] = {}
        self.unknown_load_epoch: int = 0

    def reset(self) -> None:
        self.pending.clear()
        self.pending_by_key.clear()
        self.pending_by_base.clear()
        self.unknown_load_epoch = 0


class DSEPass(Pass):
    """Dead store elimination using alias analysis."""

    def __init__(self):
        super().__init__()
        self._stores_analyzed = 0
        self._stores_eliminated = 0
        self._loads_seen = 0
        self._use_def_ctx: Optional[UseDefContext] = None
        self._alias: Optional[AliasAnalysis] = None

    @property
    def name(self) -> str:
        return "dse"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        self._stores_analyzed = 0
        self._stores_eliminated = 0
        self._loads_seen = 0

        if not config.enabled:
            return hir

        self._use_def_ctx = UseDefContext(hir)
        restrict_ptr = config.options.get("restrict_ptr", False)
        self._alias = AliasAnalysis(self._use_def_ctx, restrict_ptr=restrict_ptr)

        state = DSEState(self._alias)
        new_body = self._transform_statements(hir.body, state)

        if self._metrics:
            self._metrics.custom = {
                "stores_analyzed": self._stores_analyzed,
                "stores_eliminated": self._stores_eliminated,
                "loads_seen": self._loads_seen,
                "alias_queries": self._alias.alias_queries,
                "alias_cache_hits": self._alias.alias_cache_hits,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=hir.num_ssa_values,
            num_vec_ssa_values=hir.num_vec_ssa_values,
        )

    def _transform_statements(self, stmts: list[Statement], state: DSEState) -> list[Statement]:
        result: list[Optional[Statement]] = []

        for stmt in stmts:
            if isinstance(stmt, Op):
                transformed = self._transform_op(stmt, state, result)
                if transformed is not None:
                    result.append(transformed)
            elif isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt, state))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, state))
            else:
                result.append(stmt)

        return [s for s in result if s is not None]

    def _transform_op(self, op: Op, state: DSEState, result: list[Optional[Statement]]) -> Optional[Op]:
        assert self._alias is not None

        if op.opcode in ("load", "vload", "vgather"):
            self._loads_seen += 1
            addr_key = self._alias.normalize(op.operands[0])
            width = 1 if op.opcode == "load" else VLEN
            self._mark_load(state, addr_key, width)
            return op

        if op.opcode in ("store", "vstore"):
            self._stores_analyzed += 1
            addr_key = self._alias.normalize(op.operands[0])
            width = 1 if op.opcode == "store" else VLEN
            self._process_store(state, addr_key, width, op.opcode, result)
            return op

        return op

    def _mark_load(self, state: DSEState, addr_key: Optional[AddrKey], width: int) -> None:
        assert self._alias is not None

        if addr_key is None:
            # Unknown load may alias everything; avoid scanning by bumping epoch.
            state.unknown_load_epoch += 1
            return

        if width == 1:
            pending = state.pending_by_key.get((addr_key, 1, "store"))
            if pending is not None:
                pending.used_by_load = True
            return

        # Vector load: check overlapping ranges within the same base only.
        base_list = state.pending_by_base.get(addr_key.base, [])
        for pending in base_list:
            alias = self._alias.alias_keys(addr_key, width, pending.addr_key, pending.width)
            if alias != AliasResult.NO_ALIAS:
                pending.used_by_load = True

    def _process_store(self, state: DSEState, addr_key: Optional[AddrKey], width: int, opcode: str,
                       result: list[Optional[Statement]]) -> None:
        assert self._alias is not None
        if addr_key is not None:
            key = (addr_key, width, opcode)
            pending = state.pending_by_key.get(key)
            if pending is not None:
                used = pending.used_by_load or (pending.unknown_epoch != state.unknown_load_epoch)
                if not used:
                    result[pending.stmt_index] = None
                    self._stores_eliminated += 1
                # Remove from base list
                base_list = state.pending_by_base.get(addr_key.base)
                if base_list is not None:
                    try:
                        base_list.remove(pending)
                    except ValueError:
                        pass
                state.pending_by_key.pop(key, None)
                # Also remove from pending list (used for fallback only)
                try:
                    state.pending.remove(pending)
                except ValueError:
                    pass

        new_pending = PendingStore(addr_key, width, len(result), False, state.unknown_load_epoch)
        state.pending.append(new_pending)
        if addr_key is not None:
            state.pending_by_key[(addr_key, width, opcode)] = new_pending
            state.pending_by_base.setdefault(addr_key.base, []).append(new_pending)

    def _transform_for_loop(self, loop: ForLoop, state: DSEState) -> ForLoop:
        loop_state = DSEState(self._alias)
        new_body = self._transform_statements(loop.body, loop_state)

        # Conservative: do not eliminate across loops
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

    def _transform_if(self, if_stmt: If, state: DSEState) -> If:
        then_state = DSEState(self._alias)
        else_state = DSEState(self._alias)

        new_then_body = self._transform_statements(if_stmt.then_body, then_state)
        new_else_body = self._transform_statements(if_stmt.else_body, else_state)

        # Conservative: do not eliminate across control-flow merges
        state.reset()

        return If(
            cond=if_stmt.cond,
            then_body=new_then_body,
            then_yields=if_stmt.then_yields,
            else_body=new_else_body,
            else_yields=if_stmt.else_yields,
            results=if_stmt.results,
        )
