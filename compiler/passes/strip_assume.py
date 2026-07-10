"""Erase compile-time-only HIR assumptions after their last consumer."""

from __future__ import annotations

from ..hir import ForLoop, HIRFunction, If, Op, Statement
from ..pass_manager import Pass, PassConfig, count_statements


def is_hir_assume_op(stmt: Statement) -> bool:
    """Return whether *stmt* is a compile-time-only assumption."""

    return isinstance(stmt, Op) and stmt.engine == "meta"


class StripAssumePass(Pass):
    """Remove assumptions before the following DCE computes runtime liveness."""

    @property
    def name(self) -> str:
        return "strip-assume"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        assert self._metrics is not None
        self._metrics.ir_size_before = count_statements(hir)
        self._metrics.ssa_count_before = hir.num_ssa_values

        if not config.enabled:
            self._metrics.ir_size_after = self._metrics.ir_size_before
            self._metrics.ssa_count_after = hir.num_ssa_values
            return hir

        body, removed = self._strip_body(hir.body)
        rewritten = HIRFunction(
            name=hir.name,
            body=body,
            num_ssa_values=hir.num_ssa_values,
            num_vec_ssa_values=hir.num_vec_ssa_values,
        )
        self._metrics.ir_size_after = count_statements(rewritten)
        self._metrics.ssa_count_after = rewritten.num_ssa_values
        self._metrics.custom = {"assumptions_removed": removed}
        return rewritten

    @classmethod
    def _strip_body(
        cls, body: list[Statement]
    ) -> tuple[list[Statement], int]:
        result: list[Statement] = []
        removed = 0
        for stmt in body:
            if is_hir_assume_op(stmt):
                assert isinstance(stmt, Op)
                if stmt.result is not None:
                    raise ValueError("HIR assumptions cannot define SSA values")
                removed += 1
                continue
            if isinstance(stmt, ForLoop):
                nested, nested_removed = cls._strip_body(stmt.body)
                removed += nested_removed
                result.append(ForLoop(
                    counter=stmt.counter,
                    start=stmt.start,
                    end=stmt.end,
                    iter_args=stmt.iter_args,
                    body_params=stmt.body_params,
                    body=nested,
                    yields=stmt.yields,
                    results=stmt.results,
                    pragma_unroll=stmt.pragma_unroll,
                ))
                continue
            if isinstance(stmt, If):
                then_body, then_removed = cls._strip_body(stmt.then_body)
                else_body, else_removed = cls._strip_body(stmt.else_body)
                removed += then_removed + else_removed
                result.append(If(
                    cond=stmt.cond,
                    then_body=then_body,
                    then_yields=stmt.then_yields,
                    else_body=else_body,
                    else_yields=stmt.else_yields,
                    results=stmt.results,
                ))
                continue
            result.append(stmt)
        return result, removed
