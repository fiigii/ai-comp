"""Promote statically addressed local-memory regions to scalar SSA values."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from vm import VLEN

from ..hir import (
    Const,
    ForLoop,
    HIRFunction,
    If,
    Op,
    SSAValue,
    Statement,
    Value,
    Variable,
    VectorSSAValue,
)
from ..local_memory import (
    LocalAddressRelation,
    LocalPointerProvenance,
    StaticLocalMemoryRegion,
    WORD_MASK,
    collect_local_memory_markers,
    is_local_memory_marker,
    parse_static_local_memory_marker,
)
from ..pass_manager import Pass, PassConfig, count_statements
from ..range_analysis import RangeAnalysis
from ..use_def import UseDefContext


_SCALAR_MEMORY_OPS = frozenset(("load", "store"))
_VECTOR_MEMORY_OPS = frozenset(("vload", "vstore", "vgather"))
_ALL_MEMORY_OPS = _SCALAR_MEMORY_OPS | _VECTOR_MEMORY_OPS


@dataclass(frozen=True)
class _Occurrence:
    statement: Statement
    top_level_index: int
    control_flow_depth: int


@dataclass
class _Candidate:
    region: StaticLocalMemoryRegion
    tainted: set[Variable]
    provenance: LocalPointerProvenance
    accesses: dict[int, int] = field(default_factory=dict)
    legal: bool = True
    rejection_reason: Optional[str] = None


class LocalMem2RegPass(Pass):
    """SROA + mem2reg for language-declared local-memory regions.

    The first implementation intentionally handles only flat scalar accesses.
    Local-memory accesses inside retained ``If`` or ``ForLoop`` statements are
    not supported; the entire region is left unpromoted because this pass does
    not construct phi values for control-flow merges or loop-carried state.
    A region is promoted atomically: every access that may touch the region
    after its marker must be a scalar load/store at a statically normalized
    in-range offset.  Otherwise all accesses for that region are preserved.
    """

    @property
    def name(self) -> str:
        return "local-mem2reg"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        assert self._metrics is not None
        self._metrics.ir_size_before = count_statements(hir)
        self._metrics.ssa_count_before = hir.num_ssa_values

        if not config.enabled:
            self._metrics.ir_size_after = self._metrics.ir_size_before
            self._metrics.ssa_count_after = hir.num_ssa_values
            return hir

        markers = collect_local_memory_markers(hir)
        if not markers:
            self._metrics.ir_size_after = self._metrics.ir_size_before
            self._metrics.ssa_count_after = hir.num_ssa_values
            self._metrics.custom = {
                "regions_seen": 0,
                "regions_promoted": 0,
                "regions_rejected": 0,
                "markers_removed": 0,
                "loads_promoted": 0,
                "stores_removed": 0,
                "rejection_reasons": {},
            }
            return hir

        static_regions = [
            region
            for marker in markers
            if (region := parse_static_local_memory_marker(marker)) is not None
        ]
        occurrences = list(self._iter_occurrences(hir))
        use_def = UseDefContext(hir)

        # Shared lazily-built value ranges: only programs with dynamic
        # derived addresses pay for the analysis (it proves accesses that
        # lie entirely outside the region, so they need not reject it).
        ranges_box: list[Optional[RangeAnalysis]] = [None]

        def ranges_provider() -> RangeAnalysis:
            if ranges_box[0] is None:
                ranges_box[0] = RangeAnalysis(hir)
            return ranges_box[0]

        candidates: list[_Candidate] = []
        for region in static_regions:
            # The first version is deliberately flat.  A marker under retained
            # control flow has path-dependent initialization semantics.
            if region.marker.control_flow_depth != 0:
                continue
            candidate = _Candidate(
                region=region,
                tainted=self._compute_pointer_taint(region.base, use_def),
                provenance=LocalPointerProvenance(
                    hir, region.base, use_def,
                    ranges_provider=ranges_provider,
                ),
            )
            self._check_legality(occurrences, candidate)
            candidates.append(candidate)

        self._reject_overlapping_access_plans(candidates)

        legal = [candidate for candidate in candidates if candidate.legal]
        rejected_static = len(static_regions) - len(legal)
        invalid_markers = len(markers) - len(static_regions)

        rewritten, loads_promoted, stores_removed = self._rewrite(hir, legal)

        self._metrics.ir_size_after = count_statements(rewritten)
        self._metrics.ssa_count_after = rewritten.num_ssa_values
        reasons: dict[str, int] = {}
        for candidate in candidates:
            if candidate.legal:
                continue
            reason = candidate.rejection_reason or "unknown"
            reasons[reason] = reasons.get(reason, 0) + 1
        if invalid_markers:
            reasons["invalid_marker"] = invalid_markers
        nested_regions = sum(
            region.marker.control_flow_depth != 0 for region in static_regions
        )
        if nested_regions:
            reasons["marker_in_control_flow"] = nested_regions

        self._metrics.custom = {
            "regions_seen": len(markers),
            "regions_promoted": len(legal),
            "regions_rejected": rejected_static + invalid_markers,
            "markers_removed": len(markers),
            "loads_promoted": loads_promoted,
            "stores_removed": stores_removed,
            "rejection_reasons": reasons,
        }
        return rewritten

    @staticmethod
    def _iter_occurrences(hir: HIRFunction) -> Iterator[_Occurrence]:
        def walk(
            body: list[Statement], top_level_index: int, depth: int
        ) -> Iterator[_Occurrence]:
            for stmt in body:
                yield _Occurrence(stmt, top_level_index, depth)
                if isinstance(stmt, ForLoop):
                    yield from walk(stmt.body, top_level_index, depth + 1)
                elif isinstance(stmt, If):
                    yield from walk(stmt.then_body, top_level_index, depth + 1)
                    yield from walk(stmt.else_body, top_level_index, depth + 1)

        for index, stmt in enumerate(hir.body):
            yield _Occurrence(stmt, index, 0)
            if isinstance(stmt, ForLoop):
                yield from walk(stmt.body, index, 1)
            elif isinstance(stmt, If):
                yield from walk(stmt.then_body, index, 1)
                yield from walk(stmt.else_body, index, 1)

    @staticmethod
    def _compute_pointer_taint(
        base: SSAValue,
        use_def: UseDefContext,
    ) -> set[Variable]:
        """Find values derived from the marker's exact SSA base.

        Provenance starts at the marker's SSA base, not at AliasAnalysis's
        canonical memory-slot root. A later reload from the same pointer slot
        is a new value because that slot may have been overwritten.
        """

        tainted: set[Variable] = {base}
        pending: list[Variable] = [base]

        def add(value: Variable) -> None:
            if value not in tainted:
                tainted.add(value)
                pending.append(value)

        while pending:
            value = pending.pop()
            for use in use_def.get_uses(value):
                stmt = use.statement
                if isinstance(stmt, Op):
                    # A loaded value is data, not an address derived from the
                    # address used to load it.
                    if stmt.result is None or stmt.opcode in _ALL_MEMORY_OPS:
                        continue
                    add(stmt.result)
                elif isinstance(stmt, ForLoop):
                    if use.use_kind in ("start", "end"):
                        # The induction variable is initialized from start and
                        # controlled by end. Loop results are also control-
                        # dependent on the trip count, so conservatively keep
                        # both paths tainted even when the loop precedes the
                        # local-memory marker.
                        add(stmt.counter)
                        for result in stmt.results:
                            add(result)
                        continue
                    if use.use_kind not in ("iter_arg", "yield"):
                        continue
                    index = use.operand_index
                    if 0 <= index < len(stmt.body_params):
                        add(stmt.body_params[index])
                    if 0 <= index < len(stmt.results):
                        add(stmt.results[index])
                elif isinstance(stmt, If) and use.use_kind in (
                    "then_yield", "else_yield"
                ):
                    index = use.operand_index
                    if 0 <= index < len(stmt.results):
                        add(stmt.results[index])
        return tainted

    @staticmethod
    def _address_relation(
        address: Value,
        candidate: _Candidate,
    ) -> LocalAddressRelation:
        relation = candidate.provenance.classify(address)
        if relation.kind == "unrelated" and isinstance(
            address, (SSAValue, VectorSSAValue)
        ) and address in candidate.tainted:
            return LocalAddressRelation("dynamic")
        return relation

    @staticmethod
    def _reject(candidate: _Candidate, reason: str) -> None:
        candidate.legal = False
        candidate.rejection_reason = reason

    @classmethod
    def _reject_overlapping_access_plans(
        cls, candidates: list[_Candidate]
    ) -> None:
        """Reject regions that claim the same memory operation.

        Valid local-memory contracts are disjoint. Detecting a shared access
        also keeps rewrite deterministic for malformed or overlapping markers.
        """

        owner: dict[int, _Candidate] = {}
        conflicted: set[int] = set()
        for candidate in candidates:
            if not candidate.legal:
                continue
            for statement_id in candidate.accesses:
                previous = owner.setdefault(statement_id, candidate)
                if previous is not candidate:
                    conflicted.add(id(previous))
                    conflicted.add(id(candidate))

        for candidate in candidates:
            if candidate.legal and id(candidate) in conflicted:
                cls._reject(candidate, "overlapping_regions")

    def _check_legality(
        self,
        occurrences: list[_Occurrence],
        candidate: _Candidate,
    ) -> None:
        marker_index = candidate.region.marker.top_level_index
        length = candidate.region.length

        for occurrence in occurrences:
            if occurrence.top_level_index <= marker_index:
                continue
            stmt = occurrence.statement

            if isinstance(stmt, ForLoop):
                special_values = (
                    [stmt.start, stmt.end]
                    + list(stmt.iter_args)
                    + list(stmt.yields)
                )
                if any(value in candidate.tainted for value in special_values):
                    self._reject(candidate, "tainted_control_flow")
                    return
                continue
            if isinstance(stmt, If):
                special_values = (
                    [stmt.cond]
                    + list(stmt.then_yields)
                    + list(stmt.else_yields)
                )
                if any(value in candidate.tainted for value in special_values):
                    self._reject(candidate, "tainted_control_flow")
                    return
                continue
            if not isinstance(stmt, Op) or is_local_memory_marker(stmt):
                continue

            tainted_operands = [
                index for index, operand in enumerate(stmt.operands)
                if isinstance(operand, (SSAValue, VectorSSAValue))
                and operand in candidate.tainted
            ]

            if stmt.opcode in _ALL_MEMORY_OPS:
                if not stmt.operands:
                    self._reject(candidate, "malformed_memory_op")
                    return

                # Storing an address-derived value lets the region pointer
                # escape, irrespective of the destination address.
                if stmt.opcode in ("store", "vstore") and 1 in tainted_operands:
                    self._reject(candidate, "tainted_escape")
                    return

                relation = self._address_relation(
                    stmt.operands[0], candidate
                )
                if relation.kind == "unrelated":
                    continue
                if relation.kind == "dynamic":
                    # A range-bounded dynamic offset that lies entirely
                    # outside the region is an ordinary access to other
                    # memory: preserve it, like static out-of-range ones.
                    width = (VLEN if stmt.opcode in _VECTOR_MEMORY_OPS
                             else 1)
                    if (relation.offset_range is not None
                            and relation.offset_range[0] >= length
                            and relation.offset_range[1] + width - 1
                                <= WORD_MASK):
                        continue
                    self._reject(candidate, "dynamic_address")
                    return

                assert relation.offset is not None
                offset = relation.offset
                if stmt.opcode in _SCALAR_MEMORY_OPS:
                    if 0 <= offset < length:
                        if occurrence.control_flow_depth != 0:
                            # Retained If/ForLoop local accesses require phi
                            # values, which this straight-line pass does not
                            # construct. Reject the whole region atomically.
                            self._reject(candidate, "access_in_control_flow")
                            return
                        if stmt.opcode == "load" and not isinstance(
                            stmt.result, SSAValue
                        ):
                            self._reject(candidate, "malformed_scalar_load")
                            return
                        if stmt.opcode == "store" and len(stmt.operands) != 2:
                            self._reject(candidate, "malformed_scalar_store")
                            return
                        candidate.accesses[id(stmt)] = offset
                    # A statically out-of-range scalar access is outside the
                    # declared region and is deliberately preserved.
                    continue

                # Contiguous vector operations block only if their footprint
                # overlaps the region. Derived vector gathers are rejected as
                # dynamic above; a statically related gather is unsupported.
                if stmt.opcode == "vgather":
                    self._reject(candidate, "vector_access")
                    return
                if any(((offset + lane) & WORD_MASK) < length
                       for lane in range(VLEN)):
                    self._reject(candidate, "vector_access")
                    return
                continue

            if not tainted_operands:
                continue
            if stmt.opcode in ("+", "-"):
                # Address derivation is allowed.  A subsequent dynamic access
                # is rejected by the memory-op handling above.
                continue
            if stmt.opcode == "select":
                self._reject(candidate, "tainted_select")
                return
            self._reject(candidate, "tainted_escape")
            return

    @staticmethod
    def _resolve(value: Value, replacements: dict[Variable, Value]) -> Value:
        seen: set[Variable] = set()
        while isinstance(value, (SSAValue, VectorSSAValue)) and value in replacements:
            if value in seen:
                break
            seen.add(value)
            value = replacements[value]
        return value

    def _rewrite(
        self,
        hir: HIRFunction,
        legal: list[_Candidate],
    ) -> tuple[HIRFunction, int, int]:
        replacements: dict[Variable, Value] = {}
        states: dict[int, dict[int, Value]] = {
            id(candidate): {} for candidate in legal
        }
        access_plan: dict[int, tuple[_Candidate, int]] = {}
        for candidate in legal:
            for statement_id, offset in candidate.accesses.items():
                access_plan[statement_id] = (candidate, offset)
        loads_promoted = 0
        stores_removed = 0

        def rewrite_body(body: list[Statement]) -> list[Statement]:
            nonlocal loads_promoted, stores_removed
            result: list[Statement] = []
            for stmt in body:
                if is_local_memory_marker(stmt):
                    continue
                if isinstance(stmt, Op):
                    planned = access_plan.get(id(stmt))
                    if planned is not None:
                        candidate, offset = planned
                        state = states[id(candidate)]
                        if stmt.opcode == "load":
                            assert isinstance(stmt.result, SSAValue)
                            replacements[stmt.result] = state.get(
                                offset, Const(0)
                            )
                            loads_promoted += 1
                        else:
                            value = self._resolve(
                                stmt.operands[1], replacements
                            )
                            if isinstance(value, Const):
                                value = Const(value.value & WORD_MASK)
                            state[offset] = value
                            stores_removed += 1
                        continue
                    result.append(Op(
                        stmt.opcode,
                        stmt.result,
                        [self._resolve(value, replacements) for value in stmt.operands],
                        stmt.engine,
                    ))
                elif isinstance(stmt, ForLoop):
                    result.append(ForLoop(
                        counter=stmt.counter,
                        start=self._resolve(stmt.start, replacements),
                        end=self._resolve(stmt.end, replacements),
                        iter_args=[self._resolve(v, replacements) for v in stmt.iter_args],
                        body_params=stmt.body_params,
                        body=rewrite_body(stmt.body),
                        yields=[self._resolve(v, replacements) for v in stmt.yields],
                        results=stmt.results,
                        pragma_unroll=stmt.pragma_unroll,
                    ))
                elif isinstance(stmt, If):
                    result.append(If(
                        cond=self._resolve(stmt.cond, replacements),
                        then_body=rewrite_body(stmt.then_body),
                        then_yields=[
                            self._resolve(v, replacements) for v in stmt.then_yields
                        ],
                        else_body=rewrite_body(stmt.else_body),
                        else_yields=[
                            self._resolve(v, replacements) for v in stmt.else_yields
                        ],
                        results=stmt.results,
                    ))
                else:
                    result.append(stmt)
            return result

        return HIRFunction(
            name=hir.name,
            body=rewrite_body(hir.body),
            num_ssa_values=hir.num_ssa_values,
            num_vec_ssa_values=hir.num_vec_ssa_values,
        ), loads_promoted, stores_removed
