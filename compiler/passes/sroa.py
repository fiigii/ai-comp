"""SROA (Scalar Replacement of Aggregates) for flat HIR.

One promotion pass organized as REGION QUALIFICATION x ACCESS REWRITING.

Qualification -- when can a region's values be tracked as SSA:

- LOCAL by contract (``assume_local_memory``): private, unobservable,
  zero-initialized. The pass owns every read AND write, so stores become
  state updates and disappear from memory entirely.
- READ-ONLY by proof: object-size analysis proves speculative reads in bounds,
  then alias analysis (under the program's declared restrict contract) refutes
  every store against the window, so reads can be materialized from a snapshot.

Rewriting -- how a qualified access is materialized:

- Constant offset: direct value substitution. Local regions read their
  tracked state (zero cost); read-only windows read a shared preload.
- Dynamic bounded offset (a select "table lookup"): the value becomes a
  select tree over the window, indexed by the bits of ``offset - lo``.
  Local regions use the CURRENT state snapshot as leaves (no loads at
  all); read-only windows use preloads (W loads amortized across uses,
  so the rewrite must strictly reduce load traffic).
- Dynamic stores (select scatter) are not implemented: a dynamic store
  still disqualifies its local region.

Select bits are recovered from the offset's affine form -- one boolean
atom per power-of-two coefficient, with provably-constant selects and
multipliers folded through by range refinement -- and fall back to
explicit shift/mask extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from vm import VLEN

from ..alias_analysis import AddrKey, AliasAnalysis, AliasResult
from ..hir import (
    Const,
    ForLoop,
    Halt,
    HIRFunction,
    If,
    Op,
    Pause,
    SSAValue,
    Statement,
    Value,
    Variable,
    VectorSSAValue,
    WORD_MASK,
)
from ..mir import MBundle
from ..object_size import ObjectSizeAnalysis
from ..local_memory import (
    StaticLocalMemoryRegion,
    collect_local_memory_markers,
    is_local_memory_marker,
    parse_static_local_memory_marker,
)
from ..pointer_provenance import AddressRelation, PointerProvenance
from ..pass_manager import Pass, PassConfig, count_statements
from ..range_analysis import RangeAnalysis
from ..recurrence import RecurrenceAnalysis
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
class _LocalRegion:
    """A contract-qualified (local) region and its planned accesses."""

    region: StaticLocalMemoryRegion
    tainted: set[Variable]
    provenance: PointerProvenance
    # id(stmt) -> constant offset for static loads/stores
    accesses: dict[int, int] = field(default_factory=dict)
    # id(stmt) -> (lo, width) for bounded dynamic loads
    dynamic_reads: dict[int, tuple[int, int]] = field(default_factory=dict)
    legal: bool = True
    rejection_reason: Optional[str] = None


class SROAPass(Pass):
    """Promote qualified memory regions to SSA values.

    The first implementation intentionally handles only flat scalar
    accesses. Local-memory accesses inside retained ``If``/``ForLoop``
    statements are not supported (no phi construction); such a region is
    left unpromoted atomically.
    """

    def __init__(self):
        super().__init__()
        self._next_ssa_id = 0

    @property
    def name(self) -> str:
        return "sroa"

    def _new_ssa(self, name: Optional[str] = None) -> SSAValue:
        value = SSAValue(self._next_ssa_id, name)
        self._next_ssa_id += 1
        return value

    # ==================================================================
    # Driver
    # ==================================================================

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        assert self._metrics is not None
        self._metrics.ir_size_before = count_statements(hir)
        self._metrics.ssa_count_before = hir.num_ssa_values

        if not config.enabled:
            self._metrics.ir_size_after = self._metrics.ir_size_before
            self._metrics.ssa_count_after = hir.num_ssa_values
            return hir

        self._next_ssa_id = hir.num_ssa_values

        hir, local_metrics = self._promote_local_regions(hir, config)

        window_metrics: dict = {}
        if config.options.get("table_promotion", False):
            hir, window_metrics = self._promote_readonly_windows(hir, config)

        self._metrics.ir_size_after = count_statements(hir)
        self._metrics.ssa_count_after = hir.num_ssa_values
        self._metrics.custom = {**local_metrics, **window_metrics}
        return hir

    # ==================================================================
    # Shared select engine (dynamic-offset rewriting)
    # ==================================================================

    @staticmethod
    def _affine_window_bits(
        value: Value,
        lo: int,
        depth: int,
        recurrence: RecurrenceAnalysis,
        ranges: RangeAnalysis,
        drop_atom: Optional[SSAValue] = None,
    ) -> Optional[list[Value]]:
        """Select bits of (value - drop_atom - lo), LSB first.

        Every non-point atom of the affine form must carry a distinct
        power-of-two coefficient below 2**depth and be provably boolean;
        point atoms fold into the constant, which must equal lo exactly.
        ``drop_atom`` (the region base of an address expression) must
        appear with coefficient exactly 1.
        """
        expr = recurrence.affine_of(value)
        if expr is None:
            return None
        const = expr.const
        bits: list[Value] = [Const(0)] * depth
        seen: set[int] = set()
        dropped = drop_atom is None
        for atom, coeff in expr.terms.items():
            if drop_atom is not None and atom == drop_atom:
                if coeff != 1:
                    return None
                dropped = True
                continue
            atom_lo, atom_hi = ranges.range_of(atom)
            if atom_lo == atom_hi:
                const = (const + coeff * atom_lo) & WORD_MASK
                continue
            if coeff == 0 or coeff & (coeff - 1):
                return None      # not a single power of two (or negative)
            bit_pos = coeff.bit_length() - 1
            if bit_pos >= depth or bit_pos in seen:
                return None
            if not (atom_lo >= 0 and atom_hi <= 1):
                return None
            seen.add(bit_pos)
            bits[bit_pos] = atom
        if not dropped or const != (lo & WORD_MASK):
            return None
        return bits

    def _fallback_window_bits(
        self,
        offset_of: Value,
        subtrahend: Value,
        depth: int,
        ops: list[Op],
    ) -> list[Value]:
        """Explicit shift/mask bit extraction of (offset_of - subtrahend)."""
        offset = self._new_ssa("sroa_off")
        ops.append(Op("-", offset, [offset_of, subtrahend], "alu"))
        bits: list[Value] = []
        current: Value = offset
        for k in range(depth):
            if k > 0:
                shifted = self._new_ssa("sroa_shr")
                ops.append(Op(">>", shifted, [offset, Const(k)], "alu"))
                current = shifted
            bit = self._new_ssa("sroa_bit")
            ops.append(Op("&", bit, [current, Const(1)], "alu"))
            bits.append(bit)
        return bits

    def _emit_select_tree(
        self,
        leaves: list[Value],
        bits: list[Value],
        ops: list[Op],
    ) -> tuple[Value, int]:
        """Reduce 2**depth leaves along the bits; returns (root, selects)."""
        current = list(leaves)
        selects = 0
        for bit in bits:
            nxt: list[Value] = []
            if isinstance(bit, Const):
                take = bit.value & 1
                for j in range(0, len(current), 2):
                    nxt.append(current[j + take])
            else:
                for j in range(0, len(current), 2):
                    sel = self._new_ssa("sroa_sel")
                    ops.append(Op("select", sel,
                                  [bit, current[j + 1], current[j]], "flow"))
                    selects += 1
                    nxt.append(sel)
            current = nxt
        return current[0], selects

    # ==================================================================
    # Phase A: local regions (contract-qualified, read/write)
    # ==================================================================

    def _promote_local_regions(
        self, hir: HIRFunction, config: PassConfig
    ) -> tuple[HIRFunction, dict]:
        max_window = int(config.options.get("max_window", 8))

        markers = collect_local_memory_markers(hir)
        if not markers:
            return hir, {
                "regions_seen": 0,
                "regions_promoted": 0,
                "regions_rejected": 0,
                "markers_removed": 0,
                "loads_promoted": 0,
                "stores_removed": 0,
                "dynamic_loads_promoted": 0,
                "rejection_reasons": {},
            }

        static_regions = [
            region
            for marker in markers
            if (region := parse_static_local_memory_marker(marker)) is not None
        ]
        occurrences = list(self._iter_occurrences(hir))
        use_def = UseDefContext(hir)

        # Shared lazily-built analyses: only programs with dynamic derived
        # addresses pay for them.
        lazy: dict = {}

        def ranges_provider() -> RangeAnalysis:
            if "ranges" not in lazy:
                lazy["ranges"] = RangeAnalysis(hir)
            return lazy["ranges"]

        def recurrence_provider() -> RecurrenceAnalysis:
            if "recurrence" not in lazy:
                lazy["recurrence"] = RecurrenceAnalysis(
                    hir.body, use_def, max_terms=8, max_depth=24,
                    ranges=ranges_provider())
            return lazy["recurrence"]

        candidates: list[_LocalRegion] = []
        for region in static_regions:
            # A marker under retained control flow has path-dependent
            # initialization semantics; the flat pass rejects it.
            if region.marker.control_flow_depth != 0:
                continue
            candidate = _LocalRegion(
                region=region,
                tainted=self._compute_pointer_taint(region.base, use_def),
                provenance=PointerProvenance(
                    hir, region.base, use_def,
                    ranges_provider=ranges_provider,
                ),
            )
            self._check_legality(occurrences, candidate, max_window)
            candidates.append(candidate)

        self._reject_overlapping_access_plans(candidates)

        legal = [candidate for candidate in candidates if candidate.legal]
        rejected_static = len(static_regions) - len(legal)
        invalid_markers = len(markers) - len(static_regions)

        rewritten, counters = self._rewrite_local(
            hir, legal, ranges_provider, recurrence_provider)

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

        return rewritten, {
            "regions_seen": len(markers),
            "regions_promoted": len(legal),
            "regions_rejected": rejected_static + invalid_markers,
            "markers_removed": len(markers),
            "loads_promoted": counters["loads"],
            "stores_removed": counters["stores"],
            "dynamic_loads_promoted": counters["dynamic_loads"],
            "rejection_reasons": reasons,
        }

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
                elif isinstance(stmt, If):
                    if use.use_kind == "condition":
                        # A branch on the pointer value can reconstruct it
                        # (if base then 1 else 0): every result is derived.
                        for result in stmt.results:
                            add(result)
                        continue
                    if use.use_kind in ("then_yield", "else_yield"):
                        index = use.operand_index
                        if 0 <= index < len(stmt.results):
                            add(stmt.results[index])
        return tainted

    @staticmethod
    def _address_relation(
        address: Value,
        candidate: _LocalRegion,
    ) -> AddressRelation:
        relation = candidate.provenance.classify(address)
        if relation.kind == "unrelated" and isinstance(
            address, (SSAValue, VectorSSAValue)
        ) and address in candidate.tainted:
            return AddressRelation("dynamic")
        return relation

    @staticmethod
    def _reject(candidate: _LocalRegion, reason: str) -> None:
        candidate.legal = False
        candidate.rejection_reason = reason

    @classmethod
    def _reject_overlapping_access_plans(
        cls, candidates: list[_LocalRegion]
    ) -> None:
        """Reject regions that claim the same memory operation.

        Valid local-memory contracts are disjoint. Detecting a shared access
        also keeps rewrite deterministic for malformed or overlapping markers.
        """

        owner: dict[int, _LocalRegion] = {}
        conflicted: set[int] = set()
        for candidate in candidates:
            if not candidate.legal:
                continue
            planned = list(candidate.accesses) + list(candidate.dynamic_reads)
            for statement_id in planned:
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
        candidate: _LocalRegion,
        max_window: int,
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
            if (not isinstance(stmt, Op)
                    or is_local_memory_marker(stmt)
                    or stmt.engine == "meta"):
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
                    width = (VLEN if stmt.opcode in _VECTOR_MEMORY_OPS
                             else 1)
                    # A range-bounded dynamic offset entirely outside the
                    # region is an ordinary access to other memory:
                    # preserve it, like static out-of-range ones.
                    if (relation.offset_range is not None
                            and relation.offset_range[0] >= length
                            and relation.offset_range[1] + width - 1
                                <= WORD_MASK):
                        continue
                    # A bounded dynamic scalar LOAD entirely inside the
                    # region is promotable as a select over the tracked
                    # state (the dynamic-read quadrant). Stores would need
                    # select scatter; they still reject.
                    if (stmt.opcode == "load"
                            and isinstance(stmt.result, SSAValue)
                            and occurrence.control_flow_depth == 0
                            and relation.offset_range is not None):
                        read_lo, read_hi = relation.offset_range
                        read_width = read_hi - read_lo + 1
                        if (read_lo >= 0 and read_hi < length
                                and read_width <= max_window):
                            candidate.dynamic_reads[id(stmt)] = (
                                read_lo, read_width)
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

    def _rewrite_local(
        self,
        hir: HIRFunction,
        legal: list[_LocalRegion],
        ranges_provider,
        recurrence_provider,
    ) -> tuple[HIRFunction, dict]:
        replacements: dict[Variable, Value] = {}
        states: dict[int, dict[int, Value]] = {
            id(candidate): {} for candidate in legal
        }
        access_plan: dict[int, tuple[_LocalRegion, int]] = {}
        dynamic_plan: dict[int, tuple[_LocalRegion, int, int]] = {}
        for candidate in legal:
            for statement_id, offset in candidate.accesses.items():
                access_plan[statement_id] = (candidate, offset)
            for statement_id, (lo, width) in candidate.dynamic_reads.items():
                dynamic_plan[statement_id] = (candidate, lo, width)
        counters = {"loads": 0, "stores": 0, "dynamic_loads": 0}

        def rewrite_dynamic_read(stmt: Op, candidate: _LocalRegion,
                                 lo: int, width: int) -> list[Op]:
            """Select over the CURRENT state snapshot (no memory access)."""
            state = states[id(candidate)]
            depth = (width - 1).bit_length()
            leaf_count = 1 << depth
            leaves = [
                state.get(lo + min(j, width - 1), Const(0))
                for j in range(leaf_count)
            ]
            ops: list[Op] = []
            if depth == 0:
                replacement: Value = leaves[0]
            else:
                address = self._resolve(stmt.operands[0], replacements)
                bits = self._affine_window_bits(
                    address, lo, depth,
                    recurrence_provider(), ranges_provider(),
                    drop_atom=candidate.region.base,
                )
                if bits is None:
                    relative = self._new_ssa("sroa_rel")
                    resolved_base = self._resolve(
                        candidate.region.base, replacements)
                    ops.append(Op("-", relative,
                                  [address, resolved_base], "alu"))
                    bits = self._fallback_window_bits(
                        relative, Const(lo & WORD_MASK), depth, ops)
                else:
                    # Affine atoms come from the pre-rewrite IR; any of them
                    # may itself have been promoted away by another region.
                    bits = [self._resolve(bit, replacements) for bit in bits]
                replacement, _ = self._emit_select_tree(leaves, bits, ops)
            assert isinstance(stmt.result, SSAValue)
            replacements[stmt.result] = replacement
            counters["dynamic_loads"] += 1
            return ops

        def rewrite_body(body: list[Statement]) -> list[Statement]:
            result: list[Statement] = []
            for stmt in body:
                if is_local_memory_marker(stmt):
                    if isinstance(stmt, Op) and stmt.result is not None:
                        # Preserve malformed metadata so StripAssume/lowering
                        # can diagnose the invalid SSA definition.
                        result.append(stmt)
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
                            counters["loads"] += 1
                        else:
                            value = self._resolve(
                                stmt.operands[1], replacements
                            )
                            if isinstance(value, Const):
                                value = Const(value.value & WORD_MASK)
                            state[offset] = value
                            counters["stores"] += 1
                        continue
                    dynamic = dynamic_plan.get(id(stmt))
                    if dynamic is not None:
                        result.extend(rewrite_dynamic_read(stmt, *dynamic))
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
            num_ssa_values=max(hir.num_ssa_values, self._next_ssa_id),
            num_vec_ssa_values=hir.num_vec_ssa_values,
        ), counters

    # ==================================================================
    # Phase B: read-only windows (proof-qualified, read promotion)
    # ==================================================================

    def _promote_readonly_windows(
        self, hir: HIRFunction, config: PassConfig
    ) -> tuple[HIRFunction, dict]:
        max_window = int(config.options.get("max_window", 8))
        share_window = int(config.options.get("share_window", 4))
        repreload_gap = int(config.options.get("repreload_gap", 10000))
        restrict_ptr = bool(config.options.get("restrict_ptr", False))

        stats = {
            "windows_promoted": 0,
            "window_loads_replaced": 0,
            "window_preloads": 0,
            "window_selects": 0,
            "window_bit_fallbacks": 0,
            "window_rejections": {},
        }

        def reject_window(reason: str) -> None:
            rejections = stats["window_rejections"]
            rejections[reason] = rejections.get(reason, 0) + 1

        if any(isinstance(s, (ForLoop, If)) for s in hir.body):
            return hir, stats

        use_def = UseDefContext(hir)
        ranges = RangeAnalysis(hir)
        alias = AliasAnalysis(use_def, restrict_ptr=restrict_ptr)
        object_sizes = ObjectSizeAnalysis(hir, use_def)
        recurrence = RecurrenceAnalysis(hir.body, use_def,
                                        max_terms=8, max_depth=24,
                                        ranges=ranges)

        # Memory-clobber barriers split the body into epochs: a pause hands
        # control to the host, which may rewrite memory, so a snapshot must
        # never be reused across one.
        epoch_of: list[int] = []
        epoch = 0
        for stmt in hir.body:
            epoch_of.append(epoch)
            if isinstance(stmt, (Pause, Halt)):
                epoch += 1

        # Preloading widens one dynamic access into a read of every window
        # slot. Width-one windows do not speculate a new address; wider ones
        # require a trusted memory-object extent from the frontend/ABI.

        # --- Discover candidate loads: load(base + i), range(i) small ---
        windows: dict[tuple, list] = {}
        stores: list[Op] = []
        for idx, stmt in enumerate(hir.body):
            if not isinstance(stmt, Op):
                continue
            if stmt.opcode in ("store", "vstore"):
                stores.append(stmt)
                continue
            if stmt.opcode != "load" or not isinstance(stmt.result, SSAValue):
                continue
            addr = stmt.operands[0]
            if not isinstance(addr, SSAValue):
                continue
            def_loc = use_def.get_def(addr)
            if def_loc is None or not isinstance(def_loc.statement, Op):
                continue
            addr_op = def_loc.statement
            if addr_op.opcode != "+" or len(addr_op.operands) != 2:
                continue
            choices: list[tuple[Value, Value, int, int]] = []
            for base, index in (addr_op.operands, addr_op.operands[::-1]):
                if not isinstance(base, (SSAValue, Const)):
                    continue
                lo, hi = ranges.range_of(index)
                width = hi - lo + 1
                if width > max_window:
                    continue
                choices.append((base, index, lo, width))
            if not choices:
                continue
            eligible = [
                choice for choice in choices
                if (choice[3] == 1
                    or object_sizes.contains_window(
                        choice[0], choice[2], choice[3]))
            ]
            if not eligible:
                reject_window("object_bounds")
                continue
            base, index, lo, width = eligible[0]
            windows.setdefault(
                (base, lo, width, epoch_of[idx]), []).append(
                (idx, stmt, index))

        if not windows:
            return hir, stats

        # --- Immutability: the window must be refutable against EVERY
        # store. Without the restrict contract, an unrefutable store
        # conservatively disqualifies the window (never the program).
        store_keys = [
            (alias.normalize(s.operands[0]),
             VLEN if s.opcode == "vstore" else 1)
            for s in stores
        ]

        def window_is_readonly(base: Value, lo: int, width: int) -> bool:
            base_key = alias.normalize(base)
            if base_key is None:
                return False
            window_key = AddrKey(base_key.base,
                                 (base_key.offset + lo) & WORD_MASK)
            for skey, swidth in store_keys:
                if alias.alias_keys(window_key, width,
                                    skey, swidth) != AliasResult.NO_ALIAS:
                    return False
            return True

        # --- Cost gate + legality ---
        # Wide windows are re-preloaded per use cluster (to keep live ranges
        # short), so the load-engine saving is uses - width * clusters; it
        # must be strictly positive for the rewrite to reduce load traffic.
        def cluster_starts(width: int, positions: list[int]) -> dict[int, int]:
            clusters: dict[int, int] = {}
            if width <= share_window:
                for pos in positions:
                    clusters[pos] = positions[0]
                return clusters
            start = positions[0]
            prev = positions[0]
            for pos in positions:
                if pos - prev > repreload_gap:
                    start = pos
                clusters[pos] = start
                prev = pos
            return clusters

        # Cost gate: promotion trades load-engine work for flow/alu work.
        # Track the block's per-engine lower bound and only accept windows
        # that never worsen it (the select trees land on the single-slot
        # flow engine; small blocks are easily made flow-bound).
        engine_work = {engine: 0 for engine in MBundle.SLOT_LIMITS}
        for stmt in hir.body:
            if isinstance(stmt, Op) and stmt.engine in engine_work:
                engine_work[stmt.engine] += 1

        def engine_bound(work: dict) -> int:
            return max(
                (amount + MBundle.SLOT_LIMITS[engine] - 1)
                // MBundle.SLOT_LIMITS[engine]
                for engine, amount in work.items()
            )

        baseline_bound = engine_bound(engine_work)
        # Integer ceilings make the max bound zero-tolerant: on a large
        # block, a handful of preload address adds on the binding engine
        # would outweigh hundreds of saved loads on a non-binding one.
        # Allow proportional slack (floored, so small blocks stay strict:
        # their measured select regressions must keep rejecting).
        slack_pct = int(config.options.get("window_bound_slack_pct", 1))
        bound_budget = baseline_bound + baseline_bound * slack_pct // 100
        # The gate runs before SLP: when a window has enough isomorphic
        # uses, its select trees vectorize into lane-wise vselects, so
        # their flow cost is charged at 1/VLEN. Small windows stay scalar
        # and pay full price (a handful of lookups cannot amortize the
        # single-slot flow engine).
        vector_uses = int(config.options.get("window_vector_uses", 4 * VLEN))
        use_bits: dict[int, Optional[list]] = {}

        def select_count(bits: list) -> int:
            depth = len(bits)
            return sum(
                1 << (depth - 1 - k)
                for k, bit in enumerate(bits)
                if not isinstance(bit, Const)
            )

        approved: list[tuple] = []
        for (base, lo, width, _epoch), uses in sorted(
                windows.items(), key=lambda kv: kv[1][0][0]):
            positions = sorted(u[0] for u in uses)
            clusters = cluster_starts(width, positions)
            n_clusters = len(set(clusters.values()))
            if len(uses) <= width * n_clusters:
                reject_window("amortization")
                continue
            if not window_is_readonly(base, lo, width):
                reject_window("may_write")
                continue

            depth = (width - 1).bit_length()
            delta_flow = 0
            delta_alu = width * n_clusters  # preload address adds
            window_bits: dict[int, Optional[list]] = {}
            for _, load_op, index in uses:
                if depth == 0:
                    window_bits[id(load_op)] = []
                    continue
                bits = self._affine_window_bits(
                    index, lo, depth, recurrence, ranges)
                window_bits[id(load_op)] = bits
                if bits is None:
                    delta_flow += (1 << depth) - 1
                    delta_alu += 2 * depth   # sub + shifts + masks
                else:
                    delta_flow += select_count(bits)
            if len(uses) >= vector_uses:
                delta_flow = (delta_flow + VLEN - 1) // VLEN
            candidate_work = dict(engine_work)
            candidate_work["load"] += width * n_clusters - len(uses)
            candidate_work["flow"] += delta_flow
            candidate_work["alu"] += delta_alu
            if engine_bound(candidate_work) > bound_budget:
                reject_window("engine_bound")
                continue

            engine_work = candidate_work
            use_bits.update(window_bits)
            approved.append((base, lo, width, uses, clusters))
        if not approved:
            return hir, stats

        # A window base that is itself one of the loads being replaced would
        # leave the emitted preloads referencing a deleted definition (a
        # chained indirection): drop such windows conservatively.
        all_replaced = {
            load_op.result
            for _, _, _, uses, _ in approved
            for _, load_op, _ in uses
        }
        approved = [w for w in approved if w[0] not in all_replaced]
        if not approved:
            return hir, stats

        insertions: dict[int, list[Op]] = {}
        replacements: dict[int, list[Op]] = {}
        replaced_ids: set[int] = set()
        # (base, absolute offset, cluster position) -> preloaded SSA
        preload_cache: dict[tuple, SSAValue] = {}

        def preload(base: Value, off: int, cluster_pos: int) -> SSAValue:
            key = (base, off & WORD_MASK, cluster_pos)
            cached = preload_cache.get(key)
            if cached is not None:
                return cached
            addr = self._new_ssa("win_addr")
            val = self._new_ssa("win_val")
            insertions.setdefault(cluster_pos, []).extend([
                Op("+", addr, [base, Const(off & WORD_MASK)], "alu"),
                Op("load", val, [addr], "load"),
            ])
            preload_cache[key] = val
            stats["window_preloads"] += 1
            return val

        for base, lo, width, uses, clusters in approved:
            depth = (width - 1).bit_length()
            for stmt_idx, load_op, index in uses:
                cluster_pos = clusters[stmt_idx]
                leaf_count = 1 << depth
                leaves: list[Value] = [
                    preload(base, lo + min(j, width - 1), cluster_pos)
                    for j in range(leaf_count)
                ]
                ops: list[Op] = []
                if depth == 0:
                    replacement: Value = leaves[0]
                else:
                    bits = use_bits[id(load_op)]
                    if bits is None:
                        bits = self._fallback_window_bits(
                            index, Const(lo & WORD_MASK), depth, ops)
                        stats["window_bit_fallbacks"] += 1
                    replacement, selects = self._emit_select_tree(
                        leaves, bits, ops)
                    stats["window_selects"] += selects
                use_def.replace_all_uses(load_op.result, replacement,
                                         auto_invalidate=False)
                replacements[stmt_idx] = ops
                replaced_ids.add(stmt_idx)
                stats["window_loads_replaced"] += 1
            stats["windows_promoted"] += 1

        if not replaced_ids:
            return hir, stats

        new_body: list[Statement] = []
        for idx, stmt in enumerate(hir.body):
            if idx in insertions:
                new_body.extend(insertions[idx])
            if idx in replaced_ids:
                new_body.extend(replacements[idx])
                continue
            new_body.append(stmt)

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=max(hir.num_ssa_values, self._next_ssa_id),
            num_vec_ssa_values=hir.num_vec_ssa_values,
        ), stats
