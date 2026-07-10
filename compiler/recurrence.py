"""
Recurrence analysis for flat SSA bodies.

Provides a small symbolic affine evaluator and chain-recurrence discovery
used by strength-reduction passes (see compiler/passes/slsr.py).

An affine expression is a linear combination of atoms plus a constant:

    expr = sum(coeff_i * atom_i) + const        (all arithmetic mod 2**32)

where atoms are SSA values whose definitions the evaluator does not expand
(loads, selects, non-affine ops, values in an explicit stop set, or values
beyond the expansion budget).

A chain link is a value of the form

    y = A * x + s * v + C        (A >= 2 const, s in {+1, -1}, v optional)

discovered by matching the affine form of y: exactly one atom with |coeff|
>= 2 (the chain variable x) and at most one atom with coeff +-1 (the step
variable v). Links whose x is itself a link form chains; strength-reduction
passes can then apply a change of variable along the chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .hir import SSAValue, Const, Value, Op, Statement
from .use_def import UseDefContext

_M = 0xFFFFFFFF


@dataclass
class AffineExpr:
    """terms[atom] * atom + const, mod 2**32. None coeffs never stored."""
    terms: dict[SSAValue, int]
    const: int
    # ids of Ops the expansion walked through (the expression's interior)
    interior_op_ids: set[int] = field(default_factory=set)


@dataclass
class ChainLink:
    """y = A * x + sign * v + C  (v may be None)."""
    y: SSAValue
    x: SSAValue
    a: int
    v: Optional[SSAValue]
    sign: int              # +1 or -1 (sign of v's coefficient)
    c: int
    pos: int               # statement index of y's definition
    interior_op_ids: set[int]
    # When this link was produced by merging a zero-C partial link
    # (y = partial + C), the partial's value: the caller retires the
    # covered partial from its link set so the two do not coexist (a
    # mixed zero-C/full-C member list breaks fixpoint root solving).
    covered: Optional[SSAValue] = None


class RecurrenceAnalysis:
    """Affine evaluation and chain discovery over a flat SSA body."""

    def __init__(self, body: list[Statement], use_def: UseDefContext,
                 max_terms: int = 6, max_depth: int = 8):
        self._use_def = use_def
        self._max_terms = max_terms
        self._max_depth = max_depth
        self._def_op: dict[SSAValue, tuple[int, Op]] = {}
        for pos, stmt in enumerate(body):
            if isinstance(stmt, Op) and stmt.result is not None:
                self._def_op[stmt.result] = (pos, stmt)

    def def_position(self, ssa: SSAValue) -> Optional[int]:
        entry = self._def_op.get(ssa)
        return entry[0] if entry is not None else None

    def affine_of(self, value: Value,
                  stop: Optional[set[SSAValue]] = None,
                  _depth: int = 0) -> Optional[AffineExpr]:
        """Affine form of `value`, or None when the expansion overflows the
        term budget. Atoms in `stop` are never expanded.

        Not memoized across calls with different stop sets; callers that
        discover links in program order pass their growing stop set so
        already-recognized chain values stay atomic.
        """
        if isinstance(value, Const):
            return AffineExpr({}, value.value & _M)
        if not isinstance(value, SSAValue):
            return None

        if _depth < self._max_depth and (stop is None or value not in stop):
            entry = self._def_op.get(value)
            if entry is not None:
                pos, op = entry
                expanded = self._expand_op(op, stop, _depth)
                if expanded is not None:
                    expanded.interior_op_ids.add(id(op))
                    return expanded
        # Atomic
        return AffineExpr({value: 1}, 0)

    def _expand_op(self, op: Op, stop: Optional[set[SSAValue]],
                   depth: int) -> Optional[AffineExpr]:
        if op.result is None or len(op.operands) != 2:
            return None
        opcode = op.opcode
        a, b = op.operands

        if opcode in ("+", "-"):
            ea = self.affine_of(a, stop, depth + 1)
            eb = self.affine_of(b, stop, depth + 1)
            if ea is None or eb is None:
                return None
            return self._combine(ea, eb, 1 if opcode == "+" else -1)

        if opcode == "*":
            const_side, expr_side = None, None
            if isinstance(b, Const):
                const_side, expr_side = b.value & _M, a
            elif isinstance(a, Const):
                const_side, expr_side = a.value & _M, b
            if const_side is None:
                return None
            e = self.affine_of(expr_side, stop, depth + 1)
            if e is None:
                return None
            return self._scale(e, const_side)

        if opcode == "<<":
            # Shifts are not commutative: amount must be the right operand
            if not isinstance(b, Const) or not (0 <= b.value < 32):
                return None
            e = self.affine_of(a, stop, depth + 1)
            if e is None:
                return None
            return self._scale(e, (1 << b.value) & _M)

        return None

    def _combine(self, ea: AffineExpr, eb: AffineExpr, sign: int) -> Optional[AffineExpr]:
        terms = dict(ea.terms)
        for atom, coeff in eb.terms.items():
            new = (terms.get(atom, 0) + sign * coeff) % (1 << 32)
            if new == 0:
                terms.pop(atom, None)
            else:
                terms[atom] = new
        if len(terms) > self._max_terms:
            return None
        return AffineExpr(terms, (ea.const + sign * eb.const) & _M,
                          ea.interior_op_ids | eb.interior_op_ids)

    @staticmethod
    def _scale(e: AffineExpr, factor: int) -> AffineExpr:
        terms = {}
        for atom, coeff in e.terms.items():
            new = (coeff * factor) % (1 << 32)
            if new != 0:
                terms[atom] = new
        return AffineExpr(terms, (e.const * factor) & _M,
                          set(e.interior_op_ids))

    @staticmethod
    def _signed(coeff: int) -> int:
        """Interpret a mod-2**32 coefficient as a small signed integer."""
        return coeff - (1 << 32) if coeff > (1 << 31) else coeff

    def match_link(self, op: Op, pos: int,
                   stop: set[SSAValue],
                   known_links: Optional[dict[SSAValue, "ChainLink"]] = None,
                   ) -> Optional[ChainLink]:
        """Try to view `op` as a chain link y = A*x + s*v + C.

        `stop` should contain the results of already-discovered links so
        chains stay link-by-link instead of being flattened into one big
        affine expression.

        When the affine form is a single coeff-1 atom L plus a constant and
        L is a known zero-C link (a greedily frozen partial expression like
        t = 3*x + v seen from y = t + 5), the two are merged into the full
        link y = A*x + s*v + C so a constant added on top of a chain value
        is still discovered regardless of association order.
        """
        if op.result is None:
            return None
        # Only additive tops can be links. Pure scalings (*, <<) must stay
        # expandable interiors: matching them as degenerate links would put
        # them in the discovery stop set and cut the real chain's expansion
        # (and a C == 0 scaling link saves nothing anyway).
        if op.opcode not in ("+", "-"):
            return None
        expr = self.affine_of(op.result, stop)
        if expr is None or id(op) not in expr.interior_op_ids:
            return None

        # Merge case: {L: 1} + C where L is a known C == 0 link
        if (known_links is not None and expr.const != 0
                and len(expr.terms) == 1):
            (atom, coeff), = expr.terms.items()
            inner = known_links.get(atom)
            if (self._signed(coeff) == 1 and inner is not None
                    and inner.c == 0):
                return ChainLink(
                    y=op.result, x=inner.x, a=inner.a, v=inner.v,
                    sign=inner.sign, c=expr.const, pos=pos,
                    interior_op_ids=(set(expr.interior_op_ids)
                                     | set(inner.interior_op_ids)),
                    covered=atom)

        x = None
        a_coeff = None
        v = None
        sign = 1
        for atom, coeff in expr.terms.items():
            s = self._signed(coeff)
            if abs(s) >= 2:
                if x is not None:
                    return None  # more than one candidate chain variable
                x, a_coeff = atom, s
            elif s in (1, -1):
                if v is not None:
                    return None  # ambiguous step variable
                v, sign = atom, s
            else:
                return None
        if x is None or a_coeff < 2:
            # Negative A or no chain variable: not a supported recurrence
            return None
        return ChainLink(y=op.result, x=x, a=a_coeff, v=v, sign=sign,
                         c=expr.const, pos=pos,
                         interior_op_ids=set(expr.interior_op_ids))


def find_chain_links(body: list[Statement],
                     use_def: UseDefContext,
                     max_terms: int = 6,
                     max_depth: int = 8) -> dict[SSAValue, ChainLink]:
    """Discover chain links in program order.

    Expansion stops at previously discovered link results, so a chain is
    represented link by link even when intermediate links are single-use
    (a full expansion would otherwise flatten the whole chain into one
    affine expression with large coefficients).
    """
    analysis = RecurrenceAnalysis(body, use_def,
                                  max_terms=max_terms, max_depth=max_depth)
    links: dict[SSAValue, ChainLink] = {}
    stop: set[SSAValue] = set()
    for pos, stmt in enumerate(body):
        if not isinstance(stmt, Op):
            continue
        link = analysis.match_link(stmt, pos, stop, known_links=links)
        if link is not None:
            links[link.y] = link
            stop.add(link.y)
            if link.covered is not None:
                # Retire the merged partial: it stays in the stop set (so
                # later expansions still treat it as a boundary) but is no
                # longer a chain member itself.
                links.pop(link.covered, None)
    return links
