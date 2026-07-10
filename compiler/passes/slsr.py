"""
Straight-Line Strength Reduction (SLSR, HIR) Pass

Generalized change-of-variable strength reduction over chain recurrences

    y = A * x + s*v + C        (A >= 2 const, s in {+1,-1}, v optional)

discovered by the recurrence analysis (compiler/recurrence.py) rather than
by syntactic shape matching, so the pass is independent of how earlier
passes (parity rewrites, mul distribution) happen to spell the chain.

The rewrite tracks xt = x + k with per-link compensation constants

    k(y) = A*k(x) - C   (mod 2**32),  k(root) chosen per policy (below)

so each transformed link is  xt' = A*xt + s*v  -- the constant addend
disappears, saving one op per link with C != 0. Uses of a chain value are
rewired with the inverse compensation: +(o, y) becomes +(o - k, xt), where
(o - k) folds into the constant for Const o and is a hoisted op for SSA o,
cached globally by (operand, signed delta) so structurally identical
chains (e.g. unrolled parallel lanes) share it. A nonzero root offset is
materialized through the same cache (xt_root = x + k_root), so its cost is
accounted exactly like any other compensation op.

Root-offset policies evaluated globally, best total wins:
- k_root = 0: roots are reused as-is (free); per-link ks differ, so each
  compensated use position pays one shared op per (operand, k) pair.
  Wins when many parallel chains share operands and k sequences.
- k_root = fixpoint C/(A-1) (when the chain is uniform and the congruence
  k*(A-1) = C mod 2**32 is solvable): every link gets the same k, so all
  compensations collapse to one shared op, at the price of one root add
  per distinct root. Wins for long single chains.

Approval is a fixpoint iteration: a chain member whose value must stay
alive because a NON-approved candidate link still consumes it loses its
saving (the old computation survives); rejections cascade until stable.
Scheduling effects are not modeled statically; cycle-count regression
tests guard against rewrites that hurt the schedule.
"""

from __future__ import annotations

from math import gcd
from typing import Optional

from ..hir import SSAValue, Const, Value, Op, ForLoop, If, Statement, HIRFunction
from ..pass_manager import Pass, PassConfig
from ..recurrence import ChainLink, find_chain_links
from ..use_def import UseDefContext

_M = 0xFFFFFFFF
_MOD = 1 << 32


def _solve_fixpoint_k(members: list[ChainLink]) -> Optional[int]:
    """k with k = A_l*k - C_l (mod 2**32) for every link, or None.

    Each link contributes the congruence (A_l - 1)*k = C_l (mod 2**32),
    whose solution set (when solvable) is k = k_l (mod 2**32 / g_l) with
    g_l = gcd(A_l - 1, 2**32). The sets are intersected with CRT for
    non-coprime moduli, so a common solution is found even when no single
    link's smallest representative satisfies the others.
    """
    k, mod = 0, 1  # running solution set: k (mod mod)
    for link in members:
        a1 = (link.a - 1) & _M
        c = link.c & _M
        if a1 == 0:
            return None  # unreachable for A >= 2
        g = gcd(a1, _MOD)
        if c % g != 0:
            return None
        m_l = _MOD // g
        if m_l > 1:
            inv = pow((a1 // g) % m_l, -1, m_l)
            k_l = ((c // g) * inv) % m_l
        else:
            k_l = 0
        # Intersect {k mod mod} with {k_l mod m_l}
        g2 = gcd(mod, m_l)
        if (k_l - k) % g2 != 0:
            return None
        m2 = m_l // g2
        if m2 > 1:
            t = (((k_l - k) // g2) % m2) * pow((mod // g2) % m2, -1, m2) % m2
        else:
            t = 0
        lcm = mod // g2 * m_l
        k = (k + mod * t) % lcm
        mod = lcm
    # Safety: verify against every congruence
    for link in members:
        if (((link.a - 1) * k) - link.c) % _MOD != 0:
            return None
    return k


class SLSRPass(Pass):
    """Strength-reduce affine chain recurrences via change of variable."""

    def __init__(self):
        super().__init__()
        self._next_ssa_id = 0

    @property
    def name(self) -> str:
        return "slsr"

    def _new_ssa(self, name: Optional[str] = None) -> SSAValue:
        v = SSAValue(self._next_ssa_id, name)
        self._next_ssa_id += 1
        return v

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        if not config.enabled:
            return hir
        if any(isinstance(s, (ForLoop, If)) for s in hir.body):
            self._add_metric_message("non-flat HIR detected, skipping")
            return hir

        min_savings = int(config.options.get("min_savings", 1))
        max_terms = int(config.options.get("max_terms", 6))
        max_depth = int(config.options.get("max_depth", 8))

        self._next_ssa_id = hir.num_ssa_values
        use_def = UseDefContext(hir)

        links = find_chain_links(hir.body, use_def,
                                 max_terms=max_terms, max_depth=max_depth)
        empty_metrics = {"links_rewritten": 0, "chains": 0,
                         "links_found": len(links), "policy": "none"}
        if not links:
            if self._metrics:
                self._metrics.custom = empty_metrics
            return hir

        def_pos: dict[SSAValue, int] = {}
        for pos, stmt in enumerate(hir.body):
            if isinstance(stmt, Op) and stmt.result is not None:
                def_pos[stmt.result] = pos

        # Union of all candidate links' interior op ids: uses inside any
        # link's computation belong to old chains and are never rewired.
        all_interior: set[int] = set()
        for link in links.values():
            all_interior |= link.interior_op_ids

        op_by_id: dict[int, Op] = {
            id(stmt): stmt for stmt in hir.body if isinstance(stmt, Op)
        }

        # --- Rewritability -------------------------------------------------
        def is_dead_use(u: Op) -> bool:
            """Dead husks (e.g. address adds whose load was replaced by an
            earlier pass) are removed by the following DCE; they neither
            block a rewrite nor cost anything."""
            return u.result is not None and use_def.use_count(u.result) == 0

        def uses_ok(y: SSAValue) -> bool:
            for use in use_def.get_uses(y):
                u = use.statement
                if not isinstance(u, Op):
                    return False
                if id(u) in all_interior or is_dead_use(u):
                    continue  # old chain computation / dead husk
                if (u.opcode in ("+", "-") and len(u.operands) == 2
                        and y in u.operands):
                    other = u.operands[0] if u.operands[1] == y else u.operands[1]
                    if isinstance(other, (SSAValue, Const)) and other != y:
                        continue  # compensable use
                return False
            return True

        link_ok: dict[SSAValue, bool] = {}

        def is_rewritable(y: SSAValue, depth: int = 0) -> bool:
            if y in link_ok:
                return link_ok[y]
            if depth > 128:
                return False
            link = links.get(y)
            if link is None:
                link_ok[y] = False
                return False
            ok = uses_ok(y) and (link.x not in links
                                 or is_rewritable(link.x, depth + 1))
            link_ok[y] = ok
            return ok

        for y in links:
            is_rewritable(y)

        # --- Chain grouping -------------------------------------------------
        def chain_root(y: SSAValue) -> SSAValue:
            seen = set()
            while y in links and links[y].x in links and y not in seen:
                seen.add(y)
                y = links[y].x
            return y

        chains: dict[SSAValue, list[ChainLink]] = {}
        for y, link in links.items():
            if link_ok.get(y):
                chains.setdefault(chain_root(y), []).append(link)
        for members in chains.values():
            members.sort(key=lambda l: l.pos)

        if not chains:
            if self._metrics:
                self._metrics.custom = empty_metrics
            return hir

        # --- Cost model -----------------------------------------------------
        def chain_k_seq(members: list[ChainLink], k_root: int) -> dict[SSAValue, int]:
            ks: dict[SSAValue, int] = {}
            for l in members:
                k_prev = ks.get(l.x, k_root)
                ks[l.y] = (l.a * k_prev - l.c) & _M
            return ks

        def use_delta(u: Op, y: SSAValue, k: int) -> int:
            """Signed compensation delta applied to the other operand,
            matching the rewrite exactly: '+' uses o + (-k); '-' uses o + k."""
            if u.opcode == "+":
                return (-k) & _M
            return k & _M

        def chain_comp_pairs(members: list[ChainLink],
                             ks: dict[SSAValue, int],
                             root_value: Optional[SSAValue],
                             k_root: int) -> set[tuple]:
            """(operand, delta) pairs of hoisted ops this chain needs: one
            per compensated SSA use and, for a nonzero root offset, the
            root materialization (root_value, k_root)."""
            pairs: set[tuple] = set()
            if k_root != 0 and root_value is not None:
                pairs.add((root_value, k_root & _M))
            for l in members:
                k = ks[l.y]
                for use in use_def.get_uses(l.y):
                    u = use.statement
                    if (not isinstance(u, Op) or id(u) in all_interior
                            or is_dead_use(u)):
                        continue
                    if (u.opcode in ("+", "-") and len(u.operands) == 2
                            and l.y in u.operands):
                        other = u.operands[0] if u.operands[1] == l.y else u.operands[1]
                        if isinstance(other, SSAValue):
                            delta = use_delta(u, l.y, k)
                            if delta != 0:
                                pairs.add((other, delta))
            return pairs

        def emit_ops(link: ChainLink) -> int:
            """Ops the rewrite emits for one link (mul, plus add/sub if v)."""
            return 2 if link.v is not None else 1

        def compensable_uses(y: SSAValue):
            """Live, non-interior +/- uses of y with their other operand."""
            for use in use_def.get_uses(y):
                u = use.statement
                if (not isinstance(u, Op) or id(u) in all_interior
                        or is_dead_use(u)):
                    continue
                if (u.opcode in ("+", "-") and len(u.operands) == 2
                        and y in u.operands):
                    other = u.operands[0] if u.operands[1] == y else u.operands[1]
                    if isinstance(other, (SSAValue, Const)) and other != y:
                        yield u, other

        def compute_dying_ops(approved: set[SSAValue]) -> set[int]:
            """Op ids of OLD chain computations that actually die after
            rewriting the approved chains, by liveness fixpoint.

            Death candidates are the approved chains' interior ops. An op
            stays alive when its result is referenced by anything that
            survives: a non-dying op, a compensation op (which reads the
            `other` operand of a rewired use), the emitted new chain (which
            reads root values and step operands), or a non-approved
            candidate link's computation. Shared interior ops (e.g. a mul
            CSE-shared between an approved and a non-approved link) are
            handled naturally: the surviving consumer keeps them alive.
            """
            dying: set[int] = set()
            rewired: dict[int, set[SSAValue]] = {}  # use op id -> members it loses
            alive_values: set[SSAValue] = set()      # referenced by NEW ops
            for root in approved:
                members = chains[root]
                alive_values.add(members[0].x)       # root feeds the new chain
                for l in members:
                    dying |= l.interior_op_ids
                    if l.v is not None:
                        alive_values.add(l.v)        # step operand feeds new op
                    for u, other in compensable_uses(l.y):
                        rewired.setdefault(id(u), set()).add(l.y)
                        if isinstance(other, SSAValue):
                            alive_values.add(other)  # comp op reads it

            changed = True
            while changed:
                changed = False
                for oid in list(dying):
                    op = op_by_id.get(oid)
                    if op is None or op.result is None:
                        dying.discard(oid)
                        changed = True
                        continue
                    if op.result in alive_values:
                        dying.discard(oid)
                        changed = True
                        continue
                    alive = False
                    for use in use_def.get_uses(op.result):
                        u = use.statement
                        if not isinstance(u, Op):
                            alive = True   # store or other statement survives
                            break
                        uid = id(u)
                        if uid in dying or is_dead_use(u):
                            continue
                        if op.result in rewired.get(uid, ()):
                            continue       # this use is rewired to the new chain
                        alive = True
                        break
                    if alive:
                        dying.discard(oid)
                        changed = True
            return dying

        def chain_contribution(members: list[ChainLink],
                               dying: set[int]) -> int:
            """Net op-count contribution of rewriting this chain, excluding
            the (globally shared) hoisted compensation ops: old interior
            ops that die minus new ops emitted."""
            interior: set[int] = set()
            emitted = 0
            for l in members:
                interior |= l.interior_op_ids
                emitted += emit_ops(l)
            return len(interior & dying) - emitted

        def evaluate_policy(use_fixpoint: bool):
            """Jointly fixpoint-iterate chain approval and old-op liveness
            under a root-offset policy.

            Returns (approved_roots, k_root_of, ks_of, total_net).
            """
            k_root_of: dict[SSAValue, int] = {}
            ks_of: dict[SSAValue, dict[SSAValue, int]] = {}
            for root, members in chains.items():
                k_root = 0
                if use_fixpoint:
                    fp = _solve_fixpoint_k(members)
                    if fp is not None:
                        k_root = fp
                k_root_of[root] = k_root
                ks_of[root] = chain_k_seq(members, k_root)

            approved = set(chains.keys())
            while True:
                dying = compute_dying_ops(approved)
                dropped = [root for root in approved
                           if chain_contribution(chains[root], dying) < min_savings]
                if not dropped:
                    break
                approved.difference_update(dropped)

            dying = compute_dying_ops(approved)
            emitted = 0
            pairs: set[tuple] = set()
            for root in approved:
                members = chains[root]
                emitted += sum(emit_ops(l) for l in members)
                # root of the chain: the x of its earliest link
                root_val = members[0].x
                pairs |= chain_comp_pairs(members, ks_of[root], root_val,
                                          k_root_of[root])
            total_net = len(dying) - emitted - len(pairs)
            return approved, k_root_of, ks_of, total_net

        zero_res = evaluate_policy(use_fixpoint=False)
        fix_res = evaluate_policy(use_fixpoint=True)
        if fix_res[3] > zero_res[3]:
            approved_roots, k_root_of, ks_of, total_net = fix_res
            policy = "fixpoint"
        else:
            approved_roots, k_root_of, ks_of, total_net = zero_res
            policy = "zero"

        if not approved_roots or total_net < min_savings:
            if self._metrics:
                self._metrics.custom = dict(empty_metrics, policy=policy)
            return hir

        # --- Rewrite --------------------------------------------------------
        xt_of: dict[SSAValue, Value] = {}
        comp_cache: dict[tuple, SSAValue] = {}
        insertions: dict[int, list[Op]] = {}
        rewritten = 0

        def hoisted_add(o: SSAValue, delta: int, at_pos: int) -> SSAValue:
            """o + delta as a hoisted, globally cached op placed after both
            o's def and at_pos (never inside the leading load/const prefix,
            which SLP's entry-broadcast placement relies on)."""
            delta &= _M
            key = (o, delta)
            cached = comp_cache.get(key)
            if cached is not None:
                return cached
            comp = self._new_ssa("slsr_comp")
            pos = max(at_pos, def_pos.get(o, -1) + 1)
            insertions.setdefault(pos, []).append(
                Op("+", comp, [o, Const(delta)], "alu")
            )
            comp_cache[key] = comp
            return comp

        approved_links = [links[l.y]
                          for root in approved_roots for l in chains[root]]
        approved_ids = {l.y for l in approved_links}
        root_of_link: dict[SSAValue, SSAValue] = {}
        for root in approved_roots:
            for l in chains[root]:
                root_of_link[l.y] = root

        for link in sorted(approved_links, key=lambda l: l.pos):
            y = link.y
            root = root_of_link[y]
            ks = ks_of[root]
            x = link.x
            if x in xt_of:
                # x is a rewritten link of the same chain (a chain-member x
                # always belongs to this chain's group, so its k sequence
                # is consistent)
                xt_prev: Value = xt_of[x]
            else:
                # Root. Do NOT cache in xt_of keyed by the bare SSA: two
                # chains may share the root value with DIFFERENT k_root
                # (fixpoint policy), and the k=0 case needs no cache at
                # all. hoisted_add's own (value, delta) cache provides the
                # correct sharing for nonzero offsets.
                k_root = k_root_of[root]
                if k_root == 0:
                    xt_prev = x  # root reused as-is
                else:
                    xt_prev = hoisted_add(x, k_root, link.pos)
            k = ks[y]
            ops: list[Op] = []
            mul = self._new_ssa("slsr_mul")
            ops.append(Op("*", mul, [xt_prev, Const(link.a)], "alu"))
            if link.v is not None:
                xt = self._new_ssa("slsr_val")
                opcode = "+" if link.sign > 0 else "-"
                ops.append(Op(opcode, xt, [mul, link.v], "alu"))
            else:
                xt = mul
            xt_of[y] = xt

            # Rewire compensable uses (records may be stale after earlier
            # link rewrites: re-check the current operands first)
            for use in list(use_def.get_uses(y)):
                u = use.statement
                if (not isinstance(u, Op) or id(u) in all_interior
                        or is_dead_use(u)):
                    continue
                if u.opcode not in ("+", "-") or len(u.operands) != 2:
                    continue
                if y not in u.operands:
                    continue
                other = u.operands[0] if u.operands[1] == y else u.operands[1]
                if not isinstance(other, (SSAValue, Const)) or other == y:
                    continue
                delta = use_delta(u, y, k)
                if isinstance(other, Const):
                    new_other: Value = Const((other.value + delta) & _M)
                elif delta == 0:
                    new_other = other
                else:
                    new_other = hoisted_add(other, delta, link.pos)
                if u.opcode == "+":
                    u.operands = [new_other, xt]
                elif u.operands[0] == y:
                    u.operands = [xt, new_other]   # y - o -> xt - (o + k)
                else:
                    u.operands = [new_other, xt]   # o - y -> (o + k) - xt
            insertions.setdefault(link.pos, []).extend(ops)
            rewritten += 1

        if not rewritten:
            if self._metrics:
                self._metrics.custom = dict(empty_metrics, policy=policy)
            return hir

        new_body: list[Statement] = []
        for pos, stmt in enumerate(hir.body):
            if pos in insertions:
                new_body.extend(insertions[pos])
            new_body.append(stmt)

        if self._metrics:
            self._metrics.custom = {
                "links_rewritten": rewritten,
                "chains": len(approved_roots),
                "links_found": len(links),
                "policy": policy,
                "comp_ops": len(comp_cache),
                "net_savings": total_net,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=max(hir.num_ssa_values, self._next_ssa_id),
            num_vec_ssa_values=hir.num_vec_ssa_values,
        )
