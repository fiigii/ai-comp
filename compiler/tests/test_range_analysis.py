"""Tests for the standalone RangeAnalysis (compiler/range_analysis.py)."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from compiler.hir import Const, ForLoop, If, Op, SSAValue
from compiler.hir_builder import HIRBuilder
from compiler.range_analysis import FULL_RANGE, RangeAnalysis
from compiler.tests.conftest import (
    DebugInfo,
    Machine,
    N_CORES,
    compile_hir_to_vliw,
)

_MASK = 0xFFFFFFFF

_INTERP_BINOPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
    "^": lambda a, b: a ^ b,
    "&": lambda a, b: a & b,
    "|": lambda a, b: a | b,
    "<<": lambda a, b: a << b,
    ">>": lambda a, b: a >> b,
    "<": lambda a, b: int(a < b),
    "==": lambda a, b: int(a == b),
}


def interpret_hir(hir, mem, record=None):
    """Reference interpreter for the scalar HIR subset.

    Mirrors the VM's semantics (every result reduced mod 2**32). When
    ``record`` is given, every runtime value each SSAValue takes is appended
    to it, giving a ground truth to check interval soundness against.
    """
    env = {}

    def val(v):
        if isinstance(v, Const):
            return v.value & _MASK
        return env[v]

    def assign(var, value):
        env[var] = value
        if record is not None and isinstance(var, SSAValue):
            record.setdefault(var, []).append(value)

    def run_body(body):
        for stmt in body:
            if isinstance(stmt, Op):
                if stmt.opcode == "load":
                    assign(stmt.result, mem[val(stmt.operands[0])] & _MASK)
                elif stmt.opcode == "store":
                    mem[val(stmt.operands[0])] = val(stmt.operands[1])
                elif stmt.opcode == "select":
                    cond, a, b = stmt.operands
                    assign(stmt.result, val(a) if val(cond) != 0 else val(b))
                elif stmt.opcode in _INTERP_BINOPS:
                    left, right = stmt.operands
                    result = _INTERP_BINOPS[stmt.opcode](val(left), val(right))
                    assign(stmt.result, result % (2 ** 32))
                else:
                    raise NotImplementedError(stmt.opcode)
            elif isinstance(stmt, ForLoop):
                counter = val(stmt.start)
                end = val(stmt.end)
                params = [val(a) for a in stmt.iter_args]
                while counter < end:
                    assign(stmt.counter, counter)
                    for param, pv in zip(stmt.body_params, params):
                        assign(param, pv)
                    run_body(stmt.body)
                    params = [val(y) for y in stmt.yields]
                    counter += 1
                for result, rv in zip(stmt.results, params):
                    assign(result, rv)
            elif isinstance(stmt, If):
                if val(stmt.cond) != 0:
                    run_body(stmt.then_body)
                    yields = stmt.then_yields
                else:
                    run_body(stmt.else_body)
                    yields = stmt.else_yields
                for result, y in zip(stmt.results, yields):
                    assign(result, val(y))
            # Pause/Halt: no effect on values

    run_body(hir.body)
    return env


class TestRangeAnalysis(unittest.TestCase):
    def test_const_and_unknown_values(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        b.store(b.const(1), x)
        ra = RangeAnalysis(b.build())

        self.assertEqual(ra.range_of(b.const(7)), (7, 7))
        self.assertEqual(ra.range_of(x), FULL_RANGE)
        self.assertEqual(ra.try_const(b.const(7)), 7)
        self.assertIsNone(ra.try_const(x))

    def test_const_wraps_to_32_bits(self):
        b = HIRBuilder()
        b.store(b.const(1), b.const(0))
        ra = RangeAnalysis(b.build())
        wrapped = (1 << 32) + 5
        self.assertEqual(ra.range_of(b.const(wrapped)), (5, 5))

    def test_masked_value_bounds_add_chain(self):
        # x & 7 is in [0, 7]; +1 shifts to [1, 8]; *2 to [2, 16].
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        masked = b.and_(x, b.const(7), "masked")
        plus = b.add(masked, b.const(1), "plus")
        doubled = b.mul(plus, b.const(2), "doubled")
        b.store(b.const(1), doubled)
        ra = RangeAnalysis(b.build())

        self.assertEqual(ra.range_of(masked), (0, 7))
        self.assertEqual(ra.range_of(plus), (1, 8))
        self.assertEqual(ra.range_of(doubled), (2, 16))

    def test_add_overflow_widens_to_full(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        big = b.add(x, b.const(1), "big")  # x may be 2**32-1: wraps
        b.store(b.const(1), big)
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(big), FULL_RANGE)

    def test_sub_underflow_widens_to_full(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        masked = b.and_(x, b.const(3), "masked")     # [0, 3]
        safe = b.sub(b.const(10), masked, "safe")    # [7, 10]
        unsafe = b.sub(masked, b.const(1), "unsafe")  # 0 - 1 wraps
        b.store(b.const(1), b.add(safe, unsafe))
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(safe), (7, 10))
        self.assertEqual(ra.range_of(unsafe), FULL_RANGE)

    def test_shift_div_mod_transfer(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        masked = b.and_(x, b.const(15), "masked")     # [0, 15]
        shl = b.shl(masked, b.const(2), "shl")        # [0, 60]
        shr = b.shr(masked, b.const(1), "shr")        # [0, 7]
        div = b.div(masked, b.const(4), "div")        # [0, 3]
        mod = b.mod(x, b.const(6), "mod")             # [0, 5]
        small_mod = b.mod(masked, b.const(100), "small_mod")  # stays [0, 15]
        b.store(b.const(1), b.add(shl, b.add(shr, b.add(div, b.add(mod, small_mod)))))
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(shl), (0, 60))
        self.assertEqual(ra.range_of(shr), (0, 7))
        self.assertEqual(ra.range_of(div), (0, 3))
        self.assertEqual(ra.range_of(mod), (0, 5))
        self.assertEqual(ra.range_of(small_mod), (0, 15))

    def test_bitwise_or_xor_bounds(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        a = b.and_(x, b.const(7), "a")     # [0, 7]
        c = b.and_(x, b.const(3), "c")     # [0, 3]
        orred = b.or_(a, c, "orred")       # [0, 7] by bit_length bound
        xored = b.xor(a, c, "xored")       # [0, 7]
        b.store(b.const(1), b.add(orred, xored))
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(orred), (0, 7))
        self.assertEqual(ra.range_of(xored), (0, 7))

    def test_comparisons_fold_to_points(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        small = b.and_(x, b.const(7), "small")            # [0, 7]
        lt_true = b.lt(small, b.const(8), "lt_true")      # (1, 1)
        lt_false = b.lt(b.const(9), b.const(5), "lt_false")
        lt_unknown = b.lt(small, b.const(4), "lt_unknown")
        b.store(b.const(1), b.add(lt_true, b.add(lt_false, lt_unknown)))
        ra = RangeAnalysis(b.build())

        self.assertEqual(ra.try_const(lt_true), 1)
        self.assertEqual(ra.try_const(lt_false), 0)
        self.assertEqual(ra.range_of(lt_unknown), (0, 1))

    def test_select_transfer(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        bit = b.and_(x, b.const(1), "bit")                 # [0, 1]
        merged = b.select(bit, b.const(10), b.const(20), "merged")
        taken = b.select(b.const(1), b.const(10), b.const(20), "taken")
        skipped = b.select(b.const(0), b.const(10), b.const(20), "skipped")
        b.store(b.const(1), b.add(merged, b.add(taken, skipped)))
        ra = RangeAnalysis(b.build())

        self.assertEqual(ra.range_of(merged), (10, 20))
        self.assertEqual(ra.try_const(taken), 10)
        self.assertEqual(ra.try_const(skipped), 20)

    def test_try_compare_helpers(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        small = b.and_(x, b.const(3), "small")   # [0, 3]
        offset = b.add(small, b.const(10), "offset")  # [10, 13]
        b.store(b.const(1), offset)
        ra = RangeAnalysis(b.build())

        self.assertIs(ra.try_compare_lt(small, offset), True)
        self.assertIs(ra.try_compare_lt(offset, small), False)
        self.assertIsNone(ra.try_compare_lt(small, b.const(2)))
        self.assertIs(ra.try_compare_eq(b.const(4), b.const(4)), True)
        self.assertIs(ra.try_compare_eq(small, b.const(9)), False)
        self.assertIsNone(ra.try_compare_eq(small, b.const(2)))

    def test_is_boolean(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        bit = b.and_(x, b.const(1), "bit")
        cmp_val = b.lt(x, b.const(5), "cmp_val")
        wide = b.and_(x, b.const(3), "wide")
        b.store(b.const(1), b.add(bit, b.add(cmp_val, wide)))
        ra = RangeAnalysis(b.build())

        self.assertTrue(ra.is_boolean(bit))
        self.assertTrue(ra.is_boolean(cmp_val))
        self.assertTrue(ra.is_boolean(b.const(1)))
        self.assertFalse(ra.is_boolean(wide))
        self.assertFalse(ra.is_boolean(x))

    def test_analysis_stays_available_with_control_flow(self):
        b = HIRBuilder()
        masked = b.and_(b.load(b.const(0), "x"), b.const(1), "masked")

        def body_fn(i, params):
            b.store(b.const(1), i)
            return []

        b.for_loop(b.const(0), b.const(4), [], body_fn)
        hir = b.build()

        ra = RangeAnalysis(hir)
        # Values outside the loop keep their straight-line precision.
        self.assertEqual(ra.range_of(masked), (0, 1))
        self.assertIs(ra.try_compare_lt(masked, b.const(2)), True)


class TestStructuredRangeAnalysis(unittest.TestCase):
    """Structured fixpoint: If joins, refinements, loop widening."""

    def test_if_results_join_both_branches(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        cond = b.lt(x, b.const(100), "cond")
        merged = b.if_stmt(
            cond,
            lambda: [b.const(10)],
            lambda: [b.const(20)],
        )[0]
        b.store(b.const(1), merged)
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(merged), (10, 20))

    def test_if_with_provable_condition_is_one_sided(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        small = b.and_(x, b.const(7), "small")            # [0, 7]
        cond = b.lt(small, b.const(8), "cond")            # provably 1
        merged = b.if_stmt(
            cond,
            lambda: [b.const(10)],
            lambda: [b.const(20)],
        )[0]
        b.store(b.const(1), merged)
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.try_const(merged), 10)

    def test_branch_refinement_narrows_compared_value(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        cond = b.lt(x, b.const(8), "cond")
        results = {}

        def then_fn():
            results["then_plus"] = b.add(x, b.const(1), "then_plus")
            return [results["then_plus"]]

        def else_fn():
            results["else_shift"] = b.shr(x, b.const(29), "else_shift")
            return [results["else_shift"]]

        b.if_stmt(cond, then_fn, else_fn)
        ra = RangeAnalysis(b.build())

        # then: x in [0, 7] -> x + 1 in [1, 8]
        self.assertEqual(ra.range_of(results["then_plus"]), (1, 8))
        # else: x >= 8 -> x >> 29 unrefined lo would be 0; lo(x) = 8 gives 0 too,
        # so assert the then-side refinement is what narrowed things.
        self.assertEqual(ra.range_of(results["else_shift"]), (0, 7))

    def test_equality_refinement(self):
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        cond = b.eq(x, b.const(5), "cond")
        results = {}

        def then_fn():
            results["copy"] = b.add(x, b.const(0), "copy")
            return [results["copy"]]

        b.if_stmt(cond, then_fn, lambda: [b.const(0)])
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(results["copy"]), (5, 5))

    def test_loop_counter_range(self):
        b = HIRBuilder()
        results = {}

        def body_fn(i, params):
            results["scaled"] = b.mul(i, b.const(2), "scaled")
            b.store(b.const(1), results["scaled"])
            return []

        b.for_loop(b.const(2), b.const(10), [], body_fn)
        ra = RangeAnalysis(b.build())
        # counter in [2, 9] -> scaled in [4, 18]
        self.assertEqual(ra.range_of(results["scaled"]), (4, 18))

    def test_wrap_select_recurrence_stabilizes_without_widening(self):
        # idx' = select(2*idx + d < 7, 2*idx + d, 0) with d in [1, 2]:
        # the flagship VM recurrence proves idx in [0, 6] in two rounds.
        b = HIRBuilder()
        x = b.load(b.const(0), "x")
        seen = {}

        def body_fn(i, params):
            idx = params[0]
            bit = b.and_(x, b.const(1), "bit")
            offset = b.add(bit, b.const(1), "offset")
            doubled = b.mul(idx, b.const(2), "doubled")
            nxt = b.add(doubled, offset, "nxt")
            in_bounds = b.lt(nxt, b.const(7), "in_bounds")
            seen["wrapped"] = b.select(in_bounds, nxt, b.const(0), "wrapped")
            return [seen["wrapped"]]

        final = b.for_loop(b.const(0), b.load(b.const(1), "n"), [b.const(0)],
                           body_fn)[0]
        b.store(b.const(1), final)
        ra = RangeAnalysis(b.build())

        self.assertEqual(ra.range_of(seen["wrapped"]), (0, 6))
        # Zero-trip is possible (end unknown), so the result joins the
        # initial Const(0) -- still [0, 6].
        self.assertEqual(ra.range_of(final), (0, 6))

    def test_masked_accumulator_reaches_threshold_fixpoint(self):
        b = HIRBuilder()
        seen = {}

        def body_fn(i, params):
            acc = params[0]
            seen["step"] = b.and_(b.add(acc, b.const(1), "inc"), b.const(7),
                                  "step")
            return [seen["step"]]

        final = b.for_loop(b.const(0), b.load(b.const(0), "n"), [b.const(0)],
                           body_fn)[0]
        b.store(b.const(1), final)
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(final), (0, 7))

    def test_unbounded_accumulator_widens_to_full(self):
        b = HIRBuilder()

        def body_fn(i, params):
            return [b.add(params[0], b.const(1), "inc")]

        final = b.for_loop(b.const(0), b.load(b.const(0), "n"), [b.const(0)],
                           body_fn)[0]
        b.store(b.const(1), final)
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(final), FULL_RANGE)

    def test_zero_trip_loop_carries_iter_args(self):
        b = HIRBuilder()

        def body_fn(i, params):
            return [b.const(999)]

        final = b.for_loop(b.const(5), b.const(5), [b.const(3)], body_fn)[0]
        b.store(b.const(1), final)
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.try_const(final), 3)

    def test_must_run_loop_takes_yield_range_only(self):
        b = HIRBuilder()

        def body_fn(i, params):
            return [b.const(9)]

        final = b.for_loop(b.const(0), b.const(4), [b.const(3)], body_fn)[0]
        b.store(b.const(1), final)
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.try_const(final), 9)

    def test_nested_loop_fixpoint(self):
        b = HIRBuilder()
        seen = {}

        def outer_body(i, outer_params):
            def inner_body(j, inner_params):
                masked = b.and_(
                    b.add(inner_params[0], j, "mix"), b.const(15), "masked"
                )
                seen["inner"] = masked
                return [masked]

            inner_final = b.for_loop(
                b.const(0), b.const(4), [outer_params[0]], inner_body
            )[0]
            return [inner_final]

        final = b.for_loop(b.const(0), b.load(b.const(0), "n"), [b.const(1)],
                           outer_body)[0]
        b.store(b.const(1), final)
        ra = RangeAnalysis(b.build())
        self.assertEqual(ra.range_of(seen["inner"]), (0, 15))
        self.assertEqual(ra.range_of(final), (0, 15))


class _ProgramFuzzer:
    """Seeded random generator of scalar HIR programs with loops and ifs.

    Dynamic loop bounds are always masked to small ranges so the reference
    interpreter terminates quickly; everything else (opcodes, operands,
    nesting, iter_args) is randomized.
    """

    _DIVISORS = (1, 2, 3, 5, 7, 9)

    def __init__(self, seed: int):
        self.rand = random.Random(seed)
        self.builder = HIRBuilder()

    def _operand(self, scope):
        if scope and self.rand.random() < 0.7:
            return self.rand.choice(scope)
        return self.builder.const(self.rand.randrange(0, 20))

    def _random_op(self, scope):
        b = self.builder
        roll = self.rand.random()
        if roll < 0.3:
            # Bias toward masks: they create the bounded values that make
            # interval reasoning non-trivial.
            mask = (1 << self.rand.randrange(1, 6)) - 1
            return b.and_(self._operand(scope), b.const(mask), "masked")
        opcode = self.rand.choice(
            ["+", "-", "*", "^", "&", "|", "<<", ">>", "%", "//", "<", "=="]
        )
        left = self._operand(scope)
        if opcode in ("%", "//"):
            right = b.const(self.rand.choice(self._DIVISORS))
        elif opcode == "<<":
            right = b.const(self.rand.randrange(0, 5))
        elif opcode == ">>":
            right = b.const(self.rand.randrange(0, 9))
        else:
            right = self._operand(scope)
        return b.alu(opcode, left, right, "fuzz")

    def _random_if(self, scope):
        b = self.builder
        cond = self.rand.choice(
            [b.lt(self._operand(scope), self._operand(scope), "cond"),
             b.eq(self._operand(scope), self._operand(scope), "cond"),
             self._operand(scope)]
        )

        def branch():
            local = list(scope)
            for _ in range(self.rand.randrange(0, 3)):
                local.append(self._random_op(local))
            return [self._operand(local)]

        return b.if_stmt(cond, branch, branch)

    def _random_loop(self, scope, depth):
        b = self.builder
        start = b.const(self.rand.randrange(0, 4))
        if self.rand.random() < 0.5:
            end = b.const(self.rand.randrange(0, 7))
        else:
            end = b.and_(self._operand(scope), b.const(7), "bound")
        min_inits = 1 if depth > 0 else 0
        inits = [self._operand(scope)
                 for _ in range(self.rand.randrange(min_inits, 3))]

        def body_fn(counter, params):
            local = list(scope) + list(params) + [counter]
            for _ in range(self.rand.randrange(1, 4)):
                local.append(self._random_op(local))
            if depth == 0 and self.rand.random() < 0.3:
                local.extend(self._random_loop(local, depth + 1))
            if self.rand.random() < 0.4:
                local.extend(self._random_if(local))
            return [self._operand(local) for _ in range(len(params))]

        return b.for_loop(start, end, inits, body_fn)

    def build(self):
        b = self.builder
        scope = [b.load(b.const(slot), f"in{slot}") for slot in range(3)]
        for _ in range(self.rand.randrange(1, 4)):
            scope.append(self._random_op(scope))
        for _ in range(self.rand.randrange(1, 3)):
            scope.extend(self._random_loop(scope, 0))
            if self.rand.random() < 0.5:
                scope.append(self._random_op(scope))
        b.store(b.const(8), self._operand(scope))
        b.store(b.const(9), self._operand(scope))
        return b.build()


class TestRangeAnalysisSoundnessFuzz(unittest.TestCase):
    """Ground-truth soundness: every runtime value of every SSA value must
    lie inside the interval the analysis assigned to it."""

    PROGRAMS = 150
    MEMS_PER_PROGRAM = 3

    def test_random_programs_stay_within_computed_ranges(self):
        for seed in range(self.PROGRAMS):
            hir = _ProgramFuzzer(seed).build()
            analysis = RangeAnalysis(hir)
            input_rand = random.Random(10_000 + seed)
            for trial in range(self.MEMS_PER_PROGRAM):
                mem = [input_rand.randrange(0, 2 ** 32) for _ in range(16)]
                record = {}
                interpret_hir(hir, mem, record)
                for ssa, values in record.items():
                    lo, hi = analysis.range_of(ssa)
                    for value in values:
                        self.assertTrue(
                            lo <= value <= hi,
                            f"seed={seed} trial={trial} {ssa}: "
                            f"value {value} outside [{lo}, {hi}]",
                        )


class TestRangeFoldInLoopsEndToEnd(unittest.TestCase):
    """Differential tests through the FULL default pipeline.

    Dynamic loop bounds keep the ForLoops alive through loop-unroll, so the
    simplify pass now range-folds inside loop bodies; the compiled VM output
    must match the reference interpreter exactly.
    """

    def _run_differential(self, hir, mem, out_slots):
        expected_mem = list(mem)
        interpret_hir(hir, expected_mem)

        instrs = compile_hir_to_vliw(hir)
        machine = Machine(list(mem), instrs, DebugInfo(scratch_map={}),
                          n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        for slot in out_slots:
            self.assertEqual(
                machine.mem[slot], expected_mem[slot],
                f"output slot {slot}: VM {machine.mem[slot]} != "
                f"reference {expected_mem[slot]}",
            )

    def test_wrap_recurrence_with_foldable_guard(self):
        # The wrap recurrence keeps idx in [0, 6]; the redundant guard
        # lt(idx2, 16) is provably true INSIDE the loop, so range_fold
        # collapses its select. A soundness bug here changes the output.
        b = HIRBuilder()
        rounds = b.load(b.const(0), "rounds")
        base = b.load(b.const(1), "base")

        def body(i, params):
            idx, acc = params
            node = b.load(b.add(base, idx, "node_addr"), "node")
            acc2 = b.xor(acc, node, "acc2")
            acc2 = b.add(acc2, b.const(3), "acc3")
            bit = b.and_(acc2, b.const(1), "bit")
            nxt = b.add(b.mul(idx, b.const(2), "dbl"),
                        b.add(bit, b.const(1), "off"), "nxt")
            in_bounds = b.lt(nxt, b.const(7), "in_bounds")
            idx2 = b.select(in_bounds, nxt, b.const(0), "idx2")
            guard = b.lt(idx2, b.const(16), "guard")
            acc3 = b.select(guard, acc2, b.const(0), "guarded")
            return [idx2, acc3]

        res = b.for_loop(b.const(0), rounds, [b.const(0), b.const(0)], body)
        b.store(b.const(30), res[0])
        b.store(b.const(31), res[1])
        hir = b.build()

        mem = [0] * 64
        mem[0] = 5           # dynamic round count
        mem[1] = 16          # table base
        for i in range(7):
            mem[16 + i] = 1000 + 37 * i
        self._run_differential(hir, mem, [30, 31])

    def test_nested_dynamic_loops_with_if(self):
        b = HIRBuilder()
        outer_n = b.load(b.const(0), "outer_n")
        inner_n = b.load(b.const(1), "inner_n")
        seed_val = b.load(b.const(2), "seed")

        def outer_body(i, outer_params):
            def inner_body(j, inner_params):
                mixed = b.and_(b.add(inner_params[0], j, "mix"),
                               b.const(31), "mixed")
                is_even = b.eq(b.and_(mixed, b.const(1), "lsb"),
                               b.const(0), "is_even")
                stepped = b.if_stmt(
                    is_even,
                    lambda: [b.add(mixed, b.const(2), "even_step")],
                    lambda: [b.mul(mixed, b.const(3), "odd_step")],
                )[0]
                return [stepped]

            inner_res = b.for_loop(
                b.const(0), inner_n, [outer_params[0]], inner_body
            )[0]
            folded = b.xor(inner_res, i, "folded")
            return [folded]

        final = b.for_loop(b.const(0), outer_n, [seed_val], outer_body)[0]
        b.store(b.const(40), final)
        hir = b.build()

        mem = [0] * 64
        mem[0] = 4
        mem[1] = 3
        mem[2] = 12345
        self._run_differential(hir, mem, [40])

    def test_loop_result_feeds_bounded_address(self):
        # The loop result is provably in [0, 7]; it indexes a table whose
        # loads must still read the right cell after folding.
        b = HIRBuilder()
        n = b.load(b.const(0), "n")
        base = b.load(b.const(1), "base")

        def body(i, params):
            nxt = b.and_(b.add(params[0], i, "step"), b.const(7), "wrapped")
            return [nxt]

        idx = b.for_loop(b.const(0), n, [b.const(0)], body)[0]
        value = b.load(b.add(base, idx, "cell_addr"), "cell")
        b.store(b.const(50), value)
        hir = b.build()

        mem = [0] * 64
        mem[0] = 6
        mem[1] = 20
        for i in range(8):
            mem[20 + i] = 7000 + i
        self._run_differential(hir, mem, [50])



class TestAnalysisBudget(unittest.TestCase):
    def test_budget_exhaustion_degrades_to_full(self):
        import compiler.range_analysis as ra_mod
        b = HIRBuilder()
        masked = b.and_(b.load(b.const(0), "x"), b.const(7), "masked")

        def body(i, params):
            return [b.and_(b.add(params[0], b.const(1), "inc"),
                           b.const(7), "step")]

        res = b.for_loop(b.const(0), b.load(b.const(1), "n"),
                         [masked], body)[0]
        b.store(b.const(2), res)
        hir = b.build()

        old_budget = ra_mod._ANALYSIS_BUDGET
        ra_mod._ANALYSIS_BUDGET = 5
        try:
            ra = RangeAnalysis(hir)
        finally:
            ra_mod._ANALYSIS_BUDGET = old_budget

        self.assertTrue(ra.budget_exhausted)
        self.assertEqual(ra.range_of(masked), FULL_RANGE)
        self.assertEqual(ra.range_of(res), FULL_RANGE)

        # With the real budget the same program is fully analyzed.
        ra_full = RangeAnalysis(hir)
        self.assertFalse(ra_full.budget_exhausted)
        self.assertEqual(ra_full.range_of(masked), (0, 7))




class TestNestedCaptureSoundness(unittest.TestCase):
    """Regression: an inner loop whose body captures a CHANGING outer SSA
    directly (not via iter_args/bounds) must not reuse a stale inner
    fixpoint (the memo formerly keyed only on bounds/inits/overlay)."""

    def _build(self, seen, dynamic_bound=False):
        b = HIRBuilder()
        rounds = (b.load(b.const(0), "rounds") if dynamic_bound
                  else b.const(3))

        def outer_body(r, outer_params):
            o = outer_params[0]

            def inner_body(k, inner_params):
                seen["step"] = b.add(inner_params[0], o, "step")
                return [seen["step"]]

            # pragma_unroll=1 keeps the nested shape alive through
            # loop-unroll so the pipeline exercises the warm-start path.
            inner_res = b.for_loop(
                b.const(0), b.const(2), [b.const(0)], inner_body,
                pragma_unroll=1,
            )[0]
            seen["inner_res"] = inner_res
            # A guard on the inner body param's derived value: folding it
            # from a stale [0, 0] param miscompiles.
            guard = b.lt(inner_res, b.const(4), "guard")
            picked = b.select(guard, b.const(1), inner_res, "picked")
            b.store(b.add(b.const(8), r, "out_slot"), picked)
            return [b.add(o, b.const(4), "o_next")]

        b.for_loop(b.const(0), rounds, [b.const(0)], outer_body)
        return b.build()

    def test_inner_params_cover_all_outer_rounds(self):
        seen = {}
        hir = self._build(seen)
        ra = RangeAnalysis(hir)
        record = {}
        interpret_hir(hir, [0] * 16, record)
        for ssa, values in record.items():
            lo, hi = ra.range_of(ssa)
            for value in values:
                self.assertTrue(
                    lo <= value <= hi,
                    f"{ssa}: runtime {value} outside [{lo}, {hi}]",
                )

    def test_full_pipeline_differential(self):
        # Dynamic outer bound + pragma-pinned inner loop: both survive to
        # Simplify, so its RangeAnalysis actually runs the nested fixpoint
        # (with static bounds the loops would unroll before Simplify).
        seen = {}
        hir = self._build(seen, dynamic_bound=True)
        mem = [0] * 64
        mem[0] = 3
        expected = list(mem)
        interpret_hir(self._build({}, dynamic_bound=True), expected)

        instrs = compile_hir_to_vliw(hir)
        machine = Machine(list(mem), instrs, DebugInfo(scratch_map={}),
                          n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        for slot in (8, 9, 10):
            self.assertEqual(machine.mem[slot], expected[slot],
                             f"output slot {slot}")


if __name__ == "__main__":
    unittest.main()
