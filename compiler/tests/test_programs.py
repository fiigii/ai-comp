"""End-to-end program correctness tests exercising the full optimization pipeline.

Tests are organized by which optimizations they exercise:
  Group 1: SLP vectorization targets
  Group 2: MAD synthesis
  Group 3: Load/store optimization
  Group 4: CSE / Simplify
  Group 5: Complex multi-pass programs
"""

import unittest

from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    HIRBuilder,
    compile_hir_to_vliw,
)
from compiler.hir import Const

MASK32 = (1 << 32) - 1


def r(x):
    """Truncate to 32-bit unsigned."""
    return x & MASK32


class TestPrograms(unittest.TestCase):
    """End-to-end program tests through the full compiler pipeline."""

    def _run(self, instrs, mem):
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    # ------------------------------------------------------------------
    # Group 1: SLP Vectorization Targets
    # ------------------------------------------------------------------

    def test_vector_add_arrays(self):
        """c[i] = a[i] + b[i] for i in 0..32 — exercises SLP (vadd)."""
        N = 32
        base_a, base_b, base_c = 0, 32, 64
        a_vals = [i * 3 + 7 for i in range(N)]
        b_vals = [i * 5 + 11 for i in range(N)]

        b = HIRBuilder()
        ba = b.const(base_a)
        bb = b.const(base_b)
        bc = b.const(base_c)

        def body(i, params):
            addr_a = b.add(ba, i, "addr_a")
            addr_b = b.add(bb, i, "addr_b")
            addr_c = b.add(bc, i, "addr_c")
            va = b.load(addr_a, "va")
            vb = b.load(addr_b, "vb")
            vc = b.add(va, vb, "vc")
            b.store(addr_c, vc)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 128
        mem[base_a:base_a + N] = a_vals
        mem[base_b:base_b + N] = b_vals
        machine = self._run(instrs, mem)

        expected = [r(a_vals[i] + b_vals[i]) for i in range(N)]
        self.assertEqual(machine.mem[base_c:base_c + N], expected)

    def test_vector_xor_shift(self):
        """out[i] = (in[i] ^ K) >> 3 for i in 0..8 — exercises SLP (vxor, vshr)."""
        N = 8
        K = 0xDEADBEEF
        base_in, base_out = 0, 8
        in_vals = [i * 1000 + 42 for i in range(N)]

        b = HIRBuilder()
        bi = b.const(base_in)
        bo = b.const(base_out)
        ck = b.const(K)
        c3 = b.const(3)

        def body(i, params):
            addr_in = b.add(bi, i, "addr_in")
            addr_out = b.add(bo, i, "addr_out")
            v = b.load(addr_in, "v")
            x = b.xor(v, ck, "x")
            s = b.shr(x, c3, "s")
            b.store(addr_out, s)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[base_in:base_in + N] = in_vals
        machine = self._run(instrs, mem)

        expected = [r(in_vals[i] ^ K) >> 3 for i in range(N)]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    def test_vector_select_relu(self):
        """out[i] = in[i] if in[i] > thresh else 0 — exercises SLP (vlt + vselect)."""
        N = 8
        thresh = 20
        base_in, base_out = 0, 8
        in_vals = [i * 7 for i in range(N)]  # 0, 7, 14, ..., 49

        b = HIRBuilder()
        bi = b.const(base_in)
        bo = b.const(base_out)
        ct = b.const(thresh)
        c0 = b.const(0)

        def body(i, params):
            addr_in = b.add(bi, i, "addr_in")
            addr_out = b.add(bo, i, "addr_out")
            v = b.load(addr_in, "v")
            cond = b.lt(ct, v, "cond")  # thresh < v  =>  v > thresh
            res = b.select(cond, v, c0, "res")
            b.store(addr_out, res)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[base_in:base_in + N] = in_vals
        machine = self._run(instrs, mem)

        expected = [v if v > thresh else 0 for v in in_vals]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    # ------------------------------------------------------------------
    # Group 2: MAD Synthesis
    # ------------------------------------------------------------------

    def test_multiply_add_array(self):
        """out[i] = a[i] * k + b[i] — exercises SLP → MAD synthesis."""
        N = 16
        k = 7
        base_a, base_b, base_out = 0, 16, 32
        a_vals = [i + 1 for i in range(N)]
        b_vals = [100 + i * 3 for i in range(N)]

        b = HIRBuilder()
        ba = b.const(base_a)
        bb = b.const(base_b)
        bo = b.const(base_out)
        ck = b.const(k)

        def body(i, params):
            addr_a = b.add(ba, i, "addr_a")
            addr_b = b.add(bb, i, "addr_b")
            addr_o = b.add(bo, i, "addr_o")
            va = b.load(addr_a, "va")
            vb = b.load(addr_b, "vb")
            prod = b.mul(va, ck, "prod")
            res = b.add(prod, vb, "res")
            b.store(addr_o, res)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[base_a:base_a + N] = a_vals
        mem[base_b:base_b + N] = b_vals
        machine = self._run(instrs, mem)

        expected = [r(a_vals[i] * k + b_vals[i]) for i in range(N)]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    def test_polynomial_horner(self):
        """out[i] = ((c3*x + c2)*x + c1)*x + c0 — exercises chained multiply-add.

        Uses pragma_unroll=1 (scalar path) because chained mul-add chains
        trigger a known SLP vectorization bug.
        """
        N = 8
        c0_v, c1_v, c2_v, c3_v = 5, 3, 7, 2
        base_x, base_out = 0, 8
        x_vals = [i + 1 for i in range(N)]

        b = HIRBuilder()
        bx = b.const(base_x)
        bo = b.const(base_out)
        cc0 = b.const(c0_v)
        cc1 = b.const(c1_v)
        cc2 = b.const(c2_v)
        cc3 = b.const(c3_v)

        def body(i, params):
            ax = b.add(bx, i, "ax")
            ao = b.add(bo, i, "ao")
            x = b.load(ax, "x")
            # Horner form: ((c3*x + c2)*x + c1)*x + c0
            t1 = b.mul(cc3, x, "t1")
            t2 = b.add(t1, cc2, "t2")
            t3 = b.mul(t2, x, "t3")
            t4 = b.add(t3, cc1, "t4")
            t5 = b.mul(t4, x, "t5")
            t6 = b.add(t5, cc0, "t6")
            b.store(ao, t6)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=1)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[base_x:base_x + N] = x_vals
        machine = self._run(instrs, mem)

        expected = [r(((c3_v * x + c2_v) * x + c1_v) * x + c0_v)
                    for x in x_vals]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    # ------------------------------------------------------------------
    # Group 3: Load/Store Optimization
    # ------------------------------------------------------------------

    def test_array_copy(self):
        """dst[i] = src[i] for i in 0..32 — exercises SLP (vload/vstore)."""
        N = 32
        base_src, base_dst = 0, 32
        src_vals = [i * 11 + 1 for i in range(N)]

        b = HIRBuilder()
        bs = b.const(base_src)
        bd = b.const(base_dst)

        def body(i, params):
            as_ = b.add(bs, i, "as")
            ad = b.add(bd, i, "ad")
            v = b.load(as_, "v")
            b.store(ad, v)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 128
        mem[base_src:base_src + N] = src_vals
        machine = self._run(instrs, mem)

        self.assertEqual(machine.mem[base_dst:base_dst + N], src_vals)

    def test_store_load_forwarding(self):
        """Store values then load them back and compute — exercises load-elim."""
        N = 16
        base_tmp, base_out = 0, 16
        vals = [i * 13 + 5 for i in range(N)]

        b = HIRBuilder()
        bt = b.const(base_tmp)
        bo = b.const(base_out)
        c1 = b.const(1)

        def body(i, params):
            at = b.add(bt, i, "at")
            ao = b.add(bo, i, "ao")
            # Store a known value
            v = b.add(i, c1, "v")  # v = i + 1
            b.store(at, v)
            # Load it back
            loaded = b.load(at, "loaded")
            # Use the loaded value
            res = b.add(loaded, c1, "res")  # res = (i+1) + 1
            b.store(ao, res)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        machine = self._run(instrs, mem)

        expected = [i + 2 for i in range(N)]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    # ------------------------------------------------------------------
    # Group 4: CSE / Simplify
    # ------------------------------------------------------------------

    def test_cse_redundant_subexpr(self):
        """Compute a[i]+b[i] twice, use both — exercises CSE deduplication."""
        N = 16
        base_a, base_b, base_c, base_d = 0, 16, 32, 48
        a_vals = [i + 10 for i in range(N)]
        b_vals = [i * 2 + 3 for i in range(N)]

        b = HIRBuilder()
        ba = b.const(base_a)
        bb = b.const(base_b)
        bc = b.const(base_c)
        bd = b.const(base_d)
        c5 = b.const(5)

        def body(i, params):
            aa = b.add(ba, i, "aa")
            ab = b.add(bb, i, "ab")
            ac = b.add(bc, i, "ac")
            ad = b.add(bd, i, "ad")
            va = b.load(aa, "va")
            vb = b.load(ab, "vb")
            # Compute a[i] + b[i] twice (redundant)
            sum1 = b.add(va, vb, "sum1")
            sum2 = b.add(va, vb, "sum2")
            # Use both: c[i] = sum1 * 5, d[i] = sum2 + 5
            r1 = b.mul(sum1, c5, "r1")
            r2 = b.add(sum2, c5, "r2")
            b.store(ac, r1)
            b.store(ad, r2)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 128
        mem[base_a:base_a + N] = a_vals
        mem[base_b:base_b + N] = b_vals
        machine = self._run(instrs, mem)

        for i in range(N):
            s = r(a_vals[i] + b_vals[i])
            self.assertEqual(machine.mem[base_c + i], r(s * 5))
            self.assertEqual(machine.mem[base_d + i], r(s + 5))

    def test_parity_classification(self):
        """out[i] = (in[i] & 1) + 1 — exercises simplify pass parity pattern."""
        N = 16
        base_in, base_out = 0, 16
        in_vals = [i * 7 + 3 for i in range(N)]

        b = HIRBuilder()
        bi = b.const(base_in)
        bo = b.const(base_out)
        c1 = b.const(1)

        def body(i, params):
            ai = b.add(bi, i, "ai")
            ao = b.add(bo, i, "ao")
            v = b.load(ai, "v")
            parity = b.and_(v, c1, "parity")
            res = b.add(parity, c1, "res")
            b.store(ao, res)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[base_in:base_in + N] = in_vals
        machine = self._run(instrs, mem)

        expected = [(v & 1) + 1 for v in in_vals]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    # ------------------------------------------------------------------
    # Group 5: Complex Multi-Pass Programs
    # ------------------------------------------------------------------

    def test_hash_one_round(self):
        """Apply one hash stage to an array — exercises SLP + MAD + CSE + scheduling.

        Uses the first HASH_STAGE: ('+', 0x7ED55D16, '+', '<<', 12)
          val = (val + c1) + (val << c3)
        """
        N = 16
        base_in, base_out = 0, 16
        in_vals = [i * 1000 + 12345 for i in range(N)]
        c1_v = 0x7ED55D16
        c3_v = 12

        b = HIRBuilder()
        bi = b.const(base_in)
        bo = b.const(base_out)
        cc1 = b.const(c1_v)
        cc3 = b.const(c3_v)

        def body(i, params):
            ai = b.add(bi, i, "ai")
            ao = b.add(bo, i, "ao")
            val = b.load(ai, "val")
            # (val + c1) + (val << c3)
            t1 = b.add(val, cc1, "t1")
            t2 = b.shl(val, cc3, "t2")
            res = b.add(t1, t2, "res")
            b.store(ao, res)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[base_in:base_in + N] = in_vals
        machine = self._run(instrs, mem)

        expected = [r(r(v + c1_v) + r(v << c3_v)) for v in in_vals]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    def test_batch_index_update(self):
        """idx = 2*idx + ((val%2==0) ? 1 : 2) — exercises parity + select + MAD + CSE."""
        N = 16
        base_idx, base_val, base_out = 0, 16, 32
        idx_vals = [i + 10 for i in range(N)]
        val_vals = [i * 3 for i in range(N)]

        b = HIRBuilder()
        bidx = b.const(base_idx)
        bval = b.const(base_val)
        bout = b.const(base_out)
        c2 = b.const(2)
        c0 = b.const(0)
        c1 = b.const(1)

        def body(i, params):
            aidx = b.add(bidx, i, "aidx")
            aval = b.add(bval, i, "aval")
            aout = b.add(bout, i, "aout")
            idx = b.load(aidx, "idx")
            val = b.load(aval, "val")
            # idx = 2*idx + (1 if val%2==0 else 2)
            mod_val = b.mod(val, c2, "mod_val")
            is_even = b.eq(mod_val, c0, "is_even")
            offset = b.select(is_even, c1, c2, "offset")
            idx_doubled = b.mul(idx, c2, "idx_doubled")
            new_idx = b.add(idx_doubled, offset, "new_idx")
            b.store(aout, new_idx)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body,
                   pragma_unroll=0)
        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[base_idx:base_idx + N] = idx_vals
        mem[base_val:base_val + N] = val_vals
        machine = self._run(instrs, mem)

        expected = [2 * idx_vals[i] + (1 if val_vals[i] % 2 == 0 else 2)
                    for i in range(N)]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    def test_multi_round_accumulate(self):
        """Two manually-unrolled rounds of array processing.

        Round 0: tmp[i] = in[i] * 3 + 7
        Round 1: out[i] = tmp[i] * 5 + 11

        Exercises multi-round store-then-load, load-elim, DSE.
        """
        N = 16
        base_in, base_tmp, base_out = 0, 16, 32
        in_vals = [i + 1 for i in range(N)]

        b = HIRBuilder()
        bin_ = b.const(base_in)
        btmp = b.const(base_tmp)
        bout = b.const(base_out)
        c3 = b.const(3)
        c5 = b.const(5)
        c7 = b.const(7)
        c11 = b.const(11)

        # Round 0: tmp[i] = in[i] * 3 + 7
        def body0(i, params):
            ai = b.add(bin_, i, "ai0")
            at = b.add(btmp, i, "at0")
            v = b.load(ai, "v0")
            prod = b.mul(v, c3, "prod0")
            res = b.add(prod, c7, "res0")
            b.store(at, res)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body0,
                   pragma_unroll=0)

        # Round 1: out[i] = tmp[i] * 5 + 11
        def body1(i, params):
            at = b.add(btmp, i, "at1")
            ao = b.add(bout, i, "ao1")
            v = b.load(at, "v1")
            prod = b.mul(v, c5, "prod1")
            res = b.add(prod, c11, "res1")
            b.store(ao, res)
            return []

        b.for_loop(start=Const(0), end=Const(N), iter_args=[], body_fn=body1,
                   pragma_unroll=0)

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 64
        mem[base_in:base_in + N] = in_vals
        machine = self._run(instrs, mem)

        expected = [r(r(v * 3 + 7) * 5 + 11) for v in in_vals]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)


class TestControlFlowPrograms(unittest.TestCase):
    """Tests exercising complex control flow through the full compiler pipeline."""

    def _run(self, instrs, mem):
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    # ------------------------------------------------------------------
    # If/else inside loops with carried state
    # ------------------------------------------------------------------

    def test_loop_conditional_accumulate(self):
        """Accumulate even-indexed values, subtract odd-indexed values.

        acc = 0
        for i in 0..N:
            if i % 2 == 0: acc += a[i]
            else:          acc -= a[i]
        mem[out] = acc

        Exercises: loop iter_args + if/else branching + phi merge.
        """
        N = 8
        base_a, addr_out = 0, 8
        a_vals = [10, 3, 20, 7, 15, 2, 30, 1]

        b = HIRBuilder()
        ba = b.const(base_a)
        out = b.const(addr_out)
        c2 = b.const(2)
        c0 = b.const(0)
        init_acc = b.const(0)

        def body(i, params):
            acc = params[0]
            addr = b.add(ba, i, "addr")
            v = b.load(addr, "v")
            mod = b.mod(i, c2, "mod")
            is_even = b.eq(mod, c0, "is_even")

            def then_fn():
                return [b.add(acc, v, "acc_add")]

            def else_fn():
                return [b.sub(acc, v, "acc_sub")]

            merged = b.if_stmt(is_even, then_fn, else_fn)
            return [merged[0]]

        results = b.for_loop(
            start=Const(0), end=Const(N), iter_args=[init_acc],
            body_fn=body, pragma_unroll=1
        )
        b.store(out, results[0])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 16
        mem[base_a:base_a + N] = a_vals
        machine = self._run(instrs, mem)

        # Python reference: even indices add, odd indices subtract
        expected = 0
        for i in range(N):
            if i % 2 == 0:
                expected = r(expected + a_vals[i])
            else:
                expected = r(expected - a_vals[i])
        self.assertEqual(machine.mem[addr_out], expected)

    def test_loop_conditional_store(self):
        """Classify array elements: store to different outputs based on threshold.

        for i in 0..N:
            if a[i] > thresh: high[cnt_hi++] = a[i]
            else:             low[cnt_lo++] = a[i]
        mem[cnt_addr] = cnt_hi

        Exercises: if/else with stores in both branches + carried counters.
        """
        N = 8
        thresh = 50
        base_a, base_high, base_low, cnt_addr = 0, 8, 16, 24
        a_vals = [30, 80, 10, 90, 55, 40, 70, 20]

        b = HIRBuilder()
        ba = b.const(base_a)
        bhi = b.const(base_high)
        blo = b.const(base_low)
        cout = b.const(cnt_addr)
        ct = b.const(thresh)
        c1 = b.const(1)
        init_hi = b.const(0)
        init_lo = b.const(0)

        def body(i, params):
            cnt_hi, cnt_lo = params[0], params[1]
            addr = b.add(ba, i, "addr")
            v = b.load(addr, "v")
            cond = b.lt(ct, v, "gt_thresh")  # thresh < v

            def then_fn():
                dst = b.add(bhi, cnt_hi, "dst_hi")
                b.store(dst, v)
                return [b.add(cnt_hi, c1, "inc_hi"), cnt_lo]

            def else_fn():
                dst = b.add(blo, cnt_lo, "dst_lo")
                b.store(dst, v)
                return [cnt_hi, b.add(cnt_lo, c1, "inc_lo")]

            merged = b.if_stmt(cond, then_fn, else_fn)
            return [merged[0], merged[1]]

        results = b.for_loop(
            start=Const(0), end=Const(N), iter_args=[init_hi, init_lo],
            body_fn=body, pragma_unroll=1
        )
        b.store(cout, results[0])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 32
        mem[base_a:base_a + N] = a_vals
        machine = self._run(instrs, mem)

        # Python reference
        ref_high, ref_low = [], []
        for v in a_vals:
            if v > thresh:
                ref_high.append(v)
            else:
                ref_low.append(v)
        self.assertEqual(machine.mem[base_high:base_high + len(ref_high)], ref_high)
        self.assertEqual(machine.mem[base_low:base_low + len(ref_low)], ref_low)
        self.assertEqual(machine.mem[cnt_addr], len(ref_high))

    # ------------------------------------------------------------------
    # Nested loops with carried state
    # ------------------------------------------------------------------

    def test_nested_loop_matrix_rowsum(self):
        """Compute row sums of a 4x4 matrix.

        for i in 0..4:
            sum = 0
            for j in 0..4:
                sum += mat[i*4 + j]
            out[i] = sum

        Exercises: nested loops with inner carried state + outer stores.
        """
        ROWS, COLS = 4, 4
        base_mat, base_out = 0, 16
        mat = [
            10, 20, 30, 40,
            1,  2,  3,  4,
            100, 200, 300, 400,
            5,  15, 25, 35,
        ]

        b = HIRBuilder()
        bmat = b.const(base_mat)
        bout = b.const(base_out)
        ccols = b.const(COLS)

        def outer_body(i, outer_params):
            init_sum = b.const(0)

            def inner_body(j, inner_params):
                s = inner_params[0]
                row_off = b.mul(i, ccols, "row_off")
                idx = b.add(row_off, j, "idx")
                addr = b.add(bmat, idx, "addr")
                v = b.load(addr, "v")
                new_s = b.add(s, v, "new_s")
                return [new_s]

            inner_results = b.for_loop(
                start=Const(0), end=Const(COLS), iter_args=[init_sum],
                body_fn=inner_body, pragma_unroll=1
            )
            out_addr = b.add(bout, i, "out_addr")
            b.store(out_addr, inner_results[0])
            return []

        b.for_loop(
            start=Const(0), end=Const(ROWS), iter_args=[],
            body_fn=outer_body, pragma_unroll=1
        )

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 32
        mem[base_mat:base_mat + ROWS * COLS] = mat
        machine = self._run(instrs, mem)

        expected = [sum(mat[i * COLS:(i + 1) * COLS]) for i in range(ROWS)]
        self.assertEqual(machine.mem[base_out:base_out + ROWS], expected)

    def test_nested_loop_carried_across_outer(self):
        """Global running sum accumulated across nested loops.

        global_sum = 0
        for i in 0..3:
            for j in 0..4:
                global_sum += mat[i*4 + j]
            out[i] = global_sum   # partial prefix sum per row

        Exercises: iter_args carried through both loop levels.
        """
        ROWS, COLS = 3, 4
        base_mat, base_out = 0, 12
        mat = [1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400]

        b = HIRBuilder()
        bmat = b.const(base_mat)
        bout = b.const(base_out)
        ccols = b.const(COLS)
        init_gsum = b.const(0)

        def outer_body(i, outer_params):
            gsum = outer_params[0]

            def inner_body(j, inner_params):
                s = inner_params[0]
                row_off = b.mul(i, ccols, "row_off")
                idx = b.add(row_off, j, "idx")
                addr = b.add(bmat, idx, "addr")
                v = b.load(addr, "v")
                return [b.add(s, v, "new_s")]

            inner_results = b.for_loop(
                start=Const(0), end=Const(COLS), iter_args=[gsum],
                body_fn=inner_body, pragma_unroll=1
            )
            out_addr = b.add(bout, i, "out_addr")
            b.store(out_addr, inner_results[0])
            return [inner_results[0]]

        results = b.for_loop(
            start=Const(0), end=Const(ROWS), iter_args=[init_gsum],
            body_fn=outer_body, pragma_unroll=1
        )

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 32
        mem[base_mat:base_mat + ROWS * COLS] = mat
        machine = self._run(instrs, mem)

        # Python reference: prefix row sums
        expected = []
        gsum = 0
        for i in range(ROWS):
            for j in range(COLS):
                gsum += mat[i * COLS + j]
            expected.append(gsum)
        self.assertEqual(machine.mem[base_out:base_out + ROWS], expected)

    # ------------------------------------------------------------------
    # If/else with complex value flow
    # ------------------------------------------------------------------

    def test_diamond_if_multi_result(self):
        """If/else returning multiple values used in downstream computation.

        if a > b:  (x, y) = (a + b, a - b)
        else:      (x, y) = (a * 2, b * 2)
        out0 = x + y
        out1 = x * y

        Exercises: multi-result if/else + downstream use of merged phis.
        """
        b = HIRBuilder()
        addr_a, addr_b = b.const(0), b.const(1)
        addr_out0, addr_out1 = b.const(2), b.const(3)
        c2 = b.const(2)

        a = b.load(addr_a, "a")
        val_b = b.load(addr_b, "b")
        cond = b.lt(val_b, a, "a_gt_b")

        def then_fn():
            return [b.add(a, val_b, "sum"), b.sub(a, val_b, "diff")]

        def else_fn():
            return [b.mul(a, c2, "a2"), b.mul(val_b, c2, "b2")]

        xy = b.if_stmt(cond, then_fn, else_fn)
        x, y = xy[0], xy[1]
        b.store(addr_out0, b.add(x, y, "xpy"))
        b.store(addr_out1, b.mul(x, y, "xty"))

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        # Test case 1: a=10, b=3 → a > b → x=13, y=7 → out0=20, out1=91
        mem1 = [10, 3, 0, 0] + [0] * 96
        m1 = self._run(instrs, mem1)
        self.assertEqual(m1.mem[2], 20)
        self.assertEqual(m1.mem[3], 91)

        # Test case 2: a=2, b=8 → a <= b → x=4, y=16 → out0=20, out1=64
        mem2 = [2, 8, 0, 0] + [0] * 96
        m2 = self._run(instrs, mem2)
        self.assertEqual(m2.mem[2], 20)
        self.assertEqual(m2.mem[3], 64)

    def test_nested_if_else(self):
        """Nested if/else: classify into 3 categories.

        if a > 100:       result = 3   (high)
        else:
            if a > 50:    result = 2   (medium)
            else:         result = 1   (low)

        Exercises: nested if/else + phi merging at multiple levels.
        """
        b = HIRBuilder()
        addr_a, addr_out = b.const(0), b.const(1)
        c100 = b.const(100)
        c50 = b.const(50)
        v1, v2, v3 = b.const(1), b.const(2), b.const(3)

        a = b.load(addr_a, "a")

        cond1 = b.lt(c100, a, "gt_100")

        def then_high():
            return [v3]

        def else_rest():
            cond2 = b.lt(c50, a, "gt_50")

            def then_med():
                return [v2]

            def else_low():
                return [v1]

            return b.if_stmt(cond2, then_med, else_low)

        result = b.if_stmt(cond1, then_high, else_rest)
        b.store(addr_out, result[0])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        for val, expected in [(150, 3), (75, 2), (30, 1), (100, 2), (50, 1)]:
            mem = [val, 0] + [0] * 98
            machine = self._run(instrs, mem)
            self.assertEqual(machine.mem[1], expected,
                             f"classify({val}): expected {expected}, got {machine.mem[1]}")

    # ------------------------------------------------------------------
    # Loop + if/else interaction patterns
    # ------------------------------------------------------------------

    def test_loop_find_max(self):
        """Find maximum value in an array.

        max_val = 0
        for i in 0..N:
            if a[i] > max_val: max_val = a[i]
        mem[out] = max_val

        Exercises: loop with conditional iter_arg update via if/else.
        """
        N = 8
        base_a, addr_out = 0, 8
        a_vals = [42, 17, 99, 3, 88, 55, 12, 76]

        b = HIRBuilder()
        ba = b.const(base_a)
        cout = b.const(addr_out)
        init_max = b.const(0)

        def body(i, params):
            cur_max = params[0]
            addr = b.add(ba, i, "addr")
            v = b.load(addr, "v")
            cond = b.lt(cur_max, v, "v_gt_max")

            def then_fn():
                return [v]

            def else_fn():
                return [cur_max]

            merged = b.if_stmt(cond, then_fn, else_fn)
            return [merged[0]]

        results = b.for_loop(
            start=Const(0), end=Const(N), iter_args=[init_max],
            body_fn=body, pragma_unroll=1
        )
        b.store(cout, results[0])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 16
        mem[base_a:base_a + N] = a_vals
        machine = self._run(instrs, mem)

        self.assertEqual(machine.mem[addr_out], max(a_vals))

    def test_loop_prefix_sum_with_reset(self):
        """Prefix sum that resets when a sentinel value is encountered.

        acc = 0
        for i in 0..N:
            if a[i] == 0: acc = 0
            else:         acc += a[i]
            out[i] = acc

        Exercises: if/else with one branch resetting carried state.
        """
        N = 8
        base_a, base_out = 0, 8
        a_vals = [5, 3, 0, 7, 2, 0, 4, 1]

        b = HIRBuilder()
        ba = b.const(base_a)
        bo = b.const(base_out)
        c0 = b.const(0)
        init_acc = b.const(0)

        def body(i, params):
            acc = params[0]
            addr = b.add(ba, i, "addr")
            v = b.load(addr, "v")
            is_zero = b.eq(v, c0, "is_zero")

            def then_fn():
                return [c0]

            def else_fn():
                return [b.add(acc, v, "new_acc")]

            merged = b.if_stmt(is_zero, then_fn, else_fn)
            out_addr = b.add(bo, i, "out_addr")
            b.store(out_addr, merged[0])
            return [merged[0]]

        b.for_loop(
            start=Const(0), end=Const(N), iter_args=[init_acc],
            body_fn=body, pragma_unroll=1
        )

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 24
        mem[base_a:base_a + N] = a_vals
        machine = self._run(instrs, mem)

        # Python reference
        expected = []
        acc = 0
        for v in a_vals:
            if v == 0:
                acc = 0
            else:
                acc += v
            expected.append(acc)
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    def test_loop_with_if_and_nested_loop(self):
        """Outer loop with conditional inner loop.

        For each row, if the first element > threshold, sum the row;
        otherwise store 0.

        Exercises: loop → if/else → inner loop (multi-level nesting).
        """
        ROWS, COLS = 4, 3
        thresh = 10
        base_mat, base_out = 0, 12
        mat = [
            15, 20, 25,   # row 0: first > 10 → sum = 60
            5,  30, 40,   # row 1: first <= 10 → 0
            12, 1,  2,    # row 2: first > 10 → sum = 15
            8,  99, 100,  # row 3: first <= 10 → 0
        ]

        b = HIRBuilder()
        bmat = b.const(base_mat)
        bout = b.const(base_out)
        ccols = b.const(COLS)
        cthresh = b.const(thresh)

        def outer_body(i, outer_params):
            # Load first element of row
            row_off = b.mul(i, ccols, "row_off")
            first_addr = b.add(bmat, row_off, "first_addr")
            first = b.load(first_addr, "first")
            cond = b.lt(cthresh, first, "first_gt_thresh")

            def then_sum():
                init_s = b.const(0)

                def inner_body(j, inner_params):
                    s = inner_params[0]
                    idx = b.add(row_off, j, "idx")
                    addr = b.add(bmat, idx, "addr")
                    v = b.load(addr, "v")
                    return [b.add(s, v, "new_s")]

                inner_results = b.for_loop(
                    start=Const(0), end=Const(COLS), iter_args=[init_s],
                    body_fn=inner_body, pragma_unroll=1
                )
                return [inner_results[0]]

            def else_zero():
                return [b.const(0)]

            merged = b.if_stmt(cond, then_sum, else_zero)
            out_addr = b.add(bout, i, "out_addr")
            b.store(out_addr, merged[0])
            return []

        b.for_loop(
            start=Const(0), end=Const(ROWS), iter_args=[],
            body_fn=outer_body, pragma_unroll=1
        )

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 32
        mem[base_mat:base_mat + ROWS * COLS] = mat
        machine = self._run(instrs, mem)

        expected = []
        for i in range(ROWS):
            row = mat[i * COLS:(i + 1) * COLS]
            if row[0] > thresh:
                expected.append(sum(row))
            else:
                expected.append(0)
        self.assertEqual(machine.mem[base_out:base_out + ROWS], expected)

    # ------------------------------------------------------------------
    # Multi-carried-value loops
    # ------------------------------------------------------------------

    def test_loop_dual_accumulator(self):
        """Two independent accumulators with different operations.

        sum_val = 0, prod_val = 1
        for i in 0..N:
            sum_val += a[i]
            prod_val *= a[i]

        Exercises: multiple carried values updated independently.
        """
        N = 5
        base_a = 0
        addr_sum, addr_prod = 5, 6
        a_vals = [2, 3, 4, 5, 6]

        b = HIRBuilder()
        ba = b.const(base_a)
        cs = b.const(addr_sum)
        cp = b.const(addr_prod)
        init_sum = b.const(0)
        init_prod = b.const(1)

        def body(i, params):
            s, p = params[0], params[1]
            addr = b.add(ba, i, "addr")
            v = b.load(addr, "v")
            return [b.add(s, v, "new_s"), b.mul(p, v, "new_p")]

        results = b.for_loop(
            start=Const(0), end=Const(N), iter_args=[init_sum, init_prod],
            body_fn=body, pragma_unroll=1
        )
        b.store(cs, results[0])
        b.store(cp, results[1])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 16
        mem[base_a:base_a + N] = a_vals
        machine = self._run(instrs, mem)

        self.assertEqual(machine.mem[addr_sum], sum(a_vals))
        prod = 1
        for v in a_vals:
            prod *= v
        self.assertEqual(machine.mem[addr_prod], r(prod))

    def test_loop_min_max_simultaneously(self):
        """Track both min and max in a single loop pass.

        min_val = MAX, max_val = 0
        for i in 0..N:
            if a[i] < min_val: min_val = a[i]
            if a[i] > max_val: max_val = a[i]

        Exercises: two conditional updates per iteration + dual carried state.
        """
        N = 6
        base_a = 0
        addr_min, addr_max = 6, 7
        a_vals = [50, 10, 80, 5, 95, 30]

        b = HIRBuilder()
        ba = b.const(base_a)
        cmin = b.const(addr_min)
        cmax = b.const(addr_max)
        init_min = b.const(0xFFFFFFFF)
        init_max = b.const(0)

        def body(i, params):
            cur_min, cur_max = params[0], params[1]
            addr = b.add(ba, i, "addr")
            v = b.load(addr, "v")

            # Update min
            is_less = b.lt(v, cur_min, "is_less")

            def then_min():
                return [v]

            def else_min():
                return [cur_min]

            new_min = b.if_stmt(is_less, then_min, else_min)

            # Update max
            is_greater = b.lt(cur_max, v, "is_greater")

            def then_max():
                return [v]

            def else_max():
                return [cur_max]

            new_max = b.if_stmt(is_greater, then_max, else_max)

            return [new_min[0], new_max[0]]

        results = b.for_loop(
            start=Const(0), end=Const(N), iter_args=[init_min, init_max],
            body_fn=body, pragma_unroll=1
        )
        b.store(cmin, results[0])
        b.store(cmax, results[1])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 16
        mem[base_a:base_a + N] = a_vals
        machine = self._run(instrs, mem)

        self.assertEqual(machine.mem[addr_min], min(a_vals))
        self.assertEqual(machine.mem[addr_max], max(a_vals))

    # ------------------------------------------------------------------
    # Collatz-style iterative computation
    # ------------------------------------------------------------------

    def test_collatz_step_array(self):
        """Apply one Collatz step to each element: n → n/2 if even, 3n+1 if odd.

        for i in 0..N:
            v = a[i]
            if v % 2 == 0: out[i] = v / 2
            else:          out[i] = 3*v + 1

        Exercises: loop body with data-dependent if/else + mixed arithmetic.
        """
        N = 8
        base_a, base_out = 0, 8
        a_vals = [6, 11, 14, 7, 22, 3, 10, 1]

        b = HIRBuilder()
        ba = b.const(base_a)
        bo = b.const(base_out)
        c2 = b.const(2)
        c3 = b.const(3)
        c1 = b.const(1)
        c0 = b.const(0)

        def body(i, params):
            addr = b.add(ba, i, "addr")
            out_addr = b.add(bo, i, "out_addr")
            v = b.load(addr, "v")
            mod = b.mod(v, c2, "mod")
            is_even = b.eq(mod, c0, "is_even")

            def then_fn():
                return [b.div(v, c2, "half")]

            def else_fn():
                t = b.mul(v, c3, "triple")
                return [b.add(t, c1, "triple_plus_1")]

            result = b.if_stmt(is_even, then_fn, else_fn)
            b.store(out_addr, result[0])
            return []

        b.for_loop(
            start=Const(0), end=Const(N), iter_args=[],
            body_fn=body, pragma_unroll=1
        )

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        mem = [0] * 24
        mem[base_a:base_a + N] = a_vals
        machine = self._run(instrs, mem)

        expected = [v // 2 if v % 2 == 0 else 3 * v + 1 for v in a_vals]
        self.assertEqual(machine.mem[base_out:base_out + N], expected)

    # ------------------------------------------------------------------
    # Sequential if/else chains (non-nested)
    # ------------------------------------------------------------------

    def test_sequential_conditionals(self):
        """Multiple sequential if/else blocks operating on the same value.

        x = mem[0]
        if x > 100: x = x - 50   else: x = x + 50
        if x > 75:  x = x * 2    else: x = x * 3
        mem[1] = x

        Exercises: CFG simplification, sequential diamond patterns.
        """
        b = HIRBuilder()
        addr0, addr1 = b.const(0), b.const(1)
        c100, c75, c50 = b.const(100), b.const(75), b.const(50)
        c2, c3 = b.const(2), b.const(3)

        x = b.load(addr0, "x")

        # First conditional
        cond1 = b.lt(c100, x, "gt_100")

        def then1():
            return [b.sub(x, c50, "x_sub")]

        def else1():
            return [b.add(x, c50, "x_add")]

        x_vals = b.if_stmt(cond1, then1, else1)
        x2 = x_vals[0]

        # Second conditional
        cond2 = b.lt(c75, x2, "gt_75")

        def then2():
            return [b.mul(x2, c2, "x_mul2")]

        def else2():
            return [b.mul(x2, c3, "x_mul3")]

        x_vals2 = b.if_stmt(cond2, then2, else2)
        b.store(addr1, x_vals2[0])

        hir = b.build()
        instrs = compile_hir_to_vliw(hir)

        # Test case 1: x=120 → 120-50=70 → 70<=75 → 70*3=210
        mem1 = [120, 0] + [0] * 98
        m1 = self._run(instrs, mem1)
        self.assertEqual(m1.mem[1], 210)

        # Test case 2: x=30 → 30+50=80 → 80>75 → 80*2=160
        mem2 = [30, 0] + [0] * 98
        m2 = self._run(instrs, mem2)
        self.assertEqual(m2.mem[1], 160)

        # Test case 3: x=200 → 200-50=150 → 150>75 → 150*2=300
        mem3 = [200, 0] + [0] * 98
        m3 = self._run(instrs, mem3)
        self.assertEqual(m3.mem[1], 300)

        # Test case 4: x=10 → 10+50=60 → 60<=75 → 60*3=180
        mem4 = [10, 0] + [0] * 98
        m4 = self._run(instrs, mem4)
        self.assertEqual(m4.mem[1], 180)


if __name__ == "__main__":
    unittest.main()
