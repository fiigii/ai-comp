"""Tests for Simplify pass (constant folding and algebraic identities)."""

import unittest

from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    HIRBuilder,
    compile_hir_to_vliw,
)
from compiler import (
    PassManager,
    PassConfig,
    SimplifyPass,
    count_statements,
)


class TestSimplifyPass(unittest.TestCase):
    """Test Simplify pass (constant folding and algebraic identities)."""

    def _run_program(self, instrs, mem):
        """Helper to run a compiled program."""
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    # --- Constant Folding Tests ---

    def test_constant_fold_add(self):
        """Test that Const(10) + Const(20) -> Const(30)."""
        b = HIRBuilder()
        addr0 = b.const(0)
        val10 = b.const(10)
        val20 = b.const(20)

        result = b.add(val10, val20, "result")
        b.store(addr0, result)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)

        # Verify correctness
        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 30)
        print("Constant fold add test passed!")

    def test_constant_fold_all_ops(self):
        """Test constant folding for all supported operations."""
        b = HIRBuilder()
        addr = [b.const(i) for i in range(12)]

        val10 = b.const(10)
        val3 = b.const(3)

        # Test all foldable ops
        r_add = b.add(val10, val3, "r_add")       # 10 + 3 = 13
        r_sub = b.sub(val10, val3, "r_sub")       # 10 - 3 = 7
        r_mul = b.mul(val10, val3, "r_mul")       # 10 * 3 = 30
        r_div = b.div(val10, val3, "r_div")       # 10 // 3 = 3
        r_mod = b.mod(val10, val3, "r_mod")       # 10 % 3 = 1
        r_xor = b.xor(val10, val3, "r_xor")       # 10 ^ 3 = 9
        r_and = b.and_(val10, val3, "r_and")      # 10 & 3 = 2
        r_or = b.or_(val10, val3, "r_or")         # 10 | 3 = 11
        r_shl = b.shl(val10, val3, "r_shl")       # 10 << 3 = 80
        r_shr = b.shr(val10, val3, "r_shr")       # 10 >> 3 = 1
        r_lt = b.lt(val3, val10, "r_lt")          # 3 < 10 = 1
        r_eq = b.eq(val3, val3, "r_eq")           # 3 == 3 = 1

        b.store(addr[0], r_add)
        b.store(addr[1], r_sub)
        b.store(addr[2], r_mul)
        b.store(addr[3], r_div)
        b.store(addr[4], r_mod)
        b.store(addr[5], r_xor)
        b.store(addr[6], r_and)
        b.store(addr[7], r_or)
        b.store(addr[8], r_shl)
        b.store(addr[9], r_shr)
        b.store(addr[10], r_lt)
        b.store(addr[11], r_eq)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [0] * 100
        machine = self._run_program(instrs, mem)

        self.assertEqual(machine.mem[0], 13)   # add
        self.assertEqual(machine.mem[1], 7)    # sub
        self.assertEqual(machine.mem[2], 30)   # mul
        self.assertEqual(machine.mem[3], 3)    # div
        self.assertEqual(machine.mem[4], 1)    # mod
        self.assertEqual(machine.mem[5], 9)    # xor
        self.assertEqual(machine.mem[6], 2)    # and
        self.assertEqual(machine.mem[7], 11)   # or
        self.assertEqual(machine.mem[8], 80)   # shl
        self.assertEqual(machine.mem[9], 1)    # shr
        self.assertEqual(machine.mem[10], 1)   # lt
        self.assertEqual(machine.mem[11], 1)   # eq
        print("Constant fold all ops test passed!")

    # --- Algebraic Identity Tests ---

    def test_identity_add_zero(self):
        """Test that x + 0 -> x and 0 + x -> x."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        x = b.load(addr0, "x")
        zero = b.const(0)

        # x + 0 -> x
        result1 = b.add(x, zero, "result1")
        # 0 + x -> x
        result2 = b.add(zero, x, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[1], 42)
        print("Identity add zero test passed!")

    def test_identity_mul_one(self):
        """Test that x * 1 -> x and 1 * x -> x."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        x = b.load(addr0, "x")
        one = b.const(1)

        # x * 1 -> x
        result1 = b.mul(x, one, "result1")
        # 1 * x -> x
        result2 = b.mul(one, x, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[1], 42)
        print("Identity mul one test passed!")

    def test_identity_mul_zero(self):
        """Test that x * 0 -> 0 and 0 * x -> 0."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        x = b.load(addr0, "x")
        zero = b.const(0)

        # x * 0 -> 0
        result1 = b.mul(x, zero, "result1")
        # 0 * x -> 0
        result2 = b.mul(zero, x, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 99] + [0] * 98
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 0)
        self.assertEqual(machine.mem[1], 0)
        print("Identity mul zero test passed!")

    def test_identity_xor_zero(self):
        """Test that x ^ 0 -> x and 0 ^ x -> x."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        x = b.load(addr0, "x")
        zero = b.const(0)

        # x ^ 0 -> x
        result1 = b.xor(x, zero, "result1")
        # 0 ^ x -> x
        result2 = b.xor(zero, x, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[1], 42)
        print("Identity xor zero test passed!")

    def test_identity_and_zero(self):
        """Test that x & 0 -> 0 and 0 & x -> 0."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        x = b.load(addr0, "x")
        zero = b.const(0)

        # x & 0 -> 0
        result1 = b.and_(x, zero, "result1")
        # 0 & x -> 0
        result2 = b.and_(zero, x, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 99] + [0] * 98
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 0)
        self.assertEqual(machine.mem[1], 0)
        print("Identity and zero test passed!")

    def test_identity_or_zero(self):
        """Test that x | 0 -> x and 0 | x -> x."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        x = b.load(addr0, "x")
        zero = b.const(0)

        # x | 0 -> x
        result1 = b.or_(x, zero, "result1")
        # 0 | x -> x
        result2 = b.or_(zero, x, "result2")

        b.store(addr0, result1)
        b.store(addr1, result2)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)

        instrs = compile_hir_to_vliw(transformed)
        mem = [42, 0] + [0] * 98
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[1], 42)
        print("Identity or zero test passed!")

    # --- Integration Tests ---

    def test_simplify_preserves_semantics(self):
        """Test that simplify pass preserves program semantics."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        x = b.load(addr0, "x")
        y = b.load(addr1, "y")

        # Expression with simplifiable subexpressions
        zero = b.const(0)
        one = b.const(1)

        # (x + 0) * (y * 1) + (10 + 20)
        t1 = b.add(x, zero, "t1")
        t2 = b.mul(y, one, "t2")
        t3 = b.mul(t1, t2, "t3")
        val10 = b.const(10)
        val20 = b.const(20)
        t4 = b.add(val10, val20, "t4")
        result = b.add(t3, t4, "result")

        b.store(addr0, result)

        hir = b.build()

        # Without simplify
        instrs_no_simplify = compile_hir_to_vliw(hir)
        mem_no_simplify = [5, 7] + [0] * 98
        machine_no_simplify = self._run_program(instrs_no_simplify, mem_no_simplify)

        # With simplify
        pm = PassManager()
        pm.add_pass(SimplifyPass())
        transformed = pm.run(hir)
        instrs_simplify = compile_hir_to_vliw(transformed)
        mem_simplify = [5, 7] + [0] * 98
        machine_simplify = self._run_program(instrs_simplify, mem_simplify)

        # Both should produce same result: (5 * 7) + 30 = 35 + 30 = 65
        self.assertEqual(machine_no_simplify.mem[0], machine_simplify.mem[0])
        self.assertEqual(machine_simplify.mem[0], 65)
        print("Simplify preserves semantics test passed!")

    def test_simplify_metrics(self):
        """Test that simplify pass reports correct metrics."""
        b = HIRBuilder()
        addr0 = b.const(0)
        x = b.load(addr0, "x")
        zero = b.const(0)
        val10 = b.const(10)
        val20 = b.const(20)

        # x + 0 -> identity simplification
        t1 = b.add(x, zero, "t1")
        # 10 + 20 -> constant fold
        t2 = b.add(val10, val20, "t2")

        result = b.add(t1, t2, "result")
        b.store(addr0, result)

        hir = b.build()

        simplify_pass = SimplifyPass()
        pm = PassManager()
        pm.add_pass(simplify_pass)
        pm.run(hir)

        metrics = simplify_pass.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("constants_folded", metrics.custom)
        self.assertIn("identities_simplified", metrics.custom)
        self.assertGreater(metrics.custom["constants_folded"], 0)
        self.assertGreater(metrics.custom["identities_simplified"], 0)
        print(f"Simplify metrics: {metrics.custom}")
        print("Simplify metrics test passed!")

    def test_simplify_config_disable(self):
        """Test that simplify pass can be disabled via config."""
        b = HIRBuilder()
        addr0 = b.const(0)
        val10 = b.const(10)
        val20 = b.const(20)

        # This would normally be folded
        result = b.add(val10, val20, "result")
        b.store(addr0, result)

        hir = b.build()
        original_count = count_statements(hir)

        # With simplify disabled
        pm_disabled = PassManager()
        pm_disabled.add_pass(SimplifyPass())
        pm_disabled.config["simplify"] = PassConfig(name="simplify", enabled=False)
        result_disabled = pm_disabled.run(hir)

        # Disabled should not change anything
        self.assertEqual(count_statements(result_disabled), original_count)
        print("Simplify config disable test passed!")

    # --- Peephole Optimization Tests ---

    def test_simplify_mod2_to_and1(self):
        """Test %(x, 2) -> &(x, 1) strength reduction."""
        b = HIRBuilder()
        addr0 = b.const(0)
        x = b.load(addr0, "x")
        two = b.const(2)
        result = b.mod(x, two, "result")  # x % 2
        b.store(addr0, result)

        hir = b.build()

        # Apply simplify pass
        pm = PassManager()
        pm.add_pass(SimplifyPass())
        simplified = pm.run(hir)

        # Verify the mod was converted to and
        # Check by running the program - should get same result
        instrs = compile_hir_to_vliw(simplified)

        # Test with various values
        for val in [0, 1, 2, 3, 4, 5, 100, 255]:
            mem = [val] + [0] * 99
            machine = self._run_program(instrs, mem)
            self.assertEqual(machine.mem[0], val % 2, f"Failed for input {val}")

        print("Simplify mod2 to and1 test passed!")

    def test_simplify_mul2_to_shift(self):
        """Test *(x, 2) -> <<(x, 1) strength reduction."""
        b = HIRBuilder()
        addr0 = b.const(0)
        x = b.load(addr0, "x")
        two = b.const(2)
        result = b.mul(x, two, "result")  # x * 2
        b.store(addr0, result)

        hir = b.build()

        # Apply simplify pass
        pm = PassManager()
        pm.add_pass(SimplifyPass())
        simplified = pm.run(hir)

        # Verify by running the program
        instrs = compile_hir_to_vliw(simplified)

        for val in [0, 1, 2, 3, 100, 1000]:
            mem = [val] + [0] * 99
            machine = self._run_program(instrs, mem)
            expected = (val * 2) & 0xFFFFFFFF  # 32-bit wrap
            self.assertEqual(machine.mem[0], expected, f"Failed for input {val}")

        print("Simplify mul2 to shift test passed!")

    def test_simplify_mul_power_of_2(self):
        """Test *(x, 16) -> <<(x, 4) and other powers of 2."""
        # Test multiply by 16 (2^4)
        b = HIRBuilder()
        addr0 = b.const(0)
        x = b.load(addr0, "x")
        sixteen = b.const(16)
        result = b.mul(x, sixteen, "result")  # x * 16
        b.store(addr0, result)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        simplified = pm.run(hir)

        instrs = compile_hir_to_vliw(simplified)

        for val in [0, 1, 2, 5, 100]:
            mem = [val] + [0] * 99
            machine = self._run_program(instrs, mem)
            expected = (val * 16) & 0xFFFFFFFF
            self.assertEqual(machine.mem[0], expected, f"Failed for input {val}")

        print("Simplify mul power of 2 test passed!")

    def test_simplify_select_to_multiply(self):
        """Test select(cond, x, 0) -> *(x, cond) when cond is boolean."""
        b = HIRBuilder()
        addr0 = b.const(0)
        addr1 = b.const(1)
        x = b.load(addr0, "x")
        y = b.load(addr1, "y")

        # cond = x < 5 (produces 0 or 1)
        five = b.const(5)
        cond = b.lt(x, five, "cond")

        # select(cond, y, 0) should become y * cond
        zero = b.const(0)

        def then_fn():
            return [y]

        def else_fn():
            return [zero]

        results = b.if_stmt(cond, then_fn, else_fn)
        b.store(addr0, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        simplified = pm.run(hir)

        instrs = compile_hir_to_vliw(simplified)

        # Test: when x < 5, result should be y; when x >= 5, result should be 0
        test_cases = [
            (0, 42, 42),   # 0 < 5, so select y=42
            (3, 100, 100), # 3 < 5, so select y=100
            (5, 42, 0),    # 5 >= 5, so select 0
            (10, 100, 0),  # 10 >= 5, so select 0
        ]

        for x_val, y_val, expected in test_cases:
            mem = [x_val, y_val] + [0] * 98
            machine = self._run_program(instrs, mem)
            self.assertEqual(machine.mem[0], expected,
                           f"Failed for x={x_val}, y={y_val}, expected {expected}")

        print("Simplify select to multiply test passed!")

    def test_simplify_parity_pattern(self):
        """Test full parity pattern: %(x,2) + ==(mod,0) + select(even,1,2) -> &(x,1) + 1."""
        b = HIRBuilder()
        addr0 = b.const(0)
        x = b.load(addr0, "x")

        # Parity pattern: offset = select(x % 2 == 0, 1, 2)
        two = b.const(2)
        mod = b.mod(x, two, "mod")  # x % 2 -> & 1 (boolean)

        zero = b.const(0)
        even = b.eq(mod, zero, "even")  # == 0, tracked as negated boolean

        one = b.const(1)
        two_const = b.const(2)

        # select(even, 1, 2) should become (x & 1) + 1
        def then_fn():
            return [one]

        def else_fn():
            return [two_const]

        results = b.if_stmt(even, then_fn, else_fn)
        b.store(addr0, results[0])

        hir = b.build()

        pm = PassManager()
        pm.add_pass(SimplifyPass())
        simplified = pm.run(hir)

        instrs = compile_hir_to_vliw(simplified)

        # Test: when x is even, result is 1; when x is odd, result is 2
        test_cases = [
            (0, 1),   # 0 is even -> 1
            (1, 2),   # 1 is odd -> 2
            (2, 1),   # 2 is even -> 1
            (3, 2),   # 3 is odd -> 2
            (100, 1), # 100 is even -> 1
            (101, 2), # 101 is odd -> 2
        ]

        for x_val, expected in test_cases:
            mem = [x_val] + [0] * 99
            machine = self._run_program(instrs, mem)
            self.assertEqual(machine.mem[0], expected,
                           f"Failed for x={x_val}, expected {expected}")

        print("Simplify parity pattern test passed!")


if __name__ == "__main__":
    unittest.main()
