"""Tests for Use-Def Chain Infrastructure."""

import unittest

from compiler.tests.conftest import HIRBuilder
from compiler import (
    Const,
    UseDefContext,
    DefLocation,
    UseLocation,
    SSAValue,
)
from compiler.hir import ForLoop, If, Op, VectorSSAValue


class TestUseDefBasicOps(unittest.TestCase):
    """Test use-def chains for basic Op statements."""

    def test_const_load_def(self):
        """Test that const_load creates a definition."""
        b = HIRBuilder()
        val = b.const_load(42, "val")
        b.halt()
        hir = b.build()

        ctx = UseDefContext(hir)
        def_loc = ctx.get_def(val)

        self.assertIsNotNone(def_loc)
        self.assertEqual(def_loc.def_kind, "result")
        self.assertIsInstance(def_loc.statement, Op)
        self.assertEqual(def_loc.statement.result, val)

    def test_op_uses_operands(self):
        """Test that operations record uses of their operands."""
        b = HIRBuilder()
        a = b.const_load(10, "a")
        b_val = b.const_load(20, "b")
        c = b.add(a, b_val, "c")
        b.halt()
        hir = b.build()

        ctx = UseDefContext(hir)

        # Check uses of 'a'
        uses_a = ctx.get_uses(a)
        self.assertEqual(len(uses_a), 1)
        self.assertEqual(uses_a[0].use_kind, "operand")
        self.assertEqual(uses_a[0].operand_index, 0)

        # Check uses of 'b'
        uses_b = ctx.get_uses(b_val)
        self.assertEqual(len(uses_b), 1)
        self.assertEqual(uses_b[0].use_kind, "operand")
        self.assertEqual(uses_b[0].operand_index, 1)

    def test_unused_value_no_uses(self):
        """Test that unused values have no uses."""
        b = HIRBuilder()
        used = b.const_load(10, "used")
        unused = b.const_load(20, "unused")
        addr = b.const_load(0, "addr")
        b.store(addr, used)
        hir = b.build()

        ctx = UseDefContext(hir)

        self.assertTrue(ctx.has_uses(used))
        self.assertFalse(ctx.has_uses(unused))
        self.assertEqual(ctx.use_count(used), 1)
        self.assertEqual(ctx.use_count(unused), 0)

    def test_store_uses(self):
        """Test that store records uses of address and value."""
        b = HIRBuilder()
        addr = b.const_load(0, "addr")
        val = b.const_load(42, "val")
        b.store(addr, val)
        hir = b.build()

        ctx = UseDefContext(hir)

        # Both addr and val are used by store
        self.assertTrue(ctx.has_uses(addr))
        self.assertTrue(ctx.has_uses(val))

        uses_addr = ctx.get_uses(addr)
        uses_val = ctx.get_uses(val)
        self.assertEqual(len(uses_addr), 1)
        self.assertEqual(len(uses_val), 1)

    def test_multiple_uses(self):
        """Test value used multiple times."""
        b = HIRBuilder()
        x = b.const_load(5, "x")
        y = b.add(x, x, "y")  # x used twice
        addr = b.const_load(0, "addr")
        b.store(addr, y)
        hir = b.build()

        ctx = UseDefContext(hir)

        uses_x = ctx.get_uses(x)
        self.assertEqual(len(uses_x), 2)
        self.assertEqual(ctx.use_count(x), 2)


class TestUseDefForLoop(unittest.TestCase):
    """Test use-def chains for ForLoop statements."""

    def test_loop_counter_def(self):
        """Test that loop counter is defined by the loop."""
        b = HIRBuilder()

        def body(i, params):
            return []

        b.for_loop(start=Const(0), end=Const(10), iter_args=[], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)

        # Find the loop
        loop = hir.body[0]
        self.assertIsInstance(loop, ForLoop)

        def_loc = ctx.get_def(loop.counter)
        self.assertIsNotNone(def_loc)
        self.assertEqual(def_loc.def_kind, "counter")
        self.assertEqual(def_loc.statement, loop)

    def test_loop_body_params_def(self):
        """Test that body_params are defined by the loop."""
        b = HIRBuilder()
        init = b.const_load(0, "init")

        def body(i, params):
            s = params[0]
            return [b.add(s, i, "new_s")]

        results = b.for_loop(start=Const(0), end=Const(10), iter_args=[init], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)
        loop = hir.body[1]  # After const_load
        self.assertIsInstance(loop, ForLoop)

        for param in loop.body_params:
            def_loc = ctx.get_def(param)
            self.assertIsNotNone(def_loc)
            self.assertEqual(def_loc.def_kind, "body_param")

    def test_loop_results_def(self):
        """Test that loop results are defined by the loop."""
        b = HIRBuilder()
        init = b.const_load(0, "init")

        def body(i, params):
            s = params[0]
            return [b.add(s, i, "new_s")]

        results = b.for_loop(start=Const(0), end=Const(10), iter_args=[init], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)
        loop = hir.body[1]
        self.assertIsInstance(loop, ForLoop)

        for result in loop.results:
            def_loc = ctx.get_def(result)
            self.assertIsNotNone(def_loc)
            self.assertEqual(def_loc.def_kind, "loop_result")

    def test_loop_iter_args_uses(self):
        """Test that iter_args are uses."""
        b = HIRBuilder()
        init = b.const_load(0, "init")

        def body(i, params):
            return [params[0]]

        b.for_loop(start=Const(0), end=Const(10), iter_args=[init], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)

        # init should be used as iter_arg
        uses = ctx.get_uses(init)
        self.assertEqual(len(uses), 1)
        self.assertEqual(uses[0].use_kind, "iter_arg")

    def test_loop_yields_uses(self):
        """Test that yields are uses."""
        b = HIRBuilder()
        init = b.const_load(0, "init")

        def body(i, params):
            s = params[0]
            new_s = b.add(s, i, "new_s")
            return [new_s]

        b.for_loop(start=Const(0), end=Const(10), iter_args=[init], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)
        loop = hir.body[1]
        self.assertIsInstance(loop, ForLoop)

        # Find the new_s value in yields
        for y in loop.yields:
            uses = ctx.get_uses(y)
            yield_uses = [u for u in uses if u.use_kind == "yield"]
            self.assertEqual(len(yield_uses), 1)

    def test_loop_start_end_ssa_uses(self):
        """Test that SSA values in start/end are uses."""
        b = HIRBuilder()
        start_val = b.const_load(0, "start")
        end_val = b.const_load(10, "end")

        def body(i, params):
            return []

        b.for_loop(start=start_val, end=end_val, iter_args=[], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)

        start_uses = ctx.get_uses(start_val)
        end_uses = ctx.get_uses(end_val)

        start_kind = [u.use_kind for u in start_uses]
        end_kind = [u.use_kind for u in end_uses]

        self.assertIn("start", start_kind)
        self.assertIn("end", end_kind)


class TestUseDefIf(unittest.TestCase):
    """Test use-def chains for If statements."""

    def test_if_results_def(self):
        """Test that if results are defined by the if."""
        b = HIRBuilder()
        cond = b.const_load(1, "cond")

        def then_fn():
            return [b.const_load(100, "then_val")]

        def else_fn():
            return [b.const_load(200, "else_val")]

        results = b.if_stmt(cond, then_fn, else_fn)
        hir = b.build()

        ctx = UseDefContext(hir)
        if_stmt = hir.body[1]  # After const_load for cond
        self.assertIsInstance(if_stmt, If)

        for result in if_stmt.results:
            def_loc = ctx.get_def(result)
            self.assertIsNotNone(def_loc)
            self.assertEqual(def_loc.def_kind, "if_result")

    def test_if_cond_use(self):
        """Test that condition is a use."""
        b = HIRBuilder()
        cond = b.const_load(1, "cond")

        def then_fn():
            return []

        def else_fn():
            return []

        b.if_stmt(cond, then_fn, else_fn)
        hir = b.build()

        ctx = UseDefContext(hir)

        uses = ctx.get_uses(cond)
        cond_uses = [u for u in uses if u.use_kind == "condition"]
        self.assertEqual(len(cond_uses), 1)

    def test_if_yields_uses(self):
        """Test that then_yields and else_yields are uses."""
        b = HIRBuilder()
        cond = b.const_load(1, "cond")

        def then_fn():
            then_val = b.const_load(100, "then_val")
            return [then_val]

        def else_fn():
            else_val = b.const_load(200, "else_val")
            return [else_val]

        results = b.if_stmt(cond, then_fn, else_fn)
        hir = b.build()

        ctx = UseDefContext(hir)
        if_stmt = hir.body[1]
        self.assertIsInstance(if_stmt, If)

        # Check then_yields
        for y in if_stmt.then_yields:
            uses = ctx.get_uses(y)
            yield_uses = [u for u in uses if u.use_kind == "then_yield"]
            self.assertEqual(len(yield_uses), 1)

        # Check else_yields
        for y in if_stmt.else_yields:
            uses = ctx.get_uses(y)
            yield_uses = [u for u in uses if u.use_kind == "else_yield"]
            self.assertEqual(len(yield_uses), 1)


class TestUseDefNested(unittest.TestCase):
    """Test use-def chains for nested structures."""

    def test_nested_loop_in_loop(self):
        """Test nested loops."""
        b = HIRBuilder()
        init = b.const_load(0, "init")

        def outer_body(i, outer_params):
            outer_s = outer_params[0]

            def inner_body(j, inner_params):
                inner_s = inner_params[0]
                return [b.add(inner_s, j, "inner_sum")]

            inner_results = b.for_loop(
                start=Const(0), end=Const(3),
                iter_args=[outer_s], body_fn=inner_body
            )
            return [inner_results[0]]

        results = b.for_loop(start=Const(0), end=Const(2), iter_args=[init], body_fn=outer_body)
        addr = b.const_load(0, "addr")
        b.store(addr, results[0])
        hir = b.build()

        ctx = UseDefContext(hir)

        # Just ensure no errors and results are defined
        self.assertTrue(ctx.has_uses(results[0]))

    def test_loop_in_if(self):
        """Test loop inside if branch."""
        b = HIRBuilder()
        cond = b.const_load(1, "cond")

        def then_fn():
            init = b.const_load(0, "init")

            def body(i, params):
                return [b.add(params[0], i, "sum")]

            results = b.for_loop(start=Const(0), end=Const(5), iter_args=[init], body_fn=body)
            return [results[0]]

        def else_fn():
            return [b.const_load(999, "else_val")]

        results = b.if_stmt(cond, then_fn, else_fn)
        addr = b.const_load(0, "addr")
        b.store(addr, results[0])
        hir = b.build()

        ctx = UseDefContext(hir)

        # Ensure results from if are tracked
        self.assertTrue(ctx.has_uses(results[0]))

    def test_if_in_loop(self):
        """Test if inside loop body."""
        b = HIRBuilder()
        init = b.const_load(0, "init")

        def body(i, params):
            s = params[0]
            # Conditional increment
            cond = b.lt(i, Const(3), "cond")

            def then_fn():
                return [b.add(s, Const(1), "then_val")]

            def else_fn():
                return [s]

            if_results = b.if_stmt(cond, then_fn, else_fn)
            return [if_results[0]]

        results = b.for_loop(start=Const(0), end=Const(5), iter_args=[init], body_fn=body)
        addr = b.const_load(0, "addr")
        b.store(addr, results[0])
        hir = b.build()

        ctx = UseDefContext(hir)

        self.assertTrue(ctx.has_uses(results[0]))


class TestUseDefParent(unittest.TestCase):
    """Test parent tracking functionality."""

    def test_parent_tracking_basic(self):
        """Test that parent list and index are tracked correctly."""
        b = HIRBuilder()
        a = b.const_load(10, "a")
        c = b.add(a, a, "c")
        b.halt()
        hir = b.build()

        ctx = UseDefContext(hir)

        # First op
        parent_info = ctx.get_parent(hir.body[0])
        self.assertIsNotNone(parent_info)
        parent_list, idx = parent_info
        self.assertIs(parent_list, hir.body)
        self.assertEqual(idx, 0)

        # Second op
        parent_info = ctx.get_parent(hir.body[1])
        self.assertIsNotNone(parent_info)
        parent_list, idx = parent_info
        self.assertIs(parent_list, hir.body)
        self.assertEqual(idx, 1)

    def test_parent_tracking_in_loop(self):
        """Test that statements inside loop have correct parent."""
        b = HIRBuilder()

        def body(i, params):
            val = b.const_load(42, "val")  # Inside loop
            return []

        b.for_loop(start=Const(0), end=Const(5), iter_args=[], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)
        loop = hir.body[0]
        self.assertIsInstance(loop, ForLoop)

        # Check that the const_load inside has the loop body as parent
        inner_op = loop.body[0]
        parent_info = ctx.get_parent(inner_op)
        self.assertIsNotNone(parent_info)
        parent_list, idx = parent_info
        self.assertIs(parent_list, loop.body)
        self.assertEqual(idx, 0)


class TestUseDefUpdate(unittest.TestCase):
    """Test update functionality."""

    def test_invalidate_rebuilds(self):
        """Test that invalidate causes rebuild on next query."""
        b = HIRBuilder()
        a = b.const_load(10, "a")
        b.halt()
        hir = b.build()

        ctx = UseDefContext(hir)
        ctx.get_def(a)  # Force build
        self.assertTrue(ctx._built)

        ctx.invalidate()
        self.assertFalse(ctx._built)

        ctx.get_def(a)  # Should rebuild
        self.assertTrue(ctx._built)

    def test_rebuild_with_new_hir(self):
        """Test rebuild with new HIR."""
        b1 = HIRBuilder()
        a1 = b1.const_load(10, "a")
        b1.halt()
        hir1 = b1.build()

        b2 = HIRBuilder()
        a2 = b2.const_load(20, "a")
        b2.halt()
        hir2 = b2.build()

        ctx = UseDefContext(hir1)
        def_loc1 = ctx.get_def(a1)
        self.assertIsNotNone(def_loc1)
        self.assertIs(def_loc1.parent_list, hir1.body)

        ctx.rebuild(hir2)
        def_loc2 = ctx.get_def(a2)
        self.assertIsNotNone(def_loc2)
        # After rebuild, the def should be in the new HIR's body
        self.assertIs(def_loc2.parent_list, hir2.body)
        self.assertIsNot(def_loc2.parent_list, hir1.body)

    def test_replace_all_uses_op_operands(self):
        """Test replacing uses in op operands."""
        b = HIRBuilder()
        old_val = b.const_load(10, "old")
        new_val = b.const_load(20, "new")
        result = b.add(old_val, old_val, "result")  # Uses old_val twice
        addr = b.const_load(0, "addr")
        b.store(addr, result)
        hir = b.build()

        ctx = UseDefContext(hir)

        # Verify old_val has uses
        self.assertEqual(ctx.use_count(old_val), 2)

        # Replace all uses
        count = ctx.replace_all_uses(old_val, new_val)
        self.assertEqual(count, 2)

        # Verify the IR was modified
        add_op = hir.body[2]  # The add operation
        self.assertIsInstance(add_op, Op)
        self.assertEqual(add_op.operands[0], new_val)
        self.assertEqual(add_op.operands[1], new_val)

    def test_replace_all_uses_loop_iter_args(self):
        """Test replacing uses in loop iter_args."""
        b = HIRBuilder()
        old_init = b.const_load(0, "old_init")
        new_init = b.const_load(100, "new_init")

        def body(i, params):
            return [params[0]]

        b.for_loop(start=Const(0), end=Const(5), iter_args=[old_init], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)
        count = ctx.replace_all_uses(old_init, new_init)

        self.assertEqual(count, 1)
        loop = hir.body[2]  # After two const_loads
        self.assertIsInstance(loop, ForLoop)
        self.assertEqual(loop.iter_args[0], new_init)


class TestUseDefGetAllDefs(unittest.TestCase):
    """Test get_all_defs iteration."""

    def test_get_all_defs(self):
        """Test iterating over all definitions."""
        b = HIRBuilder()
        a = b.const_load(10, "a")
        c = b.add(a, a, "c")
        b.halt()
        hir = b.build()

        ctx = UseDefContext(hir)
        all_defs = list(ctx.get_all_defs())

        # Should have definitions for both 'a' and 'c'
        self.assertEqual(len(all_defs), 2)
        ssa_values = [ssa for ssa, def_loc in all_defs]
        self.assertIn(a, ssa_values)
        self.assertIn(c, ssa_values)


class TestUseDefConst(unittest.TestCase):
    """Test that Const values are not tracked as uses."""

    def test_const_not_tracked_as_use(self):
        """Test that Const operands don't appear in uses."""
        b = HIRBuilder()
        val = b.const_load(10, "val")
        result = b.add(val, Const(5), "result")  # Const(5) is an operand
        b.halt()
        hir = b.build()

        ctx = UseDefContext(hir)

        # val has one use (first operand of add)
        uses = ctx.get_uses(val)
        self.assertEqual(len(uses), 1)
        self.assertEqual(uses[0].operand_index, 0)

        # Const(5) should not be tracked
        # (We can't query it directly, but we can verify the add only has one use recorded for val)

    def test_replace_all_uses_with_const(self):
        """Test replacing SSA uses with a Const value."""
        b = HIRBuilder()
        val = b.const_load(10, "val")
        result = b.add(val, val, "result")  # val used twice
        addr = b.const_load(0, "addr")
        b.store(addr, result)
        hir = b.build()

        ctx = UseDefContext(hir)

        # Replace val with Const(42)
        count = ctx.replace_all_uses(val, Const(42))
        self.assertEqual(count, 2)

        # Verify the IR was modified
        add_op = hir.body[1]  # The add operation
        self.assertIsInstance(add_op, Op)
        self.assertEqual(add_op.operands[0], Const(42))
        self.assertEqual(add_op.operands[1], Const(42))


class TestReplaceAllUsesIntegration(unittest.TestCase):
    """Integration tests for replace_all_uses in control flow contexts."""

    def test_replace_in_forloop_yields(self):
        """Test replacing a value used in ForLoop yields."""
        b = HIRBuilder()
        init = b.const_load(0, "init")
        new_val = b.const_load(42, "new_val")

        def body(i, params):
            # The yield will use params[0] (which we want to replace)
            return [params[0]]

        results = b.for_loop(start=Const(0), end=Const(3), iter_args=[init], body_fn=body)
        hir = b.build()

        # Find the loop and get the yield SSA
        loop = hir.body[2]  # After two const_loads
        self.assertIsInstance(loop, ForLoop)
        old_yield = loop.yields[0]

        ctx = UseDefContext(hir)
        count = ctx.replace_all_uses(old_yield, new_val)

        # Should have replaced the yield
        self.assertEqual(count, 1)
        self.assertEqual(loop.yields[0], new_val)

    def test_replace_in_forloop_start(self):
        """Test replacing a value used as ForLoop start."""
        b = HIRBuilder()
        old_start = b.const_load(0, "old_start")
        new_start = b.const_load(5, "new_start")

        def body(i, params):
            return []

        b.for_loop(start=old_start, end=Const(10), iter_args=[], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)
        count = ctx.replace_all_uses(old_start, new_start)

        self.assertEqual(count, 1)
        loop = hir.body[2]  # After two const_loads
        self.assertIsInstance(loop, ForLoop)
        self.assertEqual(loop.start, new_start)

    def test_replace_in_forloop_end(self):
        """Test replacing a value used as ForLoop end."""
        b = HIRBuilder()
        old_end = b.const_load(10, "old_end")
        new_end = b.const_load(20, "new_end")

        def body(i, params):
            return []

        b.for_loop(start=Const(0), end=old_end, iter_args=[], body_fn=body)
        hir = b.build()

        ctx = UseDefContext(hir)
        count = ctx.replace_all_uses(old_end, new_end)

        self.assertEqual(count, 1)
        loop = hir.body[2]  # After two const_loads
        self.assertIsInstance(loop, ForLoop)
        self.assertEqual(loop.end, new_end)

    def test_replace_in_if_cond(self):
        """Test replacing a value used as If condition."""
        b = HIRBuilder()
        old_cond = b.const_load(1, "old_cond")
        new_cond = b.const_load(0, "new_cond")

        def then_fn():
            return []

        def else_fn():
            return []

        b.if_stmt(old_cond, then_fn, else_fn)
        hir = b.build()

        ctx = UseDefContext(hir)
        count = ctx.replace_all_uses(old_cond, new_cond)

        self.assertEqual(count, 1)
        if_stmt = hir.body[2]  # After two const_loads
        self.assertIsInstance(if_stmt, If)
        self.assertEqual(if_stmt.cond, new_cond)

    def test_replace_in_if_then_yields(self):
        """Test replacing a value used in If then_yields."""
        b = HIRBuilder()
        cond = b.const_load(1, "cond")
        old_then_val = b.const_load(100, "old_then")
        new_val = b.const_load(999, "new_val")

        def then_fn():
            return [old_then_val]

        def else_fn():
            return [b.const_load(200, "else_val")]

        results = b.if_stmt(cond, then_fn, else_fn)
        hir = b.build()

        # Find the if statement
        if_stmt = None
        for stmt in hir.body:
            if isinstance(stmt, If):
                if_stmt = stmt
                break
        self.assertIsNotNone(if_stmt)

        ctx = UseDefContext(hir)
        count = ctx.replace_all_uses(old_then_val, new_val)

        self.assertEqual(count, 1)
        self.assertEqual(if_stmt.then_yields[0], new_val)

    def test_replace_in_if_else_yields(self):
        """Test replacing a value used in If else_yields."""
        b = HIRBuilder()
        cond = b.const_load(1, "cond")
        old_else_val = b.const_load(200, "old_else")
        new_val = b.const_load(999, "new_val")

        def then_fn():
            return [b.const_load(100, "then_val")]

        def else_fn():
            return [old_else_val]

        results = b.if_stmt(cond, then_fn, else_fn)
        hir = b.build()

        # Find the if statement
        if_stmt = None
        for stmt in hir.body:
            if isinstance(stmt, If):
                if_stmt = stmt
                break
        self.assertIsNotNone(if_stmt)

        ctx = UseDefContext(hir)
        count = ctx.replace_all_uses(old_else_val, new_val)

        self.assertEqual(count, 1)
        self.assertEqual(if_stmt.else_yields[0], new_val)

    def test_replace_vector_ssa_in_operand(self):
        """Test replacing VectorSSAValue used in vector op."""
        b = HIRBuilder()
        addr = b.const_load(0, "addr")
        v1 = b.vload(addr, "v1")  # First vector load
        v2 = b.vload(addr, "v2")  # Second vector load (to replace with)
        v3 = b.vadd(v1, v1, "v3")  # Uses v1 twice
        b.vstore(addr, v3)
        hir = b.build()

        ctx = UseDefContext(hir)

        # Verify v1 has uses
        self.assertEqual(ctx.use_count(v1), 2)

        # Replace v1 with v2
        count = ctx.replace_all_uses(v1, v2)
        self.assertEqual(count, 2)

        # Find the vadd op and verify operands were replaced
        vadd_op = hir.body[3]  # After addr, v1, v2 loads
        self.assertIsInstance(vadd_op, Op)
        self.assertEqual(vadd_op.opcode, "v+")
        self.assertEqual(vadd_op.operands[0], v2)
        self.assertEqual(vadd_op.operands[1], v2)


if __name__ == "__main__":
    unittest.main()
