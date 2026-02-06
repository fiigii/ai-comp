"""Tests for tree-level cache optimization pass."""

import unittest

from compiler import PassManager, PassConfig
from compiler.hir import Op, SSAValue, Const, ForLoop, If
from compiler.hir_builder import HIRBuilder
from compiler.passes import TreeLevelCachePass
from compiler.use_def import UseDefContext


def _find_header_load(body, slot: int):
    for stmt in body:
        if isinstance(stmt, Op) and stmt.opcode == "load" and stmt.result is not None:
            addr = stmt.operands[0]
            if isinstance(addr, Const) and addr.value == slot:
                return stmt.result
    return None


def _count_opcodes(body, opcode):
    count = 0
    for stmt in body:
        if isinstance(stmt, Op):
            if stmt.opcode == opcode:
                count += 1
        elif isinstance(stmt, ForLoop):
            count += _count_opcodes(stmt.body, opcode)
        elif isinstance(stmt, If):
            count += _count_opcodes(stmt.then_body, opcode)
            count += _count_opcodes(stmt.else_body, opcode)
    return count


def _count_dynamic_node_loads(hir, forest_values_p: SSAValue) -> int:
    use_def = UseDefContext(hir)
    count = 0
    for stmt in hir.body:
        if not isinstance(stmt, Op) or stmt.opcode != "load":
            continue
        addr = stmt.operands[0]
        if not isinstance(addr, SSAValue):
            continue
        def_loc = use_def.get_def(addr)
        if def_loc is None or not isinstance(def_loc.statement, Op):
            continue
        op = def_loc.statement
        if op.opcode != "+" or len(op.operands) != 2:
            continue
        a, b = op.operands
        if a == forest_values_p and isinstance(b, SSAValue):
            count += 1
        elif b == forest_values_p and isinstance(a, SSAValue):
            count += 1
    return count


def _count_const_forest_loads(hir, forest_values_p: SSAValue) -> int:
    use_def = UseDefContext(hir)
    count = 0
    for stmt in hir.body:
        if not isinstance(stmt, Op) or stmt.opcode != "load":
            continue
        addr = stmt.operands[0]
        if not isinstance(addr, SSAValue):
            continue
        def_loc = use_def.get_def(addr)
        if def_loc is None or not isinstance(def_loc.statement, Op):
            continue
        op = def_loc.statement
        if op.opcode != "+" or len(op.operands) != 2:
            continue
        a, b = op.operands
        if a == forest_values_p and isinstance(b, Const):
            count += 1
        elif b == forest_values_p and isinstance(a, Const):
            count += 1
    return count


class TestTreeLevelCachePass(unittest.TestCase):
    def test_replaces_early_node_loads(self):
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        inp_indices_p = b.load(b.const(5), "inp_indices_p")
        b.pause()

        batch_size = 2
        rounds = 3

        for r in range(rounds):
            for i in range(batch_size):
                idx_addr = b.add(inp_indices_p, b.const(i), f"idx_addr_r{r}_{i}")
                idx = b.load(idx_addr, f"idx_r{r}_{i}")
                node_addr = b.add(forest_values_p, idx, f"node_addr_r{r}_{i}")
                node_val = b.load(node_addr, f"node_val_r{r}_{i}")
                tmp = b.add(node_val, b.const(1), f"tmp_r{r}_{i}")
                b.store(b.const(100 + r * batch_size + i), tmp)

        hir = b.build()

        forest_before = _find_header_load(hir.body, 4)
        self.assertIsNotNone(forest_before)
        before_dynamic = _count_dynamic_node_loads(hir, forest_before)
        self.assertEqual(before_dynamic, batch_size * rounds)

        pm = PassManager()
        pm.add_pass(TreeLevelCachePass())
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": 2}
        )
        transformed = pm.run(hir)

        forest_after = _find_header_load(transformed.body, 4)
        self.assertIsNotNone(forest_after)
        after_dynamic = _count_dynamic_node_loads(transformed, forest_after)
        self.assertEqual(after_dynamic, batch_size * (rounds - 2))

        preload_count = _count_const_forest_loads(transformed, forest_after)
        self.assertEqual(preload_count, (1 << 2) - 1)

        select_count = _count_opcodes(transformed.body, "select")
        self.assertEqual(select_count, batch_size)


if __name__ == "__main__":
    unittest.main()
