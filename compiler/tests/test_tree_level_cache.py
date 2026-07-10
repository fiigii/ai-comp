"""Tests for tree-level cache optimization pass."""

import json
import os
import random
import tempfile
import unittest

from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    build_mem_image,
    reference_kernel2,
    Tree,
    Input,
    KernelBuilder,
    compile_hir_to_vliw,
)
from compiler import PassManager, PassConfig
from compiler.compile import PASS_REGISTRY
from compiler.hir import Op, SSAValue, Const, ForLoop, HIRFunction, If
from compiler.hir_builder import HIRBuilder
from compiler.passes import TreeLevelCachePass
from compiler.use_def import UseDefContext
from programs.tree_hash import build_tree_hash_kernel

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "pass_config.json"
)


def _load_default_config():
    with open(_DEFAULT_CONFIG_PATH) as f:
        return json.load(f)


def _run_hir_pipeline_to_tree_cache(hir, tlc_options=None):
    """Run the default HIR pass prefix up to and including tree-level-cache.

    Uses the pipeline order and per-pass options from the default
    pass_config.json. Returns (transformed_hir, tree_level_cache_pass) so
    tests can inspect the pass metrics from the run. If tlc_options is
    given, it overrides the tree-level-cache options.
    """
    config_data = _load_default_config()
    pipeline = config_data["pipeline"]
    prefix = pipeline[:pipeline.index("tree-level-cache")]

    pm = PassManager()
    for name in prefix:
        pm.add_pass(PASS_REGISTRY[name]())
        opts = config_data["passes"].get(name, {})
        pm.config[name] = PassConfig(
            name=name,
            enabled=opts.get("enabled", True),
            options=opts.get("options", {}),
        )
    tlc = TreeLevelCachePass()
    pm.add_pass(tlc)
    if tlc_options is None:
        opts = config_data["passes"]["tree-level-cache"]
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache",
            enabled=opts.get("enabled", True),
            options=opts.get("options", {}),
        )
    else:
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options=tlc_options
        )
    transformed = pm.run(hir)
    return transformed, tlc


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
    def test_skips_without_promoted_zero_root(self):
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        inp_indices_p = b.load(b.const(5), "inp_indices_p")
        idx = b.load(inp_indices_p, "idx")
        node_addr = b.add(forest_values_p, idx, "node_addr")
        node_val = b.load(node_addr, "node_val")
        b.store(b.const(100), node_val)
        hir = b.build()

        pm = PassManager()
        tlc = TreeLevelCachePass()
        pm.add_pass(tlc)
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": 1}
        )
        transformed = pm.run(hir)

        self.assertEqual(transformed.body, hir.body)
        self.assertTrue(any("promoted zero-root" in msg
                            for msg in tlc.get_metrics().messages))

    def test_later_node_index_must_follow_root_recurrence(self):
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        root_node = b.load(
            b.add(forest_values_p, b.const(0)), "root_node"
        )
        unrelated_idx = b.load(b.const(20), "unrelated_idx")
        node_val = b.load(
            b.add(forest_values_p, unrelated_idx), "node_val"
        )
        b.store(b.const(100), b.add(root_node, node_val))
        hir = b.build()

        pm = PassManager()
        tlc = TreeLevelCachePass()
        pm.add_pass(tlc)
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": 1}
        )
        transformed = pm.run(hir)

        self.assertEqual(transformed.body, hir.body)
        self.assertTrue(any("recurrence does not match" in msg
                            for msg in tlc.get_metrics().messages))

    def test_store_to_forest_blocks_cache(self):
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        first = b.load(b.add(forest_values_p, b.const(0)), "first")
        b.store(forest_values_p, b.const(99))
        second = b.load(b.add(forest_values_p, b.const(0)), "second")
        b.store(b.const(30), b.add(first, second))
        hir = b.build()

        pm = PassManager()
        tlc = TreeLevelCachePass()
        pm.add_pass(tlc)
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": 1}
        )
        transformed = pm.run(hir)

        self.assertEqual(transformed.body, hir.body)
        self.assertTrue(any("forest may be modified" in msg
                            for msg in tlc.get_metrics().messages))

    def test_wrapped_store_to_forest_blocks_cache(self):
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        first = b.load(forest_values_p, "first")
        wrapped = b.add(
            forest_values_p, b.const(1 << 32), "wrapped_forest_root"
        )
        b.store(wrapped, b.const(99))
        second = b.load(forest_values_p, "second")
        b.store(b.const(30), b.add(first, second))
        hir = b.build()

        pm = PassManager()
        tlc = TreeLevelCachePass()
        pm.add_pass(tlc)
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": 1}
        )
        transformed = pm.run(hir)

        self.assertEqual(transformed.body, hir.body)
        self.assertTrue(any(
            "forest may be modified" in message
            for message in tlc.get_metrics().messages
        ))

    def test_selected_forest_store_blocks_cache_end_to_end(self):
        def build_program():
            b = HIRBuilder()
            forest_values_p = b.load(b.const(4), "forest_values_p")
            other_values_p = b.load(b.const(7), "other_values_p")
            condition = b.load(b.const(8), "condition")
            first = b.load(forest_values_p, "first_root")
            selected = b.select(
                condition, forest_values_p, other_values_p, "selected_base"
            )
            replacement = b.vbroadcast(b.const(99), "replacement")
            b.vstore(selected, replacement)
            second = b.load(forest_values_p, "second_root")
            b.store(b.const(30), b.add(first, second, "root_sum"))
            return b.build()

        _, tlc = _run_hir_pipeline_to_tree_cache(build_program())
        self.assertTrue(any(
            "forest may be modified" in message
            for message in tlc.get_metrics().messages
        ))

        instrs = compile_hir_to_vliw(build_program())
        mem = [0] * 64
        mem[4] = 16
        mem[7] = 32
        mem[8] = 1
        mem[16] = 2
        machine = Machine(
            mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES
        )
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        self.assertEqual(machine.mem[16], 99)
        self.assertEqual(machine.mem[30], 101)

    def test_vector_store_crossing_forest_base_blocks_cache_end_to_end(self):
        def build_program():
            b = HIRBuilder()
            forest_values_p = b.load(b.const(4), "forest_values_p")
            first = b.load(forest_values_p, "first_root")
            before_forest = b.sub(
                forest_values_p, b.const(1), "before_forest"
            )
            replacement = b.vbroadcast(b.const(99), "replacement")
            b.vstore(before_forest, replacement)
            second = b.load(forest_values_p, "second_root")
            b.store(b.const(30), b.add(first, second, "root_sum"))
            return b.build()

        _, tlc = _run_hir_pipeline_to_tree_cache(build_program())
        self.assertTrue(any(
            "forest may be modified" in message
            for message in tlc.get_metrics().messages
        ))

        instrs = compile_hir_to_vliw(build_program())
        mem = [0] * 64
        mem[4] = 16
        mem[16] = 2
        machine = Machine(
            mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES
        )
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        self.assertEqual(machine.mem[16], 99)
        self.assertEqual(machine.mem[30], 101)

    def test_dynamic_subtracted_store_blocks_cache_end_to_end(self):
        def build_program():
            b = HIRBuilder()
            forest_values_p = b.load(b.const(4), "forest_values_p")
            dynamic_offset = b.load(b.const(8), "dynamic_offset")
            first = b.load(forest_values_p, "first_root")
            dynamic_address = b.sub(
                forest_values_p, dynamic_offset, "dynamic_address"
            )
            replacement = b.vbroadcast(b.const(99), "replacement")
            b.vstore(dynamic_address, replacement)
            second = b.load(forest_values_p, "second_root")
            b.store(b.const(30), b.add(first, second, "root_sum"))
            return b.build()

        _, tlc = _run_hir_pipeline_to_tree_cache(build_program())
        self.assertTrue(any(
            "forest may be modified" in message
            for message in tlc.get_metrics().messages
        ))

        instrs = compile_hir_to_vliw(build_program())
        mem = [0] * 64
        mem[4] = 16
        mem[8] = 0
        mem[16] = 2
        machine = Machine(
            mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES
        )
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        self.assertEqual(machine.mem[16], 99)
        self.assertEqual(machine.mem[30], 101)

    def test_reloaded_header_alias_store_blocks_cache_end_to_end(self):
        """A repeated load of header slot 4 still aliases the forest base.

        The unrelated scalar store bumps CSE's memory epoch so the second
        header load remains a distinct SSA value. The vector store is used so
        load-elim cannot forward it to the later scalar root load. Without the
        alias-based forest guard, both root loads are replaced by a preload
        before the vstore and the output is 2 + 2 instead of 2 + 99.
        """

        def build_program():
            b = HIRBuilder()
            forest_values_p = b.load(b.const(4), "forest_values_p")
            first = b.load(forest_values_p, "first_root")
            b.store(b.const(40), b.const(1))
            reloaded_forest_p = b.load(b.const(4), "reloaded_forest_p")
            replacement = b.vbroadcast(b.const(99), "replacement")
            b.vstore(reloaded_forest_p, replacement)
            second = b.load(forest_values_p, "second_root")
            b.store(b.const(30), b.add(first, second, "root_sum"))
            return b.build()

        _, tlc = _run_hir_pipeline_to_tree_cache(build_program())
        self.assertTrue(any(
            "forest may be modified" in msg
            for msg in tlc.get_metrics().messages
        ))

        instrs = compile_hir_to_vliw(build_program())
        mem = [0] * 64
        mem[4] = 10
        mem[10] = 2
        machine = Machine(
            mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES
        )
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        self.assertEqual(machine.mem[10], 99)
        self.assertEqual(machine.mem[30], 101)

    def test_root_only_program_preloads_only_root(self):
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        root = b.load(b.add(forest_values_p, b.const(0)), "root")
        b.store(b.const(30), root)

        pm = PassManager()
        tlc = TreeLevelCachePass()
        pm.add_pass(tlc)
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": 4}
        )
        transformed = pm.run(b.build())

        forest_after = _find_header_load(transformed.body, 4)
        self.assertEqual(_count_const_forest_loads(transformed, forest_after), 1)

    def test_unused_bound_compare_does_not_prove_wrap(self):
        # Comparisons that are not part of the matched index recurrence must
        # not influence wrap inference at all: the no-wrap chain transforms
        # exactly as it would without the stray compares, and wrap_period
        # stays None (regression: the old inference scanned every '<' const
        # in the body).
        def build(with_compares: bool):
            b = HIRBuilder()
            forest_values_p = b.load(b.const(4), "forest_values_p")
            idx = b.const(0)
            for round_index in range(4):
                node_val = b.load(
                    b.add(forest_values_p, idx), f"node_val_{round_index}"
                )
                branch = b.and_(node_val, b.const(1), f"branch_{round_index}")
                offset = b.add(branch, b.const(1), f"offset_{round_index}")
                doubled = b.mul(idx, b.const(2), f"doubled_{round_index}")
                idx = b.add(doubled, offset, f"next_idx_{round_index}")
                if with_compares:
                    _ = b.lt(idx, b.const(7), f"unused_bound_{round_index}")
            b.store(b.const(100), idx)
            return b.build()

        results = {}
        for with_compares in (False, True):
            pm = PassManager()
            tlc = TreeLevelCachePass()
            pm.add_pass(tlc)
            pm.config["tree-level-cache"] = PassConfig(
                name="tree-level-cache", enabled=True, options={"levels": 1}
            )
            transformed = pm.run(build(with_compares))
            results[with_compares] = (transformed, tlc.get_metrics().custom)

        for with_compares, (_, metrics) in results.items():
            self.assertIsNone(metrics.get("wrap_period"),
                              f"with_compares={with_compares}")
            self.assertEqual(metrics.get("node_loads_replaced"), 1,
                             f"with_compares={with_compares}")
        # The stray compares themselves are untouched; the node-load rewrite
        # is identical in both variants.
        plain_loads = [s for s in results[False][0].body
                       if isinstance(s, Op) and s.opcode == "load"]
        compared_loads = [s for s in results[True][0].body
                          if isinstance(s, Op) and s.opcode == "load"]
        self.assertEqual(len(plain_loads), len(compared_loads))

    def test_replaces_early_node_loads(self):
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        b.pause()

        batch_size = 2
        rounds = 3

        indices = [b.const(0) for _ in range(batch_size)]

        for r in range(rounds):
            for i in range(batch_size):
                idx = indices[i]
                node_addr = b.add(forest_values_p, idx, f"node_addr_r{r}_{i}")
                node_val = b.load(node_addr, f"node_val_r{r}_{i}")
                tmp = b.add(node_val, b.const(1), f"tmp_r{r}_{i}")
                b.store(b.const(100 + r * batch_size + i), tmp)
                branch = b.and_(tmp, b.const(1), f"branch_r{r}_{i}")
                offset = b.add(branch, b.const(1), f"offset_r{r}_{i}")
                doubled = b.mul(idx, b.const(2), f"doubled_r{r}_{i}")
                indices[i] = b.add(doubled, offset, f"next_idx_r{r}_{i}")

        hir = b.build()

        forest_before = _find_header_load(hir.body, 4)
        self.assertIsNotNone(forest_before)
        before_dynamic = _count_dynamic_node_loads(hir, forest_before)
        self.assertEqual(before_dynamic, batch_size * (rounds - 1))

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

    def test_replaces_levels_again_after_wrap_period(self):
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        b.pause()

        batch_size = 2
        rounds = 5

        indices = [b.const(0) for _ in range(batch_size)]

        # The wrap period is derived from the wrap checks on the matched
        # chains (n_nodes=7 -> period log2(7+1) = 3); this unrelated compare
        # is ignored by the inference.
        _ = b.lt(b.const(0), b.const(7), "period_hint")

        for r in range(rounds):
            for i in range(batch_size):
                idx = indices[i]
                node_addr = b.add(forest_values_p, idx, f"node_addr_r{r}_{i}")
                node_val = b.load(node_addr, f"node_val_r{r}_{i}")
                tmp = b.add(node_val, b.const(1), f"tmp_r{r}_{i}")
                b.store(b.const(200 + r * batch_size + i), tmp)
                branch = b.and_(tmp, b.const(1), f"branch_r{r}_{i}")
                offset = b.add(branch, b.const(1), f"offset_r{r}_{i}")
                doubled = b.mul(idx, b.const(2), f"doubled_r{r}_{i}")
                next_idx = b.add(doubled, offset, f"next_idx_r{r}_{i}")
                in_bounds = b.lt(next_idx, b.const(7), f"in_bounds_r{r}_{i}")
                indices[i] = b.select(
                    in_bounds, next_idx, b.const(0), f"wrapped_r{r}_{i}"
                )

        hir = b.build()

        forest_before = _find_header_load(hir.body, 4)
        self.assertIsNotNone(forest_before)
        before_dynamic = _count_dynamic_node_loads(hir, forest_before)
        self.assertEqual(before_dynamic, batch_size * (rounds - 1))

        pm = PassManager()
        pm.add_pass(TreeLevelCachePass())
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": 2}
        )
        transformed = pm.run(hir)

        forest_after = _find_header_load(transformed.body, 4)
        self.assertIsNotNone(forest_after)
        after_dynamic = _count_dynamic_node_loads(transformed, forest_after)
        # With period=3 and levels=2, cached rounds are phases 0 and 1:
        # rounds 0,1,3,4 replaced, round 2 remains.
        self.assertEqual(after_dynamic, batch_size * 1)

        preload_count = _count_const_forest_loads(transformed, forest_after)
        self.assertEqual(preload_count, (1 << 2) - 1)

    def _build_wrap_chain(self, rounds: int, bounds_per_round: list):
        """One-lane zero-root chain; bounds_per_round[r] wraps the index
        produced after round r (None omits the wrap check)."""
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        idx = b.const(0)
        for r in range(rounds):
            node_val = b.load(b.add(forest_values_p, idx), f"node_val_{r}")
            b.store(b.const(200 + r), node_val)
            branch = b.and_(node_val, b.const(1), f"branch_{r}")
            offset = b.add(branch, b.const(1), f"offset_{r}")
            doubled = b.mul(idx, b.const(2), f"doubled_{r}")
            next_idx = b.add(doubled, offset, f"next_idx_{r}")
            bound = bounds_per_round[r]
            if bound is None:
                idx = next_idx
            else:
                in_bounds = b.lt(next_idx, b.const(bound), f"in_bounds_{r}")
                idx = b.select(in_bounds, next_idx, b.const(0), f"wrapped_{r}")
        b.store(b.const(100), idx)
        return b.build()

    def _run_tlc(self, hir, levels: int = 4):
        pm = PassManager()
        tlc = TreeLevelCachePass()
        pm.add_pass(tlc)
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": levels}
        )
        return pm.run(hir), tlc

    def test_unrelated_compare_does_not_poison_wrap_period(self):
        # Regression: the old inference took max() over every '<' constant in
        # the body, so one unrelated bounded compare (101 not a power of two)
        # yielded wrap_period=None while the wrap selects still peeled, and
        # the wrapped round 3 was cached at level 3 over out-of-forest
        # preloads. The period must come from the chain's own wrap checks.
        hir = self._build_wrap_chain(4, [7, 7, 7, 7])
        # An unrelated live comparison against a non-power-of-two bound.
        b_extra = hir.body
        pval = next(s.result for s in b_extra
                    if isinstance(s, Op) and s.opcode == "load")
        cmp_op = Op("<", SSAValue(hir.num_ssa_values, "pval_cmp"),
                    [pval, Const(100)], "alu")
        store_op = Op("store", None, [Const(101), cmp_op.result], "store")
        hir = HIRFunction(hir.name, [cmp_op, store_op] + list(b_extra),
                          hir.num_ssa_values + 1, hir.num_vec_ssa_values)

        transformed, tlc = self._run_tlc(hir)
        metrics = tlc.get_metrics().custom

        self.assertEqual(metrics.get("wrap_period"), 3)
        self.assertEqual(metrics.get("node_loads_replaced"), 4)
        # Round 3 wrapped back to the root (phase 0): only levels 0..2 are
        # ever preloaded, never the out-of-forest nodes 7..14.
        self.assertEqual(metrics.get("preloads_inserted"), 7)
        self.assertEqual(metrics.get("higher_level_rounds"), [])
        use_def = UseDefContext(transformed)
        forest_after = _find_header_load(transformed.body, 4)
        for stmt in transformed.body:
            if not (isinstance(stmt, Op) and stmt.opcode == "load"):
                continue
            addr = stmt.operands[0]
            if not isinstance(addr, SSAValue):
                continue
            def_loc = use_def.get_def(addr)
            if def_loc is None or not isinstance(def_loc.statement, Op):
                continue
            addr_op = def_loc.statement
            if addr_op.opcode != "+" or forest_after not in addr_op.operands:
                continue
            for operand in addr_op.operands:
                if isinstance(operand, Const):
                    self.assertLess(operand.value, 7,
                                    "preload beyond the 7-node forest")

    def test_mixed_wrap_bounds_reject(self):
        # Two lanes wrapped against different bounds cannot share one wrap
        # period; the pass must skip instead of guessing.
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        indices = [b.const(0), b.const(0)]
        lane_bounds = [7, 15]
        for r in range(2):
            for lane in range(2):
                idx = indices[lane]
                node_val = b.load(
                    b.add(forest_values_p, idx), f"node_val_r{r}_{lane}"
                )
                b.store(b.const(200 + r * 2 + lane), node_val)
                branch = b.and_(node_val, b.const(1), f"branch_r{r}_{lane}")
                offset = b.add(branch, b.const(1), f"offset_r{r}_{lane}")
                doubled = b.mul(idx, b.const(2), f"doubled_r{r}_{lane}")
                next_idx = b.add(doubled, offset, f"next_idx_r{r}_{lane}")
                in_bounds = b.lt(next_idx, b.const(lane_bounds[lane]),
                                 f"in_bounds_r{r}_{lane}")
                indices[lane] = b.select(
                    in_bounds, next_idx, b.const(0), f"wrapped_r{r}_{lane}"
                )
        hir = b.build()

        transformed, tlc = self._run_tlc(hir)

        self.assertEqual(transformed.body, hir.body)
        self.assertTrue(any("inconsistent wrap-check bounds" in msg
                            for msg in tlc.get_metrics().messages))

    def test_missing_wrap_check_at_wrap_round_rejects(self):
        # Bound 7 -> period 3, so the index feeding round 3 must carry the
        # wrap check (it provably wraps there). Omitting exactly that check
        # leaves the chain matchable but the wrap schedule unprovable.
        hir = self._build_wrap_chain(4, [7, 7, None, 7])

        transformed, tlc = self._run_tlc(hir)

        self.assertEqual(transformed.body, hir.body)
        self.assertTrue(any("missing wrap check at wrap round 3" in msg
                            for msg in tlc.get_metrics().messages))


class TestTreeLevelCacheOptions(unittest.TestCase):
    """Tests for zero-root recurrence, post-wrap caching, and branch reuse."""

    def _run_machine(self, instrs, mem, debug_info=None):
        if debug_info is None:
            debug_info = DebugInfo(scratch_map={})
        machine = Machine(mem, instrs, debug_info, n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_local_memory_contract_small_kernel(self):
        """The default kernel declares zero-initialized private index state."""
        forest_height, batch_size, rounds = 4, 16, 4
        random.seed(42)
        forest = Tree.generate(forest_height)
        n_nodes = len(forest.values)
        self.assertEqual(n_nodes, 31)
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)
        ref_mem = list(mem)

        # End-to-end correctness with the default config.
        kb = KernelBuilder()
        kb.build_kernel(forest.height, n_nodes, batch_size, rounds)
        machine = self._run_machine(kb.instrs, mem, kb.debug_info())

        for _ in reference_kernel2(ref_mem):
            pass
        inp_values_p = ref_mem[6]
        self.assertEqual(
            machine.mem[inp_values_p:inp_values_p + batch_size],
            ref_mem[inp_values_p:inp_values_p + batch_size],
            "kernel output does not match reference with local index state",
        )

        # Generic local-memory promotion exposes the zero-root SSA values;
        # TreeLevelCache consumes those values without reading the contract.
        hir = build_tree_hash_kernel(forest_height, n_nodes, batch_size, rounds)
        _, tlc = _run_hir_pipeline_to_tree_cache(hir)
        metrics = tlc.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.custom.get("root_indices_proven"), batch_size)

    def test_default_pipeline_promotes_all_index_accesses_before_memory_opts(self):
        config_data = _load_default_config()
        pipeline = config_data["pipeline"]
        promotion_index = pipeline.index("local-mem2reg")
        self.assertLess(promotion_index, pipeline.index("load-elim"))
        self.assertLess(promotion_index, pipeline.index("dse"))

        manager = PassManager()
        promotion = None
        for name in pipeline[:promotion_index + 1]:
            compiler_pass = PASS_REGISTRY[name]()
            manager.add_pass(compiler_pass)
            options = config_data["passes"].get(name, {})
            manager.config[name] = PassConfig(
                name=name,
                enabled=options.get("enabled", True),
                options=options.get("options", {}),
            )
            if name == "local-mem2reg":
                promotion = compiler_pass

        hir = build_tree_hash_kernel(
            forest_height=10,
            n_nodes=(1 << 11) - 1,
            batch_size=256,
            rounds=16,
        )
        manager.run(hir)

        self.assertIsNotNone(promotion)
        metrics = promotion.get_metrics()
        self.assertEqual(metrics.custom.get("loads_promoted"), 4096)
        self.assertEqual(metrics.custom.get("stores_removed"), 4096)

    def test_post_wrap_levels_metrics(self):
        """post_wrap_levels extends caching to post-wrap rounds: with
        wrap_period=3 (forest_height=2), rounds=5, levels=1 and
        post_wrap_levels=2, rounds 0 (pre-wrap phase 0) and 3, 4 (post-wrap
        phases 0, 1) are cached; rounds 1 and 2 are not."""
        forest_height, batch_size, rounds = 2, 8, 5
        n_nodes = (1 << (forest_height + 1)) - 1  # 7
        wrap_period = forest_height + 1  # 3
        levels, post_wrap_levels = 1, 2

        hir = build_tree_hash_kernel(forest_height, n_nodes, batch_size, rounds)
        transformed, tlc = _run_hir_pipeline_to_tree_cache(hir, {
            "levels": levels,
            "post_wrap_levels": post_wrap_levels,
        })
        metrics = tlc.get_metrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.custom.get("wrap_period"), wrap_period)

        cached_rounds = sum(
            1 for r in range(rounds)
            if (r % wrap_period) < (levels if r < wrap_period
                                    else post_wrap_levels)
        )
        self.assertEqual(cached_rounds, 3)  # rounds 0, 3, 4
        self.assertEqual(
            metrics.custom.get("node_loads_replaced"),
            cached_rounds * batch_size,
        )

        forest_after = _find_header_load(transformed.body, 4)
        self.assertIsNotNone(forest_after)
        self.assertEqual(
            _count_dynamic_node_loads(transformed, forest_after),
            (rounds - cached_rounds) * batch_size,
        )

    def test_post_wrap_levels_end_to_end(self):
        """Compiling with levels=1, post_wrap_levels=2 (so post-wrap rounds
        use the cached select trees) still yields correct results."""
        forest_height, batch_size, rounds = 2, 8, 5
        cfg = _load_default_config()
        cfg["passes"]["tree-level-cache"]["options"]["levels"] = 1
        cfg["passes"]["tree-level-cache"]["options"]["post_wrap_levels"] = 2
        with tempfile.NamedTemporaryFile("w", suffix=".json",
                                         delete=False) as tf:
            json.dump(cfg, tf)
            tmp_path = tf.name
        try:
            random.seed(7)
            forest = Tree.generate(forest_height)
            inp = Input.generate(forest, batch_size, rounds)
            mem = build_mem_image(forest, inp)
            ref_mem = list(mem)

            kb = KernelBuilder()
            kb.build_kernel(forest.height, len(forest.values), batch_size,
                            rounds, pass_config=tmp_path)
            machine = self._run_machine(kb.instrs, mem, kb.debug_info())
        finally:
            os.unlink(tmp_path)

        for _ in reference_kernel2(ref_mem):
            pass
        inp_values_p = ref_mem[6]
        self.assertEqual(
            machine.mem[inp_values_p:inp_values_p + batch_size],
            ref_mem[inp_values_p:inp_values_p + batch_size],
            "kernel output does not match reference with post_wrap_levels=2",
        )

    def _build_two_round_chain(self):
        """Build a hand-written 2-round index-update chain in the standard
        shape: next = 2*idx + (bit + 1), wrapped by select(next < 7, next, 0).
        Returns (hir, idx1, idx2, bit0, bit1)."""
        b = HIRBuilder()
        idx0 = b.load(b.const(100), "idx0")
        val0 = b.load(b.const(101), "val0")
        bit0 = b.and_(val0, b.const(1), "bit0")
        off0 = b.add(bit0, b.const(1), "off0")
        dbl0 = b.mul(idx0, b.const(2), "dbl0")
        nxt0 = b.add(dbl0, off0, "nxt0")
        inb0 = b.lt(nxt0, b.const(7), "inb0")
        idx1 = b.select(inb0, nxt0, b.const(0), "idx1")

        val1 = b.load(b.const(102), "val1")
        bit1 = b.and_(val1, b.const(1), "bit1")
        off1 = b.add(bit1, b.const(1), "off1")
        dbl1 = b.mul(idx1, b.const(2), "dbl1")
        nxt1 = b.add(dbl1, off1, "nxt1")
        inb1 = b.lt(nxt1, b.const(7), "inb1")
        idx2 = b.select(inb1, nxt1, b.const(0), "idx2")
        b.store(b.const(103), idx2)
        return b.build(), idx1, idx2, bit0, bit1

    def test_extract_branch_bits_from_index_chain(self):
        """_extract_branch_bits recovers the branch bits (LSB-first, most
        recent round first) from a standard-shape index chain, and the
        select tree built from them needs no offset arithmetic."""
        hir, idx1, idx2, bit0, bit1 = self._build_two_round_chain()
        use_def = UseDefContext(hir)
        p = TreeLevelCachePass()
        p._next_ssa_id = hir.num_ssa_values

        update1 = p._peel_index_update(idx1, use_def)
        update2 = p._peel_index_update(idx2, use_def)
        self.assertIsNotNone(update1)
        self.assertIsNotNone(update2)
        updates = {idx1: update1, idx2: update2}

        bits2 = p._extract_branch_bits(idx2, 2, updates)
        bits1 = p._extract_branch_bits(idx1, 1, updates)
        self.assertEqual(bits2, [bit1, bit0])
        self.assertEqual(bits1, [bit0])

        node_vals = [SSAValue(1000 + i, f"n{i}") for i in range(7)]

        # The verified branch bits are reused: only selects are emitted.
        ops = []
        self.assertIsNotNone(bits2)
        p._build_select_for_level(2, node_vals, ops, tuple(bits2))
        self.assertEqual([op.opcode for op in ops],
                         ["select", "select", "select"])

    def test_branch_bit_reuse_avoids_offset_sub_in_pass_output(self):
        """Running the pass on a flat HIR whose index chain has the standard
        shape must emit select trees with no '-' offset ops at all."""
        b = HIRBuilder()
        forest_values_p = b.load(b.const(4), "forest_values_p")
        batch_size = 2
        rounds = 2
        cur_idx = [b.const(0) for _ in range(batch_size)]
        for r in range(rounds):
            for i in range(batch_size):
                idx = cur_idx[i]
                node_addr = b.add(forest_values_p, idx, f"node_addr_r{r}_{i}")
                node_val = b.load(node_addr, f"node_val_r{r}_{i}")
                bit = b.and_(node_val, b.const(1), f"bit_r{r}_{i}")
                off = b.add(bit, b.const(1), f"off_r{r}_{i}")
                dbl = b.mul(idx, b.const(2), f"dbl_r{r}_{i}")
                nxt = b.add(dbl, off, f"nxt_r{r}_{i}")
                inb = b.lt(nxt, b.const(7), f"inb_r{r}_{i}")
                fidx = b.select(inb, nxt, b.const(0), f"fidx_r{r}_{i}")
                cur_idx[i] = fidx
                b.store(b.const(200 + r * batch_size + i), fidx)
        hir = b.build()

        selects_before = _count_opcodes(hir.body, "select")
        self.assertEqual(_count_opcodes(hir.body, "-"), 0)

        pm = PassManager()
        tlc = TreeLevelCachePass()
        pm.add_pass(tlc)
        pm.config["tree-level-cache"] = PassConfig(
            name="tree-level-cache", enabled=True, options={"levels": 2}
        )
        transformed = pm.run(hir)

        metrics = tlc.get_metrics()
        self.assertEqual(metrics.custom.get("node_loads_replaced"),
                         batch_size * rounds)
        # Branch-bit reuse: no '-' offset op is needed anywhere; each
        # level-1 replacement adds exactly one select.
        self.assertEqual(_count_opcodes(transformed.body, "-"), 0)
        self.assertEqual(_count_opcodes(transformed.body, "select"),
                         selects_before + batch_size)

if __name__ == "__main__":
    unittest.main()
