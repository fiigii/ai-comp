"""Tests for late removal of compile-time-only HIR assumptions."""

from __future__ import annotations

import json
import os
import tempfile

from compiler import HIRBuilder, PassConfig, PassManager, compile_hir_to_vliw
from compiler.hir import If, Op
from compiler.passes import DCEPass, LocalMem2RegPass, StripAssumePass
from compiler.tests.conftest import DebugInfo, Machine, N_CORES


def _cleanup_with_promotion_disabled(hir):
    manager = PassManager()
    manager.add_pass(LocalMem2RegPass())
    manager.add_pass(StripAssumePass())
    manager.add_pass(DCEPass())
    manager.config["local-mem2reg"] = PassConfig(
        name="local-mem2reg", enabled=False, options={}
    )
    manager.config["strip-assume"] = PassConfig(
        name="strip-assume", enabled=True, options={}
    )
    manager.config["dce"] = PassConfig(name="dce", enabled=True, options={})
    return manager.run(hir)


def test_disabled_promotion_does_not_keep_marker_only_if_alive():
    builder = HIRBuilder()
    condition = builder.load(builder.const(0), "condition")
    base = builder.load(builder.const(1), "base")

    def then_body():
        builder.assume_local_memory(base, builder.const(4))
        return []

    builder.if_stmt(condition, then_body, lambda: [])

    transformed = _cleanup_with_promotion_disabled(builder.build())

    assert transformed.body == []


def test_disabled_promotion_does_not_keep_marker_only_loop_alive():
    builder = HIRBuilder()
    base = builder.load(builder.const(1), "base")

    def body(_counter, _params):
        builder.assume_local_memory(base, builder.const(4))
        return []

    builder.for_loop(
        builder.const(0), builder.const(2), [], body, pragma_unroll=1
    )

    transformed = _cleanup_with_promotion_disabled(builder.build())

    assert transformed.body == []


def test_strip_preserves_live_control_flow_and_store():
    builder = HIRBuilder()
    condition = builder.load(builder.const(0), "condition")
    base = builder.load(builder.const(1), "base")

    def then_body():
        builder.assume_local_memory(base, builder.const(4))
        builder.store(builder.const(10), builder.const(7))
        return []

    builder.if_stmt(condition, then_body, lambda: [])

    transformed = _cleanup_with_promotion_disabled(builder.build())

    assert not any(
        isinstance(stmt, Op) and stmt.engine == "meta"
        for statement in transformed.body
        for stmt in (
            statement.then_body if isinstance(statement, If) else [statement]
        )
    )
    assert any(isinstance(stmt, If) for stmt in transformed.body)


def _config_with_local_cleanup_disabled():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pass_config.json"
    )
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["passes"]["local-mem2reg"]["enabled"] = False
    config["passes"]["strip-assume"]["enabled"] = False
    return config


def _write_temporary_config(config):
    file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    with file:
        json.dump(config, file)
    return file.name


def _run_local_round_trip(config):
    builder = HIRBuilder()
    base = builder.load(builder.const(10), "base")
    builder.assume_local_memory(base, builder.const(1))
    initial = builder.load(base, "initial")
    updated = builder.add(initial, builder.const(5), "updated")
    builder.store(base, updated)
    reloaded = builder.load(base, "reloaded")
    builder.store(builder.const(0), reloaded)

    temporary_config = _write_temporary_config(config)
    try:
        instructions = compile_hir_to_vliw(
            builder.build(), pass_config=temporary_config
        )
    finally:
        os.unlink(temporary_config)

    memory = [0] * 64
    memory[10] = 32
    memory[32] = 0
    machine = Machine(
        memory,
        instructions,
        DebugInfo(scratch_map={}),
        n_cores=N_CORES,
    )
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()
    return machine


def test_disabled_promotion_keeps_local_accesses_as_ordinary_memory():
    machine = _run_local_round_trip(_config_with_local_cleanup_disabled())

    assert machine.mem[0] == 5


def test_lowering_fallback_when_strip_assume_is_omitted():
    config = _config_with_local_cleanup_disabled()
    config["pipeline"].remove("strip-assume")
    machine = _run_local_round_trip(config)

    assert machine.mem[0] == 5


def test_default_pipeline_strips_assumptions_before_cleanup_dce():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "pass_config.json"
    )
    with open(config_path) as config_file:
        pipeline = json.load(config_file)["pipeline"]

    promotion_index = pipeline.index("local-mem2reg")
    assert pipeline[promotion_index:promotion_index + 6] == [
        "local-mem2reg",
        "strip-assume",
        "dce",
        "load-elim",
        "dse",
        "tree-level-cache",
    ]
