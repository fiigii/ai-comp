"""Tests for Dead Store Elimination pass."""

import unittest

from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    HIRBuilder,
    compile_hir_to_vliw,
)
from compiler import PassManager, PassConfig
from compiler.passes import DSEPass
from compiler.hir import Op, ForLoop, If


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


class TestDSEPass(unittest.TestCase):
    """Tests for dead store elimination (store-store forwarding)."""

    def _run_program(self, instrs, mem):
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_dse_basic_elim(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        addr = b.add(base, b.const(0), "addr")
        b.store(addr, b.const(11))
        b.store(addr, b.const(22))
        loaded = b.load(addr, "loaded")
        b.store(b.const(3), loaded)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        pm.config["dse"] = PassConfig(name="dse", enabled=True, options={"restrict_ptr": True})
        transformed = pm.run(hir)

        # One store should be eliminated
        self.assertEqual(
            _count_opcodes(hir.body, "store") - 1,
            _count_opcodes(transformed.body, "store"),
        )

        # Validate semantics
        instrs = compile_hir_to_vliw(transformed)
        mem = [10, 0, 0, 0] + [0] * 96
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[3], 22)

    def test_dse_preserves_store_if_load_between(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        addr = b.add(base, b.const(0), "addr")
        b.store(addr, b.const(11))
        loaded = b.load(addr, "loaded")
        b.store(addr, b.const(22))
        b.store(b.const(3), loaded)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        pm.config["dse"] = PassConfig(name="dse", enabled=True, options={"restrict_ptr": True})
        transformed = pm.run(hir)

        # Store should remain due to intervening load
        self.assertEqual(
            _count_opcodes(hir.body, "store"),
            _count_opcodes(transformed.body, "store"),
        )

    def test_dse_no_alias_different_offsets(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        addr0 = b.add(base, b.const(0), "addr0")
        addr1 = b.add(base, b.const(1), "addr1")
        b.store(addr0, b.const(11))
        b.store(addr1, b.const(22))
        loaded = b.load(addr0, "loaded")
        b.store(b.const(3), loaded)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        pm.config["dse"] = PassConfig(name="dse", enabled=True, options={"restrict_ptr": True})
        transformed = pm.run(hir)

        # Stores should remain since addresses differ
        self.assertEqual(
            _count_opcodes(hir.body, "store"),
            _count_opcodes(transformed.body, "store"),
        )


if __name__ == "__main__":
    unittest.main()
