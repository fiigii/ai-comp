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

    def test_dynamic_scalar_load_marks_may_alias_store(self):
        b = HIRBuilder()
        base = b.load(b.const(10), "base")
        store_index = b.load(b.const(1), "store_index")
        load_index = b.load(b.const(2), "load_index")
        store_addr = b.add(base, store_index, "store_addr")
        b.store(store_addr, b.const(42))
        load_addr = b.add(base, load_index, "load_addr")
        loaded = b.load(load_addr, "loaded")
        b.store(b.const(0), loaded)
        b.store(store_addr, b.const(99))
        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        pm.config["dse"] = PassConfig(
            name="dse", enabled=True, options={"restrict_ptr": True}
        )
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(transformed.body, "store"),
            _count_opcodes(hir.body, "store"),
        )
        mem = [0] * 128
        mem[1] = 3
        mem[2] = 3
        mem[10] = 64
        machine = self._run_program(compile_hir_to_vliw(transformed), mem)
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[67], 99)

    def test_dynamic_vector_load_marks_may_alias_store(self):
        b = HIRBuilder()
        base = b.load(b.const(10), "base")
        store_index = b.load(b.const(1), "store_index")
        load_index = b.load(b.const(2), "load_index")
        store_addr = b.add(base, store_index, "store_addr")
        b.store(store_addr, b.const(42))
        load_addr = b.add(base, load_index, "load_addr")
        loaded = b.vload(load_addr, "loaded")
        lane_zero = b.vextract(loaded, 0, "lane_zero")
        b.store(b.const(0), lane_zero)
        b.store(store_addr, b.const(99))
        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        pm.config["dse"] = PassConfig(
            name="dse", enabled=True, options={"restrict_ptr": True}
        )
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(transformed.body, "store"),
            _count_opcodes(hir.body, "store"),
        )
        mem = [0] * 128
        mem[1] = 3
        mem[2] = 3
        mem[10] = 64
        machine = self._run_program(compile_hir_to_vliw(transformed), mem)
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[67], 99)

    def test_wrapped_scalar_load_marks_same_address_store(self):
        b = HIRBuilder()
        base = b.load(b.const(10), "base")
        b.store(base, b.const(42))
        wrapped_address = b.add(base, b.const(1 << 32), "wrapped_address")
        loaded = b.load(wrapped_address, "loaded")
        b.store(b.const(0), loaded)
        b.store(base, b.const(99))
        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        pm.config["dse"] = PassConfig(
            name="dse", enabled=True, options={"restrict_ptr": True}
        )
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(transformed.body, "store"),
            _count_opcodes(hir.body, "store"),
        )
        mem = [0] * 128
        mem[10] = 64
        machine = self._run_program(compile_hir_to_vliw(transformed), mem)
        self.assertEqual(machine.mem[0], 42)
        self.assertEqual(machine.mem[64], 99)

    def test_pointer_slot_overwrite_keeps_stores_to_both_loaded_addresses(self):
        b = HIRBuilder()
        first_base = b.load(b.const(5), "first_base")
        b.store(first_base, b.const(11))
        b.store(b.const(5), b.const(24))
        second_base = b.load(b.const(5), "second_base")
        b.store(second_base, b.const(22))
        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        pm.config["dse"] = PassConfig(
            name="dse",
            enabled=True,
            options={"restrict_ptr": True},
        )
        transformed = pm.run(hir)

        # Reloading a pointer slot produces a new SSA value. Its runtime
        # address can differ after the intervening store, so neither pointed
        # store overwrites the other.
        self.assertEqual(
            _count_opcodes(hir.body, "store"),
            _count_opcodes(transformed.body, "store"),
        )

        mem = [0] * 64
        mem[5] = 16
        machine = self._run_program(compile_hir_to_vliw(transformed), mem)
        self.assertEqual(machine.mem[16], 11)
        self.assertEqual(machine.mem[24], 22)

    def test_equivalent_constant_slot_load_marks_store_used(self):
        b = HIRBuilder()
        literal_base = b.load(b.const(4), "literal_base")
        computed_slot = b.add(b.const(2), b.const(2), "computed_slot")
        computed_base = b.load(computed_slot, "computed_base")
        b.store(literal_base, b.const(11))
        loaded = b.load(computed_base, "loaded")
        b.store(b.const(8), loaded)
        b.store(literal_base, b.const(22))
        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        pm.config["dse"] = PassConfig(
            name="dse",
            enabled=True,
            options={"restrict_ptr": True},
        )
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(hir.body, "store"),
            _count_opcodes(transformed.body, "store"),
        )

        mem = [0] * 64
        mem[4] = 16
        machine = self._run_program(compile_hir_to_vliw(transformed), mem)
        self.assertEqual(machine.mem[8], 11)

    def test_pause_keeps_store_before_later_overwrite(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        b.store(base, b.const(11))
        b.pause()
        b.store(base, b.const(22))
        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(hir.body, "store"),
            _count_opcodes(transformed.body, "store"),
        )

    def test_halt_keeps_store_before_unreachable_overwrite(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        b.store(base, b.const(11))
        b.halt()
        b.store(base, b.const(22))
        hir = b.build()

        pm = PassManager()
        pm.add_pass(DSEPass())
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(hir.body, "store"),
            _count_opcodes(transformed.body, "store"),
        )


if __name__ == "__main__":
    unittest.main()
