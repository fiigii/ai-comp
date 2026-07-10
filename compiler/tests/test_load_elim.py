"""Tests for HIR load elimination pass."""

import unittest

from compiler.tests.conftest import (
    Machine,
    DebugInfo,
    N_CORES,
    HIRBuilder,
    compile_hir_to_vliw,
)
from compiler import PassManager, PassConfig
from compiler.passes import LoadElimPass
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


class TestLoadElimPass(unittest.TestCase):
    """Tests for load elimination (store-to-load forwarding)."""

    def _run_program(self, instrs, mem):
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_load_elim_basic_forward(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        val = b.load(b.const(1), "val")
        addr = b.add(base, b.const(3), "addr")
        b.store(addr, val)
        loaded = b.load(addr, "loaded")
        summed = b.add(loaded, val, "summed")
        b.store(b.const(2), summed)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        pm.config["load-elim"] = PassConfig(name="load-elim", enabled=True, options={"restrict_ptr": True})
        transformed = pm.run(hir)

        # One load should be eliminated
        self.assertEqual(
            _count_opcodes(hir.body, "load") - 1,
            _count_opcodes(transformed.body, "load"),
        )

        # Validate semantics
        instrs = compile_hir_to_vliw(transformed)
        mem = [10, 7, 0] + [0] * 97
        machine = self._run_program(instrs, mem)
        self.assertEqual(machine.mem[2], 14)

    def test_load_elim_no_alias_different_offsets(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        val0 = b.load(b.const(1), "val0")
        val1 = b.load(b.const(2), "val1")

        addr0 = b.add(base, b.const(0), "addr0")
        addr1 = b.add(base, b.const(1), "addr1")
        b.store(addr0, val0)
        b.store(addr1, val1)  # different offset, no alias
        loaded = b.load(addr0, "loaded")
        b.store(b.const(3), loaded)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        pm.config["load-elim"] = PassConfig(name="load-elim", enabled=True, options={"restrict_ptr": True})
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(hir.body, "load") - 1,
            _count_opcodes(transformed.body, "load"),
        )

    def test_load_elim_blocks_on_may_alias(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        val0 = b.load(b.const(1), "val0")

        addr0 = b.add(base, b.const(0), "addr0")
        b.store(addr0, val0)

        idx = b.load(b.const(2), "idx")
        addr_unknown = b.add(base, idx, "addr_unknown")
        b.store(addr_unknown, b.const(9))  # may alias

        loaded = b.load(addr0, "loaded")
        b.store(b.const(3), loaded)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        pm.config["load-elim"] = PassConfig(name="load-elim", enabled=True, options={"restrict_ptr": False})
        transformed = pm.run(hir)

        # With restrict_ptr disabled, load should remain due to may-alias store
        self.assertEqual(
            _count_opcodes(hir.body, "load"),
            _count_opcodes(transformed.body, "load"),
        )

    def test_load_elim_no_alias_distinct_bases(self):
        b = HIRBuilder()
        base_a = b.load(b.const(5), "base_a")
        base_b = b.load(b.const(6), "base_b")
        val_a = b.load(b.const(1), "val_a")
        val_b = b.load(b.const(2), "val_b")

        addr_a = b.add(base_a, b.const(0), "addr_a")
        addr_b = b.add(base_b, b.const(0), "addr_b")
        b.store(addr_a, val_a)
        b.store(addr_b, val_b)  # distinct memslot base

        loaded = b.load(addr_a, "loaded")
        b.store(b.const(3), loaded)

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        pm.config["load-elim"] = PassConfig(name="load-elim", enabled=True, options={"restrict_ptr": True})
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(hir.body, "load") - 1,
            _count_opcodes(transformed.body, "load"),
        )

    def test_load_elim_vector_vload(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        addr = b.add(base, b.const(0), "addr")
        vec0 = b.vload(addr, "vec0")
        b.vstore(addr, vec0)
        vec1 = b.vload(addr, "vec1")
        b.vstore(b.const(16), vec1)  # use vec1

        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        pm.config["load-elim"] = PassConfig(name="load-elim", enabled=True, options={"restrict_ptr": True})
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(hir.body, "vload") - 1,
            _count_opcodes(transformed.body, "vload"),
        )

    def test_pointer_slot_overwrite_blocks_forwarding_across_reloads(self):
        b = HIRBuilder()
        first_base = b.load(b.const(5), "first_base")
        b.store(first_base, b.const(42))
        b.store(b.const(5), b.const(24))
        second_base = b.load(b.const(5), "second_base")
        loaded = b.load(second_base, "loaded")
        b.store(b.const(8), loaded)
        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        pm.config["load-elim"] = PassConfig(
            name="load-elim",
            enabled=True,
            options={"restrict_ptr": True},
        )
        transformed = pm.run(hir)

        # The two values loaded from slot 5 may be different, so the final
        # load cannot be forwarded from the store through first_base.
        self.assertTrue(
            any(
                isinstance(stmt, Op)
                and stmt.opcode == "load"
                and stmt.result == loaded
                for stmt in transformed.body
            )
        )

        mem = [0] * 64
        mem[5] = 16
        mem[24] = 7
        machine = self._run_program(compile_hir_to_vliw(transformed), mem)
        self.assertEqual(machine.mem[8], 7)
        self.assertEqual(machine.mem[16], 42)

    def test_equivalent_constant_slot_address_blocks_bad_forwarding(self):
        b = HIRBuilder()
        literal_base = b.load(b.const(4), "literal_base")
        computed_slot = b.add(b.const(2), b.const(2), "computed_slot")
        computed_base = b.load(computed_slot, "computed_base")
        b.store(literal_base, b.const(11))
        b.store(computed_base, b.const(22))
        loaded = b.load(literal_base, "loaded")
        b.store(b.const(8), loaded)
        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        pm.config["load-elim"] = PassConfig(
            name="load-elim",
            enabled=True,
            options={"restrict_ptr": True},
        )
        transformed = pm.run(hir)

        self.assertTrue(any(
            isinstance(stmt, Op)
            and stmt.opcode == "load"
            and stmt.result == loaded
            for stmt in transformed.body
        ))

        mem = [0] * 64
        mem[4] = 16
        machine = self._run_program(compile_hir_to_vliw(transformed), mem)
        self.assertEqual(machine.mem[8], 22)

    def test_pause_blocks_forwarding(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        b.store(base, b.const(11))
        b.pause()
        loaded = b.load(base, "loaded")
        b.store(b.const(1), loaded)
        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        transformed = pm.run(hir)

        self.assertTrue(
            any(
                isinstance(stmt, Op)
                and stmt.opcode == "load"
                and stmt.result == loaded
                for stmt in transformed.body
            )
        )

    def test_halt_blocks_forwarding_into_unreachable_code(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        b.store(base, b.const(11))
        b.halt()
        loaded = b.load(base, "loaded")
        b.store(b.const(1), loaded)
        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        transformed = pm.run(hir)

        self.assertTrue(
            any(
                isinstance(stmt, Op)
                and stmt.opcode == "load"
                and stmt.result == loaded
                for stmt in transformed.body
            )
        )


class TestRangeBasedForwarding(unittest.TestCase):
    """A store whose dynamic offset range is disjoint from a reloaded slot
    must not block forwarding -- proven purely by value ranges around the
    shared base, with NO restrict_ptr contract."""

    def _run_program(self, instrs, mem):
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_range_disjoint_dynamic_store_allows_forwarding(self):
        b = HIRBuilder()
        base = b.load(b.const(0), "base")
        x = b.load(b.const(1), "x")
        val = b.load(b.const(2), "val")
        i = b.and_(x, b.const(7), "i")                       # [0, 7]
        b.store(b.add(base, b.const(2), "dst"), val)
        st_off = b.add(i, b.const(8), "st_off")              # [8, 15]
        b.store(b.add(base, st_off, "st_addr"), b.const(999))
        loaded = b.load(b.add(base, b.const(2), "reload"), "loaded")
        b.store(b.const(3), loaded)
        hir = b.build()

        pm = PassManager()
        pm.add_pass(LoadElimPass())
        pm.config["load-elim"] = PassConfig(
            name="load-elim", enabled=True, options={"restrict_ptr": False})
        transformed = pm.run(hir)

        self.assertEqual(
            _count_opcodes(hir.body, "load") - 1,
            _count_opcodes(transformed.body, "load"),
            "the load must forward across the range-disjoint dynamic store")

        mem = [0] * 64
        mem[0] = 16   # base of the table
        mem[1] = 41   # i = 41 & 7 = 1 -> dynamic store lands at 16 + 9 = 25
        mem[2] = 7    # the stored-then-reloaded value
        machine = self._run_program(compile_hir_to_vliw(transformed), mem)
        self.assertEqual(machine.mem[3], 7)
        self.assertEqual(machine.mem[18], 7)
        self.assertEqual(machine.mem[25], 999)


if __name__ == "__main__":
    unittest.main()
