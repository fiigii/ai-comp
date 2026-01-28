"""
Tests for vector instruction support.
"""

import pytest
from compiler.hir import SSAValue, VectorSSAValue, Const, HIRFunction, Op
from compiler.hir_builder import HIRBuilder
from compiler.lowering import lower_to_lir, VLEN
from compiler.codegen import compile_to_vliw
from compiler.passes.phi_elimination import eliminate_phis
from compiler.lir import LIROpcode


def _build_vgather_hir() -> HIRFunction:
    b = HIRBuilder()
    addr_base = b.const(0)
    addr_indices = b.const(16)
    vec_indices = b.vload(addr_indices)
    vec_base = b.vbroadcast(addr_base)
    vec_addr = b.vadd(vec_base, vec_indices)

    vec_out = b._new_vec_ssa("gather")
    b._emit(Op("vgather", vec_out, [vec_addr], "load"))
    b.vstore(b.const(64), vec_out)
    b.halt()
    return b.build()




class TestVectorTypes:
    """Tests for vector SSA types."""

    def test_vector_ssa_value_creation(self):
        """VectorSSAValue should be distinct from SSAValue."""
        scalar = SSAValue(0, "scalar")
        vector = VectorSSAValue(0, "vector")

        assert isinstance(scalar, SSAValue)
        assert isinstance(vector, VectorSSAValue)
        assert not isinstance(scalar, VectorSSAValue)
        assert not isinstance(vector, SSAValue)

    def test_vector_ssa_repr(self):
        """VectorSSAValue should have a distinct repr."""
        v = VectorSSAValue(5, "data")
        assert "vec5" in repr(v)
        assert "data" in repr(v)


class TestVectorHIRBuilder:
    """Tests for vector builder methods."""

    def test_vbroadcast(self):
        """vbroadcast should create a vector from a scalar."""
        b = HIRBuilder()
        scalar = b.const(42)
        vec = b.vbroadcast(scalar)

        assert isinstance(vec, VectorSSAValue)
        hir = b.build()
        assert hir.num_vec_ssa_values == 1

    def test_vload_vstore(self):
        """vload and vstore should handle vector memory operations."""
        b = HIRBuilder()
        addr = b.const(100)
        vec = b.vload(addr)
        b.vstore(addr, vec)

        assert isinstance(vec, VectorSSAValue)
        hir = b.build()
        assert hir.num_vec_ssa_values == 1

    def test_vector_binary_ops(self):
        """Vector binary ops should produce vector results."""
        b = HIRBuilder()
        addr1 = b.const(0)
        addr2 = b.const(8)
        a = b.vload(addr1)
        c = b.vload(addr2)

        # Test all binary ops
        result_add = b.vadd(a, c)
        result_sub = b.vsub(a, c)
        result_mul = b.vmul(a, c)
        result_xor = b.vxor(a, c)

        assert all(isinstance(r, VectorSSAValue) for r in [
            result_add, result_sub, result_mul, result_xor
        ])

    def test_multiply_add(self):
        """multiply_add should perform fused multiply-add."""
        b = HIRBuilder()
        addr = b.const(0)
        a = b.vload(addr)
        c = b.vload(addr)
        d = b.vload(addr)

        result = b.multiply_add(a, c, d)
        assert isinstance(result, VectorSSAValue)

    def test_vselect(self):
        """vselect should select per-lane."""
        b = HIRBuilder()
        addr = b.const(0)
        cond = b.vload(addr)
        a = b.vload(addr)
        c = b.vload(addr)

        result = b.vselect(cond, a, c)
        assert isinstance(result, VectorSSAValue)

    def test_vextract(self):
        """vextract should extract a scalar from a vector."""
        b = HIRBuilder()
        addr = b.const(0)
        vec = b.vload(addr)
        scalar = b.vextract(vec, 3)

        assert isinstance(scalar, SSAValue)
        assert not isinstance(scalar, VectorSSAValue)

    def test_vinsert(self):
        """vinsert should insert a scalar into a vector."""
        b = HIRBuilder()
        addr = b.const(0)
        vec = b.vload(addr)
        scalar = b.const(99)
        new_vec = b.vinsert(vec, scalar, 5)

        assert isinstance(new_vec, VectorSSAValue)


class TestVectorLowering:
    """Tests for vector instruction lowering."""

    def test_vbroadcast_lowering(self):
        """vbroadcast should lower to LIR with vector dest."""
        b = HIRBuilder()
        scalar = b.const(42)
        vec = b.vbroadcast(scalar)
        b.vstore(b.const(100), vec)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)

        # Find vbroadcast instruction
        vbroadcast_found = False
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.opcode == LIROpcode.VBROADCAST:
                    vbroadcast_found = True
                    # Dest should be a list of 8 addresses
                    assert isinstance(inst.dest, list)
                    assert len(inst.dest) == VLEN
                    # Operand should be a single scalar address
                    assert isinstance(inst.operands[0], int)

        assert vbroadcast_found

    def test_vload_lowering(self):
        """vload should lower to LIR with vector dest."""
        b = HIRBuilder()
        addr = b.const(0)
        vec = b.vload(addr)
        b.vstore(b.const(100), vec)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)

        # Find vload instruction
        vload_found = False
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.opcode == LIROpcode.VLOAD:
                    vload_found = True
                    assert isinstance(inst.dest, list)
                    assert len(inst.dest) == VLEN

        assert vload_found

    def test_vadd_lowering(self):
        """vadd should lower with vector operands."""
        b = HIRBuilder()
        addr1 = b.const(0)
        addr2 = b.const(8)
        a = b.vload(addr1)
        c = b.vload(addr2)
        result = b.vadd(a, c)
        b.vstore(b.const(16), result)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)

        # Find vadd instruction
        vadd_found = False
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.opcode == LIROpcode.VADD:
                    vadd_found = True
                    # Dest and operands should be lists
                    assert isinstance(inst.dest, list)
                    assert len(inst.dest) == VLEN
                    assert isinstance(inst.operands[0], list)
                    assert isinstance(inst.operands[1], list)

        assert vadd_found

    def test_vextract_lowering(self):
        """vextract should lower to COPY instruction."""
        b = HIRBuilder()
        addr = b.const(0)
        vec = b.vload(addr)
        scalar = b.vextract(vec, 3)
        b.store(b.const(100), scalar)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)

        # Find COPY instruction (vextract expansion)
        copy_found = False
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.opcode == LIROpcode.COPY:
                    copy_found = True
                    # Dest should be scalar, operand should be scalar
                    assert isinstance(inst.dest, int)
                    assert isinstance(inst.operands[0], int)

        assert copy_found

    def test_vinsert_lowering(self):
        """vinsert should lower to multiple COPY instructions."""
        b = HIRBuilder()
        addr = b.const(0)
        vec = b.vload(addr)
        scalar = b.const(99)
        new_vec = b.vinsert(vec, scalar, 5)
        b.vstore(b.const(100), new_vec)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)

        # Should have VLEN COPY instructions for vinsert
        copy_count = 0
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.opcode == LIROpcode.COPY:
                    copy_count += 1

        assert copy_count == VLEN

    def test_vgather_lowering_to_load_offset(self):
        """vgather should lower to VLEN load_offset instructions."""
        hir = _build_vgather_hir()
        lir = lower_to_lir(hir)

        load_offset = []
        load_dests = []
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.opcode == LIROpcode.LOAD_OFFSET:
                    load_offset.append(inst)
                if inst.opcode == LIROpcode.LOAD and isinstance(inst.dest, int):
                    load_dests.append(inst.dest)

        assert len(load_offset) == VLEN
        dest_base = load_offset[0].dest
        assert all(inst.dest == dest_base for inst in load_offset)
        offsets = sorted(inst.operands[1] for inst in load_offset)
        assert offsets == list(range(VLEN))
        assert all(d not in range(dest_base, dest_base + VLEN) for d in load_dests)


class TestVectorCodegen:
    """Tests for vector instruction codegen."""

    def test_vload_codegen(self):
        """vload should generate correct VLIW slot."""
        b = HIRBuilder()
        addr = b.const(0)
        vec = b.vload(addr)
        b.vstore(b.const(8), vec)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        bundles = compile_to_vliw(lir)

        # Find vload bundle
        vload_found = False
        for bundle in bundles:
            if "load" in bundle:
                for slot in bundle["load"]:
                    if slot[0] == "vload":
                        vload_found = True
                        # Format: ("vload", dest_base, addr)
                        assert len(slot) == 3

        assert vload_found

    def test_vstore_codegen(self):
        """vstore should generate correct VLIW slot."""
        b = HIRBuilder()
        scalar = b.const(42)
        vec = b.vbroadcast(scalar)
        addr = b.const(100)
        b.vstore(addr, vec)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        bundles = compile_to_vliw(lir)

        # Find vstore bundle
        vstore_found = False
        for bundle in bundles:
            if "store" in bundle:
                for slot in bundle["store"]:
                    if slot[0] == "vstore":
                        vstore_found = True
                        # Format: ("vstore", addr, src_base)
                        assert len(slot) == 3

        assert vstore_found

    def test_vadd_codegen(self):
        """vadd should generate scalar op with base addresses."""
        b = HIRBuilder()
        addr = b.const(0)
        a = b.vload(addr)
        c = b.vload(addr)
        result = b.vadd(a, c)
        b.vstore(b.const(24), result)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        bundles = compile_to_vliw(lir)

        # Find add bundle (vadd becomes "+" on valu engine)
        add_found = False
        for bundle in bundles:
            if "valu" in bundle:
                for slot in bundle["valu"]:
                    if slot[0] == "+":
                        add_found = True
                        # Format: ("+", dest_base, a_base, b_base)
                        assert len(slot) == 4

        assert add_found

    def test_vbroadcast_codegen(self):
        """vbroadcast should generate correct VLIW slot."""
        b = HIRBuilder()
        scalar = b.const(42)
        vec = b.vbroadcast(scalar)
        b.vstore(b.const(100), vec)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        bundles = compile_to_vliw(lir)

        # Find vbroadcast bundle
        vbroadcast_found = False
        for bundle in bundles:
            if "valu" in bundle:
                for slot in bundle["valu"]:
                    if slot[0] == "vbroadcast":
                        vbroadcast_found = True
                        # Format: ("vbroadcast", dest_base, scalar)
                        assert len(slot) == 3

        assert vbroadcast_found

    def test_multiply_add_codegen(self):
        """multiply_add should generate correct VLIW slot."""
        b = HIRBuilder()
        addr = b.const(0)
        a = b.vload(addr)
        c = b.vload(addr)
        d = b.vload(addr)
        result = b.multiply_add(a, c, d)
        b.vstore(b.const(100), result)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        bundles = compile_to_vliw(lir)

        # Find multiply_add bundle
        ma_found = False
        for bundle in bundles:
            if "valu" in bundle:
                for slot in bundle["valu"]:
                    if slot[0] == "multiply_add":
                        ma_found = True
                        # Format: ("multiply_add", dest_base, a, b, c)
                        assert len(slot) == 5

        assert ma_found

    def test_vselect_codegen(self):
        """vselect should generate correct VLIW slot."""
        b = HIRBuilder()
        addr = b.const(0)
        cond = b.vload(addr)
        a = b.vload(addr)
        c = b.vload(addr)
        result = b.vselect(cond, a, c)
        b.vstore(b.const(100), result)
        b.halt()

        hir = b.build()
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        bundles = compile_to_vliw(lir)

        # Find vselect bundle
        vselect_found = False
        for bundle in bundles:
            if "flow" in bundle:
                for slot in bundle["flow"]:
                    if slot[0] == "vselect":
                        vselect_found = True
                        # Format: ("vselect", dest_base, cond_base, a_base, b_base)
                        assert len(slot) == 5

        assert vselect_found

    def test_load_offset_codegen(self):
        """load_offset should generate correct VLIW slot."""
        hir = _build_vgather_hir()
        lir = lower_to_lir(hir)
        eliminate_phis(lir)
        bundles = compile_to_vliw(lir)

        load_offset_found = False
        for bundle in bundles:
            if "load" in bundle:
                for slot in bundle["load"]:
                    if slot[0] == "load_offset":
                        load_offset_found = True
                        assert len(slot) == 4

        assert load_offset_found


class TestVectorDCE:
    """Tests for dead code elimination with vectors."""

    def test_dead_vload_eliminated(self):
        """Unused vload should be eliminated by DCE."""
        from compiler.passes.dce import DCEPass
        from compiler.pass_manager import PassConfig

        b = HIRBuilder()
        addr = b.const(0)
        _dead_vec = b.vload(addr)  # Not used
        b.halt()

        hir = b.build()

        dce = DCEPass()
        config = PassConfig(name="test", enabled=True)
        optimized = dce.run(hir, config)

        # Should have eliminated the vload
        assert len(optimized.body) < len(hir.body)

    def test_vstore_not_eliminated(self):
        """vstore should not be eliminated (side effect)."""
        from compiler.passes.dce import DCEPass
        from compiler.pass_manager import PassConfig

        b = HIRBuilder()
        addr = b.const(0)
        vec = b.vload(addr)
        b.vstore(b.const(100), vec)
        b.halt()

        hir = b.build()

        dce = DCEPass()
        config = PassConfig(name="test", enabled=True)
        optimized = dce.run(hir, config)

        # vstore should be preserved
        vstore_found = any(
            hasattr(stmt, 'opcode') and stmt.opcode == 'vstore'
            for stmt in optimized.body
        )
        assert vstore_found


class TestVectorCSE:
    """Tests for common subexpression elimination with vectors."""

    def test_duplicate_vbroadcast_eliminated(self):
        """Duplicate vbroadcast with same scalar should be CSE'd."""
        from compiler.passes.cse import CSEPass
        from compiler.pass_manager import PassConfig

        b = HIRBuilder()
        scalar = b.const(42)
        vec1 = b.vbroadcast(scalar)
        vec2 = b.vbroadcast(scalar)  # Duplicate
        result = b.vadd(vec1, vec2)
        b.vstore(b.const(100), result)
        b.halt()

        hir = b.build()

        cse = CSEPass()
        config = PassConfig(name="test", enabled=True)
        optimized = cse.run(hir, config)

        # Count vbroadcast ops
        vbroadcast_count = sum(
            1 for stmt in optimized.body
            if hasattr(stmt, 'opcode') and stmt.opcode == 'vbroadcast'
        )

        # Should have eliminated one
        assert vbroadcast_count < 2

    def test_vstore_increments_epoch(self):
        """vstore should increment memory epoch, preventing load CSE across it."""
        from compiler.passes.cse import CSEPass
        from compiler.pass_manager import PassConfig

        b = HIRBuilder()
        addr = b.const(0)
        vec1 = b.vload(addr)
        b.vstore(addr, vec1)  # Clobbers memory
        vec2 = b.vload(addr)  # Should NOT be CSE'd with vec1
        b.vstore(b.const(100), vec2)
        b.halt()

        hir = b.build()

        cse = CSEPass()
        config = PassConfig(name="test", enabled=True)
        optimized = cse.run(hir, config)

        # Should still have 2 vloads (not CSE'd)
        vload_count = sum(
            1 for stmt in optimized.body
            if hasattr(stmt, 'opcode') and stmt.opcode == 'vload'
        )
        assert vload_count == 2

    def test_cse_vector_binding_rewrites_operands(self):
        """CSE should rewrite VectorSSAValue operands when eliminating duplicates.

        Regression test: when CSE eliminates a duplicate vector op, subsequent
        uses of the eliminated value must be rewritten to use the surviving value.
        """
        from compiler.passes.cse import CSEPass
        from compiler.pass_manager import PassConfig
        from compiler import compile_hir_to_vliw
        from problem import Machine, DebugInfo, N_CORES

        b = HIRBuilder()

        # Create two identical vbroadcast ops
        scalar = b.const(42)
        vec1 = b.vbroadcast(scalar)  # First vbroadcast
        vec2 = b.vbroadcast(scalar)  # Duplicate - should be eliminated

        # Use both vectors in an add - after CSE, should use vec1 twice
        result = b.vadd(vec1, vec2)

        addr_out = b.const(0)
        b.vstore(addr_out, result)
        b.halt()

        hir = b.build()

        # Apply CSE
        cse = CSEPass()
        config = PassConfig(name="test", enabled=True)
        optimized = cse.run(hir, config)

        # Verify CSE eliminated one vbroadcast
        vbroadcast_count = sum(
            1 for stmt in optimized.body
            if hasattr(stmt, 'opcode') and stmt.opcode == 'vbroadcast'
        )
        assert vbroadcast_count == 1, "CSE should eliminate duplicate vbroadcast"

        # Compile and run to verify correctness
        instrs = compile_hir_to_vliw(optimized)

        mem = [0] * 100
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # result = vbroadcast(42) + vbroadcast(42) = [84, 84, 84, 84, 84, 84, 84, 84]
        for i in range(8):
            assert machine.mem[i] == 84, f"lane {i}: expected 84, got {machine.mem[i]}"

    def test_cse_vector_vadd_binding(self):
        """CSE should correctly handle duplicate vadd operations."""
        from compiler.passes.cse import CSEPass
        from compiler.pass_manager import PassConfig
        from compiler import compile_hir_to_vliw
        from problem import Machine, DebugInfo, N_CORES

        b = HIRBuilder()

        addr_a = b.const(0)
        addr_b = b.const(8)
        addr_out = b.const(16)

        vec_a = b.vload(addr_a)
        vec_b = b.vload(addr_b)

        # Two identical vadd operations
        sum1 = b.vadd(vec_a, vec_b)  # First
        sum2 = b.vadd(vec_a, vec_b)  # Duplicate - should be eliminated

        # Use both in another operation
        result = b.vadd(sum1, sum2)

        b.vstore(addr_out, result)
        b.halt()

        hir = b.build()

        # Apply CSE
        cse = CSEPass()
        config = PassConfig(name="test", enabled=True)
        optimized = cse.run(hir, config)

        # Verify CSE eliminated one vadd (3 total -> 2)
        vadd_count = sum(
            1 for stmt in optimized.body
            if hasattr(stmt, 'opcode') and stmt.opcode == 'v+'
        )
        assert vadd_count == 2, f"Expected 2 v+ ops after CSE, got {vadd_count}"

        # Compile and run
        instrs = compile_hir_to_vliw(optimized)

        mem = [0] * 100
        # a = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            mem[i] = i + 1
        # b = [10, 10, 10, 10, 10, 10, 10, 10]
        for i in range(8):
            mem[8 + i] = 10

        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # result = (a + b) + (a + b) = 2 * (a + b)
        # = 2 * [11, 12, 13, 14, 15, 16, 17, 18] = [22, 24, 26, 28, 30, 32, 34, 36]
        for i in range(8):
            expected = 2 * ((i + 1) + 10)
            assert machine.mem[16 + i] == expected, f"lane {i}: expected {expected}, got {machine.mem[16 + i]}"

    def test_cse_scalar_vector_id_collision(self):
        """CSE should not confuse scalar and vector SSA IDs that happen to match.

        Regression test: scalar SSA IDs start at 0, vector SSA IDs also start at 0.
        CSE bindings must be kept separate to avoid cross-kind rewrites.
        """
        from compiler.passes.cse import CSEPass
        from compiler.pass_manager import PassConfig
        from compiler import compile_hir_to_vliw
        from problem import Machine, DebugInfo, N_CORES

        b = HIRBuilder()

        # First scalar and first vector both have id=0
        # (add creates scalar v0, vload creates vec0)
        addr = b.add(Const(0), Const(0), "addr")  # SSAValue(id=0)
        vec = b.vload(addr)     # VectorSSAValue(id=0)

        # Create duplicate operations for both
        addr_dup = b.add(Const(0), Const(0), "addr_dup")  # SSAValue(id=1) - same expr, should CSE to v0
        vec_dup = b.vload(addr)     # VectorSSAValue(id=1) - same, should CSE to vec0

        # Use the duplicates to verify bindings work correctly
        addr_out = Const(8)
        result = b.vadd(vec, vec_dup)  # Should use vec0 twice after CSE
        b.vstore(addr_out, result)

        # Also verify scalar binding works
        scalar_sum = b.add(addr, addr_dup)  # Should use v0 twice after CSE
        b.store(Const(100), scalar_sum)

        b.halt()

        hir = b.build()

        # Verify IDs overlap
        assert hir.num_ssa_values > 0, "Should have scalar SSA values"
        assert hir.num_vec_ssa_values > 0, "Should have vector SSA values"

        # Apply CSE
        cse = CSEPass()
        config = PassConfig(name="test", enabled=True)
        optimized = cse.run(hir, config)

        # Compile and run
        instrs = compile_hir_to_vliw(optimized)

        mem = [0] * 200
        # mem[0..7] = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            mem[i] = i + 1

        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        # Vector result at mem[8..15] = vec + vec = 2 * [1..8]
        for i in range(8):
            expected = 2 * (i + 1)
            assert machine.mem[8 + i] == expected, f"vec lane {i}: expected {expected}, got {machine.mem[8 + i]}"

        # Scalar result at mem[100] = 0 + 0 = 0 (the address values)
        assert machine.mem[100] == 0, f"scalar: expected 0, got {machine.mem[100]}"


class TestVectorKernel:
    """End-to-end tests running vector kernels on the VM."""

    def _run_kernel(self, hir, mem):
        """Compile and run a kernel, returning the machine state."""
        from compiler import compile_hir_to_vliw
        from problem import Machine, DebugInfo, N_CORES

        instrs = compile_hir_to_vliw(hir)
        machine = Machine(mem, instrs, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()
        return machine

    def test_vector_add_kernel(self):
        """Test kernel: c[0:8] = a[0:8] + b[0:8]"""
        b = HIRBuilder()

        # Memory layout:
        # [0..7]   = a (input)
        # [8..15]  = b (input)
        # [16..23] = c (output)

        addr_a = b.const(0)
        addr_b = b.const(8)
        addr_c = b.const(16)

        # Load vectors
        vec_a = b.vload(addr_a)
        vec_b = b.vload(addr_b)

        # Add
        vec_c = b.vadd(vec_a, vec_b)

        # Store result
        b.vstore(addr_c, vec_c)
        b.halt()

        hir = b.build()

        # Set up memory
        mem = [0] * 100
        # a = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            mem[i] = i + 1
        # b = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(8):
            mem[8 + i] = (i + 1) * 10

        machine = self._run_kernel(hir, mem)

        # Verify c = a + b
        for i in range(8):
            expected = (i + 1) + (i + 1) * 10
            assert machine.mem[16 + i] == expected, f"c[{i}]: expected {expected}, got {machine.mem[16 + i]}"

    def test_vector_broadcast_multiply(self):
        """Test kernel: result[0:8] = input[0:8] * scalar"""
        b = HIRBuilder()

        # Memory layout:
        # [0..7]   = input vector
        # [8]      = scalar
        # [16..23] = output

        addr_input = b.const(0)
        addr_scalar = b.const(8)
        addr_output = b.const(16)

        # Load input vector
        vec_input = b.vload(addr_input)

        # Load and broadcast scalar
        scalar = b.load(addr_scalar)
        vec_scalar = b.vbroadcast(scalar)

        # Multiply
        vec_result = b.vmul(vec_input, vec_scalar)

        # Store
        b.vstore(addr_output, vec_result)
        b.halt()

        hir = b.build()

        # Set up memory
        mem = [0] * 100
        # input = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            mem[i] = i + 1
        # scalar = 3
        mem[8] = 3

        machine = self._run_kernel(hir, mem)

        # Verify result = input * 3
        for i in range(8):
            expected = (i + 1) * 3
            assert machine.mem[16 + i] == expected, f"result[{i}]: expected {expected}, got {machine.mem[16 + i]}"

    def test_vector_multiply_add_kernel(self):
        """Test kernel: d[0:8] = a[0:8] * b[0:8] + c[0:8]"""
        b = HIRBuilder()

        # Memory layout:
        # [0..7]   = a
        # [8..15]  = b
        # [16..23] = c
        # [24..31] = d (output)

        addr_a = b.const(0)
        addr_b = b.const(8)
        addr_c = b.const(16)
        addr_d = b.const(24)

        # Load vectors
        vec_a = b.vload(addr_a)
        vec_b = b.vload(addr_b)
        vec_c = b.vload(addr_c)

        # Fused multiply-add
        vec_d = b.multiply_add(vec_a, vec_b, vec_c)

        # Store
        b.vstore(addr_d, vec_d)
        b.halt()

        hir = b.build()

        # Set up memory
        mem = [0] * 100
        # a = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            mem[i] = i + 1
        # b = [2, 2, 2, 2, 2, 2, 2, 2]
        for i in range(8):
            mem[8 + i] = 2
        # c = [100, 100, 100, 100, 100, 100, 100, 100]
        for i in range(8):
            mem[16 + i] = 100

        machine = self._run_kernel(hir, mem)

        # Verify d = a * b + c = a * 2 + 100
        for i in range(8):
            expected = (i + 1) * 2 + 100
            assert machine.mem[24 + i] == expected, f"d[{i}]: expected {expected}, got {machine.mem[24 + i]}"

    def test_vector_select_kernel(self):
        """Test kernel: result[i] = cond[i] ? a[i] : b[i]"""
        b = HIRBuilder()

        # Memory layout:
        # [0..7]   = cond (0 or 1)
        # [8..15]  = a
        # [16..23] = b
        # [24..31] = result

        addr_cond = b.const(0)
        addr_a = b.const(8)
        addr_b = b.const(16)
        addr_result = b.const(24)

        # Load vectors
        vec_cond = b.vload(addr_cond)
        vec_a = b.vload(addr_a)
        vec_b = b.vload(addr_b)

        # Select
        vec_result = b.vselect(vec_cond, vec_a, vec_b)

        # Store
        b.vstore(addr_result, vec_result)
        b.halt()

        hir = b.build()

        # Set up memory
        mem = [0] * 100
        # cond = [1, 0, 1, 0, 1, 0, 1, 0]
        for i in range(8):
            mem[i] = 1 if i % 2 == 0 else 0
        # a = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(8):
            mem[8 + i] = (i + 1) * 10
        # b = [100, 200, 300, 400, 500, 600, 700, 800]
        for i in range(8):
            mem[16 + i] = (i + 1) * 100

        machine = self._run_kernel(hir, mem)

        # Verify result[i] = (i % 2 == 0) ? a[i] : b[i]
        for i in range(8):
            if i % 2 == 0:
                expected = (i + 1) * 10
            else:
                expected = (i + 1) * 100
            assert machine.mem[24 + i] == expected, f"result[{i}]: expected {expected}, got {machine.mem[24 + i]}"

    def test_vector_extract_insert_kernel(self):
        """Test kernel: extract lane, modify, insert back."""
        b = HIRBuilder()

        # Memory layout:
        # [0..7]   = input
        # [8..15]  = output (same as input but lane 3 doubled)

        addr_input = b.const(0)
        addr_output = b.const(8)

        # Load vector
        vec = b.vload(addr_input)

        # Extract lane 3
        val = b.vextract(vec, 3)

        # Double it
        two = b.const(2)
        doubled = b.mul(val, two)

        # Insert back at lane 3
        vec_modified = b.vinsert(vec, doubled, 3)

        # Store
        b.vstore(addr_output, vec_modified)
        b.halt()

        hir = b.build()

        # Set up memory
        mem = [0] * 100
        # input = [10, 20, 30, 40, 50, 60, 70, 80]
        for i in range(8):
            mem[i] = (i + 1) * 10

        machine = self._run_kernel(hir, mem)

        # Verify output: same as input but lane 3 (40) is doubled (80)
        for i in range(8):
            if i == 3:
                expected = 40 * 2  # doubled
            else:
                expected = (i + 1) * 10
            assert machine.mem[8 + i] == expected, f"output[{i}]: expected {expected}, got {machine.mem[8 + i]}"

    def test_vector_reduction_kernel(self):
        """Test kernel: sum all elements of a vector using extract."""
        b = HIRBuilder()

        # Memory layout:
        # [0..7] = input vector
        # [8]    = sum output

        addr_input = b.const(0)
        addr_output = b.const(8)

        # Load vector
        vec = b.vload(addr_input)

        # Extract and sum all lanes
        sum_val = b.vextract(vec, 0)
        for lane in range(1, 8):
            val = b.vextract(vec, lane)
            sum_val = b.add(sum_val, val)

        # Store sum
        b.store(addr_output, sum_val)
        b.halt()

        hir = b.build()

        # Set up memory
        mem = [0] * 100
        # input = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            mem[i] = i + 1

        machine = self._run_kernel(hir, mem)

        # Verify sum = 1+2+3+4+5+6+7+8 = 36
        expected = sum(range(1, 9))
        assert machine.mem[8] == expected, f"sum: expected {expected}, got {machine.mem[8]}"

    def test_vector_chain_operations(self):
        """Test kernel: chain of vector operations."""
        b = HIRBuilder()

        # Compute: result = ((a + b) * c) ^ d
        # Memory layout:
        # [0..7]   = a
        # [8..15]  = b
        # [16..23] = c
        # [24..31] = d
        # [32..39] = result

        addr_a = b.const(0)
        addr_b = b.const(8)
        addr_c = b.const(16)
        addr_d = b.const(24)
        addr_result = b.const(32)

        # Load all vectors
        vec_a = b.vload(addr_a)
        vec_b = b.vload(addr_b)
        vec_c = b.vload(addr_c)
        vec_d = b.vload(addr_d)

        # Compute
        t1 = b.vadd(vec_a, vec_b)
        t2 = b.vmul(t1, vec_c)
        vec_result = b.vxor(t2, vec_d)

        # Store
        b.vstore(addr_result, vec_result)
        b.halt()

        hir = b.build()

        # Set up memory
        mem = [0] * 100
        for i in range(8):
            mem[i] = i + 1       # a = [1..8]
            mem[8 + i] = 10      # b = [10, 10, ...]
            mem[16 + i] = 2      # c = [2, 2, ...]
            mem[24 + i] = 0xFF   # d = [255, 255, ...]

        machine = self._run_kernel(hir, mem)

        # Verify result = ((a + 10) * 2) ^ 0xFF
        for i in range(8):
            expected = (((i + 1) + 10) * 2) ^ 0xFF
            assert machine.mem[32 + i] == expected, f"result[{i}]: expected {expected}, got {machine.mem[32 + i]}"

    def test_vector_loop_kernel(self):
        """Test kernel: vector operations inside a loop."""
        b = HIRBuilder()

        # Compute: for each of 4 iterations, add a constant vector to accumulator
        # Memory layout:
        # [0..7]   = initial accumulator (zeros)
        # [8..15]  = increment vector
        # [16..23] = final result

        addr_acc = b.const(0)
        addr_inc = b.const(8)
        addr_result = b.const(16)

        zero = b.const(0)
        four = b.const(4)

        # Load increment vector (constant across loop)
        vec_inc = b.vload(addr_inc)

        # Initial accumulator
        vec_acc_init = b.vload(addr_acc)

        # Note: ForLoop doesn't directly support VectorSSAValue as iter_args/yields
        # So we work around by loading/storing each iteration
        # This tests vector ops in loop body

        def loop_body(i, params):
            # Load current accumulator from memory
            current_acc = b.vload(addr_acc)
            # Add increment
            new_acc = b.vadd(current_acc, vec_inc)
            # Store back
            b.vstore(addr_acc, new_acc)
            return []

        b.for_loop(
            start=zero,
            end=four,
            iter_args=[],
            body_fn=loop_body
        )

        # Copy result to output location
        final = b.vload(addr_acc)
        b.vstore(addr_result, final)
        b.halt()

        hir = b.build()

        # Set up memory
        mem = [0] * 100
        # acc starts at zeros
        for i in range(8):
            mem[i] = 0
        # increment = [1, 2, 3, 4, 5, 6, 7, 8]
        for i in range(8):
            mem[8 + i] = i + 1

        machine = self._run_kernel(hir, mem)

        # After 4 iterations, result = inc * 4
        for i in range(8):
            expected = (i + 1) * 4
            assert machine.mem[16 + i] == expected, f"result[{i}]: expected {expected}, got {machine.mem[16 + i]}"
