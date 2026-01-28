"""
Tests for Register Allocation Pass

Covers:
- Basic liveness analysis
- Live interval building
- Linear scan allocation
- LIR rewriting
- LOAD_OFFSET handling (the main bug that was fixed)
- Vector base detection
- COPY materialization
"""

import pytest
from compiler.lir import LIRFunction, LIRInst, LIROpcode, BasicBlock
from compiler.passes.register_allocation import (
    RegisterAllocationPass,
    _compute_liveness,
    _build_live_intervals,
    _linear_scan,
    _rewrite_lir,
    _collect_defs,
    _collect_uses,
    _materialize_zero_for_copies,
)
from compiler.pass_manager import PassConfig


def make_lir(blocks_dict: dict, entry: str = "entry") -> LIRFunction:
    """Helper to create LIR from a dict of block definitions."""
    lir = LIRFunction(entry=entry)
    for name, (insts, term) in blocks_dict.items():
        block = BasicBlock(name)
        block.instructions = insts
        block.terminator = term
        lir.blocks[name] = block
    return lir


class TestCollectDefs:
    """Tests for _collect_defs function."""

    def test_scalar_def(self):
        """Scalar instruction defines one address."""
        inst = LIRInst(LIROpcode.ADD, 5, [3, 4], "alu")
        defs = set()
        _collect_defs(inst, defs)
        assert defs == {5}

    def test_vector_def(self):
        """Vector instruction defines 8 consecutive addresses."""
        inst = LIRInst(LIROpcode.VLOAD, [10, 11, 12, 13, 14, 15, 16, 17], [5], "load")
        defs = set()
        _collect_defs(inst, defs)
        assert defs == {10, 11, 12, 13, 14, 15, 16, 17}

    def test_load_offset_def(self):
        """LOAD_OFFSET defines dest + offset, not just dest."""
        # This was the main bug: LOAD_OFFSET with dest=10, offset=3 writes to 13
        inst = LIRInst(LIROpcode.LOAD_OFFSET, 10, [20, 3], "load")
        defs = set()
        _collect_defs(inst, defs)
        assert defs == {13}  # 10 + 3 = 13

    def test_load_offset_zero_offset(self):
        """LOAD_OFFSET with offset=0 defines dest."""
        inst = LIRInst(LIROpcode.LOAD_OFFSET, 10, [20, 0], "load")
        defs = set()
        _collect_defs(inst, defs)
        assert defs == {10}

    def test_no_dest(self):
        """Instructions without dest define nothing."""
        inst = LIRInst(LIROpcode.STORE, None, [5, 10], "store")
        defs = set()
        _collect_defs(inst, defs)
        assert defs == set()


class TestCollectUses:
    """Tests for _collect_uses function."""

    def test_scalar_uses(self):
        """Scalar instruction uses operands."""
        inst = LIRInst(LIROpcode.ADD, 5, [3, 4], "alu")
        uses = set()
        _collect_uses(inst, uses)
        assert uses == {3, 4}

    def test_const_no_uses(self):
        """CONST operands are immediates, not scratch references."""
        inst = LIRInst(LIROpcode.CONST, 5, [42], "load")
        uses = set()
        _collect_uses(inst, uses)
        assert uses == set()

    def test_load_offset_uses(self):
        """LOAD_OFFSET uses scratch[addr + offset], not just addr."""
        inst = LIRInst(LIROpcode.LOAD_OFFSET, 10, [20, 3], "load")
        uses = set()
        _collect_uses(inst, uses)
        # The actual scratch read is scratch[addr + offset] = scratch[20 + 3] = scratch[23]
        assert uses == {23}

    def test_vector_uses(self):
        """Vector operands add all elements as uses."""
        inst = LIRInst(LIROpcode.VADD, [20, 21, 22, 23, 24, 25, 26, 27],
                       [[10, 11, 12, 13, 14, 15, 16, 17], [0, 1, 2, 3, 4, 5, 6, 7]], "valu")
        uses = set()
        _collect_uses(inst, uses)
        # Should include both vector operands: 0-7 and 10-17
        expected = set(range(8)) | set(range(10, 18))
        assert uses == expected

    def test_jump_no_uses(self):
        """JUMP operands are labels, not scratch references."""
        inst = LIRInst(LIROpcode.JUMP, None, ["target_block"], "flow")
        uses = set()
        _collect_uses(inst, uses)
        assert uses == set()

    def test_cond_jump_uses_condition(self):
        """COND_JUMP uses the condition scratch, not the target labels."""
        inst = LIRInst(LIROpcode.COND_JUMP, None, [5, "true_block", "false_block"], "flow")
        uses = set()
        _collect_uses(inst, uses)
        assert uses == {5}


class TestLivenessAnalysis:
    """Tests for _compute_liveness function."""

    def test_simple_block(self):
        """Single block with def then use."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [10], "load"),  # def s0
                    LIRInst(LIROpcode.CONST, 1, [20], "load"),  # def s1
                    LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),   # use s0, s1, def s2
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        liveness = _compute_liveness(lir)
        info = liveness["entry"]
        # Nothing is live at entry (all defs before uses)
        assert info.live_in == set()
        assert info.live_out == set()

    def test_use_before_def_in_gen(self):
        """Use before def should be in gen set."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.ADD, 1, [0, 0], "alu"),  # use s0 before def
                    LIRInst(LIROpcode.CONST, 0, [10], "load"), # def s0
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        liveness = _compute_liveness(lir)
        info = liveness["entry"]
        assert 0 in info.gen  # s0 used before defined
        assert 0 in info.live_in  # s0 must be live at entry

    def test_loop_liveness(self):
        """Values used in loop body should be live throughout."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [100], "load"),  # def s0 = loop bound
                    LIRInst(LIROpcode.CONST, 1, [0], "load"),    # def s1 = counter
                ],
                LIRInst(LIROpcode.JUMP, None, ["header"], "flow")
            ),
            "header": (
                [
                    LIRInst(LIROpcode.LT, 2, [1, 0], "alu"),  # use s0, s1
                ],
                LIRInst(LIROpcode.COND_JUMP, None, [2, "body", "exit"], "flow")
            ),
            "body": (
                [
                    LIRInst(LIROpcode.ADD, 1, [1, 3], "alu"),  # increment s1
                ],
                LIRInst(LIROpcode.JUMP, None, ["header"], "flow")
            ),
            "exit": (
                [],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            ),
        })
        liveness = _compute_liveness(lir)
        # s0 (loop bound) should be live at header entry
        assert 0 in liveness["header"].live_in


class TestLiveIntervalBuilding:
    """Tests for _build_live_intervals function."""

    def test_simple_intervals(self):
        """Basic interval building for scalars."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [10], "load"),  # idx 0: def s0
                    LIRInst(LIROpcode.CONST, 1, [20], "load"),  # idx 1: def s1
                    LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),   # idx 2: use s0, s1, def s2
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        liveness = _compute_liveness(lir)
        intervals = _build_live_intervals(lir, liveness)

        # Convert to dict for easier checking
        interval_map = {i.vreg: (i.start, i.end, i.is_vector) for i in intervals}

        assert 0 in interval_map  # s0
        assert 1 in interval_map  # s1
        assert 2 in interval_map  # s2
        # s0 defined at 0, used at 2
        assert interval_map[0][0] <= 0 and interval_map[0][1] >= 2

    def test_vector_intervals(self):
        """Vector intervals track only the base address."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [5], "load"),
                    LIRInst(LIROpcode.VLOAD, [10, 11, 12, 13, 14, 15, 16, 17], [0], "load"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        liveness = _compute_liveness(lir)
        intervals = _build_live_intervals(lir, liveness)

        interval_map = {i.vreg: i for i in intervals}
        # Only base address 10 should have an interval, marked as vector
        assert 10 in interval_map
        assert interval_map[10].is_vector
        # Non-base addresses should not have separate intervals
        for addr in [11, 12, 13, 14, 15, 16, 17]:
            assert addr not in interval_map

    def test_load_offset_vector_detection(self):
        """LOAD_OFFSET with offsets 0-7 should be detected as vector."""
        # This tests the vgather pattern
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [100], "load"),  # some address
                    # 8 LOAD_OFFSET instructions with same dest, offsets 0-7
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 0], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 1], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 2], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 3], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 4], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 5], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 6], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 7], "load"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        liveness = _compute_liveness(lir)
        intervals = _build_live_intervals(lir, liveness)

        interval_map = {i.vreg: i for i in intervals}
        # dest=20 should be treated as a vector base
        assert 20 in interval_map
        assert interval_map[20].is_vector


class TestLinearScan:
    """Tests for _linear_scan function."""

    def test_non_overlapping_reuse(self):
        """Non-overlapping intervals should reuse registers."""
        from compiler.passes.register_allocation import LiveInterval

        intervals = [
            LiveInterval(vreg=0, start=0, end=2, is_vector=False),
            LiveInterval(vreg=1, start=3, end=5, is_vector=False),  # starts after 0 ends
        ]
        allocation, max_preg = _linear_scan(intervals)

        # Both should map to same physical register since they don't overlap
        assert allocation[0] == allocation[1]
        assert max_preg == 0  # Only one register used

    def test_overlapping_different_regs(self):
        """Overlapping intervals need different registers."""
        from compiler.passes.register_allocation import LiveInterval

        intervals = [
            LiveInterval(vreg=0, start=0, end=5, is_vector=False),
            LiveInterval(vreg=1, start=2, end=7, is_vector=False),  # overlaps with 0
        ]
        allocation, max_preg = _linear_scan(intervals)

        assert allocation[0] != allocation[1]
        assert max_preg >= 1  # At least 2 registers used

    def test_vector_allocation_contiguous(self):
        """Vector registers need 8 contiguous slots."""
        from compiler.passes.register_allocation import LiveInterval

        intervals = [
            LiveInterval(vreg=10, start=0, end=5, is_vector=True),
        ]
        allocation, max_preg = _linear_scan(intervals)

        # Vector base should be allocated
        assert 10 in allocation
        # Next allocation would be at base + 8
        assert max_preg >= 7  # 0-7 used for the vector


class TestRewriteLIR:
    """Tests for _rewrite_lir function."""

    def test_scalar_rewrite(self):
        """Scalar destinations and operands are rewritten."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 10, [42], "load"),
                    LIRInst(LIROpcode.ADD, 11, [10, 10], "alu"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        allocation = {10: 0, 11: 1}
        _rewrite_lir(lir, allocation)

        insts = lir.blocks["entry"].instructions
        assert insts[0].dest == 0  # s10 -> s0
        assert insts[1].dest == 1  # s11 -> s1
        assert insts[1].operands == [0, 0]  # operands also rewritten

    def test_vector_rewrite(self):
        """Vector destinations are rewritten with new base."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [5], "load"),
                    LIRInst(LIROpcode.VLOAD, [10, 11, 12, 13, 14, 15, 16, 17], [0], "load"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        allocation = {0: 0, 10: 20}
        _rewrite_lir(lir, allocation)

        insts = lir.blocks["entry"].instructions
        # Vector dest should be rewritten with new contiguous range
        assert insts[1].dest == [20, 21, 22, 23, 24, 25, 26, 27]

    def test_load_offset_vector_rewrite(self):
        """LOAD_OFFSET with vector base has dest rewritten correctly."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [100], "load"),
                    # 8 LOAD_OFFSET instructions that form a vector
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 0], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 1], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 2], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 3], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 4], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 5], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 6], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 7], "load"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        allocation = {0: 0, 20: 50}  # Vector base 20 -> 50
        _rewrite_lir(lir, allocation)

        insts = lir.blocks["entry"].instructions
        # All LOAD_OFFSET should have dest=50 (the new vector base)
        for i in range(1, 9):
            assert insts[i].dest == 50, f"Instruction {i} has wrong dest"
        # Offsets should be unchanged
        for i in range(1, 9):
            assert insts[i].operands[1] == i - 1


class TestMaterializeZero:
    """Tests for _materialize_zero_for_copies function."""

    def test_no_copies_no_change(self):
        """No COPY instructions means no changes."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [42], "load"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        max_preg = _materialize_zero_for_copies(lir, 0)
        assert max_preg == 0  # Unchanged
        assert len(lir.blocks["entry"].instructions) == 1  # No instruction added

    def test_copy_converted_to_add(self):
        """COPY instructions are converted to ADD with zero."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [42], "load"),
                    LIRInst(LIROpcode.COPY, 1, [0], "alu"),  # copy s0 to s1
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        max_preg = _materialize_zero_for_copies(lir, 1)

        insts = lir.blocks["entry"].instructions
        # Should have const 0 inserted at the beginning
        assert insts[0].opcode == LIROpcode.CONST
        assert insts[0].operands == [0]
        zero_scratch = insts[0].dest

        # Original COPY should be converted to ADD
        copy_inst = insts[2]  # After the inserted const
        assert copy_inst.opcode == LIROpcode.ADD
        assert copy_inst.operands == [0, zero_scratch]
        assert max_preg == zero_scratch


class TestRegisterAllocationPass:
    """Integration tests for the full RegisterAllocationPass."""

    def test_simple_program(self):
        """Test allocation on a simple program."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [10], "load"),
                    LIRInst(LIROpcode.CONST, 1, [20], "load"),
                    LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),
                    LIRInst(LIROpcode.STORE, None, [0, 2], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 2

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        # Should complete without error
        assert result is not None
        # max_scratch_used should be updated
        assert result.max_scratch_used >= 0

    def test_vector_program(self):
        """Test allocation with vector operations."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [5], "load"),
                    LIRInst(LIROpcode.VLOAD, [10, 11, 12, 13, 14, 15, 16, 17], [0], "load"),
                    LIRInst(LIROpcode.VSTORE, None, [0, [10, 11, 12, 13, 14, 15, 16, 17]], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 17

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        assert result is not None

    def test_load_offset_vgather_pattern(self):
        """Test the vgather pattern that caused the original bug."""
        # This is the pattern that was failing before the fix:
        # 8 LOAD_OFFSET instructions with same dest, consecutive offsets
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [100], "load"),  # address base
                    LIRInst(LIROpcode.CONST, 1, [5], "load"),    # some other value
                    # vgather pattern: 8 LOAD_OFFSET with dest=20, offsets 0-7
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 0], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 1], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 2], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 3], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 4], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 5], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 6], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 7], "load"),
                    # Use the gathered values
                    LIRInst(LIROpcode.VSTORE, None, [1, [20, 21, 22, 23, 24, 25, 26, 27]], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 27

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        # Verify all LOAD_OFFSET have the same dest after rewriting
        insts = result.blocks["entry"].instructions
        load_offset_dests = set()
        for inst in insts:
            if inst.opcode == LIROpcode.LOAD_OFFSET:
                load_offset_dests.add(inst.dest)

        # All LOAD_OFFSET should have the same dest (the allocated vector base)
        assert len(load_offset_dests) == 1, f"Expected 1 unique dest, got {load_offset_dests}"

    def test_register_reuse(self):
        """Test that registers are reused when intervals don't overlap."""
        # Create a program where many virtual registers have non-overlapping lifetimes
        lir = make_lir({
            "entry": (
                [
                    # First use s0, s1 to compute s2
                    LIRInst(LIROpcode.CONST, 0, [10], "load"),
                    LIRInst(LIROpcode.CONST, 1, [20], "load"),
                    LIRInst(LIROpcode.ADD, 2, [0, 1], "alu"),
                    LIRInst(LIROpcode.STORE, None, [0, 2], "store"),  # s0, s1, s2 dead after
                    # Now s3, s4 can reuse s0, s1's registers
                    LIRInst(LIROpcode.CONST, 3, [30], "load"),
                    LIRInst(LIROpcode.CONST, 4, [40], "load"),
                    LIRInst(LIROpcode.ADD, 5, [3, 4], "alu"),
                    LIRInst(LIROpcode.STORE, None, [3, 5], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 5

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        # Physical registers used should be less than virtual registers
        metrics = pass_obj._metrics
        if metrics and metrics.custom:
            physical = metrics.custom.get("physical_regs_used", 0)
            virtual = metrics.custom.get("virtual_regs", 0)
            # We should see some register reuse
            assert physical <= virtual


class TestEdgeCases:
    """Test edge cases and potential issues."""

    def test_empty_program(self):
        """Empty program should not crash."""
        lir = make_lir({
            "entry": ([], LIRInst(LIROpcode.HALT, None, [], "flow"))
        })
        lir.max_scratch_used = -1

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        assert result is not None

    def test_only_constants(self):
        """Program with only constant loads."""
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [1], "load"),
                    LIRInst(LIROpcode.CONST, 1, [2], "load"),
                    LIRInst(LIROpcode.CONST, 2, [3], "load"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 2

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        assert result is not None

    def test_load_offset_partial_offsets(self):
        """LOAD_OFFSET with non-complete offset set is treated as scalars."""
        # Only offsets 0, 1, 2 - not a full vector
        lir = make_lir({
            "entry": (
                [
                    LIRInst(LIROpcode.CONST, 0, [100], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 0], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 1], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 20, [0, 2], "load"),
                    # Use as scalars
                    LIRInst(LIROpcode.ADD, 30, [20, 21], "alu"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 30

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        # Should handle without treating as vector
        assert result is not None


class TestConstructedVectorRegressions:
    """
    Regression tests for constructed vectors.

    These test cases cover scenarios where vectors are built lane-by-lane
    (e.g., via vinsert lowering) and must maintain contiguity after allocation.
    """

    def test_constructed_vector_in_vadd(self):
        """
        Regression: constructed vector used in VADD (not just VSTORE).

        If a vector is built lane-by-lane and fed to VADD, it must be detected
        as a vector base to maintain contiguity. Otherwise, lanes could be
        allocated non-contiguously, but codegen uses only op[0] as the base.
        """
        # Simulate a constructed vector [20, 21, 22, ..., 27] used in VADD
        constructed_vec = [20, 21, 22, 23, 24, 25, 26, 27]
        another_vec = [30, 31, 32, 33, 34, 35, 36, 37]
        result_vec = [40, 41, 42, 43, 44, 45, 46, 47]

        lir = make_lir({
            "entry": (
                [
                    # Define both vectors via scalar defs (simulating vinsert lowering)
                    LIRInst(LIROpcode.CONST, 20, [1], "load"),
                    LIRInst(LIROpcode.CONST, 21, [2], "load"),
                    LIRInst(LIROpcode.CONST, 22, [3], "load"),
                    LIRInst(LIROpcode.CONST, 23, [4], "load"),
                    LIRInst(LIROpcode.CONST, 24, [5], "load"),
                    LIRInst(LIROpcode.CONST, 25, [6], "load"),
                    LIRInst(LIROpcode.CONST, 26, [7], "load"),
                    LIRInst(LIROpcode.CONST, 27, [8], "load"),
                    LIRInst(LIROpcode.CONST, 30, [10], "load"),
                    LIRInst(LIROpcode.CONST, 31, [20], "load"),
                    LIRInst(LIROpcode.CONST, 32, [30], "load"),
                    LIRInst(LIROpcode.CONST, 33, [40], "load"),
                    LIRInst(LIROpcode.CONST, 34, [50], "load"),
                    LIRInst(LIROpcode.CONST, 35, [60], "load"),
                    LIRInst(LIROpcode.CONST, 36, [70], "load"),
                    LIRInst(LIROpcode.CONST, 37, [80], "load"),
                    # Use constructed vectors in VADD (not VSTORE!)
                    LIRInst(LIROpcode.VADD, result_vec, [constructed_vec, another_vec], "valu"),
                    # Store result
                    LIRInst(LIROpcode.CONST, 0, [100], "load"),
                    LIRInst(LIROpcode.VSTORE, None, [0, result_vec], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 47

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        # Find the VADD instruction and verify operands are contiguous
        vadd_inst = None
        for inst in result.blocks["entry"].instructions:
            if inst.opcode == LIROpcode.VADD:
                vadd_inst = inst
                break

        assert vadd_inst is not None, "VADD instruction not found"

        # Check that both operands are contiguous vectors
        for op in vadd_inst.operands:
            if isinstance(op, list) and len(op) == 8:
                base = op[0]
                expected = [base + i for i in range(8)]
                assert op == expected, f"Vector operand not contiguous: {op}, expected {expected}"

    def test_constructed_vector_in_vselect(self):
        """
        Regression: constructed vector used in VSELECT condition/operands.

        Similar to VADD, VSELECT takes vector operands that must stay contiguous.
        """
        cond_vec = [10, 11, 12, 13, 14, 15, 16, 17]
        true_vec = [20, 21, 22, 23, 24, 25, 26, 27]
        false_vec = [30, 31, 32, 33, 34, 35, 36, 37]
        result_vec = [40, 41, 42, 43, 44, 45, 46, 47]

        lir = make_lir({
            "entry": (
                [
                    # Define all vectors via scalar defs
                    *[LIRInst(LIROpcode.CONST, 10 + i, [i], "load") for i in range(8)],
                    *[LIRInst(LIROpcode.CONST, 20 + i, [100 + i], "load") for i in range(8)],
                    *[LIRInst(LIROpcode.CONST, 30 + i, [200 + i], "load") for i in range(8)],
                    # Use in VSELECT
                    LIRInst(LIROpcode.VSELECT, result_vec, [cond_vec, true_vec, false_vec], "flow"),
                    # Store
                    LIRInst(LIROpcode.CONST, 0, [999], "load"),
                    LIRInst(LIROpcode.VSTORE, None, [0, result_vec], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 47

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        # Find VSELECT and verify all operands are contiguous
        vselect_inst = None
        for inst in result.blocks["entry"].instructions:
            if inst.opcode == LIROpcode.VSELECT:
                vselect_inst = inst
                break

        assert vselect_inst is not None
        for op in vselect_inst.operands:
            if isinstance(op, list) and len(op) == 8:
                base = op[0]
                expected = [base + i for i in range(8)]
                assert op == expected, f"VSELECT operand not contiguous: {op}"

    def test_load_offset_with_constructed_address_vector(self):
        """
        Regression: LOAD_OFFSET with address vector that was constructed lane-wise.

        LOAD_OFFSET reads from scratch[addr + offset], so if addr is a constructed
        vector (not from VLOAD), it must maintain contiguity for correctness.
        """
        # Simulate: address vector built lane-by-lane, then used in vgather-style LOAD_OFFSET
        addr_vec_base = 50

        lir = make_lir({
            "entry": (
                [
                    # Build address vector lane-by-lane (simulating computed addresses)
                    LIRInst(LIROpcode.CONST, 50, [100], "load"),  # addr[0]
                    LIRInst(LIROpcode.CONST, 51, [108], "load"),  # addr[1]
                    LIRInst(LIROpcode.CONST, 52, [116], "load"),  # addr[2]
                    LIRInst(LIROpcode.CONST, 53, [124], "load"),  # addr[3]
                    LIRInst(LIROpcode.CONST, 54, [132], "load"),  # addr[4]
                    LIRInst(LIROpcode.CONST, 55, [140], "load"),  # addr[5]
                    LIRInst(LIROpcode.CONST, 56, [148], "load"),  # addr[6]
                    LIRInst(LIROpcode.CONST, 57, [156], "load"),  # addr[7]
                    # Use in LOAD_OFFSET (vgather pattern)
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 0], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 1], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 2], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 3], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 4], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 5], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 6], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 7], "load"),
                    # Store gathered results
                    LIRInst(LIROpcode.CONST, 0, [500], "load"),
                    LIRInst(LIROpcode.VSTORE, None, [0, [70, 71, 72, 73, 74, 75, 76, 77]], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 77

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        # Verify LOAD_OFFSET instructions all have same dest after allocation
        load_offset_dests = set()
        for inst in result.blocks["entry"].instructions:
            if inst.opcode == LIROpcode.LOAD_OFFSET:
                load_offset_dests.add(inst.dest)

        assert len(load_offset_dests) == 1, \
            f"LOAD_OFFSET dests should all be same vector base, got {load_offset_dests}"

        # Verify the VSTORE operand is contiguous
        for inst in result.blocks["entry"].instructions:
            if inst.opcode == LIROpcode.VSTORE:
                vec_op = inst.operands[1]
                base = vec_op[0]
                expected = [base + i for i in range(8)]
                assert vec_op == expected, f"VSTORE vector not contiguous: {vec_op}"

    def test_multiply_add_with_constructed_vectors(self):
        """
        Regression: MULTIPLY_ADD with constructed vector operands.

        MULTIPLY_ADD takes 3 vector operands: dest = a * b + c
        All must maintain contiguity.
        """
        vec_a = [10, 11, 12, 13, 14, 15, 16, 17]
        vec_b = [20, 21, 22, 23, 24, 25, 26, 27]
        vec_c = [30, 31, 32, 33, 34, 35, 36, 37]
        result = [40, 41, 42, 43, 44, 45, 46, 47]

        lir = make_lir({
            "entry": (
                [
                    # Define vectors via scalar defs
                    *[LIRInst(LIROpcode.CONST, 10 + i, [i + 1], "load") for i in range(8)],
                    *[LIRInst(LIROpcode.CONST, 20 + i, [i + 10], "load") for i in range(8)],
                    *[LIRInst(LIROpcode.CONST, 30 + i, [i + 100], "load") for i in range(8)],
                    # MULTIPLY_ADD
                    LIRInst(LIROpcode.MULTIPLY_ADD, result, [vec_a, vec_b, vec_c], "valu"),
                    # Store
                    LIRInst(LIROpcode.CONST, 0, [999], "load"),
                    LIRInst(LIROpcode.VSTORE, None, [0, result], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 47

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result_lir = pass_obj.run(lir, config)

        # Find MULTIPLY_ADD and verify all operands are contiguous
        mad_inst = None
        for inst in result_lir.blocks["entry"].instructions:
            if inst.opcode == LIROpcode.MULTIPLY_ADD:
                mad_inst = inst
                break

        assert mad_inst is not None
        for op in mad_inst.operands:
            if isinstance(op, list) and len(op) == 8:
                base = op[0]
                expected = [base + i for i in range(8)]
                assert op == expected, f"MULTIPLY_ADD operand not contiguous: {op}"

    def test_load_offset_address_operand_contiguity(self):
        """
        Regression: LOAD_OFFSET addr+offset reads must stay contiguous.

        LOAD_OFFSET reads from scratch[addr + offset], so if we have 8 LOAD_OFFSET
        instructions with the same addr base and offsets 0-7, the actual reads are
        from scratch[addr+0], scratch[addr+1], ..., scratch[addr+7].

        This tests that the address operand vector stays contiguous after allocation.
        """
        # The address vector is at scratch[50..57]
        # Each LOAD_OFFSET reads scratch[50+offset] = scratch[50], scratch[51], etc.
        addr_vec_base = 50

        lir = make_lir({
            "entry": (
                [
                    # Build address vector lane-by-lane
                    LIRInst(LIROpcode.CONST, 50, [100], "load"),
                    LIRInst(LIROpcode.CONST, 51, [200], "load"),
                    LIRInst(LIROpcode.CONST, 52, [300], "load"),
                    LIRInst(LIROpcode.CONST, 53, [400], "load"),
                    LIRInst(LIROpcode.CONST, 54, [500], "load"),
                    LIRInst(LIROpcode.CONST, 55, [600], "load"),
                    LIRInst(LIROpcode.CONST, 56, [700], "load"),
                    LIRInst(LIROpcode.CONST, 57, [800], "load"),
                    # vgather pattern: LOAD_OFFSET reads scratch[addr+offset]
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 0], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 1], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 2], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 3], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 4], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 5], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 6], "load"),
                    LIRInst(LIROpcode.LOAD_OFFSET, 70, [addr_vec_base, 7], "load"),
                    # Store gathered results
                    LIRInst(LIROpcode.CONST, 0, [999], "load"),
                    LIRInst(LIROpcode.VSTORE, None, [0, [70, 71, 72, 73, 74, 75, 76, 77]], "store"),
                ],
                LIRInst(LIROpcode.HALT, None, [], "flow")
            )
        })
        lir.max_scratch_used = 77

        pass_obj = RegisterAllocationPass()
        config = PassConfig(name="register-allocation")
        result = pass_obj.run(lir, config)

        # Collect the LOAD_OFFSET instructions and verify addr+offset forms a contiguous range
        load_offset_insts = [
            inst for inst in result.blocks["entry"].instructions
            if inst.opcode == LIROpcode.LOAD_OFFSET
        ]
        assert len(load_offset_insts) == 8

        # All LOAD_OFFSET should have the same addr base (the allocated address vector base)
        addr_bases = set()
        for inst in load_offset_insts:
            addr_bases.add(inst.operands[0])
        assert len(addr_bases) == 1, f"Address bases should be same, got {addr_bases}"

        # The actual addresses read are addr + offset, which should form a contiguous range
        addr_base = list(addr_bases)[0]
        actual_addrs = sorted([inst.operands[0] + inst.operands[1] for inst in load_offset_insts])
        expected_addrs = [addr_base + i for i in range(8)]
        assert actual_addrs == expected_addrs, \
            f"Address reads should be contiguous: expected {expected_addrs}, got {actual_addrs}"

        # The address vector lanes must actually be stored in that contiguous range too.
        # Otherwise LOAD_OFFSET will read from the wrong scratch locations.
        lane_values = {100, 200, 300, 400, 500, 600, 700, 800}
        addr_lane_defs = [
            inst for inst in result.blocks["entry"].instructions
            if inst.opcode == LIROpcode.CONST and inst.operands and inst.operands[0] in lane_values
        ]
        assert len(addr_lane_defs) == 8
        addr_lane_dests = sorted(inst.dest for inst in addr_lane_defs)
        expected_lane_dests = [addr_base + i for i in range(8)]
        assert addr_lane_dests == expected_lane_dests, \
            f"Address lane dests should be contiguous at {expected_lane_dests}, got {addr_lane_dests}"
