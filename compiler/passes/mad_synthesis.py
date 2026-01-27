"""
Multiply-Add (MAD) Synthesis Pass

Combines vector multiply (v*) and vector add (v+) operations into fused
multiply_add instructions for better performance.

Pattern: v+(v*(a, b), c) or v+(c, v*(a, b)) -> multiply_add(a, b, c)
"""

from typing import Optional

from ..hir import (
    SSAValue, VectorSSAValue, Variable, Const, Value, Op, ForLoop, If,
    Halt, Pause, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..use_def import UseDefContext


class MADSynthesisPass(Pass):
    """
    Fuses vector multiply and add operations into multiply_add.

    The multiply_add instruction computes (a * b) + c in a single operation,
    which can be more efficient than separate v* and v+ instructions.

    Key constraint: Only fuse when the v* result has exactly one user (the v+).
    This ensures we don't break the data flow graph.
    """

    def __init__(self):
        super().__init__()
        self._patterns_matched = 0
        self._ops_fused = 0

    @property
    def name(self) -> str:
        return "mad-synthesis"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        self._init_metrics()
        self._patterns_matched = 0
        self._ops_fused = 0

        if not config.enabled:
            return hir

        # Build use-def context for use count queries
        use_def_ctx = UseDefContext(hir)

        # Transform the function body
        new_body = self._transform_statements(hir.body, use_def_ctx)

        # Record metrics
        if self._metrics:
            self._metrics.custom = {
                "patterns_matched": self._patterns_matched,
                "ops_fused": self._ops_fused,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=hir.num_ssa_values,
            num_vec_ssa_values=hir.num_vec_ssa_values
        )

    def _transform_statements(
        self,
        stmts: list[Statement],
        use_def_ctx: UseDefContext
    ) -> list[Statement]:
        """Transform a list of statements, fusing MAD patterns.

        Uses two passes:
        1. First pass: identify all v* ops that should be fused (and skipped)
        2. Second pass: emit result, replacing v+ with multiply_add and skipping fused v*
        """
        # First pass: identify fusion opportunities
        # Maps v+ op id -> (mul_op, add_operand) for fusable patterns
        fusion_map: dict[int, tuple[Op, Value]] = {}
        # Set of v* ops to skip
        skip_ops: set[int] = set()

        for stmt in stmts:
            if isinstance(stmt, Op) and stmt.opcode == "v+":
                mul_op, add_operand = self._try_find_fusion(stmt, use_def_ctx)
                if mul_op is not None:
                    fusion_map[id(stmt)] = (mul_op, add_operand)
                    skip_ops.add(id(mul_op))

        # Second pass: emit transformed statements
        result = []
        for stmt in stmts:
            if isinstance(stmt, Op):
                if id(stmt) in skip_ops:
                    # Skip this v* op - it was fused
                    continue

                if id(stmt) in fusion_map:
                    # Replace v+ with multiply_add
                    mul_op, add_operand = fusion_map[id(stmt)]
                    self._patterns_matched += 1
                    self._ops_fused += 1

                    mul_left, mul_right = mul_op.operands[0], mul_op.operands[1]
                    fused_op = Op(
                        opcode="multiply_add",
                        result=stmt.result,
                        operands=[mul_left, mul_right, add_operand],
                        engine="valu"
                    )
                    result.append(fused_op)
                else:
                    result.append(stmt)

            elif isinstance(stmt, ForLoop):
                result.append(self._transform_for_loop(stmt, use_def_ctx))

            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, use_def_ctx))

            else:
                # Halt, Pause
                result.append(stmt)

        return result

    def _try_find_fusion(
        self,
        op: Op,
        use_def_ctx: UseDefContext
    ) -> tuple[Optional[Op], Optional[Value]]:
        """
        Try to find a fusable v* for a v+ operation.

        Returns (mul_op, add_operand) if fusable, (None, None) otherwise.
        """
        if len(op.operands) != 2:
            return None, None

        left, right = op.operands[0], op.operands[1]

        # Check if left operand is from a v* with single use
        mul_op, add_operand = self._find_fusable_mul(left, right, use_def_ctx)

        if mul_op is None:
            # Try the other way: v+(c, v*(a, b))
            mul_op, add_operand = self._find_fusable_mul(right, left, use_def_ctx)

        return mul_op, add_operand

    def _find_fusable_mul(
        self,
        potential_mul_result: Value,
        other_operand: Value,
        use_def_ctx: UseDefContext
    ) -> tuple[Optional[Op], Optional[Value]]:
        """
        Check if potential_mul_result comes from a v* with single use.

        Returns (mul_op, other_operand) if fusable, (None, None) otherwise.
        """
        if not isinstance(potential_mul_result, VectorSSAValue):
            return None, None

        # Find the defining op
        def_loc = use_def_ctx.get_def(potential_mul_result)
        if def_loc is None:
            return None, None

        def_stmt = def_loc.statement
        if not isinstance(def_stmt, Op):
            return None, None

        if def_stmt.opcode != "v*":
            return None, None

        # Check that the v* has exactly one use (the v+ we're fusing)
        use_count = use_def_ctx.use_count(potential_mul_result)
        if use_count != 1:
            return None, None

        return def_stmt, other_operand

    def _transform_for_loop(self, loop: ForLoop, use_def_ctx: UseDefContext) -> ForLoop:
        """Transform a ForLoop, fusing MAD patterns in its body."""
        new_body = self._transform_statements(loop.body, use_def_ctx)

        return ForLoop(
            counter=loop.counter,
            start=loop.start,
            end=loop.end,
            iter_args=loop.iter_args,
            body_params=loop.body_params,
            body=new_body,
            yields=loop.yields,
            results=loop.results,
            pragma_unroll=loop.pragma_unroll
        )

    def _transform_if(self, if_stmt: If, use_def_ctx: UseDefContext) -> If:
        """Transform an If statement, fusing MAD patterns in its branches."""
        new_then = self._transform_statements(if_stmt.then_body, use_def_ctx)
        new_else = self._transform_statements(if_stmt.else_body, use_def_ctx)

        return If(
            cond=if_stmt.cond,
            then_body=new_then,
            then_yields=if_stmt.then_yields,
            else_body=new_else,
            else_yields=if_stmt.else_yields,
            results=if_stmt.results
        )
