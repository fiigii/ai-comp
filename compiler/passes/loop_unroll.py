"""
Loop Unroll Pass

Unrolls ForLoops with static trip counts, either fully or partially.
"""

from ..hir import (
    SSAValue, VectorSSAValue, Const, Operand, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..ssa_context import SSARenumberContext


class LoopUnrollPass(Pass):
    """
    Unroll ForLoops with static trip counts.

    Each loop can have a pragma_unroll attribute controlling its unrolling:
    - pragma_unroll=0: Full unroll (default, backward compatible)
    - pragma_unroll=1: Disable unrolling on this loop
    - pragma_unroll=N (N>1): Partial unroll by factor N

    Loops with dynamic bounds are left unchanged.

    Options:
        max_trip_count: Maximum trip count to fully unroll (default: 10000000).
                        Loops with larger trip counts are skipped.
    """

    def __init__(self):
        super().__init__()
        self._loops_processed = 0
        self._loops_fully_unrolled = 0
        self._loops_partially_unrolled = 0
        self._loops_skipped = 0

    @property
    def name(self) -> str:
        return "loop-unroll"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        # Initialize metrics
        self._init_metrics()
        self._loops_processed = 0
        self._loops_fully_unrolled = 0
        self._loops_partially_unrolled = 0
        self._loops_skipped = 0

        # Very large default - fully unroll all static loops
        # If scratch space is exhausted, use pragma_unroll on individual loops
        max_trip_count = config.options.get("max_trip_count", 10000000)

        # Create SSA renumber context starting from current max
        max_vec_id = self._max_vector_id(hir.body)
        ctx = SSARenumberContext(hir.num_ssa_values, max_vec_id + 1)

        # Transform body recursively
        new_body = self._transform_statements(hir.body, max_trip_count, ctx)

        # Record custom metrics
        if self._metrics:
            self._metrics.custom = {
                "loops_processed": self._loops_processed,
                "fully_unrolled": self._loops_fully_unrolled,
                "partially_unrolled": self._loops_partially_unrolled,
                "skipped": self._loops_skipped,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=ctx.next_id,
            num_vec_ssa_values=ctx.next_vec_id
        )

    def _max_vector_id(self, body: list[Statement]) -> int:
        """Find the maximum VectorSSAValue id defined in the body."""
        max_id = -1
        for stmt in body:
            if isinstance(stmt, Op) and isinstance(stmt.result, VectorSSAValue):
                max_id = max(max_id, stmt.result.id)
            elif isinstance(stmt, ForLoop):
                max_id = max(max_id, self._max_vector_id(stmt.body))
            elif isinstance(stmt, If):
                max_id = max(max_id, self._max_vector_id(stmt.then_body))
                max_id = max(max_id, self._max_vector_id(stmt.else_body))
        return max_id

    def _transform_statements(
        self, stmts: list[Statement], max_trip: int, ctx: SSARenumberContext
    ) -> list[Statement]:
        """Transform a list of statements, unrolling loops as appropriate."""
        result = []
        for stmt in stmts:
            # First, remap operands based on result_bindings from previous unrolled loops
            stmt = self._apply_bindings(stmt, ctx.result_bindings)

            if isinstance(stmt, ForLoop):
                result.extend(self._unroll_loop(stmt, max_trip, ctx))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, max_trip, ctx))
            else:
                result.append(stmt)
        return result

    def _resolve_ssa_binding(self, value: Operand, bindings: dict[int, Operand]) -> Operand:
        """Resolve SSA bindings transitively (old -> new -> ...)."""
        while isinstance(value, SSAValue) and value.id in bindings:
            value = bindings[value.id]
        return value

    def _apply_bindings(self, stmt: Statement, bindings: dict[int, Operand]) -> Statement:
        """Apply SSA value bindings to a statement's operands."""
        if not bindings:
            return stmt

        if isinstance(stmt, Op):
            new_operands = []
            for op in stmt.operands:
                if isinstance(op, SSAValue) and op.id in bindings:
                    new_operands.append(self._resolve_ssa_binding(op, bindings))
                else:
                    new_operands.append(op)
            return Op(stmt.opcode, stmt.result, new_operands, stmt.engine)

        elif isinstance(stmt, ForLoop):
            # Remap iter_args
            new_iter_args = []
            for arg in stmt.iter_args:
                if isinstance(arg, SSAValue) and arg.id in bindings:
                    new_iter_args.append(self._resolve_ssa_binding(arg, bindings))
                else:
                    new_iter_args.append(arg)
            new_yields = [
                self._resolve_ssa_binding(y, bindings) if isinstance(y, SSAValue) and y.id in bindings else y
                for y in stmt.yields
            ]
            return ForLoop(
                counter=stmt.counter,
                start=stmt.start if not isinstance(stmt.start, SSAValue) or stmt.start.id not in bindings
                      else self._resolve_ssa_binding(stmt.start, bindings),
                end=stmt.end if not isinstance(stmt.end, SSAValue) or stmt.end.id not in bindings
                    else self._resolve_ssa_binding(stmt.end, bindings),
                iter_args=new_iter_args,
                body_params=stmt.body_params,
                body=stmt.body,
                yields=new_yields,
                results=stmt.results,
                pragma_unroll=stmt.pragma_unroll
            )

        elif isinstance(stmt, If):
            if isinstance(stmt.cond, SSAValue) and stmt.cond.id in bindings:
                new_cond = self._resolve_ssa_binding(stmt.cond, bindings)
            else:
                new_cond = stmt.cond
            new_then_yields = [
                self._resolve_ssa_binding(y, bindings) if isinstance(y, SSAValue) and y.id in bindings else y
                for y in stmt.then_yields
            ]
            new_else_yields = [
                self._resolve_ssa_binding(y, bindings) if isinstance(y, SSAValue) and y.id in bindings else y
                for y in stmt.else_yields
            ]
            return If(
                cond=new_cond,
                then_body=stmt.then_body,
                then_yields=new_then_yields,
                else_body=stmt.else_body,
                else_yields=new_else_yields,
                results=stmt.results
            )

        return stmt

    def _transform_if(
        self, if_stmt: If, max_trip: int, ctx: SSARenumberContext
    ) -> If:
        """Recursively transform If statement bodies."""
        then_body = self._transform_statements(if_stmt.then_body, max_trip, ctx)
        else_body = self._transform_statements(if_stmt.else_body, max_trip, ctx)
        return If(
            cond=if_stmt.cond,
            then_body=then_body,
            then_yields=[self._resolve_ssa_binding(y, ctx.result_bindings) for y in if_stmt.then_yields],
            else_body=else_body,
            else_yields=[self._resolve_ssa_binding(y, ctx.result_bindings) for y in if_stmt.else_yields],
            results=if_stmt.results
        )

    def _unroll_loop(
        self, loop: ForLoop, max_trip: int, ctx: SSARenumberContext
    ) -> list[Statement]:
        """Unroll a single ForLoop if possible.

        Uses bottom-up approach: transform inner loops first, then this loop.
        Respects per-loop pragma_unroll setting.
        """
        self._loops_processed += 1

        # FIRST: Transform the body (inner loops) before deciding about this loop
        # This ensures inner loops are unrolled before outer loops (bottom-up)
        transformed_body = self._transform_statements(loop.body, max_trip, ctx)
        transformed_yields = [self._resolve_ssa_binding(y, ctx.result_bindings) for y in loop.yields]

        # Create a loop with the transformed body for potential use below
        # Preserve pragma_unroll from original loop
        transformed_loop = ForLoop(
            counter=loop.counter,
            start=loop.start,
            end=loop.end,
            iter_args=loop.iter_args,
            body_params=loop.body_params,
            body=transformed_body,
            yields=transformed_yields,
            results=loop.results,
            pragma_unroll=loop.pragma_unroll
        )

        # Check pragma_unroll=1 (unrolling disabled)
        if loop.pragma_unroll == 1:
            self._loops_skipped += 1
            self._add_metric_message(f"Loop skipped: pragma_unroll=1")
            return [transformed_loop]

        # Check if loop has static bounds
        if not isinstance(loop.start, Const) or not isinstance(loop.end, Const):
            # Dynamic bounds - can't unroll, return with transformed body
            self._loops_skipped += 1
            start_str = loop.start.value if isinstance(loop.start, Const) else "dynamic"
            end_str = loop.end.value if isinstance(loop.end, Const) else "dynamic"
            self._add_metric_message(f"Loop skipped: dynamic bounds (start={start_str}, end={end_str})")
            return [transformed_loop]

        trip_count = loop.end.value - loop.start.value

        if trip_count <= 0:
            # Empty loop - just bind results to iter_args
            self._add_metric_message(f"Loop eliminated: zero iterations (trip_count={trip_count})")
            for result_ssa, iter_arg in zip(loop.results, loop.iter_args):
                ctx.bind_result(result_ssa.id, iter_arg)
            return []

        # Read pragma from loop
        pragma = loop.pragma_unroll

        # Determine actual unroll factor based on pragma
        if pragma == 0:
            # Full unroll - but only if trip count is within limit
            if trip_count > max_trip:
                # Too large for full unroll - return with transformed body
                self._loops_skipped += 1
                self._add_metric_message(f"Loop skipped: trip_count {trip_count} > max_trip_count {max_trip}")
                return [transformed_loop]
            actual_factor = trip_count
        else:
            # Partial unroll by pragma factor
            if trip_count % pragma != 0:
                # Factor doesn't divide evenly - skip unrolling this loop
                self._loops_skipped += 1
                self._add_metric_message(f"Loop skipped: trip_count {trip_count} not divisible by pragma_unroll {pragma}")
                return [transformed_loop]
            actual_factor = pragma

        # Fully unroll (using pre-transformed body)
        if actual_factor == trip_count:
            self._loops_fully_unrolled += 1
            self._add_metric_message(f"Loop fully unrolled: trip_count={trip_count}")
            return self._fully_unroll_pretransformed(transformed_loop, trip_count, ctx)

        # Partial unroll (using pre-transformed body)
        self._loops_partially_unrolled += 1
        self._add_metric_message(f"Loop partially unrolled: trip_count={trip_count}, factor={actual_factor}")
        return self._partially_unroll_pretransformed(transformed_loop, actual_factor, ctx)

    def _fully_unroll_pretransformed(
        self, loop: ForLoop, trip_count: int, ctx: SSARenumberContext
    ) -> list[Statement]:
        """Replace loop with trip_count copies of body.

        The loop body is already transformed (inner loops already unrolled).
        """
        result = []

        # Track current values for loop-carried variables
        # Initially: current_params[i] = iter_args[i]
        current_params = list(loop.iter_args)

        for i in range(trip_count):
            # Create remapping: counter -> const(start + i), body_params -> current_params
            remap: dict[int, Operand] = {}

            # Map counter to constant
            counter_value = loop.start.value + i
            remap[self._remap_key(loop.counter)] = Const(counter_value)

            # Map body_params to current values
            for param, current in zip(loop.body_params, current_params):
                remap[self._remap_key(param)] = current

            # Clone body with remapping (no recursion since body is already transformed)
            cloned_body, new_yields = self._clone_body_no_recurse(
                loop.body, loop.yields, remap, ctx
            )
            result.extend(cloned_body)

            # Update current_params for next iteration
            current_params = new_yields

        # Bind results to final yields
        for result_ssa, final_val in zip(loop.results, current_params):
            ctx.bind_result(result_ssa.id, final_val)

        return result

    def _partially_unroll_pretransformed(
        self, loop: ForLoop, factor: int, ctx: SSARenumberContext
    ) -> list[Statement]:
        """Create a new loop with factor copies of the body per iteration.

        The loop body is already transformed (inner loops already unrolled).
        """
        trip_count = loop.end.value - loop.start.value
        new_trip_count = trip_count // factor

        # New loop SSA values
        new_counter = ctx.new_ssa("unroll_i")
        new_body_params = [ctx.new_ssa(f"unroll_param_{j}") for j in range(len(loop.body_params))]
        new_results = [ctx.new_ssa(f"unroll_result_{j}") for j in range(len(loop.results))]

        # Build unrolled body
        unrolled_body: list[Statement] = []
        current_params = list(new_body_params)

        for j in range(factor):
            remap: dict[int, Operand] = {}

            # Compute counter for this unrolled iteration:
            # original_counter = new_counter * factor + j + start
            counter_ssa = ctx.new_ssa(f"iter_{j}_counter")

            # First compute: new_counter * factor
            if factor > 1:
                scaled = ctx.new_ssa(f"scaled_{j}")
                unrolled_body.append(Op("*", scaled, [new_counter, Const(factor)], "alu"))
                # Then add offset: scaled + (start + j)
                unrolled_body.append(Op("+", counter_ssa, [scaled, Const(loop.start.value + j)], "alu"))
            else:
                # factor == 1, just add start
                unrolled_body.append(Op("+", counter_ssa, [new_counter, Const(loop.start.value)], "alu"))

            remap[self._remap_key(loop.counter)] = counter_ssa

            # Map body_params to current values
            for param, current in zip(loop.body_params, current_params):
                remap[self._remap_key(param)] = current

            # Clone body with remapping (no recursion since body is already transformed)
            cloned_body, new_yields = self._clone_body_no_recurse(
                loop.body, loop.yields, remap, ctx
            )
            unrolled_body.extend(cloned_body)
            current_params = new_yields

        # Bind old results to new results so subsequent statements use the new ones
        for old_result, new_result in zip(loop.results, new_results):
            ctx.bind_result(old_result.id, new_result)

        # Create new ForLoop with unrolled body
        # Set pragma_unroll=1 to prevent re-unrolling
        return [ForLoop(
            counter=new_counter,
            start=Const(0),
            end=Const(new_trip_count),
            iter_args=loop.iter_args,
            body_params=new_body_params,
            body=unrolled_body,
            yields=current_params,
            results=new_results,
            pragma_unroll=1
        )]

    def _clone_body_no_recurse(
        self,
        body: list[Statement],
        yields: list[Operand],
        remap: dict,
        ctx: SSARenumberContext
    ) -> tuple[list[Statement], list[Operand]]:
        """Clone a body of statements without recursing into nested structures.

        This does not call _unroll_loop on nested ForLoops because the body is
        already transformed.
        """
        cloned = []
        for stmt in body:
            cloned.extend(self._clone_stmt_no_recurse(stmt, remap, ctx))

        new_yields = [self._remap_operand(y, remap) for y in yields]
        return cloned, new_yields

    def _clone_stmt_no_recurse(
        self,
        stmt: Statement,
        remap: dict,
        ctx: SSARenumberContext
    ) -> list[Statement]:
        """Clone a single statement without recursing into nested loop unrolling."""
        if isinstance(stmt, Op):
            new_operands = [self._remap_operand(op, remap) for op in stmt.operands]
            if stmt.result:
                if isinstance(stmt.result, VectorSSAValue):
                    new_result = ctx.new_vec_ssa(stmt.result.name)
                    remap[self._remap_key(stmt.result)] = new_result
                else:
                    new_result = ctx.new_ssa(stmt.result.name)
                    remap[self._remap_key(stmt.result)] = new_result
            else:
                new_result = None
            return [Op(stmt.opcode, new_result, new_operands, stmt.engine)]

        elif isinstance(stmt, ForLoop):
            # Clone the ForLoop structure without recursing into unroll logic
            new_counter = ctx.new_ssa(stmt.counter.name)
            new_body_params = [ctx.new_ssa(p.name) for p in stmt.body_params]
            new_results = [ctx.new_ssa(r.name) for r in stmt.results]

            # Remap iter_args
            new_iter_args = [self._remap_operand(arg, remap) for arg in stmt.iter_args]

            # Build inner remap for the loop body
            inner_remap = dict(remap)
            inner_remap[self._remap_key(stmt.counter)] = new_counter
            for old_param, new_param in zip(stmt.body_params, new_body_params):
                inner_remap[self._remap_key(old_param)] = new_param

            # Clone body (no recurse into unroll)
            cloned_body = []
            for s in stmt.body:
                cloned_body.extend(self._clone_stmt_no_recurse(s, inner_remap, ctx))

            # Remap yields
            new_yields = [self._remap_operand(y, inner_remap) for y in stmt.yields]

            # Remap start/end
            new_start = self._remap_operand(stmt.start, remap) if isinstance(stmt.start, SSAValue) else stmt.start
            new_end = self._remap_operand(stmt.end, remap) if isinstance(stmt.end, SSAValue) else stmt.end

            new_loop = ForLoop(
                counter=new_counter,
                start=new_start,
                end=new_end,
                iter_args=new_iter_args,
                body_params=new_body_params,
                body=cloned_body,
                yields=new_yields,
                results=new_results,
                pragma_unroll=stmt.pragma_unroll
            )

            # Map old results to new results
            for old_res, new_res in zip(stmt.results, new_results):
                remap[self._remap_key(old_res)] = new_res

            return [new_loop]

        elif isinstance(stmt, If):
            # Clone If statement
            new_cond = self._remap_operand(stmt.cond, remap)

            # Clone branches
            then_remap = dict(remap)
            cloned_then = []
            for s in stmt.then_body:
                cloned_then.extend(self._clone_stmt_no_recurse(s, then_remap, ctx))

            else_remap = dict(remap)
            cloned_else = []
            for s in stmt.else_body:
                cloned_else.extend(self._clone_stmt_no_recurse(s, else_remap, ctx))

            # Remap yields
            new_then_yields = [self._remap_operand(y, then_remap) for y in stmt.then_yields]

            new_else_yields = [self._remap_operand(y, else_remap) for y in stmt.else_yields]

            # New results
            new_results = [ctx.new_ssa(r.name) for r in stmt.results]
            for old_res, new_res in zip(stmt.results, new_results):
                remap[self._remap_key(old_res)] = new_res

            return [If(
                cond=new_cond,
                then_body=cloned_then,
                then_yields=new_then_yields,
                else_body=cloned_else,
                else_yields=new_else_yields,
                results=new_results
            )]

        elif isinstance(stmt, Halt):
            return [Halt()]

        elif isinstance(stmt, Pause):
            return [Pause()]

        else:
            return [stmt]

    def _remap_key(self, op: Operand) -> object:
        """Build a remap key for scalar or vector SSA values."""
        if isinstance(op, VectorSSAValue):
            return ("vec", op.id)
        if isinstance(op, SSAValue):
            return op.id
        return None

    def _remap_operand(self, op: Operand, remap: dict) -> Operand:
        """Remap an operand using the SSA remapping."""
        if isinstance(op, (SSAValue, VectorSSAValue)):
            key = self._remap_key(op)
            if key in remap:
                return remap[key]
            return op
        return op  # Const stays as-is
