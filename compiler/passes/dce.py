"""
Dead Code Elimination (DCE) Pass

Removes operations whose results are never used.
Uses backward mark-and-sweep analysis on SSA values with UseDefContext.
"""

from ..hir import (
    SSAValue, VectorSSAValue, Variable, Const, Value, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig
from ..use_def import UseDefContext


# Operations that have side effects and should never be eliminated
SIDE_EFFECT_OPS = {"store", "vstore"}


class DCEPass(Pass):
    """
    Dead Code Elimination using backward mark-and-sweep with UseDefContext.

    Pass 1: Compute live set (backward traversal)
    - Mark SSA values as live if used by side-effect ops or other live ops
    - Uses UseDefContext for efficient use queries

    Pass 2: Filter dead code (forward traversal)
    - Keep statements if they have side effects or their result is live
    """

    def __init__(self):
        super().__init__()
        self._ops_eliminated = 0
        self._loops_eliminated = 0
        self._ifs_eliminated = 0

    @property
    def name(self) -> str:
        return "dce"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        # Initialize metrics
        self._init_metrics()
        self._ops_eliminated = 0
        self._loops_eliminated = 0
        self._ifs_eliminated = 0

        # Check if pass is enabled
        if not config.enabled:
            return hir

        # Build use-def context for efficient queries
        use_def_ctx = UseDefContext(hir)

        # Pass 1: Compute live set (backward analysis)
        live = self._compute_live_set(hir.body, use_def_ctx)

        # Pass 2: Filter dead code (forward sweep)
        new_body = self._filter_dead_code(hir.body, live, use_def_ctx)

        # Record custom metrics
        if self._metrics:
            self._metrics.custom = {
                "ops_eliminated": self._ops_eliminated,
                "loops_eliminated": self._loops_eliminated,
                "ifs_eliminated": self._ifs_eliminated,
            }

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=hir.num_ssa_values,
            num_vec_ssa_values=hir.num_vec_ssa_values
        )

    def _compute_live_set(
        self, body: list[Statement], ctx: UseDefContext
    ) -> set[Variable]:
        """
        Compute the set of live SSA values using backward traversal.

        An SSA value is live if:
        - It's used as operand to a side-effect op (store, vstore)
        - It's used as operand to another live operation
        - It's a ForLoop/If result that's used downstream
        - It's a ForLoop yield that feeds a live result
        - It's an If condition when the If is needed
        """
        live: set[Variable] = set()
        self._mark_live_backward(body, live, ctx)
        return live

    def _mark_live_backward(
        self, body: list[Statement], live: set[Variable], ctx: UseDefContext
    ) -> None:
        """Traverse statements in reverse, marking live SSA values."""
        for stmt in reversed(body):
            if isinstance(stmt, Op):
                self._mark_op_live(stmt, live)
            elif isinstance(stmt, Halt):
                # Halt is always live (terminates program)
                pass
            elif isinstance(stmt, Pause):
                # Pause is always live (debug sync)
                pass
            elif isinstance(stmt, ForLoop):
                self._mark_forloop_live(stmt, live, ctx)
            elif isinstance(stmt, If):
                self._mark_if_live(stmt, live, ctx)

    def _mark_operands_live(
        self, operands: list[Value], live: set[Variable]
    ) -> None:
        """Mark all SSA operands as live."""
        for operand in operands:
            if isinstance(operand, (SSAValue, VectorSSAValue)):
                live.add(operand)

    def _mark_op_live(self, op: Op, live: set[Variable]) -> None:
        """Mark an Op's operands as live if the op is live."""
        # Op is live if:
        # 1. It has side effects (store, vstore, result=None)
        # 2. Its result is in the live set
        is_live = False

        if self._is_side_effect_op(op):
            is_live = True
        elif op.result is not None:
            is_live = op.result in live

        if is_live:
            self._mark_operands_live(op.operands, live)

    def _mark_forloop_live(
        self, loop: ForLoop, live: set[Variable], ctx: UseDefContext
    ) -> None:
        """Mark a ForLoop's components as live if needed."""
        # Build map from body_param to its index
        body_param_to_idx = {p: i for i, p in enumerate(loop.body_params)}

        # First, process the loop body to determine what's live inside
        body_live: set[Variable] = set()

        # If any result is live, mark the corresponding yield as live
        for i, result in enumerate(loop.results):
            if result in live:
                if i < len(loop.yields):
                    yield_val = loop.yields[i]
                    if isinstance(yield_val, (SSAValue, VectorSSAValue)):
                        body_live.add(yield_val)

        # Iteratively compute live set until fixed point
        # This is needed because body_params[i] = yields[i] on subsequent iterations
        # So if body_params[i] is live, yields[i] must also be live
        while True:
            prev_size = len(body_live)

            # Mark any body_param that's live -> also mark corresponding yield as live
            for ssa in list(body_live):
                if ssa in body_param_to_idx:
                    idx = body_param_to_idx[ssa]
                    if idx < len(loop.yields):
                        yield_val = loop.yields[idx]
                        if isinstance(yield_val, (SSAValue, VectorSSAValue)):
                            body_live.add(yield_val)

            # Run backward analysis on body
            self._mark_live_backward(loop.body, body_live, ctx)

            # Check for fixed point
            if len(body_live) == prev_size:
                break

        # Check if loop has side effects in body
        has_side_effects = self._has_side_effects(loop.body)

        # Loop is live if any result is used OR body has side effects
        loop_is_live = has_side_effects or any(r in live for r in loop.results)

        if loop_is_live:
            # Mark yields as live (they feed back to body_params)
            for y in loop.yields:
                if isinstance(y, (SSAValue, VectorSSAValue)):
                    body_live.add(y)

            # Mark start/end as live
            if isinstance(loop.start, (SSAValue, VectorSSAValue)):
                live.add(loop.start)
            if isinstance(loop.end, (SSAValue, VectorSSAValue)):
                live.add(loop.end)

            # Propagate body_live to outer live set
            # For body_params that are live, mark the corresponding iter_args as live
            # For other SSA values, propagate to outer scope
            for ssa in body_live:
                if ssa == loop.counter:
                    # Counter is local to loop, don't propagate
                    continue
                elif ssa in body_param_to_idx:
                    # Body param is live - mark corresponding iter_arg as live
                    idx = body_param_to_idx[ssa]
                    if idx < len(loop.iter_args):
                        arg = loop.iter_args[idx]
                        if isinstance(arg, (SSAValue, VectorSSAValue)):
                            live.add(arg)
                else:
                    # Value defined outside loop - propagate to outer scope
                    live.add(ssa)

    def _mark_if_live(
        self, if_stmt: If, live: set[Variable], ctx: UseDefContext
    ) -> None:
        """Mark an If's components as live if needed."""
        # First, process both branches to determine what's live inside
        then_live: set[Variable] = set()
        else_live: set[Variable] = set()

        # If any result is live, mark the corresponding yields as live
        for i, result in enumerate(if_stmt.results):
            if result in live:
                if i < len(if_stmt.then_yields):
                    then_val = if_stmt.then_yields[i]
                    if isinstance(then_val, (SSAValue, VectorSSAValue)):
                        then_live.add(then_val)
                if i < len(if_stmt.else_yields):
                    else_val = if_stmt.else_yields[i]
                    if isinstance(else_val, (SSAValue, VectorSSAValue)):
                        else_live.add(else_val)

        # Recursively mark live in branches
        self._mark_live_backward(if_stmt.then_body, then_live, ctx)
        self._mark_live_backward(if_stmt.else_body, else_live, ctx)

        # Check if branches have side effects
        then_has_side_effects = self._has_side_effects(if_stmt.then_body)
        else_has_side_effects = self._has_side_effects(if_stmt.else_body)

        # If is live if any result is used OR either branch has side effects
        if_is_live = (then_has_side_effects or else_has_side_effects or
                      any(r in live for r in if_stmt.results))

        if if_is_live:
            # Mark condition as live
            if isinstance(if_stmt.cond, (SSAValue, VectorSSAValue)):
                live.add(if_stmt.cond)

            # Propagate branch-live to outer live set
            # (No local definitions in if branches besides results)
            live.update(then_live)
            live.update(else_live)

    def _is_side_effect_op(self, op: Op) -> bool:
        """Check if an operation has side effects."""
        return op.opcode in SIDE_EFFECT_OPS or op.result is None

    def _has_side_effects(self, body: list[Statement]) -> bool:
        """Check if a body contains any side-effect operations."""
        for stmt in body:
            if isinstance(stmt, Op):
                if self._is_side_effect_op(stmt):
                    return True
            elif isinstance(stmt, Halt):
                return True
            elif isinstance(stmt, Pause):
                return True
            elif isinstance(stmt, ForLoop):
                if self._has_side_effects(stmt.body):
                    return True
            elif isinstance(stmt, If):
                if (self._has_side_effects(stmt.then_body) or
                        self._has_side_effects(stmt.else_body)):
                    return True
        return False

    def _filter_dead_code(
        self, body: list[Statement], live: set[Variable], ctx: UseDefContext
    ) -> list[Statement]:
        """Filter out dead statements, keeping only live ones."""
        result = []

        for stmt in body:
            if isinstance(stmt, Op):
                filtered = self._filter_op(stmt, live, ctx)
                if filtered is not None:
                    result.append(filtered)
            elif isinstance(stmt, Halt):
                result.append(stmt)
            elif isinstance(stmt, Pause):
                result.append(stmt)
            elif isinstance(stmt, ForLoop):
                filtered = self._filter_forloop(stmt, live, ctx)
                if filtered is not None:
                    result.append(filtered)
            elif isinstance(stmt, If):
                filtered = self._filter_if(stmt, live, ctx)
                if filtered is not None:
                    result.append(filtered)

        return result

    def _filter_op(
        self, op: Op, live: set[Variable], ctx: UseDefContext
    ) -> Op | None:
        """Filter an Op - keep if it has side effects or result is live."""
        if self._is_side_effect_op(op):
            return op
        if op.result is not None:
            # Check if result is live using our computed live set
            if op.result in live:
                return op
            # Additional quick check: if UseDefContext shows no uses at all,
            # definitely dead (redundant but provides a sanity check)
            if not ctx.has_uses(op.result):
                self._ops_eliminated += 1
                return None

        # Dead code - eliminate
        self._ops_eliminated += 1
        return None

    def _filter_forloop(
        self, loop: ForLoop, live: set[Variable], ctx: UseDefContext
    ) -> ForLoop | None:
        """Filter a ForLoop - keep if any result is live or body has side effects."""
        has_side_effects = self._has_side_effects(loop.body)
        any_result_live = any(r in live for r in loop.results)

        if not has_side_effects and not any_result_live:
            # Dead loop - eliminate
            self._loops_eliminated += 1
            return None

        # Build map from body_param to its index
        body_param_to_idx = {p: i for i, p in enumerate(loop.body_params)}

        # Compute what's live in the loop body
        body_live: set[Variable] = set()

        # Results that are live make corresponding yields live
        for i, result in enumerate(loop.results):
            if result in live:
                if i < len(loop.yields):
                    yield_val = loop.yields[i]
                    if isinstance(yield_val, (SSAValue, VectorSSAValue)):
                        body_live.add(yield_val)

        # If body has side effects, all yields are potentially live
        if has_side_effects:
            for y in loop.yields:
                if isinstance(y, (SSAValue, VectorSSAValue)):
                    body_live.add(y)

        # Iteratively compute live set until fixed point
        while True:
            prev_size = len(body_live)

            # Mark any body_param that's live -> also mark corresponding yield as live
            for ssa in list(body_live):
                if ssa in body_param_to_idx:
                    idx = body_param_to_idx[ssa]
                    if idx < len(loop.yields):
                        yield_val = loop.yields[idx]
                        if isinstance(yield_val, (SSAValue, VectorSSAValue)):
                            body_live.add(yield_val)

            # Propagate liveness backward through body
            self._mark_live_backward(loop.body, body_live, ctx)

            # Check for fixed point
            if len(body_live) == prev_size:
                break

        # Filter the body
        new_body = self._filter_dead_code(loop.body, body_live, ctx)

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

    def _filter_if(
        self, if_stmt: If, live: set[Variable], ctx: UseDefContext
    ) -> If | None:
        """Filter an If - keep if any result is live or branches have side effects."""
        then_has_side_effects = self._has_side_effects(if_stmt.then_body)
        else_has_side_effects = self._has_side_effects(if_stmt.else_body)
        any_result_live = any(r in live for r in if_stmt.results)

        if not then_has_side_effects and not else_has_side_effects and not any_result_live:
            # Dead if - eliminate
            self._ifs_eliminated += 1
            return None

        # Compute what's live in each branch
        then_live: set[Variable] = set()
        else_live: set[Variable] = set()

        # Results that are live make corresponding yields live
        for i, result in enumerate(if_stmt.results):
            if result in live:
                if i < len(if_stmt.then_yields):
                    then_val = if_stmt.then_yields[i]
                    if isinstance(then_val, (SSAValue, VectorSSAValue)):
                        then_live.add(then_val)
                if i < len(if_stmt.else_yields):
                    else_val = if_stmt.else_yields[i]
                    if isinstance(else_val, (SSAValue, VectorSSAValue)):
                        else_live.add(else_val)

        # If branch has side effects, all its yields are potentially live
        if then_has_side_effects:
            for y in if_stmt.then_yields:
                if isinstance(y, (SSAValue, VectorSSAValue)):
                    then_live.add(y)
        if else_has_side_effects:
            for y in if_stmt.else_yields:
                if isinstance(y, (SSAValue, VectorSSAValue)):
                    else_live.add(y)

        # Propagate liveness backward through branches
        self._mark_live_backward(if_stmt.then_body, then_live, ctx)
        self._mark_live_backward(if_stmt.else_body, else_live, ctx)

        # Filter each branch
        new_then_body = self._filter_dead_code(if_stmt.then_body, then_live, ctx)
        new_else_body = self._filter_dead_code(if_stmt.else_body, else_live, ctx)

        return If(
            cond=if_stmt.cond,
            then_body=new_then_body,
            then_yields=if_stmt.then_yields,
            else_body=new_else_body,
            else_yields=if_stmt.else_yields,
            results=if_stmt.results
        )
