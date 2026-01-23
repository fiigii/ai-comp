"""
Dead Code Elimination (DCE) Pass

Removes operations whose results are never used.
Uses backward mark-and-sweep analysis on SSA values.
"""

from ..hir import (
    SSAValue, Const, Operand, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)
from ..pass_manager import Pass, PassConfig


# Operations that have side effects and should never be eliminated
SIDE_EFFECT_OPS = {"store"}


class DCEPass(Pass):
    """
    Dead Code Elimination using backward mark-and-sweep.

    Pass 1: Compute live set (backward traversal)
    - Mark SSA values as live if used by side-effect ops or other live ops

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

        # Pass 1: Compute live set (backward analysis)
        live = self._compute_live_set(hir.body)

        # Pass 2: Filter dead code (forward sweep)
        new_body = self._filter_dead_code(hir.body, live)

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
            num_ssa_values=hir.num_ssa_values
        )

    def _compute_live_set(self, body: list[Statement]) -> set[int]:
        """
        Compute the set of live SSA value IDs using backward traversal.

        An SSA value is live if:
        - It's used as operand to a side-effect op (store)
        - It's used as operand to Halt/Pause
        - It's used as operand to another live operation
        - It's a ForLoop/If result that's used downstream
        - It's a ForLoop yield that feeds a live result
        - It's an If condition when the If is needed
        """
        live: set[int] = set()
        self._mark_live_backward(body, live)
        return live

    def _mark_live_backward(self, body: list[Statement], live: set[int]) -> None:
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
                self._mark_forloop_live(stmt, live)
            elif isinstance(stmt, If):
                self._mark_if_live(stmt, live)

    def _mark_op_live(self, op: Op, live: set[int]) -> None:
        """Mark an Op's operands as live if the op is live."""
        # Op is live if:
        # 1. It has side effects (store, result=None)
        # 2. Its result is in the live set
        is_live = False

        if self._is_side_effect_op(op):
            is_live = True
        elif op.result is not None and op.result.id in live:
            is_live = True

        if is_live:
            # Mark all operands as live
            for operand in op.operands:
                if isinstance(operand, SSAValue):
                    live.add(operand.id)

    def _mark_forloop_live(self, loop: ForLoop, live: set[int]) -> None:
        """Mark a ForLoop's components as live if needed."""
        # Build map from body_param id to its index
        body_param_id_to_idx = {p.id: i for i, p in enumerate(loop.body_params)}

        # First, process the loop body to determine what's live inside
        body_live: set[int] = set()

        # If any result is live, mark the corresponding yield as live
        for i, result in enumerate(loop.results):
            if result.id in live:
                if i < len(loop.yields):
                    body_live.add(loop.yields[i].id)

        # Iteratively compute live set until fixed point
        # This is needed because body_params[i] = yields[i] on subsequent iterations
        # So if body_params[i] is live, yields[i] must also be live
        while True:
            prev_size = len(body_live)

            # Mark any body_param that's live -> also mark corresponding yield as live
            for ssa_id in list(body_live):
                if ssa_id in body_param_id_to_idx:
                    idx = body_param_id_to_idx[ssa_id]
                    if idx < len(loop.yields):
                        yield_val = loop.yields[idx]
                        if isinstance(yield_val, SSAValue):
                            body_live.add(yield_val.id)

            # Run backward analysis on body
            self._mark_live_backward(loop.body, body_live)

            # Check for fixed point
            if len(body_live) == prev_size:
                break

        # Check if loop has side effects in body
        has_side_effects = self._has_side_effects(loop.body)

        # Loop is live if any result is used OR body has side effects
        loop_is_live = has_side_effects or any(r.id in live for r in loop.results)

        if loop_is_live:
            # Mark yields as live (they feed back to body_params)
            for y in loop.yields:
                if isinstance(y, SSAValue):
                    body_live.add(y.id)

            # Mark start/end as live
            if isinstance(loop.start, SSAValue):
                live.add(loop.start.id)
            if isinstance(loop.end, SSAValue):
                live.add(loop.end.id)

            # Propagate body_live to outer live set
            # For body_params that are live, mark the corresponding iter_args as live
            # For other SSA values, propagate to outer scope
            counter_id = loop.counter.id
            for ssa_id in body_live:
                if ssa_id == counter_id:
                    # Counter is local to loop, don't propagate
                    continue
                elif ssa_id in body_param_id_to_idx:
                    # Body param is live - mark corresponding iter_arg as live
                    idx = body_param_id_to_idx[ssa_id]
                    if idx < len(loop.iter_args):
                        arg = loop.iter_args[idx]
                        if isinstance(arg, SSAValue):
                            live.add(arg.id)
                else:
                    # Value defined outside loop - propagate to outer scope
                    live.add(ssa_id)

    def _mark_if_live(self, if_stmt: If, live: set[int]) -> None:
        """Mark an If's components as live if needed."""
        # First, process both branches to determine what's live inside
        then_live: set[int] = set()
        else_live: set[int] = set()

        # If any result is live, mark the corresponding yields as live
        for i, result in enumerate(if_stmt.results):
            if result.id in live:
                if i < len(if_stmt.then_yields):
                    then_live.add(if_stmt.then_yields[i].id)
                if i < len(if_stmt.else_yields):
                    else_live.add(if_stmt.else_yields[i].id)

        # Recursively mark live in branches
        self._mark_live_backward(if_stmt.then_body, then_live)
        self._mark_live_backward(if_stmt.else_body, else_live)

        # Check if branches have side effects
        then_has_side_effects = self._has_side_effects(if_stmt.then_body)
        else_has_side_effects = self._has_side_effects(if_stmt.else_body)

        # If is live if any result is used OR either branch has side effects
        if_is_live = (then_has_side_effects or else_has_side_effects or
                      any(r.id in live for r in if_stmt.results))

        if if_is_live:
            # Mark condition as live
            live.add(if_stmt.cond.id)

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

    def _filter_dead_code(self, body: list[Statement], live: set[int]) -> list[Statement]:
        """Filter out dead statements, keeping only live ones."""
        result = []

        for stmt in body:
            if isinstance(stmt, Op):
                filtered = self._filter_op(stmt, live)
                if filtered is not None:
                    result.append(filtered)
            elif isinstance(stmt, Halt):
                result.append(stmt)
            elif isinstance(stmt, Pause):
                result.append(stmt)
            elif isinstance(stmt, ForLoop):
                filtered = self._filter_forloop(stmt, live)
                if filtered is not None:
                    result.append(filtered)
            elif isinstance(stmt, If):
                filtered = self._filter_if(stmt, live)
                if filtered is not None:
                    result.append(filtered)

        return result

    def _filter_op(self, op: Op, live: set[int]) -> Op | None:
        """Filter an Op - keep if it has side effects or result is live."""
        if self._is_side_effect_op(op):
            return op
        if op.result is not None and op.result.id in live:
            return op

        # Dead code - eliminate
        self._ops_eliminated += 1
        return None

    def _filter_forloop(self, loop: ForLoop, live: set[int]) -> ForLoop | None:
        """Filter a ForLoop - keep if any result is live or body has side effects."""
        has_side_effects = self._has_side_effects(loop.body)
        any_result_live = any(r.id in live for r in loop.results)

        if not has_side_effects and not any_result_live:
            # Dead loop - eliminate
            self._loops_eliminated += 1
            return None

        # Build map from body_param id to its index
        body_param_id_to_idx = {p.id: i for i, p in enumerate(loop.body_params)}

        # Compute what's live in the loop body
        body_live: set[int] = set()

        # Results that are live make corresponding yields live
        for i, result in enumerate(loop.results):
            if result.id in live:
                if i < len(loop.yields):
                    body_live.add(loop.yields[i].id)

        # If body has side effects, all yields are potentially live
        if has_side_effects:
            for y in loop.yields:
                body_live.add(y.id)

        # Iteratively compute live set until fixed point
        while True:
            prev_size = len(body_live)

            # Mark any body_param that's live -> also mark corresponding yield as live
            for ssa_id in list(body_live):
                if ssa_id in body_param_id_to_idx:
                    idx = body_param_id_to_idx[ssa_id]
                    if idx < len(loop.yields):
                        yield_val = loop.yields[idx]
                        if isinstance(yield_val, SSAValue):
                            body_live.add(yield_val.id)

            # Propagate liveness backward through body
            self._mark_live_backward(loop.body, body_live)

            # Check for fixed point
            if len(body_live) == prev_size:
                break

        # Filter the body
        new_body = self._filter_dead_code(loop.body, body_live)

        return ForLoop(
            counter=loop.counter,
            start=loop.start,
            end=loop.end,
            iter_args=loop.iter_args,
            body_params=loop.body_params,
            body=new_body,
            yields=loop.yields,
            results=loop.results
        )

    def _filter_if(self, if_stmt: If, live: set[int]) -> If | None:
        """Filter an If - keep if any result is live or branches have side effects."""
        then_has_side_effects = self._has_side_effects(if_stmt.then_body)
        else_has_side_effects = self._has_side_effects(if_stmt.else_body)
        any_result_live = any(r.id in live for r in if_stmt.results)

        if not then_has_side_effects and not else_has_side_effects and not any_result_live:
            # Dead if - eliminate
            self._ifs_eliminated += 1
            return None

        # Compute what's live in each branch
        then_live: set[int] = set()
        else_live: set[int] = set()

        # Results that are live make corresponding yields live
        for i, result in enumerate(if_stmt.results):
            if result.id in live:
                if i < len(if_stmt.then_yields):
                    then_live.add(if_stmt.then_yields[i].id)
                if i < len(if_stmt.else_yields):
                    else_live.add(if_stmt.else_yields[i].id)

        # If branch has side effects, all its yields are potentially live
        if then_has_side_effects:
            for y in if_stmt.then_yields:
                then_live.add(y.id)
        if else_has_side_effects:
            for y in if_stmt.else_yields:
                else_live.add(y.id)

        # Propagate liveness backward through branches
        self._mark_live_backward(if_stmt.then_body, then_live)
        self._mark_live_backward(if_stmt.else_body, else_live)

        # Filter each branch
        new_then_body = self._filter_dead_code(if_stmt.then_body, then_live)
        new_else_body = self._filter_dead_code(if_stmt.else_body, else_live)

        return If(
            cond=if_stmt.cond,
            then_body=new_then_body,
            then_yields=if_stmt.then_yields,
            else_body=new_else_body,
            else_yields=if_stmt.else_yields,
            results=if_stmt.results
        )
