"""
Use-Def Chain Infrastructure for HIR

Provides efficient queries for:
1. All uses of an SSA value (def -> uses)
2. The defining instruction of an SSA value (use -> def)
3. The parent statement list of any instruction (stmt -> parent)
4. Automatic updates when IR changes
"""

from dataclasses import dataclass, field
from typing import Optional, Iterator

from .hir import SSAValue, VectorSSAValue, Variable, Const, Value, Op, ForLoop, If, Halt, Pause, Statement, HIRFunction


@dataclass
class DefLocation:
    """Where an SSA value is defined."""
    statement: Statement      # Op, ForLoop, or If
    def_kind: str             # "result", "counter", "body_param", "loop_result", "if_result"
    parent_list: list[Statement]


@dataclass
class UseLocation:
    """Where an SSA value is used."""
    statement: Statement      # The statement containing the use
    operand_index: int        # Index in operands (or -1 for special uses)
    use_kind: str             # "operand", "condition", "iter_arg", "yield", etc.
    parent_list: list[Statement]

    def __eq__(self, other):
        if not isinstance(other, UseLocation):
            return False
        return (id(self.statement) == id(other.statement) and
                self.operand_index == other.operand_index and
                self.use_kind == other.use_kind)

    def __hash__(self):
        return hash((id(self.statement), self.operand_index, self.use_kind))


@dataclass
class UseDefContext:
    """
    Context for use-def chain queries on HIR.

    Builds chains lazily on first query and caches results.
    Call invalidate() when the IR changes.
    """
    hir: HIRFunction
    _built: bool = field(default=False, repr=False)

    # Mappings (built lazily) - use SSAValue/VectorSSAValue objects directly as keys
    _defs: dict[Variable, DefLocation] = field(default_factory=dict, repr=False)
    _uses: dict[Variable, list[UseLocation]] = field(default_factory=dict, repr=False)
    _parents: dict[int, tuple[list[Statement], int]] = field(default_factory=dict, repr=False)

    def _ensure_built(self) -> None:
        """Build the use-def chains if not already built."""
        if not self._built:
            self._build()

    def _build(self) -> None:
        """Build all use-def chain mappings."""
        self._defs.clear()
        self._uses.clear()
        self._parents.clear()
        self._traverse(self.hir.body)
        self._built = True

    def _traverse(self, body: list[Statement]) -> None:
        """Recursively traverse statements and build mappings."""
        for idx, stmt in enumerate(body):
            self._parents[id(stmt)] = (body, idx)
            if isinstance(stmt, Op):
                self._process_op(stmt, body)
            elif isinstance(stmt, ForLoop):
                self._process_forloop(stmt, body)
            elif isinstance(stmt, If):
                self._process_if(stmt, body)
            # Halt and Pause don't define or use SSA values

    def _add_def(self, ssa: Variable, def_loc: DefLocation) -> None:
        """Record a definition."""
        self._defs[ssa] = def_loc

    def _add_use(self, ssa: Variable, use_loc: UseLocation) -> None:
        """Record a use."""
        if ssa not in self._uses:
            self._uses[ssa] = []
        self._uses[ssa].append(use_loc)

    def _process_op(self, op: Op, parent_list: list[Statement]) -> None:
        """Process an Op statement for definitions and uses."""
        # Definition: op.result (if not None)
        if op.result is not None:
            self._add_def(op.result, DefLocation(op, "result", parent_list))

        # Uses: each operand (skip Const values)
        for idx, operand in enumerate(op.operands):
            if isinstance(operand, (SSAValue, VectorSSAValue)):
                self._add_use(operand, UseLocation(op, idx, "operand", parent_list))

    def _process_forloop(self, loop: ForLoop, parent_list: list[Statement]) -> None:
        """Process a ForLoop statement for definitions and uses."""
        # Definitions
        # counter is defined by the loop
        self._add_def(loop.counter, DefLocation(loop, "counter", parent_list))

        # body_params are defined by the loop (phi values)
        for param in loop.body_params:
            self._add_def(param, DefLocation(loop, "body_param", parent_list))

        # results are defined by the loop (values after loop completes)
        for result in loop.results:
            self._add_def(result, DefLocation(loop, "loop_result", parent_list))

        # Uses
        # start and end are uses (skip Const)
        if isinstance(loop.start, (SSAValue, VectorSSAValue)):
            self._add_use(loop.start, UseLocation(loop, -1, "start", parent_list))
        if isinstance(loop.end, (SSAValue, VectorSSAValue)):
            self._add_use(loop.end, UseLocation(loop, -1, "end", parent_list))

        # iter_args are uses (initial values entering the loop)
        for idx, arg in enumerate(loop.iter_args):
            if isinstance(arg, (SSAValue, VectorSSAValue)):
                self._add_use(arg, UseLocation(loop, idx, "iter_arg", parent_list))

        # yields are uses (values fed back to body_params)
        for idx, y in enumerate(loop.yields):
            if isinstance(y, (SSAValue, VectorSSAValue)):
                self._add_use(y, UseLocation(loop, idx, "yield", parent_list))

        # Recursively process loop body
        self._traverse(loop.body)

    def _process_if(self, if_stmt: If, parent_list: list[Statement]) -> None:
        """Process an If statement for definitions and uses."""
        # Definitions
        # results are defined by the if (phi of then/else yields)
        for result in if_stmt.results:
            self._add_def(result, DefLocation(if_stmt, "if_result", parent_list))

        # Uses
        # cond is a use
        if isinstance(if_stmt.cond, (SSAValue, VectorSSAValue)):
            self._add_use(if_stmt.cond, UseLocation(if_stmt, -1, "condition", parent_list))

        # then_yields are uses
        for idx, y in enumerate(if_stmt.then_yields):
            if isinstance(y, (SSAValue, VectorSSAValue)):
                self._add_use(y, UseLocation(if_stmt, idx, "then_yield", parent_list))

        # else_yields are uses
        for idx, y in enumerate(if_stmt.else_yields):
            if isinstance(y, (SSAValue, VectorSSAValue)):
                self._add_use(y, UseLocation(if_stmt, idx, "else_yield", parent_list))

        # Recursively process then and else bodies
        self._traverse(if_stmt.then_body)
        self._traverse(if_stmt.else_body)

    # ==================== Query API ====================

    def get_def(self, ssa: Variable) -> Optional[DefLocation]:
        """Get the DefLocation for an SSA value."""
        self._ensure_built()
        return self._defs.get(ssa)

    def get_uses(self, ssa: Variable) -> list[UseLocation]:
        """Get list of UseLocation for an SSA value."""
        self._ensure_built()
        return self._uses.get(ssa, [])

    def has_uses(self, ssa: Variable) -> bool:
        """Check if SSA value has any uses."""
        self._ensure_built()
        return ssa in self._uses and len(self._uses[ssa]) > 0

    def use_count(self, ssa: Variable) -> int:
        """Get number of uses of an SSA value."""
        self._ensure_built()
        return len(self._uses.get(ssa, []))

    def get_parent(self, stmt: Statement) -> Optional[tuple[list[Statement], int]]:
        """Get (parent_list, index) for a statement."""
        self._ensure_built()
        return self._parents.get(id(stmt))

    def get_all_defs(self) -> Iterator[tuple[Variable, DefLocation]]:
        """Iterate over all definitions."""
        self._ensure_built()
        # SSAValue/VectorSSAValue objects are used directly as keys
        for ssa, def_loc in self._defs.items():
            yield ssa, def_loc

    # ==================== Update API ====================

    def invalidate(self) -> None:
        """Mark context as stale, will rebuild on next query."""
        self._built = False

    def rebuild(self, new_hir: Optional[HIRFunction] = None) -> None:
        """Replace HIR and/or rebuild the context."""
        if new_hir is not None:
            self.hir = new_hir
        self._built = False
        self._ensure_built()

    def add_use(self, ssa: Variable, use_loc: UseLocation) -> None:
        """Incrementally add a use."""
        self._ensure_built()
        self._add_use(ssa, use_loc)

    def remove_use(self, ssa: Variable, use_loc: UseLocation) -> None:
        """Incrementally remove a use."""
        self._ensure_built()
        uses = self._uses.get(ssa, [])

        # Find and remove the matching use
        for i, u in enumerate(uses):
            if u == use_loc:
                uses.pop(i)
                return

    def replace_all_uses(self, old_ssa: Variable,
                         new_value: Value,
                         auto_invalidate: bool = True) -> int:
        """
        Replace all uses of old_ssa with new_value.

        Args:
            old_ssa: The SSA value to replace (must be SSAValue or VectorSSAValue)
            new_value: The replacement value (can be SSAValue, VectorSSAValue, or Const)
            auto_invalidate: If True (default), invalidate the context after replacement.
                           Set to False when doing batch replacements for better performance,
                           then call invalidate() manually when done.

        Returns the number of uses replaced.
        Note: This modifies the actual IR operands, not just the use-def mappings.
        After calling this, the context needs to be invalidated and rebuilt
        if accurate use-def information is needed.
        """
        self._ensure_built()

        uses = self.get_uses(old_ssa)
        count = 0

        for use_loc in uses:
            stmt = use_loc.statement

            if use_loc.use_kind == "operand":
                assert isinstance(stmt, Op)
                if use_loc.operand_index >= 0 and use_loc.operand_index < len(stmt.operands):
                    if stmt.operands[use_loc.operand_index] == old_ssa:
                        stmt.operands[use_loc.operand_index] = new_value
                        count += 1

            elif use_loc.use_kind == "start":
                assert isinstance(stmt, ForLoop)
                if stmt.start == old_ssa:
                    stmt.start = new_value
                    count += 1

            elif use_loc.use_kind == "end":
                assert isinstance(stmt, ForLoop)
                if stmt.end == old_ssa:
                    stmt.end = new_value
                    count += 1

            elif use_loc.use_kind == "iter_arg":
                assert isinstance(stmt, ForLoop)
                if use_loc.operand_index >= 0 and use_loc.operand_index < len(stmt.iter_args):
                    if stmt.iter_args[use_loc.operand_index] == old_ssa:
                        stmt.iter_args[use_loc.operand_index] = new_value
                        count += 1

            elif use_loc.use_kind == "yield":
                assert isinstance(stmt, ForLoop)
                if use_loc.operand_index >= 0 and use_loc.operand_index < len(stmt.yields):
                    if stmt.yields[use_loc.operand_index] == old_ssa:
                        stmt.yields[use_loc.operand_index] = new_value
                        count += 1

            elif use_loc.use_kind == "condition":
                assert isinstance(stmt, If)
                if stmt.cond == old_ssa:
                    stmt.cond = new_value
                    count += 1

            elif use_loc.use_kind == "then_yield":
                assert isinstance(stmt, If)
                if use_loc.operand_index >= 0 and use_loc.operand_index < len(stmt.then_yields):
                    if stmt.then_yields[use_loc.operand_index] == old_ssa:
                        stmt.then_yields[use_loc.operand_index] = new_value
                        count += 1

            elif use_loc.use_kind == "else_yield":
                assert isinstance(stmt, If)
                if use_loc.operand_index >= 0 and use_loc.operand_index < len(stmt.else_yields):
                    if stmt.else_yields[use_loc.operand_index] == old_ssa:
                        stmt.else_yields[use_loc.operand_index] = new_value
                        count += 1

        # After replacing, optionally invalidate so mappings get rebuilt
        if auto_invalidate:
            self.invalidate()
        return count
