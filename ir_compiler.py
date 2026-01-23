"""
IR Compiler for VLIW SIMD Virtual Machine

Two-level IR:
- HIR (High-Level IR): SSA form with explicit loops and branches
- LIR (Low-Level IR): Basic blocks with jumps, close to machine code

Compilation pipeline: HIR -> LIR -> VLIW assembly
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Union, Any
import json
from problem import HASH_STAGES, SLOT_LIMITS, VLEN, SCRATCH_SIZE


# =============================================================================
# HIR (High-Level IR) - SSA Form
# =============================================================================

@dataclass(frozen=True)
class SSAValue:
    """An SSA value - assigned exactly once."""
    id: int
    name: Optional[str] = None

    def __repr__(self):
        if self.name:
            return f"v{self.id}:{self.name}"
        return f"v{self.id}"


@dataclass(frozen=True)
class Const:
    """A compile-time constant."""
    value: int

    def __repr__(self):
        return f"#{self.value}"


# Type alias for operands
Operand = Union[SSAValue, Const]


@dataclass
class Op:
    """Single SSA operation: result = opcode(operands)"""
    opcode: str                      # "+", "load", "store", "select", etc.
    result: Optional[SSAValue]       # None for store/side-effects
    operands: list[Operand]
    engine: str                      # alu/valu/load/store/flow

    def __repr__(self):
        ops_str = ", ".join(str(o) for o in self.operands)
        if self.result:
            return f"{self.result} = {self.opcode}({ops_str}) [{self.engine}]"
        return f"{self.opcode}({ops_str}) [{self.engine}]"


@dataclass
class Halt:
    """Halt execution."""
    pass


@dataclass
class Pause:
    """Pause execution (for debugging sync with reference)."""
    pass


@dataclass
class ForLoop:
    """
    For loop with SSA semantics:

    for counter in range(start, end):
        body (uses body_params, yields values)

    iter_args:    Initial values entering the loop
    body_params:  SSA values available in body (from back-edge phi)
    yields:       Values at end of each iteration (fed back to body_params)
    results:      SSA values available after loop (final carried values)
    """
    counter: SSAValue
    start: Operand
    end: Operand
    iter_args: list[SSAValue]
    body_params: list[SSAValue]
    body: list  # list[Statement]
    yields: list[SSAValue]
    results: list[SSAValue]

    def __repr__(self):
        return f"ForLoop({self.counter}, {self.start}..{self.end}, body={len(self.body)} stmts)"


@dataclass
class If:
    """
    If statement with SSA semantics:

    if cond:
        then_body (yields then_vals)
    else:
        else_body (yields else_vals)
    results = phi(then_vals, else_vals)
    """
    cond: SSAValue
    then_body: list  # list[Statement]
    then_yields: list[SSAValue]
    else_body: list  # list[Statement]
    else_yields: list[SSAValue]
    results: list[SSAValue]

    def __repr__(self):
        return f"If({self.cond}, then={len(self.then_body)}, else={len(self.else_body)})"


# Statement type alias
Statement = Union[Op, Halt, Pause, ForLoop, If]


@dataclass
class HIRFunction:
    """A complete HIR function."""
    name: str
    body: list[Statement]
    num_ssa_values: int


# =============================================================================
# Pass Manager Infrastructure
# =============================================================================

@dataclass
class PassConfig:
    """Configuration for a single pass."""
    name: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)


class Pass(ABC):
    """Base class for all HIR transformation passes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the pass name for config matching."""
        pass

    @abstractmethod
    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        """Transform HIR and return new HIRFunction."""
        pass


@dataclass
class PassManager:
    """Manages and runs HIR transformation passes."""
    passes: list[Pass] = field(default_factory=list)
    config: dict[str, PassConfig] = field(default_factory=dict)
    print_after_all: bool = False

    def add_pass(self, p: Pass) -> None:
        """Register a pass."""
        self.passes.append(p)

    def load_config(self, config_path: str) -> None:
        """Load pass configs from JSON file."""
        with open(config_path) as f:
            data = json.load(f)
        for pass_name, opts in data.get("passes", {}).items():
            self.config[pass_name] = PassConfig(
                name=pass_name,
                enabled=opts.get("enabled", True),
                options=opts.get("options", {})
            )

    def run(self, hir: HIRFunction) -> HIRFunction:
        """Run all enabled passes in order."""
        if self.print_after_all:
            print("=== HIR (before passes) ===")
            print_hir(hir)

        for p in self.passes:
            cfg = self.config.get(p.name, PassConfig(name=p.name))
            if not cfg.enabled:
                continue
            hir = p.run(hir, cfg)
            if self.print_after_all:
                print(f"=== HIR (after {p.name}) ===")
                print_hir(hir)

        return hir


# =============================================================================
# SSA Renumbering Context (for passes)
# =============================================================================

@dataclass
class SSARenumberContext:
    """Track SSA value renumbering during transformations."""
    next_id: int
    result_bindings: dict[int, SSAValue] = field(default_factory=dict)

    def new_ssa(self, name: str = None) -> SSAValue:
        """Create a new unique SSA value."""
        ssa = SSAValue(self.next_id, name)
        self.next_id += 1
        return ssa

    def bind_result(self, old_id: int, new_ssa: SSAValue):
        """Map old result SSA to new value (for eliminating loop after unroll)."""
        self.result_bindings[old_id] = new_ssa


# =============================================================================
# HIR Builder - Functional SSA API
# =============================================================================

class HIRBuilder:
    """Builder for constructing HIR in SSA form."""

    def __init__(self):
        self._ssa_counter = 0
        self._statements: list[Statement] = []

    def _new_ssa(self, name: Optional[str] = None) -> SSAValue:
        """Create a new SSA value."""
        v = SSAValue(self._ssa_counter, name)
        self._ssa_counter += 1
        return v

    def _emit(self, stmt: Statement):
        """Add a statement to current context."""
        self._statements.append(stmt)

    # === Constants ===

    def const(self, value: int) -> Const:
        """Create a compile-time constant."""
        return Const(value)

    def const_load(self, value: int, name: Optional[str] = None) -> SSAValue:
        """Load an immediate constant into an SSA value.

        Note: We intentionally do NOT cache across different statement contexts
        (if/else branches, loop bodies) because that would cause the const load
        to only be emitted in one branch, leaving it undefined in the other.
        Caching is safe to re-enable if constants are hoisted to entry block.
        """
        result = self._new_ssa(name or f"c{value}")
        self._emit(Op("const", result, [Const(value)], "load"))
        return result

    # === ALU operations ===

    def alu(self, op: str, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        """Emit a binary ALU operation."""
        result = self._new_ssa(name)
        self._emit(Op(op, result, [a, b], "alu"))
        return result

    def add(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("+", a, b, name)

    def sub(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("-", a, b, name)

    def mul(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("*", a, b, name)

    def div(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("//", a, b, name)

    def mod(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("%", a, b, name)

    def xor(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("^", a, b, name)

    def and_(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("&", a, b, name)

    def or_(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("|", a, b, name)

    def shl(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("<<", a, b, name)

    def shr(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu(">>", a, b, name)

    def lt(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("<", a, b, name)

    def eq(self, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        return self.alu("==", a, b, name)

    # === Memory operations ===

    def load(self, addr: SSAValue, name: Optional[str] = None) -> SSAValue:
        """Load from memory at address."""
        result = self._new_ssa(name)
        self._emit(Op("load", result, [addr], "load"))
        return result

    def store(self, addr: SSAValue, value: SSAValue):
        """Store value to memory at address."""
        self._emit(Op("store", None, [addr, value], "store"))

    # === Flow operations ===

    def select(self, cond: SSAValue, a: SSAValue, b: SSAValue, name: Optional[str] = None) -> SSAValue:
        """Conditional select: cond ? a : b"""
        result = self._new_ssa(name)
        self._emit(Op("select", result, [cond, a, b], "flow"))
        return result

    # === Control flow ===

    def for_loop(
        self,
        start: Operand,
        end: Operand,
        iter_args: list[SSAValue],
        body_fn: Callable[[SSAValue, list[SSAValue]], list[SSAValue]]
    ) -> list[SSAValue]:
        """
        Build a for loop.

        body_fn receives (counter, body_params) and returns yield values.
        Returns the loop results (final values after loop exits).
        """
        counter = self._new_ssa("i")
        body_params = [self._new_ssa(f"loop_param_{i}") for i in range(len(iter_args))]
        results = [self._new_ssa(f"loop_result_{i}") for i in range(len(iter_args))]

        # Build body in a new statement context
        old_stmts = self._statements
        self._statements = []
        yields = body_fn(counter, body_params)
        body = self._statements
        self._statements = old_stmts

        assert len(yields) == len(iter_args), \
            f"Yield count ({len(yields)}) must match iter_args count ({len(iter_args)})"

        loop = ForLoop(
            counter=counter,
            start=start,
            end=end,
            iter_args=iter_args,
            body_params=body_params,
            body=body,
            yields=yields,
            results=results,
        )
        self._emit(loop)
        return results

    def if_stmt(
        self,
        cond: SSAValue,
        then_fn: Callable[[], list[SSAValue]],
        else_fn: Callable[[], list[SSAValue]],
    ) -> list[SSAValue]:
        """
        Build an if statement.

        Both branches must yield the same number of values.
        Returns merged SSA values.
        """
        # Build then branch
        old_stmts = self._statements
        self._statements = []
        then_yields = then_fn()
        then_body = self._statements

        # Build else branch
        self._statements = []
        else_yields = else_fn()
        else_body = self._statements
        self._statements = old_stmts

        assert len(then_yields) == len(else_yields), \
            f"Branch yield count mismatch: then={len(then_yields)}, else={len(else_yields)}"

        results = [self._new_ssa(f"if_result_{i}") for i in range(len(then_yields))]

        if_node = If(
            cond=cond,
            then_body=then_body,
            then_yields=then_yields,
            else_body=else_body,
            else_yields=else_yields,
            results=results,
        )
        self._emit(if_node)
        return results

    def halt(self):
        """Emit halt instruction."""
        self._emit(Halt())

    def pause(self):
        """Emit pause instruction."""
        self._emit(Pause())

    def build(self) -> HIRFunction:
        """Finalize and return the HIR function."""
        return HIRFunction(
            name="kernel",
            body=self._statements,
            num_ssa_values=self._ssa_counter,
        )


# =============================================================================
# Loop Unroll Pass
# =============================================================================

class LoopUnrollPass(Pass):
    """
    Unroll ForLoops with static trip counts.

    By default, fully unroll loops with static (Const) bounds.
    If 'unroll_factor' is set in config, partially unroll instead.
    The factor must divide the trip count evenly; otherwise skip unrolling.
    Loops with dynamic bounds are left unchanged.

    Options:
        unroll_factor: If set, partially unroll by this factor (must divide trip count).
                       If None, attempt full unroll.
        max_trip_count: Maximum trip count to fully unroll (default: 32).
                        Loops with larger trip counts are skipped unless unroll_factor is set.
    """

    @property
    def name(self) -> str:
        return "loop-unroll"

    def run(self, hir: HIRFunction, config: PassConfig) -> HIRFunction:
        unroll_factor = config.options.get("unroll_factor", None)  # None = full unroll
        # Very large default - fully unroll all static loops
        # If scratch space is exhausted, use a config file to set unroll_factor
        max_trip_count = config.options.get("max_trip_count", 10000000)

        # Create SSA renumber context starting from current max
        ctx = SSARenumberContext(hir.num_ssa_values)

        # Transform body recursively
        new_body = self._transform_statements(hir.body, unroll_factor, max_trip_count, ctx)

        return HIRFunction(
            name=hir.name,
            body=new_body,
            num_ssa_values=ctx.next_id
        )

    def _transform_statements(
        self, stmts: list[Statement], factor: Optional[int], max_trip: int, ctx: SSARenumberContext
    ) -> list[Statement]:
        """Transform a list of statements, unrolling loops as appropriate."""
        result = []
        for stmt in stmts:
            # First, remap operands based on result_bindings from previous unrolled loops
            stmt = self._apply_bindings(stmt, ctx.result_bindings)

            if isinstance(stmt, ForLoop):
                result.extend(self._unroll_loop(stmt, factor, max_trip, ctx))
            elif isinstance(stmt, If):
                result.append(self._transform_if(stmt, factor, max_trip, ctx))
            else:
                result.append(stmt)
        return result

    def _apply_bindings(self, stmt: Statement, bindings: dict[int, SSAValue]) -> Statement:
        """Apply SSA value bindings to a statement's operands."""
        if not bindings:
            return stmt

        if isinstance(stmt, Op):
            new_operands = []
            for op in stmt.operands:
                if isinstance(op, SSAValue) and op.id in bindings:
                    new_operands.append(bindings[op.id])
                else:
                    new_operands.append(op)
            return Op(stmt.opcode, stmt.result, new_operands, stmt.engine)

        elif isinstance(stmt, ForLoop):
            # Remap iter_args
            new_iter_args = []
            for arg in stmt.iter_args:
                if arg.id in bindings:
                    new_iter_args.append(bindings[arg.id])
                else:
                    new_iter_args.append(arg)
            return ForLoop(
                counter=stmt.counter,
                start=stmt.start if not isinstance(stmt.start, SSAValue) or stmt.start.id not in bindings
                      else bindings[stmt.start.id],
                end=stmt.end if not isinstance(stmt.end, SSAValue) or stmt.end.id not in bindings
                    else bindings[stmt.end.id],
                iter_args=new_iter_args,
                body_params=stmt.body_params,
                body=stmt.body,
                yields=stmt.yields,
                results=stmt.results
            )

        elif isinstance(stmt, If):
            new_cond = bindings.get(stmt.cond.id, stmt.cond)
            return If(
                cond=new_cond,
                then_body=stmt.then_body,
                then_yields=stmt.then_yields,
                else_body=stmt.else_body,
                else_yields=stmt.else_yields,
                results=stmt.results
            )

        return stmt

    def _transform_if(
        self, if_stmt: If, factor: Optional[int], max_trip: int, ctx: SSARenumberContext
    ) -> If:
        """Recursively transform If statement bodies."""
        return If(
            cond=if_stmt.cond,
            then_body=self._transform_statements(if_stmt.then_body, factor, max_trip, ctx),
            then_yields=if_stmt.then_yields,
            else_body=self._transform_statements(if_stmt.else_body, factor, max_trip, ctx),
            else_yields=if_stmt.else_yields,
            results=if_stmt.results
        )

    def _unroll_loop(
        self, loop: ForLoop, factor: Optional[int], max_trip: int, ctx: SSARenumberContext
    ) -> list[Statement]:
        """Unroll a single ForLoop if possible."""
        # Check if loop has static bounds
        if not isinstance(loop.start, Const) or not isinstance(loop.end, Const):
            # Dynamic bounds - recurse into body but don't unroll
            new_body = self._transform_statements(loop.body, factor, max_trip, ctx)
            return [ForLoop(
                counter=loop.counter,
                start=loop.start,
                end=loop.end,
                iter_args=loop.iter_args,
                body_params=loop.body_params,
                body=new_body,
                yields=loop.yields,
                results=loop.results
            )]

        trip_count = loop.end.value - loop.start.value

        if trip_count <= 0:
            # Empty loop - just bind results to iter_args
            for result_ssa, iter_arg in zip(loop.results, loop.iter_args):
                ctx.bind_result(result_ssa.id, iter_arg)
            return []

        # Determine actual unroll factor
        if factor is None:
            # Full unroll - but only if trip count is within limit
            if trip_count > max_trip:
                # Too large for full unroll - just recurse into body
                new_body = self._transform_statements(loop.body, factor, max_trip, ctx)
                return [ForLoop(
                    counter=loop.counter,
                    start=loop.start,
                    end=loop.end,
                    iter_args=loop.iter_args,
                    body_params=loop.body_params,
                    body=new_body,
                    yields=loop.yields,
                    results=loop.results
                )]
            actual_factor = trip_count
        elif trip_count % factor != 0:
            # Factor doesn't divide evenly - skip unrolling this loop
            new_body = self._transform_statements(loop.body, factor, max_trip, ctx)
            return [ForLoop(
                counter=loop.counter,
                start=loop.start,
                end=loop.end,
                iter_args=loop.iter_args,
                body_params=loop.body_params,
                body=new_body,
                yields=loop.yields,
                results=loop.results
            )]
        else:
            actual_factor = factor

        # Fully unroll
        if actual_factor == trip_count:
            return self._fully_unroll(loop, trip_count, ctx, factor, max_trip)

        # Partial unroll - create loop with unrolled body
        return self._partially_unroll(loop, actual_factor, ctx, factor, max_trip)

    def _fully_unroll(
        self, loop: ForLoop, trip_count: int, ctx: SSARenumberContext,
        nested_factor: Optional[int], max_trip: int
    ) -> list[Statement]:
        """Replace loop with trip_count copies of body."""
        result = []

        # Track current values for loop-carried variables
        # Initially: current_params[i] = iter_args[i]
        current_params = list(loop.iter_args)

        for i in range(trip_count):
            # Create remapping: counter -> const(start + i), body_params -> current_params
            remap: dict[int, Operand] = {}

            # Map counter to constant
            counter_value = loop.start.value + i
            counter_ssa = ctx.new_ssa(f"unroll_i{i}")
            result.append(Op("const", counter_ssa, [Const(counter_value)], "load"))
            remap[loop.counter.id] = counter_ssa

            # Map body_params to current values
            for param, current in zip(loop.body_params, current_params):
                remap[param.id] = current

            # Clone body with remapping
            cloned_body, new_yields = self._clone_body_with_remap(
                loop.body, loop.yields, remap, ctx, nested_factor, max_trip
            )
            result.extend(cloned_body)

            # Update current_params for next iteration
            current_params = new_yields

        # Bind results to final yields
        for result_ssa, final_val in zip(loop.results, current_params):
            ctx.bind_result(result_ssa.id, final_val)

        return result

    def _partially_unroll(
        self, loop: ForLoop, factor: int, ctx: SSARenumberContext,
        nested_factor: Optional[int], max_trip: int
    ) -> list[Statement]:
        """
        Create a new loop that iterates trip_count/factor times,
        with factor copies of the body per iteration.
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

            remap[loop.counter.id] = counter_ssa

            # Map body_params to current values
            for param, current in zip(loop.body_params, current_params):
                remap[param.id] = current

            # Clone body with remapping
            cloned_body, new_yields = self._clone_body_with_remap(
                loop.body, loop.yields, remap, ctx, nested_factor, max_trip
            )
            unrolled_body.extend(cloned_body)
            current_params = new_yields

        # Bind old results to new results so subsequent statements use the new ones
        for old_result, new_result in zip(loop.results, new_results):
            ctx.bind_result(old_result.id, new_result)

        # Create new ForLoop with unrolled body
        return [ForLoop(
            counter=new_counter,
            start=Const(0),
            end=Const(new_trip_count),
            iter_args=loop.iter_args,
            body_params=new_body_params,
            body=unrolled_body,
            yields=current_params,
            results=new_results
        )]

    def _clone_body_with_remap(
        self,
        body: list[Statement],
        yields: list[SSAValue],
        remap: dict[int, Operand],
        ctx: SSARenumberContext,
        nested_factor: Optional[int],
        max_trip: int
    ) -> tuple[list[Statement], list[SSAValue]]:
        """Clone a body of statements, remapping SSA values."""
        cloned = []
        for stmt in body:
            cloned.extend(self._clone_stmt(stmt, remap, ctx, nested_factor, max_trip))

        # Remap yields
        new_yields = [self._remap_operand(y, remap) for y in yields]
        # Ensure yields are SSAValue (not Const)
        final_yields = []
        for y in new_yields:
            if isinstance(y, SSAValue):
                final_yields.append(y)
            else:
                # Const yield - need to create SSA for it
                ssa = ctx.new_ssa("yield_const")
                cloned.append(Op("const", ssa, [y], "load"))
                final_yields.append(ssa)

        return cloned, final_yields

    def _clone_stmt(
        self,
        stmt: Statement,
        remap: dict[int, Operand],
        ctx: SSARenumberContext,
        nested_factor: Optional[int],
        max_trip: int
    ) -> list[Statement]:
        """Clone a single statement with SSA remapping."""
        if isinstance(stmt, Op):
            new_operands = [self._remap_operand(op, remap) for op in stmt.operands]
            if stmt.result:
                new_result = ctx.new_ssa(stmt.result.name)
                remap[stmt.result.id] = new_result
            else:
                new_result = None
            return [Op(stmt.opcode, new_result, new_operands, stmt.engine)]

        elif isinstance(stmt, ForLoop):
            # Create new SSA values for the nested loop
            new_counter = ctx.new_ssa(stmt.counter.name)
            new_body_params = [ctx.new_ssa(p.name) for p in stmt.body_params]
            new_results = [ctx.new_ssa(r.name) for r in stmt.results]

            # Remap iter_args
            new_iter_args = []
            for arg in stmt.iter_args:
                remapped = self._remap_operand(arg, remap)
                if isinstance(remapped, SSAValue):
                    new_iter_args.append(remapped)
                else:
                    # Create SSA for constant
                    ssa = ctx.new_ssa("iter_arg_const")
                    new_iter_args.append(ssa)
                    # Note: we'd need to emit this const before the loop

            # Build inner remap for the loop body
            inner_remap = dict(remap)
            inner_remap[stmt.counter.id] = new_counter
            for old_param, new_param in zip(stmt.body_params, new_body_params):
                inner_remap[old_param.id] = new_param

            # Clone body
            cloned_body = []
            for s in stmt.body:
                cloned_body.extend(self._clone_stmt(s, inner_remap, ctx, nested_factor, max_trip))

            # Remap yields
            new_yields = []
            for y in stmt.yields:
                remapped = self._remap_operand(y, inner_remap)
                if isinstance(remapped, SSAValue):
                    new_yields.append(remapped)
                else:
                    ssa = ctx.new_ssa("yield_const")
                    cloned_body.append(Op("const", ssa, [remapped], "load"))
                    new_yields.append(ssa)

            # Update outer remap with results
            for old_res, new_res in zip(stmt.results, new_results):
                remap[old_res.id] = new_res

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
                results=new_results
            )

            # Recursively unroll nested loop if it has static bounds
            return self._unroll_loop(new_loop, nested_factor, max_trip, ctx)

        elif isinstance(stmt, If):
            # Clone If statement
            new_cond = self._remap_operand(stmt.cond, remap)
            if not isinstance(new_cond, SSAValue):
                # Create SSA for constant condition (unusual but handle it)
                cond_ssa = ctx.new_ssa("cond_const")
                # We'd need to emit this before, handle specially
                new_cond = cond_ssa

            # Clone branches
            then_remap = dict(remap)
            cloned_then = []
            for s in stmt.then_body:
                cloned_then.extend(self._clone_stmt(s, then_remap, ctx, nested_factor, max_trip))

            else_remap = dict(remap)
            cloned_else = []
            for s in stmt.else_body:
                cloned_else.extend(self._clone_stmt(s, else_remap, ctx, nested_factor, max_trip))

            # Remap yields
            new_then_yields = []
            for y in stmt.then_yields:
                remapped = self._remap_operand(y, then_remap)
                if isinstance(remapped, SSAValue):
                    new_then_yields.append(remapped)
                else:
                    ssa = ctx.new_ssa("then_yield_const")
                    cloned_then.append(Op("const", ssa, [remapped], "load"))
                    new_then_yields.append(ssa)

            new_else_yields = []
            for y in stmt.else_yields:
                remapped = self._remap_operand(y, else_remap)
                if isinstance(remapped, SSAValue):
                    new_else_yields.append(remapped)
                else:
                    ssa = ctx.new_ssa("else_yield_const")
                    cloned_else.append(Op("const", ssa, [remapped], "load"))
                    new_else_yields.append(ssa)

            # New results
            new_results = [ctx.new_ssa(r.name) for r in stmt.results]
            for old_res, new_res in zip(stmt.results, new_results):
                remap[old_res.id] = new_res

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

    def _remap_operand(self, op: Operand, remap: dict[int, Operand]) -> Operand:
        """Remap an operand using the SSA remapping."""
        if isinstance(op, SSAValue):
            if op.id in remap:
                return remap[op.id]
            return op
        return op  # Const stays as-is


# =============================================================================
# LIR (Low-Level IR) - Basic Blocks with Jumps
# =============================================================================

class LIROpcode(Enum):
    """LIR opcodes."""
    # ALU
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "//"
    MOD = "%"
    XOR = "^"
    AND = "&"
    OR = "|"
    SHL = "<<"
    SHR = ">>"
    LT = "<"
    EQ = "=="

    # Load
    LOAD = "load"
    CONST = "const"

    # Store
    STORE = "store"

    # Flow
    SELECT = "select"
    JUMP = "jump"
    COND_JUMP = "cond_jump"
    HALT = "halt"
    PAUSE = "pause"

    # Pseudo-ops (eliminated before codegen)
    COPY = "copy"


@dataclass
class LIRInst:
    """A single LIR instruction."""
    opcode: LIROpcode
    dest: Optional[int]          # Scratch address for result
    operands: list               # Scratch addresses, immediates, or labels
    engine: str

    def __repr__(self):
        ops_str = ", ".join(str(o) for o in self.operands)
        if self.dest is not None:
            return f"s{self.dest} = {self.opcode.value}({ops_str}) [{self.engine}]"
        return f"{self.opcode.value}({ops_str}) [{self.engine}]"


@dataclass
class Phi:
    """Phi node for SSA merge points."""
    dest: int                        # Scratch address for result
    incoming: dict[str, int]         # block_name -> scratch address

    def __repr__(self):
        inc = ", ".join(f"{k}:s{v}" for k, v in self.incoming.items())
        return f"s{self.dest} = phi({inc})"


@dataclass
class BasicBlock:
    """A basic block in the CFG."""
    name: str
    phis: list[Phi] = field(default_factory=list)
    instructions: list[LIRInst] = field(default_factory=list)
    terminator: Optional[LIRInst] = None

    def __repr__(self):
        return f"BasicBlock({self.name}, {len(self.instructions)} insts)"


@dataclass
class LIRFunction:
    """A complete LIR function (CFG)."""
    entry: str
    blocks: dict[str, BasicBlock] = field(default_factory=dict)
    max_scratch_used: int = -1  # Highest scratch index allocated during lowering


# =============================================================================
# HIR -> LIR Lowering
# =============================================================================

class LoweringContext:
    """Context for lowering HIR to LIR."""

    def __init__(self):
        self.lir = LIRFunction(entry="entry")
        self.current_block: Optional[BasicBlock] = None
        self._block_counter = 0
        self._scratch_ptr = 0
        self._ssa_to_scratch: dict[int, int] = {}  # SSAValue.id -> scratch addr
        self._const_scratch: dict[int, int] = {}   # const value -> scratch addr
        self._pending_consts: list[tuple[int, int]] = []  # (scratch_addr, value) - deferred const loads

    def new_block(self, prefix: str = "bb") -> BasicBlock:
        """Create a new basic block."""
        name = f"{prefix}_{self._block_counter}"
        self._block_counter += 1
        block = BasicBlock(name=name)
        self.lir.blocks[name] = block
        return block

    def set_block(self, block: BasicBlock):
        """Set the current block for emission."""
        self.current_block = block

    def alloc_scratch(self, ssa: Optional[SSAValue] = None) -> int:
        """Allocate a scratch address, optionally binding to an SSA value."""
        addr = self._scratch_ptr
        self._scratch_ptr += 1
        assert self._scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        if ssa is not None:
            self._ssa_to_scratch[ssa.id] = addr
        return addr

    def get_scratch(self, ssa: SSAValue) -> int:
        """Get the scratch address for an SSA value."""
        if ssa.id not in self._ssa_to_scratch:
            # Allocate on demand
            self.alloc_scratch(ssa)
        return self._ssa_to_scratch[ssa.id]

    def get_operand(self, op: Operand) -> int:
        """Get scratch address for an operand (SSA value or const)."""
        if isinstance(op, SSAValue):
            return self.get_scratch(op)
        elif isinstance(op, Const):
            return self.get_const(op.value)
        else:
            raise ValueError(f"Unknown operand type: {op}")

    def get_const(self, value: int) -> int:
        """Get scratch address for a constant (with caching).

        Constants are deferred and emitted to the entry block later to ensure
        they dominate all uses (fixing the control-flow miscompilation bug).
        """
        if value not in self._const_scratch:
            addr = self.alloc_scratch()
            self._const_scratch[value] = addr
            # Defer const load - will be emitted to entry block later
            self._pending_consts.append((addr, value))
        return self._const_scratch[value]

    def emit_pending_consts(self):
        """Emit all pending constant loads to the entry block.

        This must be called after lowering is complete to ensure constants
        are materialized in the entry block where they dominate all uses.
        """
        if not self._pending_consts:
            return

        entry_block = self.lir.blocks[self.lir.entry]
        # Insert const loads at the beginning of the entry block
        const_insts = [
            LIRInst(LIROpcode.CONST, addr, [value], "load")
            for addr, value in self._pending_consts
        ]
        entry_block.instructions = const_insts + entry_block.instructions
        self._pending_consts = []

    def emit(self, inst: LIRInst):
        """Emit an instruction to the current block."""
        assert self.current_block is not None, "No current block"
        self.current_block.instructions.append(inst)

    def set_terminator(self, inst: LIRInst):
        """Set the terminator for the current block."""
        assert self.current_block is not None, "No current block"
        self.current_block.terminator = inst


def lower_to_lir(hir: HIRFunction) -> LIRFunction:
    """Lower HIR to LIR."""
    ctx = LoweringContext()

    # Create entry block
    entry = ctx.new_block("entry")
    ctx.lir.entry = entry.name  # Set the actual entry block name
    ctx.set_block(entry)

    # Lower all statements
    for stmt in hir.body:
        _lower_statement(stmt, ctx)

    # If the last block doesn't have a terminator, add halt
    if ctx.current_block and ctx.current_block.terminator is None:
        ctx.set_terminator(LIRInst(LIROpcode.HALT, None, [], "flow"))

    # Emit all pending constants to entry block (ensures they dominate all uses)
    ctx.emit_pending_consts()

    # Record max scratch used for phi elimination temp allocation
    ctx.lir.max_scratch_used = ctx._scratch_ptr - 1 if ctx._scratch_ptr > 0 else -1

    return ctx.lir


def _lower_statement(stmt: Statement, ctx: LoweringContext):
    """Lower a single HIR statement to LIR."""
    if isinstance(stmt, Op):
        _lower_op(stmt, ctx)
    elif isinstance(stmt, ForLoop):
        _lower_for_loop(stmt, ctx)
    elif isinstance(stmt, If):
        _lower_if(stmt, ctx)
    elif isinstance(stmt, Halt):
        ctx.set_terminator(LIRInst(LIROpcode.HALT, None, [], "flow"))
    elif isinstance(stmt, Pause):
        # Pause is a regular instruction, not a terminator
        # Execution continues after pause when run() is called again
        ctx.emit(LIRInst(LIROpcode.PAUSE, None, [], "flow"))
    else:
        raise ValueError(f"Unknown statement type: {stmt}")


def _lower_op(op: Op, ctx: LoweringContext):
    """Lower an Op to LIR instructions."""
    opcode_map = {
        "+": LIROpcode.ADD, "-": LIROpcode.SUB, "*": LIROpcode.MUL,
        "//": LIROpcode.DIV, "%": LIROpcode.MOD, "^": LIROpcode.XOR,
        "&": LIROpcode.AND, "|": LIROpcode.OR, "<<": LIROpcode.SHL,
        ">>": LIROpcode.SHR, "<": LIROpcode.LT, "==": LIROpcode.EQ,
        "load": LIROpcode.LOAD, "const": LIROpcode.CONST,
        "store": LIROpcode.STORE, "select": LIROpcode.SELECT,
    }

    lir_opcode = opcode_map.get(op.opcode)
    if lir_opcode is None:
        raise ValueError(f"Unknown opcode: {op.opcode}")

    # Get destination scratch address
    dest = None
    if op.result is not None:
        dest = ctx.get_scratch(op.result)

    # Get operand scratch addresses
    if op.opcode == "const":
        # Const has immediate value as operand
        operands = [op.operands[0].value]
    elif op.opcode == "store":
        # Store: (addr, value)
        operands = [ctx.get_operand(op.operands[0]), ctx.get_operand(op.operands[1])]
    else:
        operands = [ctx.get_operand(o) for o in op.operands]

    ctx.emit(LIRInst(lir_opcode, dest, operands, op.engine))


def _lower_for_loop(loop: ForLoop, ctx: LoweringContext):
    """Lower a ForLoop to LIR basic blocks with phis."""
    # Create blocks
    init_block = ctx.new_block("for_init")
    header_block = ctx.new_block("for_header")
    body_block = ctx.new_block("for_body")
    exit_block = ctx.new_block("for_exit")

    # Jump from current block to init
    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [init_block.name], "flow"))

    # === Init block ===
    ctx.set_block(init_block)

    # Load start value
    start_scratch = ctx.get_operand(loop.start)

    # Load end value
    end_scratch = ctx.get_operand(loop.end)

    # Allocate counter_init (copy of start for phi incoming from init)
    counter_init = ctx.alloc_scratch()
    zero_scratch = ctx.get_const(0)
    ctx.emit(LIRInst(LIROpcode.ADD, counter_init, [start_scratch, zero_scratch], "alu"))

    # Allocate iter_arg scratches (copy initial values)
    iter_arg_scratches = []
    for i, arg in enumerate(loop.iter_args):
        src = ctx.get_scratch(arg)
        dst = ctx.alloc_scratch()
        ctx.emit(LIRInst(LIROpcode.ADD, dst, [src, zero_scratch], "alu"))
        iter_arg_scratches.append(dst)

    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [header_block.name], "flow"))

    # === Header block ===
    ctx.set_block(header_block)

    # Allocate phi destinations
    counter_phi = ctx.alloc_scratch(loop.counter)
    param_scratches = [ctx.alloc_scratch(p) for p in loop.body_params]

    # We'll fill in phi incoming from body after lowering body
    # For now, set up phis with init incoming
    header_block.phis.append(Phi(counter_phi, {init_block.name: counter_init}))
    for i, param in enumerate(loop.body_params):
        header_block.phis.append(Phi(param_scratches[i], {init_block.name: iter_arg_scratches[i]}))

    # Condition: counter < end
    cond_scratch = ctx.alloc_scratch()
    ctx.emit(LIRInst(LIROpcode.LT, cond_scratch, [counter_phi, end_scratch], "alu"))
    ctx.set_terminator(LIRInst(LIROpcode.COND_JUMP, None, [cond_scratch, body_block.name, exit_block.name], "flow"))

    # === Body block ===
    ctx.set_block(body_block)

    # Lower body statements
    for stmt in loop.body:
        _lower_statement(stmt, ctx)

    # Get yield scratches
    yield_scratches = [ctx.get_scratch(y) for y in loop.yields]

    # Increment counter
    one_scratch = ctx.get_const(1)
    counter_next = ctx.alloc_scratch()
    ctx.emit(LIRInst(LIROpcode.ADD, counter_next, [counter_phi, one_scratch], "alu"))

    # Update phis with body incoming
    # Find the block that jumps back to header (could be current or a nested block)
    back_edge_block = ctx.current_block
    header_block.phis[0].incoming[back_edge_block.name] = counter_next
    for i in range(len(loop.body_params)):
        header_block.phis[i + 1].incoming[back_edge_block.name] = yield_scratches[i] if i < len(yield_scratches) else param_scratches[i]

    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [header_block.name], "flow"))

    # === Exit block ===
    ctx.set_block(exit_block)

    # Map results to param scratches (final values)
    for i, result in enumerate(loop.results):
        ctx._ssa_to_scratch[result.id] = param_scratches[i]


def _lower_if(if_stmt: If, ctx: LoweringContext):
    """Lower an If statement to LIR basic blocks with phis."""
    # Create blocks
    then_block = ctx.new_block("if_then")
    else_block = ctx.new_block("if_else")
    merge_block = ctx.new_block("if_merge")

    # Branch
    cond_scratch = ctx.get_scratch(if_stmt.cond)
    ctx.set_terminator(LIRInst(LIROpcode.COND_JUMP, None, [cond_scratch, then_block.name, else_block.name], "flow"))

    # === Then block ===
    ctx.set_block(then_block)
    for stmt in if_stmt.then_body:
        _lower_statement(stmt, ctx)
    then_yield_scratches = [ctx.get_scratch(y) for y in if_stmt.then_yields]
    then_exit_block = ctx.current_block
    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [merge_block.name], "flow"))

    # === Else block ===
    ctx.set_block(else_block)
    for stmt in if_stmt.else_body:
        _lower_statement(stmt, ctx)
    else_yield_scratches = [ctx.get_scratch(y) for y in if_stmt.else_yields]
    else_exit_block = ctx.current_block
    ctx.set_terminator(LIRInst(LIROpcode.JUMP, None, [merge_block.name], "flow"))

    # === Merge block ===
    ctx.set_block(merge_block)

    # Set up phis
    for i, result in enumerate(if_stmt.results):
        result_scratch = ctx.alloc_scratch(result)
        merge_block.phis.append(Phi(
            dest=result_scratch,
            incoming={
                then_exit_block.name: then_yield_scratches[i],
                else_exit_block.name: else_yield_scratches[i],
            }
        ))


# =============================================================================
# Phi Elimination
# =============================================================================

def _compute_parallel_copy_order(copies: list[tuple[int, int]], temp_scratch: int) -> list[tuple[int, int]]:
    """
    Compute a valid order for parallel copies that preserves semantics.

    Args:
        copies: List of (dest, src) pairs representing parallel copies
        temp_scratch: Scratch location to use for breaking cycles

    Returns:
        List of (dest, src) pairs in a valid sequential order
    """
    if not copies:
        return []

    # Build maps for analysis
    dest_to_src = {dest: src for dest, src in copies}
    src_to_dests = {}
    for dest, src in copies:
        if src not in src_to_dests:
            src_to_dests[src] = []
        src_to_dests[src].append(dest)

    result = []
    remaining = set(dest_to_src.keys())

    # Keep emitting safe copies until we can't anymore
    while remaining:
        # Find a copy whose dest is not a source for any remaining copy
        safe_dest = None
        for dest in remaining:
            # Check if this dest is used as a source by another remaining copy
            is_needed_as_source = False
            for other_dest in remaining:
                if other_dest != dest and dest_to_src[other_dest] == dest:
                    is_needed_as_source = True
                    break
            if not is_needed_as_source:
                safe_dest = dest
                break

        if safe_dest is not None:
            # Emit this copy
            result.append((safe_dest, dest_to_src[safe_dest]))
            remaining.remove(safe_dest)
        else:
            # All remaining copies form cycles - break one using temp
            # Pick any dest from remaining
            cycle_dest = next(iter(remaining))
            cycle_src = dest_to_src[cycle_dest]

            # Save the dest value (about to be overwritten) to temp first
            # This preserves the value for any other copy that reads from cycle_dest
            result.append((temp_scratch, cycle_dest))

            # Now we can safely overwrite cycle_dest
            result.append((cycle_dest, cycle_src))
            remaining.remove(cycle_dest)

            # Update any copy that used cycle_dest as source to use temp instead
            for dest in list(remaining):
                if dest_to_src[dest] == cycle_dest:
                    dest_to_src[dest] = temp_scratch

    return result


def eliminate_phis(lir: LIRFunction, temp_scratch: Optional[int] = None):
    """
    Replace phi nodes with copies at the end of predecessor blocks.

    Uses a parallel copy algorithm to handle cycles (e.g., swaps) correctly.
    This must be done before linearization since the machine doesn't have phi.

    Args:
        lir: The LIR function to transform
        temp_scratch: Scratch location for breaking cycles. If None, uses SCRATCH_SIZE-1.
    """
    if temp_scratch is None:
        temp_scratch = SCRATCH_SIZE - 1  # Reserve last scratch for temp

    # Group phis by predecessor block
    for block in lir.blocks.values():
        if not block.phis:
            continue

        # Collect copies per predecessor
        pred_copies: dict[str, list[tuple[int, int]]] = {}  # pred_name -> [(dest, src)]

        for phi in block.phis:
            for pred_name, src_scratch in phi.incoming.items():
                if pred_name not in pred_copies:
                    pred_copies[pred_name] = []
                pred_copies[pred_name].append((phi.dest, src_scratch))

        # For each predecessor, compute safe copy order and emit
        for pred_name, copies in pred_copies.items():
            pred_block = lir.blocks[pred_name]
            ordered_copies = _compute_parallel_copy_order(copies, temp_scratch)

            for dest, src in ordered_copies:
                if dest != src:  # Skip no-op copies
                    copy_inst = LIRInst(LIROpcode.COPY, dest, [src], "alu")
                    pred_block.instructions.append(copy_inst)

        block.phis = []


# =============================================================================
# LIR -> VLIW Compilation
# =============================================================================

def linearize(lir: LIRFunction) -> tuple[list[LIRInst], dict[str, int]]:
    """
    Linearize LIR into a sequence of instructions.

    Returns (instructions, label_map) where label_map maps block names to instruction indices.
    """
    instructions = []
    label_map = {}

    # Simple linearization: visit blocks in order starting from entry
    visited = set()
    worklist = [lir.entry]
    block_order = []

    while worklist:
        name = worklist.pop(0)
        if name in visited:
            continue
        visited.add(name)
        block = lir.blocks[name]
        block_order.append(block)

        # Add successors to worklist
        if block.terminator:
            if block.terminator.opcode == LIROpcode.JUMP:
                worklist.append(block.terminator.operands[0])
            elif block.terminator.opcode == LIROpcode.COND_JUMP:
                # operands: [cond, true_target, false_target]
                worklist.append(block.terminator.operands[1])
                worklist.append(block.terminator.operands[2])

    # Emit instructions in block order
    for block in block_order:
        label_map[block.name] = len(instructions)
        instructions.extend(block.instructions)
        if block.terminator:
            instructions.append(block.terminator)

    return instructions, label_map


def resolve_labels(instructions: list[LIRInst], label_map: dict[str, int]) -> list[LIRInst]:
    """Replace label references with instruction addresses."""
    resolved = []
    for inst in instructions:
        new_operands = []
        for op in inst.operands:
            if isinstance(op, str) and op in label_map:
                new_operands.append(label_map[op])
            else:
                new_operands.append(op)
        resolved.append(LIRInst(inst.opcode, inst.dest, new_operands, inst.engine))
    return resolved


def codegen(instructions: list[LIRInst], zero_scratch: int) -> list[dict]:
    """
    Generate VLIW bundles from LIR instructions.

    Simple strategy: one instruction per bundle.
    """
    bundles = []

    for inst in instructions:
        slot = _inst_to_slot(inst, zero_scratch)
        if slot is not None:
            bundle = {inst.engine: [slot]}
            bundles.append(bundle)

    return bundles


def _inst_to_slot(inst: LIRInst, zero_scratch: int) -> Optional[tuple]:
    """Convert a LIR instruction to a machine slot tuple."""
    match inst.opcode:
        # ALU operations
        case LIROpcode.ADD | LIROpcode.SUB | LIROpcode.MUL | LIROpcode.DIV | \
             LIROpcode.MOD | LIROpcode.XOR | LIROpcode.AND | LIROpcode.OR | \
             LIROpcode.SHL | LIROpcode.SHR | LIROpcode.LT | LIROpcode.EQ:
            return (inst.opcode.value, inst.dest, inst.operands[0], inst.operands[1])

        # Load operations
        case LIROpcode.CONST:
            return ("const", inst.dest, inst.operands[0])
        case LIROpcode.LOAD:
            return ("load", inst.dest, inst.operands[0])

        # Store operations
        case LIROpcode.STORE:
            return ("store", inst.operands[0], inst.operands[1])

        # Flow operations
        case LIROpcode.SELECT:
            return ("select", inst.dest, inst.operands[0], inst.operands[1], inst.operands[2])
        case LIROpcode.JUMP:
            return ("jump", inst.operands[0])
        case LIROpcode.COND_JUMP:
            # cond_jump cond, true_target, false_target
            # Machine has: cond_jump cond, target (jumps if true)
            # We need to handle false_target as fallthrough or separate jump
            # For now, emit cond_jump for true, then jump for false
            # This is handled specially below
            return None  # Handled in codegen
        case LIROpcode.HALT:
            return ("halt",)
        case LIROpcode.PAUSE:
            return ("pause",)

        # Pseudo-ops
        case LIROpcode.COPY:
            # Implement copy as add with zero
            return ("+", inst.dest, inst.operands[0], zero_scratch)

        case _:
            raise NotImplementedError(f"Codegen for {inst.opcode}")


def compile_to_vliw(lir: LIRFunction) -> list[dict]:
    """Full compilation from LIR to VLIW bundles.

    Note: Phis must be eliminated before calling this function.
    """
    # Find zero constant for copy operations
    zero_scratch = None
    for block in lir.blocks.values():
        for inst in block.instructions:
            if inst.opcode == LIROpcode.CONST and inst.operands[0] == 0:
                zero_scratch = inst.dest
                break
        if zero_scratch is not None:
            break

    # If no zero constant found, materialize one in the entry block
    if zero_scratch is None:
        # Find the maximum used scratch address to allocate a new one
        max_scratch = -1
        for block in lir.blocks.values():
            for inst in block.instructions:
                if inst.dest is not None:
                    max_scratch = max(max_scratch, inst.dest)
                # Skip CONST operands - they are immediate values, not scratch indices
                if inst.opcode != LIROpcode.CONST:
                    for op in inst.operands:
                        if isinstance(op, int) and op >= 0:
                            max_scratch = max(max_scratch, op)

        zero_scratch = max_scratch + 1
        assert zero_scratch < SCRATCH_SIZE, "Out of scratch space for zero constant"

        # Add const 0 at the beginning of entry block
        entry_block = lir.blocks[lir.entry]
        zero_inst = LIRInst(LIROpcode.CONST, zero_scratch, [0], "load")
        entry_block.instructions.insert(0, zero_inst)

    # Linearize
    instructions, label_map = linearize(lir)

    # Handle COND_JUMP specially (expand to cond_jump + jump)
    expanded = []
    for i, inst in enumerate(instructions):
        if inst.opcode == LIROpcode.COND_JUMP:
            cond, true_target, false_target = inst.operands
            # Emit conditional jump to true target
            expanded.append(LIRInst(LIROpcode.COND_JUMP, None, [cond, true_target], "flow"))
            # Emit unconditional jump to false target
            expanded.append(LIRInst(LIROpcode.JUMP, None, [false_target], "flow"))
        else:
            expanded.append(inst)

    # Recompute label map after expansion
    # We need to adjust labels since we added instructions
    new_label_map = {}
    old_to_new_idx = {}
    new_idx = 0
    old_idx = 0
    for inst in instructions:
        old_to_new_idx[old_idx] = new_idx
        if inst.opcode == LIROpcode.COND_JUMP:
            new_idx += 2  # cond_jump + jump
        else:
            new_idx += 1
        old_idx += 1

    for name, old_idx in label_map.items():
        new_label_map[name] = old_to_new_idx.get(old_idx, old_idx)

    # Resolve labels
    resolved = resolve_labels(expanded, new_label_map)

    # Generate bundles
    bundles = []
    for inst in resolved:
        if inst.opcode == LIROpcode.COND_JUMP:
            # cond_jump cond, target
            slot = ("cond_jump", inst.operands[0], inst.operands[1])
            bundles.append({"flow": [slot]})
        else:
            slot = _inst_to_slot(inst, zero_scratch)
            if slot is not None:
                bundles.append({inst.engine: [slot]})

    return bundles


# =============================================================================
# IR Printing
# =============================================================================

def print_hir(hir: HIRFunction, indent: int = 0):
    """Pretty-print HIR function."""
    prefix = "  " * indent
    print(f"{prefix}=== HIR: {hir.name} ({hir.num_ssa_values} SSA values) ===")
    for stmt in hir.body:
        _print_hir_stmt(stmt, indent)
    print()


def _print_hir_stmt(stmt: Statement, indent: int = 0):
    """Pretty-print a single HIR statement."""
    prefix = "  " * indent
    if isinstance(stmt, Op):
        print(f"{prefix}{stmt}")
    elif isinstance(stmt, ForLoop):
        print(f"{prefix}for {stmt.counter} in range({stmt.start}, {stmt.end}):")
        if stmt.iter_args:
            print(f"{prefix}  iter_args: {stmt.iter_args}")
            print(f"{prefix}  body_params: {stmt.body_params}")
        for s in stmt.body:
            _print_hir_stmt(s, indent + 1)
        if stmt.yields:
            print(f"{prefix}  yields: {stmt.yields}")
        if stmt.results:
            print(f"{prefix}  results: {stmt.results}")
    elif isinstance(stmt, If):
        print(f"{prefix}if {stmt.cond}:")
        for s in stmt.then_body:
            _print_hir_stmt(s, indent + 1)
        if stmt.then_yields:
            print(f"{prefix}  then_yields: {stmt.then_yields}")
        if stmt.else_body:
            print(f"{prefix}else:")
            for s in stmt.else_body:
                _print_hir_stmt(s, indent + 1)
            if stmt.else_yields:
                print(f"{prefix}  else_yields: {stmt.else_yields}")
        if stmt.results:
            print(f"{prefix}results: {stmt.results}")
    elif isinstance(stmt, Halt):
        print(f"{prefix}halt")
    elif isinstance(stmt, Pause):
        print(f"{prefix}pause")
    else:
        print(f"{prefix}{stmt}")


def print_lir(lir: LIRFunction):
    """Pretty-print LIR function."""
    print(f"=== LIR (entry: {lir.entry}) ===")
    for name, block in lir.blocks.items():
        print(f"\n{name}:")
        if block.phis:
            for phi in block.phis:
                print(f"  {phi}")
        for inst in block.instructions:
            print(f"  {inst}")
        if block.terminator:
            print(f"  {block.terminator}")
    print()


def print_vliw(bundles: list[dict]):
    """Pretty-print VLIW bundles."""
    print(f"=== VLIW ({len(bundles)} bundles) ===")
    for i, bundle in enumerate(bundles):
        print(f"[{i:4d}] {bundle}")
    print()


# =============================================================================
# Compilation with Debug Printing
# =============================================================================

def compile_hir_to_vliw(
    hir: HIRFunction,
    print_after_all: bool = False,
    config_path: Optional[str] = None
) -> list[dict]:
    """
    Full compilation from HIR to VLIW with optional debug printing.

    Args:
        hir: The HIR function to compile
        print_after_all: If True, print IR after each compilation phase
        config_path: Optional path to JSON config file for pass options
    """
    if print_after_all:
        print("\n" + "=" * 60)
        print("COMPILATION START")
        print("=" * 60)
        print_hir(hir)

    # Phase 0: Run HIR optimization passes
    pm = PassManager(print_after_all=print_after_all)
    pm.add_pass(LoopUnrollPass())

    if config_path:
        pm.load_config(config_path)

    hir = pm.run(hir)

    if print_after_all and pm.passes:
        print("-" * 60)
        print("After HIR passes:")
        print("-" * 60)
        print_hir(hir)

    # Phase 1: Lower HIR to LIR
    lir = lower_to_lir(hir)
    if print_after_all:
        print("-" * 60)
        print("After HIR -> LIR lowering:")
        print("-" * 60)
        print_lir(lir)

    # Phase 2: Eliminate phis
    # Use scratch slot after all allocated ones for phi temp (avoids collision)
    temp_scratch = lir.max_scratch_used + 1 if lir.max_scratch_used >= 0 else 0
    assert temp_scratch < SCRATCH_SIZE, "No room for phi temp scratch"
    eliminate_phis(lir, temp_scratch=temp_scratch)
    if print_after_all:
        print("-" * 60)
        print("After phi elimination:")
        print("-" * 60)
        print_lir(lir)

    # Phase 3: Compile to VLIW
    bundles = compile_to_vliw(lir)
    if print_after_all:
        print("-" * 60)
        print("After LIR -> VLIW codegen:")
        print("-" * 60)
        print_vliw(bundles)
        print("=" * 60)
        print("COMPILATION END")
        print("=" * 60 + "\n")

    return bundles
