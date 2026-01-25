"""
HIR Builder - Functional SSA API

Provides a builder API for constructing HIR in SSA form.
"""

from typing import Optional, Callable

from .hir import (
    SSAValue, VectorSSAValue, Const, Operand, Op, Halt, Pause, ForLoop, If, Statement, HIRFunction
)


class HIRBuilder:
    """Builder for constructing HIR in SSA form."""

    def __init__(self):
        self._ssa_counter = 0
        self._vec_ssa_counter = 0
        self._statements: list[Statement] = []

    def _new_ssa(self, name: Optional[str] = None) -> SSAValue:
        """Create a new SSA value."""
        v = SSAValue(self._ssa_counter, name)
        self._ssa_counter += 1
        return v

    def _new_vec_ssa(self, name: Optional[str] = None) -> VectorSSAValue:
        """Create a new vector SSA value."""
        v = VectorSSAValue(self._vec_ssa_counter, name)
        self._vec_ssa_counter += 1
        return v

    def _emit(self, stmt: Statement):
        """Add a statement to current context."""
        self._statements.append(stmt)

    # === Constants ===

    def const(self, value: int) -> Const:
        """Create a compile-time constant."""
        return Const(value)

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

    def load(self, addr: Operand, name: Optional[str] = None) -> SSAValue:
        """Load from memory at address."""
        result = self._new_ssa(name)
        self._emit(Op("load", result, [addr], "load"))
        return result

    def store(self, addr: Operand, value: Operand):
        """Store value to memory at address."""
        self._emit(Op("store", None, [addr, value], "store"))

    # === Flow operations ===

    def select(self, cond: Operand, a: Operand, b: Operand, name: Optional[str] = None) -> SSAValue:
        """Conditional select: cond ? a : b"""
        result = self._new_ssa(name)
        self._emit(Op("select", result, [cond, a, b], "flow"))
        return result

    # === Vector operations (VLEN=8) ===

    def valu(self, op: str, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        """Emit a binary vector ALU operation."""
        result = self._new_vec_ssa(name)
        self._emit(Op(op, result, [a, b], "valu"))
        return result

    def vadd(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v+", a, b, name)

    def vsub(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v-", a, b, name)

    def vmul(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v*", a, b, name)

    def vdiv(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v//", a, b, name)

    def vmod(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v%", a, b, name)

    def vxor(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v^", a, b, name)

    def vand(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v&", a, b, name)

    def vor(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v|", a, b, name)

    def vshl(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v<<", a, b, name)

    def vshr(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v>>", a, b, name)

    def vlt(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v<", a, b, name)

    def veq(self, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        return self.valu("v==", a, b, name)

    def vbroadcast(self, scalar: Operand, name: Optional[str] = None) -> VectorSSAValue:
        """Broadcast a scalar to all VLEN lanes."""
        result = self._new_vec_ssa(name)
        self._emit(Op("vbroadcast", result, [scalar], "valu"))
        return result


    def multiply_add(self, a: VectorSSAValue, b: VectorSSAValue, c: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        """Fused multiply-add: a * b + c."""
        result = self._new_vec_ssa(name)
        self._emit(Op("multiply_add", result, [a, b, c], "valu"))
        return result

    def vload(self, addr: Operand, name: Optional[str] = None) -> VectorSSAValue:
        """Load VLEN consecutive words from memory."""
        result = self._new_vec_ssa(name)
        self._emit(Op("vload", result, [addr], "load"))
        return result

    def vstore(self, addr: Operand, vec: VectorSSAValue):
        """Store VLEN consecutive words to memory."""
        self._emit(Op("vstore", None, [addr, vec], "store"))

    def vselect(self, cond: VectorSSAValue, a: VectorSSAValue, b: VectorSSAValue, name: Optional[str] = None) -> VectorSSAValue:
        """Per-lane vector select: cond[i] ? a[i] : b[i]."""
        result = self._new_vec_ssa(name)
        self._emit(Op("vselect", result, [cond, a, b], "flow"))
        return result

    def vextract(self, vec: VectorSSAValue, lane: int, name: Optional[str] = None) -> SSAValue:
        """Extract a scalar from a vector lane (compile-time constant lane)."""
        result = self._new_ssa(name)
        self._emit(Op("vextract", result, [vec, Const(lane)], "alu"))
        return result

    def vinsert(self, vec: VectorSSAValue, scalar: Operand, lane: int, name: Optional[str] = None) -> VectorSSAValue:
        """Insert a scalar into a vector lane (compile-time constant lane)."""
        result = self._new_vec_ssa(name)
        self._emit(Op("vinsert", result, [vec, scalar, Const(lane)], "alu"))
        return result

    # === Control flow ===

    def for_loop(
        self,
        start: Operand,
        end: Operand,
        iter_args: list[Operand],
        body_fn: Callable[[SSAValue, list[SSAValue]], list[Operand]],
        pragma_unroll: int = 0
    ) -> list[SSAValue]:
        """
        Build a for loop.

        body_fn receives (counter, body_params) and returns yield values.
        Returns the loop results (final values after loop exits).

        Args:
            pragma_unroll: Unroll pragma (0=full, 1=disabled, N>1=partial by factor N)
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
            pragma_unroll=pragma_unroll,
        )
        self._emit(loop)
        return results

    def if_stmt(
        self,
        cond: Operand,
        then_fn: Callable[[], list[Operand]],
        else_fn: Callable[[], list[Operand]],
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
            num_vec_ssa_values=self._vec_ssa_counter,
        )
