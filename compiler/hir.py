"""
HIR (High-Level IR) - SSA Form

The high-level intermediate representation uses SSA form with explicit
loops and branches. This is the input to the compilation pipeline.
"""

from dataclasses import dataclass
from typing import Optional, Union


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
class VectorSSAValue:
    """A vector SSA value (VLEN=8 elements)."""
    id: int
    name: Optional[str] = None

    def __repr__(self):
        if self.name:
            return f"vec{self.id}:{self.name}"
        return f"vec{self.id}"


@dataclass(frozen=True)
class Const:
    """A compile-time constant."""
    value: int

    def __repr__(self):
        return f"#{self.value}"


# Type alias for operands (VectorSSAValue is also a valid operand for vector ops)
Operand = Union[SSAValue, VectorSSAValue, Const]


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
    pragma_unroll: Unroll pragma (0=full, 1=disabled, N>1=partial by factor N)
    """
    counter: SSAValue
    start: Operand
    end: Operand
    iter_args: list[SSAValue]
    body_params: list[SSAValue]
    body: list  # list[Statement]
    yields: list[SSAValue]
    results: list[SSAValue]
    pragma_unroll: int = 1  # 0=full, 1=disabled (default), N>1=partial

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
    num_vec_ssa_values: int = 0
