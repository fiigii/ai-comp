"""
SSA Renumbering Context

Utility for tracking SSA value renumbering during transformations like
loop unrolling.
"""

from dataclasses import dataclass, field
from .hir import SSAValue, VectorSSAValue, Variable, Value


@dataclass
class SSARenumberContext:
    """Track SSA value renumbering during transformations."""
    next_id: int
    next_vec_id: int
    # Maps SSAValue/VectorSSAValue objects directly to their replacement values
    result_bindings: dict[Variable, Value] = field(default_factory=dict)

    def new_ssa(self, name: str = None) -> SSAValue:
        """Create a new unique SSA value."""
        ssa = SSAValue(self.next_id, name)
        self.next_id += 1
        return ssa

    def new_vec_ssa(self, name: str = None) -> VectorSSAValue:
        """Create a new unique vector SSA value."""
        vec = VectorSSAValue(self.next_vec_id, name)
        self.next_vec_id += 1
        return vec

    def bind_result(self, old_ssa: Variable, new_value: Value):
        """Map old result SSA to new value (for eliminating loop after unroll)."""
        self.result_bindings[old_ssa] = new_value
