"""
SSA Renumbering Context

Utility for tracking SSA value renumbering during transformations like
loop unrolling.
"""

from dataclasses import dataclass, field
from .hir import SSAValue, VectorSSAValue


@dataclass
class SSARenumberContext:
    """Track SSA value renumbering during transformations."""
    next_id: int
    result_bindings: dict[int, SSAValue] = field(default_factory=dict)
    vector_bindings: dict[int, VectorSSAValue] = field(default_factory=dict)

    def new_ssa(self, name: str = None) -> SSAValue:
        """Create a new unique SSA value."""
        ssa = SSAValue(self.next_id, name)
        self.next_id += 1
        return ssa

    def bind_result(self, old_id: int, new_ssa: SSAValue):
        """Map old result SSA to new value (for eliminating loop after unroll)."""
        self.result_bindings[old_id] = new_ssa

    def bind_vector_result(self, old_id: int, new_vec: VectorSSAValue):
        """Map old vector result to new vector value."""
        self.vector_bindings[old_id] = new_vec
