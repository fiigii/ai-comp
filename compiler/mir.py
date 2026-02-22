"""
MIR (Machine IR) - Bundled VLIW Instructions

The Machine IR represents bundled VLIW instructions. It sits between LIR
(after phi-elimination) and final VLIW output. The key feature is that
MachineBasicBlock contains a list of bundles, where each bundle contains
instructions that execute in parallel.

Pipeline Order:
    LIR -> phi-elimination -> LIRToMIRPass -> MIRRegisterAllocationPass -> MIRToVLIWPass
"""

from dataclasses import dataclass, field
from typing import Optional

from vm import VLEN

from .lir import LIROpcode, InstructionDefUseMixin


@dataclass
class MachineInst(InstructionDefUseMixin):
    """A single machine instruction.

    Similar to LIRInst but represents a machine-level operation.
    Before register allocation, dest/operands use virtual scratch addresses.
    After register allocation, they use physical scratch addresses.
    """
    opcode: LIROpcode
    dest: Optional[int | list[int]]  # Scratch address(es) for result
    operands: list                    # Scratch addresses, immediates, or labels
    engine: str                       # "alu", "valu", "load", "store", "flow"

    def is_terminator(self) -> bool:
        """Check if this is a control flow terminator."""
        return self.opcode in (
            LIROpcode.JUMP, LIROpcode.COND_JUMP, LIROpcode.HALT, LIROpcode.PAUSE
        )

    def __repr__(self):
        ops_str = ", ".join(str(o) for o in self.operands)
        if self.dest is not None:
            if isinstance(self.dest, list):
                return f"s{self.dest[0]}..s{self.dest[-1]} = {self.opcode.value}({ops_str}) [{self.engine}]"
            return f"s{self.dest} = {self.opcode.value}({ops_str}) [{self.engine}]"
        return f"{self.opcode.value}({ops_str}) [{self.engine}]"


@dataclass
class MBundle:
    """A bundle of machine instructions that execute in parallel.

    VLIW bundles must respect slot limits per engine.
    """
    instructions: list[MachineInst] = field(default_factory=list)

    # Slot limits per engine
    SLOT_LIMITS = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

    def _count_slots(self, engine: str) -> int:
        """Count how many slots are used by a given engine."""
        return sum(1 for inst in self.instructions if inst.engine == engine)

    def has_slot_available(self, engine: str) -> bool:
        """Check if there's an available slot for the given engine."""
        limit = self.SLOT_LIMITS.get(engine, 0)
        return self._count_slots(engine) < limit

    def add_instruction(self, inst: MachineInst) -> bool:
        """Try to add an instruction to this bundle.

        Returns True if successful, False if no slot available.
        """
        if not self.has_slot_available(inst.engine):
            return False
        self.instructions.append(inst)
        return True

    def get_all_defs(self) -> set[int]:
        """Get all scratch addresses defined in this bundle."""
        defs: set[int] = set()
        for inst in self.instructions:
            defs.update(inst.get_defs())
        return defs

    def get_all_uses(self) -> set[int]:
        """Get all scratch addresses used in this bundle."""
        uses: set[int] = set()
        for inst in self.instructions:
            uses.update(inst.get_uses())
        return uses

    def has_terminator(self) -> bool:
        """Check if this bundle contains a terminator instruction."""
        return any(inst.is_terminator() for inst in self.instructions)

    def get_terminator(self) -> Optional[MachineInst]:
        """Get the terminator instruction if present."""
        for inst in self.instructions:
            if inst.is_terminator():
                return inst
        return None

    def __repr__(self):
        return f"MBundle({len(self.instructions)} insts)"


@dataclass
class MachineBasicBlock:
    """A basic block in the Machine IR CFG."""
    name: str
    bundles: list[MBundle] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)

    def __repr__(self):
        return f"MBB({self.name}, {len(self.bundles)} bundles)"


@dataclass
class MachineFunction:
    """A complete machine function (CFG with bundled instructions)."""
    entry: str
    blocks: dict[str, MachineBasicBlock] = field(default_factory=dict)
    max_scratch_used: int = -1
    phi_eliminated: bool = False

    def get_block_order(self) -> list[str]:
        """Get blocks in reverse postorder for analysis."""
        visited: set[str] = set()
        postorder: list[str] = []

        def dfs(name: str):
            if name in visited:
                return
            visited.add(name)
            block = self.blocks.get(name)
            if block:
                for succ in block.successors:
                    dfs(succ)
                postorder.append(name)

        dfs(self.entry)
        return list(reversed(postorder))

    def total_bundles(self) -> int:
        """Count total bundles across all blocks."""
        return sum(len(block.bundles) for block in self.blocks.values())

    def total_instructions(self) -> int:
        """Count total instructions across all bundles."""
        total = 0
        for block in self.blocks.values():
            for bundle in block.bundles:
                total += len(bundle.instructions)
        return total

    def __repr__(self):
        return f"MFunc(entry={self.entry}, {len(self.blocks)} blocks, {self.total_bundles()} bundles)"
