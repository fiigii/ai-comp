"""
LIR (Low-Level IR) - Basic Blocks with Jumps

The low-level intermediate representation uses basic blocks with explicit
jumps. This is close to the machine code and is the output of lowering.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
    LOAD_OFFSET = "load_offset"
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

    # Vector ALU (engine: valu)
    VADD = "v+"
    VSUB = "v-"
    VMUL = "v*"
    VDIV = "v//"
    VMOD = "v%"
    VXOR = "v^"
    VAND = "v&"
    VOR = "v|"
    VSHL = "v<<"
    VSHR = "v>>"
    VLT = "v<"
    VEQ = "v=="
    VBROADCAST = "vbroadcast"
    MULTIPLY_ADD = "multiply_add"

    # Vector load/store
    VLOAD = "vload"
    VSTORE = "vstore"

    # Vector flow
    VSELECT = "vselect"


class InstructionDefUseMixin:
    """Mixin providing get_defs()/get_uses() for LIRInst and MachineInst.

    Both classes share identical def/use logic. This mixin expects the host
    dataclass to have `opcode: LIROpcode`, `dest`, and `operands` fields.
    """

    def get_defs(self) -> set[int]:
        """Get all scratch addresses defined by this instruction."""
        if self.dest is None:
            return set()

        # LOAD_OFFSET writes to dest + lane_offset.
        if self.opcode == LIROpcode.LOAD_OFFSET:
            offset = self.operands[1] if len(self.operands) > 1 else 0
            if isinstance(offset, int) and isinstance(self.dest, int):
                return {self.dest + offset}
            if isinstance(self.dest, int):
                return {self.dest}
            return set()

        if isinstance(self.dest, list):
            return set(self.dest)
        return {self.dest}

    def get_uses(self) -> set[int]:
        """Get all scratch addresses used by this instruction."""
        uses: set[int] = set()

        # CONST operands are immediate values.
        if self.opcode == LIROpcode.CONST:
            return uses

        # LOAD_OFFSET reads lane address from addr_base + lane_offset.
        if self.opcode == LIROpcode.LOAD_OFFSET and len(self.operands) == 2:
            addr = self.operands[0]
            offset = self.operands[1]
            if isinstance(addr, int) and isinstance(offset, int):
                uses.add(addr + offset)
            elif isinstance(addr, int):
                uses.add(addr)
            return uses

        # JUMP operands are labels, not scratch values.
        if self.opcode == LIROpcode.JUMP:
            return uses

        # COND_JUMP: first operand is condition scratch, rest are labels.
        if self.opcode == LIROpcode.COND_JUMP:
            cond = self.operands[0]
            if isinstance(cond, int):
                uses.add(cond)
            return uses

        for op in self.operands:
            if isinstance(op, int):
                uses.add(op)
            elif isinstance(op, list):
                for s in op:
                    if isinstance(s, int):
                        uses.add(s)
        return uses


@dataclass
class LIRInst(InstructionDefUseMixin):
    """A single LIR instruction.

    For scalar ops:
      - dest: int (single scratch address)
      - operands: list of int (scratch addresses) or immediates/labels

    For vector ops:
      - dest: list[int] (8 consecutive scratch addresses)
      - operands: mix of list[int] (vector) and int (scalar)
    """
    opcode: LIROpcode
    dest: Optional[int | list[int]]   # Scratch address(es) for result
    operands: list                    # Scratch addresses, immediates, or labels
    engine: str

    def __repr__(self):
        ops_str = ", ".join(str(o) for o in self.operands)
        if self.dest is not None:
            if isinstance(self.dest, list):
                return f"s{self.dest[0]}..s{self.dest[-1]} = {self.opcode.value}({ops_str}) [{self.engine}]"
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
