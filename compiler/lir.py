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
