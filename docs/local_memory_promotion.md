# Local Memory Contract and Promotion

`assume_local_memory(base, length)` is a source-level optimization contract.
`length` is measured in 32-bit words. At the annotation point the region
`[base, base + length)` is valid, non-wrapping, and logically zero-initialized.
Until function exit it is private to the current invocation, cannot escape,
and cannot be observed by another function or thread. Every access to the
region in the current invocation must use `base` or an address derived from
it; access through an independently rooted alias violates the contract. Its
final contents are unobservable. Violating the contract is undefined behavior.
The region is also unobservable at `Pause` and debug/host-inspection
boundaries; observing it there violates the same contract.

The annotation does not allocate or clear memory and lowers to no target
instruction. Ignoring it is always valid.

## Local SROA and Mem2Reg

`LocalMem2RegPass` is a general HIR optimization. It does not inspect
header slots, variable names, tree shapes, or benchmark constants.

The first implementation requires an SSA base and a positive constant length.
It promotes a region only when all accesses in its scope are scalar
loads/stores at constant, in-range offsets in flat HIR. It first proves the
whole region legal, then rewrites atomically:

```text
state[offset] = 0
load(base + offset)       -> state[offset]
store(base + offset, val) -> state[offset] = val
```

No final stores are emitted because the contract makes the region
unobservable on return. Dynamic derived addresses, pointer escape, retained
control flow, and overlapping vector accesses make the whole region fall
back to ordinary memory -- except dynamic offsets whose value range provably
lies entirely outside the region (non-wrapping, footprint included): those
are ordinary accesses to other memory and are preserved without rejecting
the region. Constant offsets follow the target's modulo-2**32 address
arithmetic; statically out-of-range accesses are preserved likewise.

The pass runs after full unrolling, Simplify, and CSE, but before generic
LoadElim/DSE. Promoting the complete local region first avoids making those
memory passes rediscover the same forwarding chains. A default-pipeline
`strip-assume` pass then removes any marker left behind when promotion is
disabled, followed immediately by DCE. This placement is a pipeline heuristic
for the current kernel; the promotion legality rules are program-independent.
If this cleanup pass is disabled or omitted by a custom pipeline, lowering
still erases the assumption marker defensively; the difference is only that
dead HIR may survive longer.
