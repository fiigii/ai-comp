# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is Anthropic's original performance engineering take-home challenge. The task is to optimize a kernel running on a simulated VLIW SIMD virtual machine. The goal is to minimize cycle count for a tree traversal + hash computation workload.

**Important:** Do not modify files in the `tests/` folder. The submission tests use a frozen copy of the simulator to prevent cheating.

## Commands

```bash
# Run submission tests (validates correctness and shows cycle count)
python tests/submission_tests.py

# Run development tests with trace output
python perf_takehome.py Tests.test_kernel_trace

# Run cycle count test
python perf_takehome.py Tests.test_kernel_cycles

# View trace in Perfetto (run after generating trace)
python watch_trace.py
# Then open http://localhost:8000 and click "Open Perfetto"
```

## Architecture

### Virtual Machine (problem.py)

A VLIW SIMD simulator with:
- **Engines**: Execute multiple slots per cycle in parallel
  - `alu` (12 slots): Scalar arithmetic (+, -, *, //, %, ^, &, |, <<, >>, <, ==)
  - `valu` (6 slots): Vector operations on VLEN=8 lanes (vbroadcast, multiply_add, vector ops)
  - `load` (2 slots): Memory reads (load, vload, const, load_offset)
  - `store` (2 slots): Memory writes (store, vstore)
  - `flow` (1 slot): Control flow, select, vselect
  - `debug` (64 slots): Debugging only, doesn't count toward cycles

- **Memory model**: All slots in a bundle read pre-bundle state; writes commit at bundle end
- **Scratch**: Per-core register file (1536 words), used like registers/cache
- **Memory**: Shared flat array of 32-bit words

### Kernel (perf_takehome.py)

`KernelBuilder.build_kernel()` generates instructions for the reference algorithm:
1. Load batch of indices and values from memory
2. For each round and batch element:
   - Look up node value at current index
   - Compute `val = myhash(val ^ node_val)` (6-stage hash)
   - Compute next index: `2*idx + (1 if val%2==0 else 2)`
   - Wrap index if out of bounds
   - Store updated values back

The baseline implementation is fully unrolled and scalar (~147k cycles). Optimization strategies include VLIW packing, vectorization (VLEN=8), loop transformations, and exploiting instruction-level parallelism.

### Memory Layout (mem image)

```
Header (words 0-6): rounds, n_nodes, batch_size, forest_height, forest_values_p, inp_indices_p, inp_values_p
Arrays: forest_values[], inp_indices[], inp_values[]
```

## Validation

Always validate submissions with:
```bash
git diff origin/main tests/  # Must be empty
python tests/submission_tests.py  # Use this cycle count
```
