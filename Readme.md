# AI-Comp (Anthropic Interview Compiler)

Anthropic published their [original performance take-home](https://github.com/anthropics/original_performance_takehome) interview challenge: optimize a kernel running on a simulated VLIW SIMD virtual machine to minimize cycle count for a tree traversal + hash computation workload.

Instead of hand-optimizing the kernel, I wrote an optimizing compiler that takes a high-level IR description of the kernel and compiles it down to efficient VLIW bundles.

## Project Structure

```
ai-comp/
├── compiler/           # Optimizing compiler (HIR → LIR → MIR → VLIW)
│   ├── passes/         # Optimization passes (DCE, CSE, SLP vectorization, etc.)
│   └── tests/          # Compiler unit and regression tests
├── programs/           # Kernel implementations using the compiler
├── vm/                 # Thin wrapper around the upstream VM simulator
├── original_performance_takehome/  # Unmodified upstream challenge code
├── tests/              # Submission correctness and speed tests
├── docs/               # Architecture and design documents
└── tools/              # Development utilities
```

## Usage

```bash
# Compile and run the tree hash kernel
python programs/tree_hash.py

# Run submission tests
python tests/submission_tests.py
```
