---
name: debug-compiler
description: Compiler development and debugging instructions
---

# Compiler Debugging Guide

## Debugging Tips

When developing the compiler, if you run into an unexplained bug, you often need to start from the generated VLIW assembly and trace the cause upward, pass by pass. Each time, you can diff the previous pass’s printed outputs (IR, DDG, metrics, etc.) against the outputs from the pass you want to debug, to see what changes the pass made and whether those changes match your expectations. You can use the following tools.

## Tools to debug compiler

```bash
# Print IR after each compilation pass
python3 perf_takehome.py --print-after-all 

# Print pass metrics and diagnostics
python3 perf_takehome.py --print-metrics 

# Print Data Dependency Graphs after each pass
python3 perf_takehome.py --print-ddg-after-all 

# Combine flags for comprehensive debugging
python3 perf_takehome.py --print-after-all --print-metrics 

# Save output to file for splitting
python3 perf_takehome.py --print-after-all --print-metrics  > dump.txt
```

## Debugging Flags

### `--print-after-all`
Prints the IR after each compilation pass. Output format:
```
------------------------------------------------------------
After PassName:
<IR dump>
```

Use this to see how the IR transforms through each pass. Helpful for:
- Verifying a pass is making expected transformations
- Finding which pass introduces a bug
- Understanding the compilation pipeline

### `--print-metrics`
Prints pass metrics and diagnostics. Output format:
```
=== Pass: PassName ===
<metrics and diagnostics>
```

Use this to see statistics about what each pass did (e.g., how many instructions were eliminated, optimizations applied).

### `--print-ddg-after-all`
Prints Data Dependency Graphs after each compilation pass. Useful for:
- Understanding data flow between operations
- Debugging scheduling issues
- Analyzing instruction dependencies

## Tools

### `tools/split_dump.py`

Splits `--print-metrics` or `--print-after-all` output into per-pass files for easier analysis.

**Usage:**
```bash
# Split from file
python3 tools/split_dump.py dump.txt -o passes/

# Split from stdin
python3 perf_takehome.py --print-after-all --print-metrics  | python3 tools/split_dump.py -o passes/
```

**Output:** Creates numbered files like `01-PassName.txt`, `02-AnotherPass.txt` in the output directory.


This makes it easy to diff passes or focus on a specific pass without scrolling through large dumps.
