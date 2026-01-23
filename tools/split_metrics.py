#!/usr/bin/env python3
"""Split compiler --print-metrics output into per-pass files."""

import argparse
import re
import sys
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Sanitize pass name for use as a filename."""
    # Replace path separators and other unsafe characters with underscore
    # Keep only alphanumeric, dash, underscore, and dot
    sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Fallback if empty
    return sanitized or 'unknown'


def parse_metrics(content: str) -> list[tuple[str, str]]:
    """Parse metrics content into list of (pass_name, section_content)."""
    # Pattern to match pass headers like "=== Pass: dce (HIR → HIR) ==="
    pattern = r'^=== Pass: (\S+)'

    sections = []
    current_name = None
    current_lines = []

    for line in content.splitlines(keepends=True):
        match = re.match(pattern, line)
        if match:
            # Save previous section
            if current_name:
                sections.append((current_name, ''.join(current_lines)))
            # Start new section
            current_name = match.group(1)
            current_lines = [line]
        elif current_name:
            current_lines.append(line)

    # Save final section
    if current_name:
        sections.append((current_name, ''.join(current_lines)))

    return sections


def main():
    parser = argparse.ArgumentParser(description='Split metrics output into per-pass files')
    parser.add_argument('input', nargs='?', help='Input file (default: stdin)')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
    args = parser.parse_args()

    # Read input
    if args.input:
        content = Path(args.input).read_text()
    else:
        content = sys.stdin.read()

    # Parse sections
    sections = parse_metrics(content)

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write per-pass files
    for i, (name, section) in enumerate(sections, 1):
        safe_name = sanitize_filename(name)
        filename = f"{i:02d}-{safe_name}.txt"
        (out_dir / filename).write_text(section)
        print(f"Wrote {filename}")


if __name__ == '__main__':
    main()
