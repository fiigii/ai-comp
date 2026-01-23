#!/usr/bin/env python3
"""Split compiler --print-metrics or --print-after-all output into per-pass files."""

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


def parse_sections(content: str) -> list[tuple[str, str]]:
    """Parse content into list of (pass_name, section_content).

    Supports formats:
    - --print-metrics: "=== Pass: NAME ... ===" headers
    - --print-after-all: "After NAME:" headers (preceded by dashes)
    - Combined: both metrics and IR dump for same pass in one section
    """
    # Pattern for --print-metrics format (primary section delimiter)
    metrics_pattern = r'^=== Pass: (\S+)'
    # Pattern for --print-after-all format
    after_all_pattern = r'^After (\S+):$'

    sections = []
    current_name = None
    current_lines = []
    prev_line_is_dashes = False

    for line in content.splitlines(keepends=True):
        # Check for --print-metrics format
        metrics_match = re.match(metrics_pattern, line)
        # Check for --print-after-all format (requires preceding dash line)
        after_match = re.match(after_all_pattern, line.strip())

        if metrics_match:
            # Save previous section
            if current_name:
                sections.append((current_name, ''.join(current_lines)))
            # Start new section
            current_name = metrics_match.group(1)
            current_lines = [line]
            prev_line_is_dashes = False
        elif after_match and prev_line_is_dashes:
            after_name = after_match.group(1).rstrip(':')
            # If this "After X:" matches current section, just continue (don't start new)
            if current_name == after_name:
                current_lines.append(line)
                prev_line_is_dashes = False
            else:
                # Different name - save previous and start new section
                if current_name:
                    sections.append((current_name, ''.join(current_lines)))
                current_name = after_name
                current_lines = ['-' * 60 + '\n', line]
                prev_line_is_dashes = False
        elif current_name:
            current_lines.append(line)
            prev_line_is_dashes = line.strip().startswith('---') and line.strip().replace('-', '') == ''
        else:
            prev_line_is_dashes = line.strip().startswith('---') and line.strip().replace('-', '') == ''

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
    sections = parse_sections(content)

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
