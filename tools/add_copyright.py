#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Script to add Microsoft copyright notice to files that don't already have one.

Usage:
    python add_copyright.py file1.py file2.py ...
    python add_copyright.py --glob "**/*.py"
    python add_copyright.py --help
"""

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List, Tuple


COPYRIGHT_NOTICE = """# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License."""


def has_copyright_notice(content: str) -> bool:
    """Check if the file already contains a Microsoft copyright notice."""
    lines = content.split("\n")

    # Check first few lines for copyright notice
    for i in range(min(10, len(lines))):
        line = lines[i].strip()
        if "Copyright (c) Microsoft Corporation" in line:
            return True

    return False


def should_add_blank_line(content: str, insert_pos: int) -> bool:
    """Determine if we should add a blank line after the copyright notice."""
    lines = content.split("\n")

    # If inserting at the very end of file, don't add blank line
    if insert_pos >= len(lines):
        return False

    # If the next line after insertion point is already blank, don't add another
    if insert_pos < len(lines) and lines[insert_pos].strip() == "":
        return False

    # If inserting at the end and file doesn't end with newline, don't add blank line
    if insert_pos == len(lines) - 1 and not content.endswith("\n"):
        return False

    return True


def find_insertion_point(content: str) -> int:
    """Find where to insert the copyright notice."""
    lines = content.split("\n")

    if not lines:
        return 0

    insert_line = 0

    # Skip shebang line if present
    if lines[0].startswith("#!"):
        insert_line = 1

    # Skip encoding declarations like # -*- coding: utf-8 -*-
    if insert_line < len(lines) and "coding:" in lines[insert_line]:
        insert_line += 1
    elif insert_line < len(lines) and "coding=" in lines[insert_line]:
        insert_line += 1

    return insert_line


def add_copyright_to_file(file_path: Path) -> bool:
    """
    Add copyright notice to a single file.

    Returns True if the file was modified, False otherwise.
    """
    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (UnicodeDecodeError, PermissionError) as e:
        print(f"Skipping {file_path}: {e}")
        return False

    # Check if copyright notice already exists
    if has_copyright_notice(content):
        print(f"Skipping {file_path}: Already has copyright notice")
        return False

    # Find where to insert the copyright notice
    lines = content.split("\n")
    insert_pos = find_insertion_point(content)

    # Prepare the copyright lines
    copyright_lines = COPYRIGHT_NOTICE.split("\n")

    # Add blank line after copyright if needed
    if should_add_blank_line(content, insert_pos):
        copyright_lines.append("")

    # Insert the copyright notice
    new_lines = lines[:insert_pos] + copyright_lines + lines[insert_pos:]
    new_content = "\n".join(new_lines)

    # Write back to file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Added copyright notice to {file_path}")
        return True
    except PermissionError as e:
        print(f"Error writing to {file_path}: {e}")
        return False


def expand_glob_patterns(patterns: List[str]) -> List[Path]:
    """Expand glob patterns to actual file paths."""
    files = []
    for pattern in patterns:
        if "*" in pattern or "?" in pattern:
            # It's a glob pattern
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                path = Path(match)
                if path.is_file():
                    files.append(path)
        else:
            # It's a regular file path
            path = Path(pattern)
            if path.is_file():
                files.append(path)
            elif path.exists():
                print(f"Warning: {pattern} is not a file, skipping")
            else:
                print(f"Warning: {pattern} does not exist, skipping")

    return files


def main():
    parser = argparse.ArgumentParser(
        description="Add Microsoft copyright notice to files that don't have one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python add_copyright.py file1.py file2.py
    python add_copyright.py --glob "**/*.py"
    python add_copyright.py --glob "src/**/*.ts" --glob "tests/**/*.py"
        """,
    )

    parser.add_argument(
        "files", nargs="*", help="Files to process (can be file paths or glob patterns)"
    )

    parser.add_argument(
        "--glob",
        action="append",
        dest="glob_patterns",
        help="Glob pattern for files to process (can be used multiple times)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    args = parser.parse_args()

    # Collect all file patterns
    all_patterns = args.files or []
    if args.glob_patterns:
        all_patterns.extend(args.glob_patterns)

    if not all_patterns:
        parser.print_help()
        return 1

    # Expand patterns to actual files
    files = expand_glob_patterns(all_patterns)

    if not files:
        print("No files found matching the given patterns")
        return 1

    print(f"Processing {len(files)} files...")

    modified_count = 0

    for file_path in files:
        if args.dry_run:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                if not has_copyright_notice(content):
                    print(f"Would add copyright notice to {file_path}")
                    modified_count += 1
                else:
                    print(f"Would skip {file_path}: Already has copyright notice")
            except Exception as e:
                print(f"Would skip {file_path}: {e}")
        else:
            if add_copyright_to_file(file_path):
                modified_count += 1

    if args.dry_run:
        print(f"\nDry run complete. Would modify {modified_count} files.")
    else:
        print(f"\nComplete. Modified {modified_count} files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
