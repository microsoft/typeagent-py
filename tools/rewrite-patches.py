#!/usr/bin/env python3
"""Rewrite git patches to strip python/ta/ prefix from all paths.

This tool is used to migrate commits from the TypeAgent monorepo to the
standalone python/ta repository. It takes git format-patch output files
and rewrites all path references to remove the python/ta/ prefix.

Usage:
    # Generate patches in the TypeAgent repo
    git format-patch <since>..HEAD --output-directory=/tmp/patches -- python/ta/

    # Rewrite the patches
    python3 tools/rewrite-patches.py /tmp/patches/*.patch

    # Apply in the new repo
    cd /path/to/new-repo
    git am /tmp/patches/*.patch
"""

import sys
from pathlib import Path


def rewrite_patch(patch_content: str) -> str:
    """Rewrite paths in a git patch to strip python/ta/ prefix.

    Args:
        patch_content: The content of a git format-patch output file

    Returns:
        The rewritten patch content with python/ta/ stripped from all paths
    """
    lines = patch_content.split("\n")
    output = []

    for line in lines:
        # Handle diff headers (e.g., "diff --git a/python/ta/foo.py b/python/ta/foo.py")
        if line.startswith("diff --git "):
            line = line.replace(" a/python/ta/", " a/")
            line = line.replace(" b/python/ta/", " b/")

        # Handle file path headers in unified diff format
        elif line.startswith("--- "):
            if line.startswith("--- a/python/ta/"):
                line = line.replace("--- a/python/ta/", "--- a/")
            elif line == "--- /dev/null":
                pass  # Leave /dev/null unchanged

        elif line.startswith("+++ "):
            if line.startswith("+++ b/python/ta/"):
                line = line.replace("+++ b/python/ta/", "+++ b/")
            elif line == "+++ /dev/null":
                pass  # Leave /dev/null unchanged

        # Handle rename/copy operations
        elif line.startswith("rename from "):
            line = line.replace("rename from python/ta/", "rename from ")
        elif line.startswith("rename to "):
            line = line.replace("rename to python/ta/", "rename to ")
        elif line.startswith("copy from "):
            line = line.replace("copy from python/ta/", "copy from ")
        elif line.startswith("copy to "):
            line = line.replace("copy to python/ta/", "copy to ")

        # Handle similarity index for renames (no path changes needed)
        # Handle index lines (no path changes needed)
        # Handle new/deleted file mode (no path changes needed)

        output.append(line)

    return "\n".join(output)


def main():
    """Main entry point for the patch rewriter."""
    if len(sys.argv) < 2:
        print(
            "Usage: rewrite-patches.py <patch-file> [patch-file ...]", file=sys.stderr
        )
        print(
            "\nRewrite git format-patch files to strip python/ta/ prefix from paths.",
            file=sys.stderr,
        )
        print("\nExample:", file=sys.stderr)
        print(
            "  git format-patch abc123..HEAD --output-directory=/tmp/patches -- python/ta/",
            file=sys.stderr,
        )
        print(
            "  python3 tools/rewrite-patches.py /tmp/patches/*.patch", file=sys.stderr
        )
        sys.exit(1)

    patch_files = sys.argv[1:]
    success_count = 0
    error_count = 0

    for patch_file in patch_files:
        try:
            path = Path(patch_file)
            if not path.exists():
                print(f"Error: File not found: {patch_file}", file=sys.stderr)
                error_count += 1
                continue

            if not path.is_file():
                print(f"Error: Not a file: {patch_file}", file=sys.stderr)
                error_count += 1
                continue

            # Read the original patch
            content = path.read_text(encoding="utf-8")

            # Rewrite paths
            rewritten = rewrite_patch(content)

            # Write back to the same file
            path.write_text(rewritten, encoding="utf-8")

            print(f"✓ Rewrote {patch_file}")
            success_count += 1

        except Exception as e:
            print(f"Error processing {patch_file}: {e}", file=sys.stderr)
            error_count += 1

    # Print summary
    print(
        f"\nProcessed {success_count + error_count} files: {success_count} successful, {error_count} errors"
    )

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
