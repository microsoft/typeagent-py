#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Release automation script for the TypeAgent Python package.

This script:
1. Creates a new release branch (release-X.Y.Z)
2. Bumps the patch version (3rd part) in pyproject.toml, or sets the whole version
3. Updates uv.lock to reflect the version change
4. Commits the change and creates a git tag in the format "v{major}.{minor}.{patch}-py"
5. Bumps version to X.Y.Z.dev (post-release development marker)
6. Updates uv.lock again for the post-release version
7. Commits the post-release version change
8. Pushes the branch and tag
9. Creates a PR for the release branch

Usage:
    python tools/release.py [version] [--dry-run] [--help] [--force]

Examples:
    python tools/release.py              # Bump patch version
    python tools/release.py 1.0.0        # Set version to 1.0.0
    python tools/release.py 1.2.3 --dry-run  # Test setting version to 1.2.3
"""

import argparse
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Tuple


def run_command(cmd: list[str], dry_run: bool = False) -> Tuple[int, str]:
    """
    Run a shell command and return (exit_code, output).

    Args:
        cmd: Command as a list of strings
        dry_run: If True, print what would be run without executing

    Returns:
        Tuple of (exit_code, output_string)
    """
    cmd_str = " ".join(shlex.quote(s) for s in cmd)

    if dry_run:
        print(f"[DRY RUN] Would run: {cmd_str}")
        return 0, ""

    print(f"Running: {cmd_str}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.stdout:
        print(result.stdout.strip("\n"))
    if result.stderr:
        print(f"stderr: {result.stderr.strip('\n')}", file=sys.stderr)

    return result.returncode, result.stdout.strip()


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """
    Parse a semantic version string into (major, minor, patch).

    Args:
        version_str: Version string like "0.1.3"

    Returns:
        Tuple of (major, minor, patch) as integers

    Raises:
        ValueError: If version format is invalid
    """
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str.strip())
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")

    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version components back into a version string."""
    return f"{major}.{minor}.{patch}"


def compare_versions(
    version1: Tuple[int, int, int], version2: Tuple[int, int, int]
) -> int:
    """
    Compare two versions.

    Args:
        version1: First version tuple (major, minor, patch)
        version2: Second version tuple (major, minor, patch)

    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    if version1 < version2:
        return -1
    elif version1 > version2:
        return 1
    else:
        return 0


def get_current_version(pyproject_path: Path) -> str:
    """
    Extract the current version from pyproject.toml.

    Args:
        pyproject_path: Path to the pyproject.toml file

    Returns:
        Current version string (with .dev suffix stripped if present)

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist
        ValueError: If version field is not found or invalid
    """
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    content = pyproject_path.read_text(encoding="utf-8")

    # Look for version = "x.y.z" in the [project] section
    version_match = re.search(
        r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE
    )

    if not version_match:
        raise ValueError("Version field not found in pyproject.toml")

    version = version_match.group(1)

    # Strip .dev suffix if present (used for post-release development versions)
    if version.endswith(".dev"):
        version = version[:-4]

    return version


def update_version_in_pyproject(
    pyproject_path: Path, new_version: str, dry_run: bool = False
) -> None:
    """
    Update the version in pyproject.toml.

    Args:
        pyproject_path: Path to the pyproject.toml file
        new_version: New version string to set
        dry_run: If True, show what would be changed without modifying the file
    """
    content = pyproject_path.read_text(encoding="utf-8")

    # Replace the version field
    new_content = re.sub(
        r'^(version\s*=\s*["\'])[^"\']+(["\'])',
        rf"\g<1>{new_version}\g<2>",
        content,
        flags=re.MULTILINE,
    )

    if content == new_content:
        raise ValueError("Failed to update version in pyproject.toml")

    if dry_run:
        print(f"[DRY RUN] Would update version to {new_version} in {pyproject_path}")
        return

    pyproject_path.write_text(new_content, encoding="utf-8")
    print(f"Updated version to {new_version} in {pyproject_path}")


def check_git_status() -> bool:
    """
    Check if the git working directory is clean.

    Returns:
        True if working directory is clean, False otherwise
    """
    exit_code, output = run_command(["git", "status", "--porcelain"])

    if exit_code != 0:
        print("Error: Failed to check git status", file=sys.stderr)
        return False

    # If there's any output, the working directory is not clean
    return len(output.strip()) == 0


def check_uv_available() -> bool:
    """Check if the 'uv' command is available."""
    return shutil.which("uv") is not None


def check_gh_available() -> bool:
    """Check if the 'gh' CLI is available."""
    return shutil.which("gh") is not None


def update_uv_lock(dry_run: bool = False) -> bool:
    """Update uv.lock by running 'uv lock'."""
    if dry_run:
        print("[DRY RUN] Would run: uv lock")
        return True

    exit_code, _ = run_command(["uv", "lock"])
    if exit_code != 0:
        print("Error: Failed to update uv.lock", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Automate the release process for TypeAgent Python package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script will:
1. Create a new release branch (release-X.Y.Z)
2. Bump the patch version in pyproject.toml (or set to specified version)
3. Update uv.lock to reflect the version change
4. Commit the change with message "Bump version to X.Y.Z"
5. Create a git tag "vX.Y.Z-py"
6. Bump version to X.Y.Z.dev (post-release development marker)
7. Update uv.lock for the post-release version
8. Commit with message "Bump version to X.Y.Z.dev for development"
9. Push the branch and tag
10. Create a PR for the release branch

The script must be run from the repository root.

Examples:
    python tools/release.py              # Bump patch version
    python tools/release.py 1.0.0        # Set version to 1.0.0
    python tools/release.py 1.2.3 --dry-run  # Test setting version
        """,
    )

    parser.add_argument(
        "version",
        nargs="?",
        help="Optional: Specific version to set (e.g., 1.0.0). If not provided, patch version will be bumped.",
    )

    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )

    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force the release even if pre-checks fail",
    )

    args = parser.parse_args()

    # Ensure we're in the right directory
    current_dir = Path.cwd()
    expected_files = ["pyproject.toml", "tools"]

    for file_name in expected_files:
        if not (current_dir / file_name).exists():
            print(
                f"Error: {file_name} not found. Please run this script from the repository root.",
                file=sys.stderr,
            )
            return 1

    # Check that uv is available
    if not check_uv_available():
        print(
            "Error: 'uv' command not found. Please install uv first.", file=sys.stderr
        )
        print(
            "  Install with: curl -LsSf https://astral.sh/uv/install.sh | sh",
            file=sys.stderr,
        )
        return 1

    # Check that gh CLI is available
    if not check_gh_available():
        print(
            "Error: 'gh' CLI not found. Please install GitHub CLI first.",
            file=sys.stderr,
        )
        print(
            "  Install with: brew install gh  (or see https://cli.github.com/)",
            file=sys.stderr,
        )
        return 1

    pyproject_path = current_dir / "pyproject.toml"

    # Check git status (unless --force)
    if not check_git_status():
        if args.force:
            print(
                "Warning: Git working directory is not clean (forced)",
            )
        else:
            print(
                "Error: Git working directory is not clean. Please commit or stash changes first.",
                file=sys.stderr,
            )
            return 1

    # Get current version
    current_version = get_current_version(pyproject_path)
    print(f"Current version: {current_version}")

    # Parse current version
    current_major, current_minor, current_patch = parse_version(current_version)

    # Determine new version
    if args.version:
        # Use provided version
        try:
            new_major, new_minor, new_patch = parse_version(args.version)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # Validate that new version is higher than current (unless --force)
        current_tuple = (current_major, current_minor, current_patch)
        new_tuple = (new_major, new_minor, new_patch)
        comparison = compare_versions(new_tuple, current_tuple)

        if comparison <= 0:
            if args.force:
                print(
                    f"Warning: New version {args.version} matches or precedes current version {current_version} (forced)"
                )
            else:
                print(
                    f"Error: New version {args.version} matches or precedes current version {current_version}",
                    file=sys.stderr,
                )
                return 1

        new_version = args.version
    else:
        # Bump patch version
        new_patch = current_patch + 1
        new_version = format_version(current_major, current_minor, new_patch)

    print(f"New version: {new_version}")

    # Create release branch
    branch_name = f"release-{new_version}"
    exit_code, _ = run_command(["git", "checkout", "-b", branch_name], args.dry_run)

    if exit_code != 0:
        print(f"Error: Failed to create branch {branch_name}", file=sys.stderr)
        return 1

    # Update pyproject.toml
    update_version_in_pyproject(pyproject_path, new_version, args.dry_run)

    # Update uv.lock
    if not update_uv_lock(args.dry_run):
        return 1

    # Git commit for release version
    exit_code, _ = run_command(
        ["git", "add", "pyproject.toml", "uv.lock"], args.dry_run
    )

    if exit_code != 0:
        print("Error: Failed to stage pyproject.toml and uv.lock", file=sys.stderr)
        return 1

    commit_message = f"Bump version to {new_version}"
    exit_code, _ = run_command(["git", "commit", "-m", commit_message], args.dry_run)

    if exit_code != 0:
        print("Error: Failed to commit changes", file=sys.stderr)
        return 1

    # Create git tag
    tag_name = f"v{new_version}-py"
    exit_code, _ = run_command(["git", "tag", tag_name], args.dry_run)

    if exit_code != 0:
        print(f"Error: Failed to create tag {tag_name}", file=sys.stderr)
        return 1

    # Post-release: bump version to X.Y.Z.dev
    post_release_version = f"{new_version}.dev"
    print(f"Post-release version: {post_release_version}")

    update_version_in_pyproject(pyproject_path, post_release_version, args.dry_run)

    # Update uv.lock for post-release version
    if not update_uv_lock(args.dry_run):
        return 1

    # Git commit for post-release version
    exit_code, _ = run_command(
        ["git", "add", "pyproject.toml", "uv.lock"], args.dry_run
    )

    if exit_code != 0:
        print("Error: Failed to stage pyproject.toml and uv.lock", file=sys.stderr)
        return 1

    post_commit_message = f"Bump version to {post_release_version} for development"
    exit_code, _ = run_command(
        ["git", "commit", "-m", post_commit_message], args.dry_run
    )

    if exit_code != 0:
        print("Error: Failed to commit post-release changes", file=sys.stderr)
        return 1

    # Push branch and tag
    exit_code, _ = run_command(
        ["git", "push", "-u", "origin", branch_name], args.dry_run
    )

    if exit_code != 0:
        print(f"Error: Failed to push branch {branch_name}", file=sys.stderr)
        return 1

    exit_code, _ = run_command(["git", "push", "--tags"], args.dry_run)

    if exit_code != 0:
        print("Error: Failed to push tags", file=sys.stderr)
        return 1

    # Create PR
    pr_title = f"Release {new_version}"
    pr_body = f"## Release {new_version}\\n\\nThis PR contains:\\n- Version bump to {new_version}\\n- Tag {tag_name}\\n- Post-release version bump to {post_release_version}"
    exit_code, pr_url = run_command(
        ["gh", "pr", "create", "--title", pr_title, "--body", pr_body], args.dry_run
    )

    if exit_code != 0:
        print("Error: Failed to create PR", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"\n[DRY RUN] Release process completed successfully!")
        print(f"Would have created branch: {branch_name}")
        print(f"Would have created tag: {tag_name}")
        print(f"Would have created PR for release {new_version}")
    else:
        print(f"\nRelease process completed successfully!")
        print(f"Created branch: {branch_name}")
        print(f"Created tag: {tag_name}")
        print(f"Created PR: {pr_url}")
        print(f"\nNext steps:")
        print(f"  1. Get the PR approved and merged")
        print(f"  2. The GitHub Actions release workflow will be triggered by the tag")

    return 0


if __name__ == "__main__":
    sys.exit(main())
