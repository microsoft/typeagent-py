#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PreToolUse hook: block git commands that mutate the repository.

Works with both Claude Code and VS Code Copilot hooks.
Reads a JSON tool invocation from stdin, checks if it contains a git
mutation, and outputs a deny decision if so.
"""

import json
import re
import shlex
import subprocess
import sys

TERMINAL_TOOL_NAMES = frozenset({"Bash", "runTerminalCommand"})

ALLOWED_SUBCOMMANDS = frozenset({
    "add",
    "blame",
    "branch",
    "cat-file",
    "config",
    "count-objects",
    "describe",
    "diff",
    "diff-tree",
    "for-each-ref",
    "fsck",
    "grep",
    "log",
    "ls-files",
    "ls-remote",
    "ls-tree",
    "name-rev",
    "push",
    "reflog",
    "remote",
    "rev-list",
    "rev-parse",
    "shortlog",
    "show",
    "show-branch",
    "show-ref",
    "stash",
    "status",
    "symbolic-ref",
    "tag",
    "version",
    "worktree",
})

BRANCH_WRITE_FLAGS = frozenset({"-d", "-D", "--delete", "-m", "-M", "--move", "-c", "-C", "--copy"})
TAG_WRITE_FLAGS = frozenset({"-d", "--delete", "-f", "--force", "-a", "-s", "--sign"})
STASH_ALLOWED_SUBCOMMANDS = frozenset({"list", "show"})
WORKTREE_ALLOWED_SUBCOMMANDS = frozenset({"list"})
REMOTE_WRITE_SUBCOMMANDS = frozenset({"add", "remove", "rm", "rename", "set-url", "set-head", "prune", "update"})
CONFIG_WRITE_INDICATORS = frozenset({"--unset", "--unset-all", "--remove-section", "--rename-section", "--replace-all", "--add"})

_GIT_ALIAS_CACHE: dict[str, str] | None = None


def _load_git_aliases() -> dict[str, str]:
    global _GIT_ALIAS_CACHE
    if _GIT_ALIAS_CACHE is not None:
        return _GIT_ALIAS_CACHE
    try:
        result = subprocess.run(
            ["git", "config", "--get-regexp", r"^alias\."],
            capture_output=True,
            text=True,
            timeout=5,
        )
        aliases: dict[str, str] = {}
        for line in result.stdout.strip().splitlines():
            parts = line.split(None, 1)
            if len(parts) == 2:
                name = parts[0].removeprefix("alias.")
                aliases[name] = parts[1].split()[0]
        _GIT_ALIAS_CACHE = aliases
    except (subprocess.TimeoutExpired, FileNotFoundError):
        _GIT_ALIAS_CACHE = {}
    return _GIT_ALIAS_CACHE


def _resolve_alias(subcommand: str) -> str:
    aliases = _load_git_aliases()
    return aliases.get(subcommand, subcommand)


def _is_readonly_git(tokens: list[str]) -> bool:
    """Check if a parsed git command is read-only."""
    if not tokens or tokens[0] != "git":
        return True

    args = tokens[1:]
    # Skip global git flags (e.g. git -C /path ...)
    while args and args[0].startswith("-"):
        flag = args.pop(0)
        if flag in ("-C", "-c", "--git-dir", "--work-tree", "--namespace"):
            if args:
                args.pop(0)
    if not args:
        return True

    subcommand = _resolve_alias(args[0])
    rest = args[1:]

    if subcommand not in ALLOWED_SUBCOMMANDS:
        return False

    if subcommand == "branch" and any(f in BRANCH_WRITE_FLAGS for f in rest):
        return False
    if subcommand == "tag" and any(f in TAG_WRITE_FLAGS for f in rest):
        return False
    if subcommand == "stash":
        stash_sub = rest[0] if rest else "push"
        if stash_sub not in STASH_ALLOWED_SUBCOMMANDS:
            return False
    if subcommand == "worktree":
        wt_sub = rest[0] if rest else ""
        if wt_sub not in WORKTREE_ALLOWED_SUBCOMMANDS:
            return False
    if subcommand == "remote":
        remote_sub = rest[0] if rest else ""
        if remote_sub in REMOTE_WRITE_SUBCOMMANDS:
            return False
    if subcommand == "config":
        if any(f in CONFIG_WRITE_INDICATORS for f in rest):
            return False
        # `git config key value` (positional set) — 2+ non-flag args means a write
        non_flag_args = [a for a in rest if not a.startswith("-")]
        if len(non_flag_args) >= 2:
            return False

    return True


_SHELL_SPLIT = re.compile(r"\s*(?:&&|\|\||[;|])\s*")


def _split_shell_commands(command: str) -> list[str]:
    return [seg.strip() for seg in _SHELL_SPLIT.split(command) if seg.strip()]


def _is_command_safe(command: str) -> bool:
    for segment in _split_shell_commands(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if tokens and tokens[0] == "git" and not _is_readonly_git(tokens):
            return False
    return True


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _allow() -> dict:
    return {"continue": True}


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        json.dump(_allow(), sys.stdout)
        return

    tool_name = payload.get("tool_name", "")
    if tool_name not in TERMINAL_TOOL_NAMES:
        json.dump(_allow(), sys.stdout)
        return

    tool_input = payload.get("tool_input", {})
    command = tool_input.get("command") or tool_input.get("input") or ""
    if not command:
        json.dump(_allow(), sys.stdout)
        return

    if _is_command_safe(command):
        json.dump(_allow(), sys.stdout)
    else:
        json.dump(
            _deny("Blocked: git mutation commands are not allowed in this repository."),
            sys.stdout,
        )


if __name__ == "__main__":
    main()
