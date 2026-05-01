# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

_script_path = Path(__file__).parent.parent / "scripts" / "block_git_mutations.py"
_spec = importlib.util.spec_from_file_location("block_git_mutations", _script_path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["block_git_mutations"] = _mod
_spec.loader.exec_module(_mod)

_is_command_safe = _mod._is_command_safe
_is_readonly_git = _mod._is_readonly_git
main = _mod.main


class TestIsReadonlyGit:
    """Unit tests for _is_readonly_git token-level checks."""

    @pytest.mark.parametrize(
        "tokens",
        [
            ["git", "status"],
            ["git", "diff"],
            ["git", "diff", "--cached"],
            ["git", "log", "--oneline"],
            ["git", "log", "-n", "5"],
            ["git", "show", "HEAD"],
            ["git", "branch"],
            ["git", "branch", "-v"],
            ["git", "branch", "--list"],
            ["git", "ls-files"],
            ["git", "ls-tree", "HEAD"],
            ["git", "rev-parse", "HEAD"],
            ["git", "describe", "--tags"],
            ["git", "remote", "-v"],
            ["git", "tag"],
            ["git", "tag", "-l"],
            ["git", "stash", "list"],
            ["git", "config", "--get", "user.name"],
            ["git", "blame", "README.md"],
            ["git", "shortlog", "-sn"],
            ["git", "worktree", "list"],
            ["git", "-C", "/some/path", "status"],
            ["git", "add", "."],
            ["git", "add", "-A"],
            ["git", "push", "origin", "main"],
        ],
    )
    def test_allows_readonly(self, tokens: list[str]) -> None:
        assert _is_readonly_git(tokens) is True

    @pytest.mark.parametrize(
        "tokens",
        [
            ["git", "commit", "-m", "test"],
            ["git", "reset", "--hard", "HEAD~1"],
            ["git", "rebase", "main"],
            ["git", "merge", "feature-branch"],
            ["git", "cherry-pick", "abc123"],
            ["git", "revert", "HEAD"],
            ["git", "stash"],
            ["git", "stash", "push"],
            ["git", "stash", "pop"],
            ["git", "stash", "drop"],
            ["git", "tag", "-a", "v1.0", "-m", "release"],
            ["git", "tag", "-d", "v1.0"],
            ["git", "branch", "-d", "old-branch"],
            ["git", "branch", "-D", "old-branch"],
            ["git", "branch", "-m", "new-name"],
            ["git", "branch", "--delete", "old-branch"],
            ["git", "remote", "add", "upstream", "url"],
            ["git", "remote", "remove", "upstream"],
            ["git", "config", "user.name", "Somebody"],
            ["git", "config", "--unset", "user.name"],
            ["git", "am"],
            ["git", "apply"],
            ["git", "checkout"],
            ["git", "switch"],
            ["git", "restore"],
            ["git", "clean"],
            ["git", "rm", "file.py"],
        ],
    )
    def test_blocks_mutations(self, tokens: list[str]) -> None:
        assert _is_readonly_git(tokens) is False


class TestIsCommandSafe:
    """Tests for full shell command parsing including chained commands and aliases."""

    def test_non_git_commands_pass(self) -> None:
        assert _is_command_safe("python -m pytest tests/") is True
        assert _is_command_safe("make check test") is True
        assert _is_command_safe("uv add stamina") is True
        assert _is_command_safe("pyright src/") is True

    def test_chained_readonly_pass(self) -> None:
        assert _is_command_safe("git status && git diff") is True
        assert _is_command_safe("git log --oneline | head -5") is True
        assert _is_command_safe("git status; git log") is True

    def test_chained_with_mutation_blocked(self) -> None:
        assert _is_command_safe("git status && git commit -m 'test'") is False
        assert _is_command_safe("git diff; git reset --hard HEAD") is False
        assert _is_command_safe("make test && git commit -m 'auto'") is False

    def test_quoted_args_handled(self) -> None:
        assert _is_command_safe("git commit -m 'this is a test'") is False
        assert _is_command_safe('git log --grep="some pattern"') is True

    def test_git_alias_resolved(self) -> None:
        fake_aliases = {"ci": "commit", "st": "status", "co": "checkout"}
        with patch.object(_mod, "_load_git_aliases", return_value=fake_aliases):
            assert _is_command_safe("git ci -m 'test'") is False
            assert _is_command_safe("git st") is True
            assert _is_command_safe("git co feature") is False

    def test_empty_command_passes(self) -> None:
        assert _is_command_safe("") is True


class TestMainHook:
    """Integration tests for the main() entry point."""

    def _run_hook(self, tool_name: str, command: str) -> dict:
        payload = json.dumps(
            {
                "tool_name": tool_name,
                "tool_input": {"command": command},
            }
        )
        import io

        with (
            patch("sys.stdin", io.StringIO(payload)),
            patch("sys.stdout", new_callable=io.StringIO) as mock_stdout,
        ):
            main()
            return json.loads(mock_stdout.getvalue())

    @pytest.mark.parametrize("tool_name", ["Bash", "runTerminalCommand"])
    @pytest.mark.parametrize(
        "command",
        [
            "git commit -m 'test'",
            "git reset --hard HEAD~1",
            "git rebase main",
            "git merge feature-branch",
            "git cherry-pick abc123",
            "git revert HEAD",
            "git stash",
            "git tag -a v1.0",
        ],
    )
    def test_blocks_mutations(self, tool_name: str, command: str) -> None:
        output = self._run_hook(tool_name, command)
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    @pytest.mark.parametrize("tool_name", ["Bash", "runTerminalCommand"])
    @pytest.mark.parametrize(
        "command",
        [
            "git status",
            "git diff",
            "git diff --cached",
            "git log --oneline",
            "git show HEAD",
            "git branch -v",
            "git add .",
            "git push origin main",
        ],
    )
    def test_allows_readonly(self, tool_name: str, command: str) -> None:
        output = self._run_hook(tool_name, command)
        assert output.get("continue") is True

    def test_ignores_non_terminal_tools(self) -> None:
        output = self._run_hook("editFiles", "git commit -m 'sneaky'")
        assert output.get("continue") is True

    def test_allows_non_git_commands(self) -> None:
        output = self._run_hook("Bash", "make check test")
        assert output.get("continue") is True
