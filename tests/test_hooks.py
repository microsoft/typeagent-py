# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import subprocess
from pathlib import Path

import pytest

HOOK_SCRIPT = Path(__file__).parent.parent / "scripts" / "block-git-mutations.sh"


def run_hook(tool_name: str, command: str) -> dict:
    """Run the hook script with a simulated tool invocation and return parsed output."""
    payload = json.dumps({
        "tool_name": tool_name,
        "tool_input": {"command": command},
    })
    result = subprocess.run(
        [str(HOOK_SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Hook exited with {result.returncode}: {result.stderr}"
    return json.loads(result.stdout)


class TestBlockGitMutations:
    """Tests for the block-git-mutations.sh PreToolUse hook."""

    @pytest.mark.parametrize("tool_name", ["Bash", "runTerminalCommand"])
    @pytest.mark.parametrize(
        "command",
        [
            "git commit -m 'test'",
            "git add .",
            "git add -A",
            "git push origin main",
            "git reset --hard HEAD~1",
            "git rebase main",
            "git merge feature-branch",
            "git cherry-pick abc123",
            "git revert HEAD",
            "git stash",
            "git tag v1.0",
        ],
    )
    def test_blocks_mutation_commands(self, tool_name: str, command: str) -> None:
        output = run_hook(tool_name, command)
        decision = output["hookSpecificOutput"]["permissionDecision"]
        assert decision == "deny", f"Expected deny for '{command}', got {decision}"

    @pytest.mark.parametrize("tool_name", ["Bash", "runTerminalCommand"])
    @pytest.mark.parametrize(
        "command",
        [
            "git status",
            "git diff",
            "git diff --cached",
            "git log --oneline",
            "git log -n 5",
            "git show HEAD",
        ],
    )
    def test_allows_read_only_commands(self, tool_name: str, command: str) -> None:
        output = run_hook(tool_name, command)
        assert "hookSpecificOutput" not in output or output.get("continue") is True

    @pytest.mark.parametrize(
        "command",
        [
            "python -m pytest tests/",
            "make check test",
            "uv add stamina",
            "pyright src/",
        ],
    )
    def test_allows_non_git_commands(self, command: str) -> None:
        output = run_hook("Bash", command)
        assert output.get("continue") is True

    def test_ignores_non_terminal_tools(self) -> None:
        output = run_hook("editFiles", "git commit -m 'sneaky'")
        assert output.get("continue") is True
