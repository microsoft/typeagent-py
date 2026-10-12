# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.chat_sessions import get_default_pager


@pytest.fixture
def pager_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    if shutil.which("git") is None:
        pytest.skip("Git is required for pager configuration tests")

    for name in (
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
        "GIT_DIR",
        "GIT_COMMON_DIR",
        "GIT_WORK_TREE",
        "GIT_PAGER",
        "PAGER",
    ):
        monkeypatch.delenv(name, raising=False)

    global_config = tmp_path / "global.gitconfig"
    global_config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.chdir(tmp_path)

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "objects").mkdir()
    (git_dir / "refs").mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git_dir / "config").write_text(
        "[core]\n\trepositoryformatversion = 1\n"
        "[extensions]\n\tworktreeConfig = true\n",
        encoding="utf-8",
    )
    return global_config


def test_get_default_pager_uses_global_config(
    pager_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pager_config.write_text("[core]\n\tpager = global-pager --flag\n", encoding="utf-8")
    monkeypatch.setenv("GIT_PAGER", "git-environment-pager")
    monkeypatch.setenv("PAGER", "environment-pager")

    assert get_default_pager() == "global-pager --flag"


@pytest.mark.parametrize("config_name", ["config", "config.worktree"])
@pytest.mark.parametrize("global_pager", [None, "global-pager --flag"])
def test_get_default_pager_ignores_repository_config(
    pager_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    global_pager: str | None,
) -> None:
    config_path = pager_config.parent / ".git" / config_name
    with config_path.open("a", encoding="utf-8") as config:
        config.write("[core]\n\tpager = repository-pager\n")
    if global_pager is not None:
        pager_config.write_text(f"[core]\n\tpager = {global_pager}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_PAGER", "environment-pager")

    result = subprocess.run(
        ["git", "config", "--get", "core.pager"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "repository-pager"
    assert get_default_pager() == (global_pager or "environment-pager")


@pytest.mark.parametrize("include_section", ["include", 'includeIf "gitdir:**/.git"'])
@pytest.mark.parametrize("global_pager", [None, "global-pager --flag"])
def test_get_default_pager_ignores_config_includes(
    pager_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_section: str,
    global_pager: str | None,
) -> None:
    included_config = pager_config.parent / "included.gitconfig"
    included_config.write_text("[core]\n\tpager = included-pager\n", encoding="utf-8")
    contents = f"[core]\n\tpager = {global_pager}\n" if global_pager is not None else ""
    contents += f"[{include_section}]\n\tpath = included.gitconfig\n"
    pager_config.write_text(contents, encoding="utf-8")
    monkeypatch.setenv("GIT_PAGER", "environment-pager")

    result = subprocess.run(
        ["git", "config", "--includes", "--get", "core.pager"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "included-pager"
    assert get_default_pager() == (global_pager or "environment-pager")


@pytest.mark.parametrize(
    ("git_pager", "pager", "expected"),
    [
        (
            "git-environment-pager --flag",
            "environment-pager",
            "git-environment-pager --flag",
        ),
        (None, "environment-pager", "environment-pager"),
        ("", "environment-pager", "environment-pager"),
        (None, None, None),
        ("", "", None),
    ],
)
def test_get_default_pager_fallbacks(
    pager_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    git_pager: str | None,
    pager: str | None,
    expected: str | None,
) -> None:
    if git_pager is not None:
        monkeypatch.setenv("GIT_PAGER", git_pager)
    if pager is not None:
        monkeypatch.setenv("PAGER", pager)
    if expected is None and sys.platform != "win32":
        expected = "less"

    assert get_default_pager() == expected


@pytest.mark.parametrize("pager", ["", "   "])
def test_get_default_pager_ignores_empty_global_config(
    pager_config: Path, monkeypatch: pytest.MonkeyPatch, pager: str
) -> None:
    pager_config.write_text(f'[core]\n\tpager = "{pager}"\n', encoding="utf-8")
    monkeypatch.setenv("GIT_PAGER", "environment-pager")

    assert get_default_pager() == "environment-pager"


def test_get_default_pager_without_git(
    pager_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pager_config.write_text("[core]\n\tpager = global-pager\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(pager_config.parent))
    monkeypatch.setenv("GIT_PAGER", "environment-pager")

    assert shutil.which("git") is None
    assert get_default_pager() == "environment-pager"
