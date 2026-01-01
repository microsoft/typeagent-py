# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility to import testdata path constants from conftest.py.

This module handles adding tests/test to sys.path so that conftest.py
can be imported from non-test code (tools/, src/).

Usage:
    from util_testdata import EPISODE_53_INDEX, EPISODE_53_ANSWERS, ...
"""

from pathlib import Path
import sys

# Add tests/test to path for conftest imports
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tests" / "test"))

# Re-export all testdata constants from conftest
from conftest import (  # type: ignore[import-not-found]  # noqa: E402
    CONFUSE_A_CAT_VTT,
    EPISODE_53_ANSWERS,
    EPISODE_53_INDEX,
    EPISODE_53_SEARCH,
    EPISODE_53_TRANSCRIPT,
    FAKE_PODCAST_TXT,
    get_repo_root,
    get_testdata_path,
    has_testdata_file,
    PARROT_SKETCH_VTT,
)

__all__ = [
    "CONFUSE_A_CAT_VTT",
    "EPISODE_53_ANSWERS",
    "EPISODE_53_INDEX",
    "EPISODE_53_SEARCH",
    "EPISODE_53_TRANSCRIPT",
    "FAKE_PODCAST_TXT",
    "PARROT_SKETCH_VTT",
    "get_repo_root",
    "get_testdata_path",
    "has_testdata_file",
]
