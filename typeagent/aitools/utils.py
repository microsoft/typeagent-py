# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities that are hard to fit in any specific module."""

from contextlib import contextmanager
import difflib
import os
import shutil
import time

import black
import colorama
import dotenv
import typechat


cap = min  # More readable name for capping a value at some limit.


@contextmanager
def timelog(label: str):
    """Context manager to log the time taken by a block of code."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print(
            f"{colorama.Style.DIM}{elapsed_time:.3f}s -- {label}{colorama.Style.RESET_ALL}"
        )


def pretty_print(obj: object, prefix: str = "", suffix: str = "") -> None:
    """Pretty-print an object using black.

    NOTE: Only works if the repr() is a valid Python expression.
    """
    line_width = cap(200, shutil.get_terminal_size().columns)
    print(
        prefix
        + black.format_str(
            repr(obj), mode=black.FileMode(line_length=line_width)
        ).rstrip()
        + suffix
    )


def load_dotenv() -> None:
    """Load environment variables from '<repo_root>/ta/.env'."""
    dn = os.path.dirname
    repo_root = dn(dn(dn(dn(dn(__file__)))))  # python/ta/typeagent/aitools/utils.py
    env_path = os.path.join(repo_root, "ts", ".env")
    dotenv.load_dotenv(env_path)
    # for k, v in os.environ.items():
    #     if "KEY" in k:
    #         print(f"{k}={v!r}")
    # print(f"Loaded {env_path}")


def create_translator[T](
    model: typechat.TypeChatLanguageModel,
    schema_class: type[T],
) -> typechat.TypeChatJsonTranslator[T]:
    """Create a TypeChat translator for a given model and schema."""
    validator = typechat.TypeChatValidator[T](schema_class)
    translator = typechat.TypeChatJsonTranslator[T](model, validator, schema_class)
    return translator


# Vibe-coded by o4-mini-high
def list_diff(label_a, a, label_b, b, max_items):
    sm = difflib.SequenceMatcher(None, a, b)
    a_out, b_out = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_slice, b_slice = a[i1:i2], b[j1:j2]
        L = max(len(a_slice), len(b_slice))
        for k in range(L):
            a_out.append(str(a_slice[k]) if k < len(a_slice) else "")
            b_out.append(str(b_slice[k]) if k < len(b_slice) else "")

    # color helpers
    def color_a(val, other):
        return (
            colorama.Fore.RED + val + colorama.Style.RESET_ALL
            if val and val != other
            else val
        )

    def color_b(val, other):
        return (
            colorama.Fore.GREEN + val + colorama.Style.RESET_ALL
            if val and val != other
            else val
        )

    # apply color
    a_cols = [color_a(a_out[i], b_out[i]) for i in range(len(a_out))]
    b_cols = [color_b(b_out[i], a_out[i]) for i in range(len(b_out))]

    # compute column widths
    widths = [max(len(a_out[i]), len(b_out[i])) for i in range(len(a_out))]

    # prepare labels
    max_label = max(len(label_a), len(label_b))
    la = label_a.ljust(max_label)
    lb = label_b.ljust(max_label)

    # split into segments
    if max_items and max_items > 0:
        segments = [
            (i, min(i + max_items, len(a_cols)))
            for i in range(0, len(a_cols), max_items)
        ]
    else:
        segments = [(0, len(a_cols))]

    # formatter for a row segment
    def fmt(row, seg_widths):
        return " ".join(f"{cell:>{w}}" for cell, w in zip(row, seg_widths))

    # print each segment
    for start, end in segments:
        seg_widths = widths[start:end]
        print(la, fmt(a_cols[start:end], seg_widths))
        print(lb, fmt(b_cols[start:end], seg_widths))
