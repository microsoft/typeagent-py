# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities that are hard to fit in any specific module."""

from contextlib import contextmanager
import os
import shutil
import time

import black
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
        print(f"{elapsed_time:.3f}s -- {label}")


def pretty_print(obj: object) -> None:
    """Pretty-print an object using black.

    NOTE: Only works if the repr() is a valid Python expression.
    """
    line_width = cap(200, shutil.get_terminal_size().columns)
    print(
        black.format_str(
            repr(obj), mode=black.FileMode(line_length=line_width)
        ).rstrip()
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
    schema: type[T],
) -> typechat.TypeChatJsonTranslator[T]:
    """Create a TypeChat translator for a given model and schema."""
    validator = typechat.TypeChatValidator[T](schema)
    return typechat.TypeChatJsonTranslator[T](model, validator, schema)
