# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities that are hard to fit in any specific module."""

import difflib
import os
import re
import shutil
import time
from contextlib import contextmanager

import black
import colorama
import dotenv
import logfire
import typechat
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic_ai import Agent, NativeOutput, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider

from .auth import get_shared_token_provider


@contextmanager
def timelog(label: str, verbose: bool = True):
    """Context manager to log the time taken by a block of code.

    With verbose=False it prints nothing."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        if verbose:
            dim = colorama.Style.DIM
            reset = colorama.Style.RESET_ALL
            print(f"{dim}{elapsed_time:.3f}s -- {label}{reset}")


def pretty_print(obj: object, prefix: str = "", suffix: str = "") -> None:
    """Pretty-print an object using black.

    NOTE: Only works if its repr() is a valid Python expression.
    """
    print(prefix + format_code(repr(obj)) + suffix)


def format_code(text: str, line_width=None) -> str:
    """Format a block of code using black, then reindent to 2 spaces.

    NOTE: The text must be a valid Python expression or code block.
    """
    if line_width is None:
        # Use the terminal width, but cap it to 200 characters.
        line_width = min(200, shutil.get_terminal_size().columns)
    formatted_text = black.format_str(
        text, mode=black.Mode(line_length=line_width)
    ).rstrip()
    return reindent(formatted_text)


def reindent(text: str) -> str:
    """Reindent a block of text from 4 to 2 spaces per indent level."""
    lines = text.splitlines()
    reindented_lines = []
    for line in lines:
        stripped_line = line.lstrip()
        twice_indent_level = (len(line) - len(stripped_line) + 1) // 2  # Round up
        reindented_lines.append(" " * twice_indent_level + stripped_line)
    return "\n".join(reindented_lines)


def load_dotenv() -> None:
    """Load environment variables from '<repo_root>/ta/.env'."""
    # Look for ".env" in current directory and up until root.
    cur_dir = os.path.abspath(os.getcwd())
    while True:
        path = os.path.join(cur_dir, ".env")
        if os.path.exists(path):
            dotenv.load_dotenv(path)
            return
        parent_dir = os.path.dirname(cur_dir)
        if parent_dir == cur_dir:
            break  # Reached filesystem root ('/').
        cur_dir = parent_dir


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
    """Print colorized diff between two sorted list of numbers."""
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


def setup_logfire():
    """Configure logfire for pydantic_ai and httpx."""

    def scrubbing_callback(m: logfire.ScrubMatch):
        """Instructions: Uncomment any block where you deem it safe to not scrub."""
        # if m.path == ('attributes', 'http.request.header.authorization'):
        #     return m.value

        # if m.path == ('attributes', 'http.request.header.api-key'):
        #     return m.value

        if (
            m.path == ("attributes", "http.request.body.text", "messages", 0, "content")
            and m.pattern_match.group(0) == "secret"
        ):
            return m.value

        # if m.path == ('attributes', 'http.response.header.azureml-model-session'):
        #     return m.value

    logfire.configure(scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback))
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)


def parse_azure_endpoint(
    endpoint_envvar: str = "AZURE_OPENAI_ENDPOINT",
) -> tuple[str, str]:
    """Parse Azure OpenAI endpoint to extract endpoint URL and API version.

    Args:
        endpoint_envvar: Name of environment variable containing the endpoint.

    Returns:
        Tuple of (endpoint_url, api_version).

    Raises:
        RuntimeError: If endpoint is not found or doesn't contain api-version.
    """
    azure_endpoint = os.getenv(endpoint_envvar)
    if not azure_endpoint:
        raise RuntimeError(f"Environment variable {endpoint_envvar} not found")

    m = re.search(r"[?,]api-version=([\d-]+(?:preview)?)", azure_endpoint)
    if not m:
        raise RuntimeError(
            f"{endpoint_envvar}={azure_endpoint} doesn't contain valid api-version field"
        )

    return azure_endpoint, m.group(1)


def get_azure_api_key(azure_api_key: str) -> str:
    """Get Azure OpenAI API key, handling 'identity' token provider case.

    Args:
        azure_api_key: The raw API key from environment variable.

    Returns:
        The API key or token to use.
    """

    # This section is rather specific to our team's setup at Microsoft.
    if azure_api_key.lower() == "identity":
        token_provider = get_shared_token_provider()
        azure_api_key = token_provider.get_token()

    return azure_api_key


def create_async_openai_client(
    endpoint_envvar: str = "AZURE_OPENAI_ENDPOINT",
    base_url: str | None = None,
):
    """Create AsyncOpenAI or AsyncAzureOpenAI client based on environment variables.

    Returns the appropriate async OpenAI client based on what credentials are available.
    Prefers OPENAI_API_KEY over AZURE_OPENAI_API_KEY.

    Args:
        endpoint_envvar: Environment variable name for Azure endpoint (default: AZURE_OPENAI_ENDPOINT).
        base_url: Optional base URL override for OpenAI client.

    Returns:
        AsyncOpenAI or AsyncAzureOpenAI client instance.

    Raises:
        RuntimeError: If neither OPENAI_API_KEY nor AZURE_OPENAI_API_KEY is set.
    """

    if openai_api_key := os.getenv("OPENAI_API_KEY"):
        return AsyncOpenAI(api_key=openai_api_key, base_url=base_url)

    elif azure_api_key := os.getenv("AZURE_OPENAI_API_KEY"):
        azure_api_key = get_azure_api_key(azure_api_key)
        azure_endpoint, api_version = parse_azure_endpoint(endpoint_envvar)

        return AsyncAzureOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
        )

    else:
        raise RuntimeError(
            "Neither OPENAI_API_KEY nor AZURE_OPENAI_API_KEY was provided."
        )


# The true return type is pydantic_ai.Agent[T], but that's an optional dependency.
def make_agent[T](cls: type[T]):
    """Create Pydantic AI agent using hardcoded preferences."""

    # Prefer straight OpenAI over Azure OpenAI.
    if os.getenv("OPENAI_API_KEY"):
        Wrapper = NativeOutput
        print(f"## Using OpenAI with {Wrapper.__name__} ##")
        model = OpenAIChatModel("gpt-4o")  # Retrieves OPENAI_API_KEY again.

    elif azure_api_key := os.getenv("AZURE_OPENAI_API_KEY"):
        azure_api_key = get_azure_api_key(azure_api_key)
        azure_endpoint, api_version = parse_azure_endpoint("AZURE_OPENAI_ENDPOINT")

        print(f"## {azure_endpoint} ##")
        Wrapper = ToolOutput

        print(f"## Using Azure {api_version} with {Wrapper.__name__} ##")
        model = OpenAIChatModel(
            "gpt-4o",
            provider=AzureProvider(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=azure_api_key,
            ),
        )

    else:
        raise RuntimeError(
            "Neither OPENAI_API_KEY nor AZURE_OPENAI_API_KEY was provided."
        )

    return Agent(model, output_type=Wrapper(cls, strict=True), retries=3)
