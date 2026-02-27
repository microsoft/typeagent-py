# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os

from dotenv import load_dotenv
import pytest

import pydantic.dataclasses
import typechat

import typeagent.aitools.utils as utils


def test_timelog():
    buf = StringIO()
    with redirect_stderr(buf):
        with utils.timelog("test block"):
            pass
    out = buf.getvalue()
    assert "test block..." in out


def test_pretty_print():
    # Use a simple object and check output is formatted by black
    obj = {"a": 1}
    buf = StringIO()
    with redirect_stdout(buf):
        utils.pretty_print(obj)
    out = buf.getvalue()
    # Should be valid Python and contain the dict
    assert out == '{"a": 1}\n', out


def test_load_dotenv(really_needs_auth):
    # Call load_dotenv and check for at least one expected key
    load_dotenv()
    assert "OPENAI_API_KEY" in os.environ or "AZURE_OPENAI_API_KEY" in os.environ


def test_create_translator():
    class DummyModel(typechat.TypeChatLanguageModel):
        async def complete(self, *args, **kwargs) -> typechat.Result:
            return typechat.Failure("dummy response")

    @pydantic.dataclasses.dataclass
    class DummySchema:
        pass

    # This will raise if the environment or typechat is not set up correctly
    translator = utils.create_translator(DummyModel(), DummySchema)
    assert hasattr(translator, "model")


class TestParseAzureEndpoint:
    """Tests for parse_azure_endpoint regex matching."""

    def test_api_version_after_question_mark(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """api-version as the first (and only) query parameter."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/gpt-4?api-version=2025-01-01-preview",
        )
        endpoint, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert version == "2025-01-01-preview"
        assert endpoint.startswith("https://")

    def test_api_version_after_ampersand(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """api-version preceded by & (not the first query parameter)."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/gpt-4?foo=bar&api-version=2025-01-01-preview",
        )
        endpoint, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert version == "2025-01-01-preview"

    def test_api_version_after_comma(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """api-version preceded by comma (alternate separator)."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/gpt-4?foo=bar,api-version=2024-06-01",
        )
        endpoint, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert version == "2024-06-01"

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError when the environment variable is not set."""
        monkeypatch.delenv("NONEXISTENT_ENDPOINT", raising=False)
        with pytest.raises(RuntimeError, match="not found"):
            utils.parse_azure_endpoint("NONEXISTENT_ENDPOINT")

    def test_no_api_version_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError when the endpoint has no api-version field."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/gpt-4",
        )
        with pytest.raises(RuntimeError, match="doesn't contain valid api-version"):
            utils.parse_azure_endpoint("TEST_ENDPOINT")
