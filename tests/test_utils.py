# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os

from dotenv import load_dotenv
import pytest

import pydantic.dataclasses
import typechat

from typeagent.aitools import utils


def test_timelog():
    buf = StringIO()
    with redirect_stderr(buf), utils.timelog("test block"):
        pass
    out = buf.getvalue()
    assert "test block..." in out


def test_pretty_print():
    obj = {"a": 1}
    buf = StringIO()
    with redirect_stdout(buf):
        utils.pretty_print(obj)
    out = buf.getvalue()
    assert out == "{'a': 1}\n", out


def test_pretty_print_nested():
    obj = {"b": [1, 2], "a": {"nested": True}}
    buf = StringIO()
    with redirect_stdout(buf):
        utils.pretty_print(obj)
    out = buf.getvalue()
    # pprint sorts keys and formats nested structures
    assert "'a'" in out
    assert "'nested'" in out


def test_format_code_simple():
    text = repr({"a": 1})
    result = utils.format_code(text)
    assert result == "{'a': 1}"


def test_format_code_nested():
    obj = {"b": [1, 2, 3], "a": {"nested": True}}
    result = utils.format_code(repr(obj))
    parsed = eval(result)
    assert parsed == obj


def test_format_code_non_literal():
    """Test that format_code gracefully handles non-literal expressions.

    Regression test for commit 59be9a5 which broke debug output when format_code()
    was called on repr() of objects containing non-literal elements (e.g., AST nodes,
    custom class instances).
    """

    # Create a custom class instance (a non-literal object whose repr() can't be
    # evaluated with ast.literal_eval)
    class CustomClass:
        pass

    obj = CustomClass()
    non_literal_repr = repr(obj)
    # This repr looks like: <__main__.CustomClass object at 0x...>

    # format_code() should handle this gracefully without raising ValueError
    result = utils.format_code(non_literal_repr)
    assert isinstance(result, str)
    assert len(result) > 0
    # The result should contain the non-literal repr (possibly wrapped in quotes)
    assert "CustomClass object" in result or "CustomClass" in result


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
        assert endpoint == "https://myhost.openai.azure.com"

    def test_api_version_after_ampersand(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """api-version preceded by & (not the first query parameter)."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/gpt-4?foo=bar&api-version=2025-01-01-preview",
        )
        _, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert version == "2025-01-01-preview"

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError when the environment variable is not set."""
        monkeypatch.delenv("NONEXISTENT_ENDPOINT", raising=False)
        with pytest.raises(RuntimeError, match="not found"):
            utils.parse_azure_endpoint("NONEXISTENT_ENDPOINT")

    def test_query_string_stripped_from_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returned endpoint should not contain query string parameters."""
        monkeypatch.setenv(
            "TEST_ENDPOINT", "https://myhost.openai.azure.com?api-version=2024-06-01"
        )
        endpoint, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert endpoint == "https://myhost.openai.azure.com"
        assert version == "2024-06-01"

    def test_query_string_stripped_with_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Query string and deployment path stripped from endpoint."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/gpt-4?api-version=2025-01-01-preview",
        )
        endpoint, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert endpoint == "https://myhost.openai.azure.com"
        assert "?" not in endpoint
        assert version == "2025-01-01-preview"

    def test_deployment_name_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deployment name is extracted from deployment-style endpoints."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/ada-002/embeddings?api-version=2025-01-01-preview",
        )
        endpoint, version, deployment = utils.parse_azure_endpoint_parts(
            "TEST_ENDPOINT"
        )
        assert endpoint == "https://myhost.openai.azure.com"
        assert version == "2025-01-01-preview"
        assert deployment == "ada-002"

    def test_query_string_stripped_multiple_params(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All query parameters stripped, not just api-version."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com?foo=bar&api-version=2024-06-01",
        )
        endpoint, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert endpoint == "https://myhost.openai.azure.com"
        assert "foo" not in endpoint
        assert version == "2024-06-01"

    def test_bare_openai_path_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Trailing /openai without /deployments/ is stripped."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai?api-version=2024-06-01",
        )
        endpoint, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert endpoint == "https://myhost.openai.azure.com"
        assert version == "2024-06-01"

    def test_apim_prefix_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """APIM prefix before /openai/deployments/ is kept."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://apim.net/openai/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview",
        )
        endpoint, version = utils.parse_azure_endpoint("TEST_ENDPOINT")
        assert endpoint == "https://apim.net/openai"
        assert version == "2025-01-01-preview"

    def test_no_api_version_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RuntimeError when the endpoint has no api-version field."""
        monkeypatch.setenv(
            "TEST_ENDPOINT", "https://myhost.openai.azure.com/openai/deployments/gpt-4"
        )
        with pytest.raises(RuntimeError, match="doesn't contain valid api-version"):
            utils.parse_azure_endpoint("TEST_ENDPOINT")

    def test_no_deployment_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Endpoint without /deployments/ yields deployment_name=None."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://myhost.openai.azure.com/openai?api-version=2024-06-01",
        )
        endpoint, version, deployment = utils.parse_azure_endpoint_parts(
            "TEST_ENDPOINT"
        )
        assert endpoint == "https://myhost.openai.azure.com"
        assert version == "2024-06-01"
        assert deployment is None

    def test_apim_style_deployment_extracted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """APIM-style URL: prefix before /openai kept, deployment name extracted."""
        monkeypatch.setenv(
            "TEST_ENDPOINT",
            "https://apim.net/openai/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview",
        )
        endpoint, version, deployment = utils.parse_azure_endpoint_parts(
            "TEST_ENDPOINT"
        )
        assert endpoint == "https://apim.net/openai"
        assert version == "2025-01-01-preview"
        assert deployment == "gpt-4o"


class TestReindent:
    def test_four_spaces_to_two(self) -> None:
        text = "def foo():\n    pass\n    return 1"
        result = utils.reindent(text)
        assert result == "def foo():\n  pass\n  return 1"

    def test_empty_string(self) -> None:
        assert utils.reindent("") == ""

    def test_no_indent(self) -> None:
        assert utils.reindent("hello") == "hello"

    def test_nested_indent(self) -> None:
        text = "a\n    b\n        c"
        result = utils.reindent(text)
        assert result == "a\n  b\n    c"


class TestTimelog:
    def test_verbose_false_no_output(self) -> None:
        buf = StringIO()
        with redirect_stderr(buf):
            with utils.timelog("silent", verbose=False):
                pass
        assert buf.getvalue() == ""

    def test_verbose_true_shows_label(self) -> None:
        buf = StringIO()
        with redirect_stderr(buf):
            with utils.timelog("myblock", verbose=True):
                pass
        assert "myblock" in buf.getvalue()


class TestListDiff:
    def test_identical_lists(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            utils.list_diff("a", [1, 2, 3], "b", [1, 2, 3], max_items=10)
        out = buf.getvalue()
        assert "1" in out
        assert "2" in out

    def test_different_lists(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            utils.list_diff("left", [1, 2], "right", [1, 3], max_items=10)
        assert buf.getvalue() != ""

    def test_no_max_items(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            utils.list_diff("a", [1], "b", [2], max_items=0)
        assert "1" in buf.getvalue() or "2" in buf.getvalue()

    def test_empty_lists(self) -> None:
        buf = StringIO()
        with redirect_stdout(buf):
            utils.list_diff("a", [], "b", [], max_items=10)
        # No output expected (nothing to diff)
        assert buf.getvalue() == ""


class TestGetAzureApiKey:
    def test_plain_key_returned_as_is(self) -> None:
        assert utils.get_azure_api_key("my-secret-key") == "my-secret-key"

    def test_uppercase_identity_not_plain(self) -> None:
        # "IDENTITY" as a plain key is not routed to token provider; only "identity"
        # (lowercased) triggers that path. Since we can't call the identity provider
        # in tests, just verify non-identity keys pass through unchanged.
        assert utils.get_azure_api_key("APIKEY123") == "APIKEY123"


class TestMakeAgent:
    def test_no_keys_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="Neither OPENAI_API_KEY"):
            utils.make_agent(str)


class TestResolveAzureModelName:
    def test_returns_model_name_when_no_deployment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://myhost.openai.azure.com/openai?api-version=2024-06-01",
        )
        result = utils.resolve_azure_model_name("gpt-4o")
        assert result == "gpt-4o"

    def test_returns_deployment_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/my-deploy/chat?api-version=2024-06-01",
        )
        result = utils.resolve_azure_model_name("gpt-4o")
        assert result == "my-deploy"
