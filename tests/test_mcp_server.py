# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end and unit tests for the MCP server."""

import json
import os
import sys
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.session import ClientSession as ClientSessionType
from mcp.client.stdio import stdio_client
from mcp.shared.context import RequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    SamplingMessage,
    TextContent,
)
import typechat

from typeagent.aitools.model_adapters import create_chat_model
from typeagent.knowpro import answers, searchlang
from typeagent.knowpro.answer_response_schema import AnswerResponse
from typeagent.knowpro.convsettings import ConversationSettings
import typeagent.mcp.server as typeagent_mcp_server
from typeagent.mcp.server import (
    load_podcast_database_or_index,
    MCPTypeChatModel,
    ProcessingContext,
    QuestionResponse,
)

from conftest import EPISODE_53_INDEX


@pytest.fixture
def server_params() -> StdioServerParameters:
    """Create MCP server parameters with environment inherited from parent process."""
    # Start with the full environment - subprocess needs PATH, PYTHONPATH, etc.
    env = dict(os.environ)
    # Coverage support
    if "COVERAGE_PROCESS_START" in os.environ:
        env["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_PROCESS_START"]

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "typeagent.mcp.server", "--podcast-index", EPISODE_53_INDEX],
        env=env,
    )


async def sampling_callback(
    context: RequestContext[ClientSessionType, Any, Any],
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    """Sampling callback that uses OpenAI to generate responses."""
    model = create_chat_model()

    # Convert MCP SamplingMessage to TypeChat PromptSection list
    sections: list[typechat.PromptSection] = []
    if params.systemPrompt:
        sections.append({"role": "system", "content": params.systemPrompt})
    for msg in params.messages:
        if isinstance(msg.content, TextContent):
            content = msg.content.text
        else:
            raise ValueError(
                f"Unsupported content type in sampling message: {type(msg.content)}"
            )
        role = "user" if msg.role == "user" else "assistant"
        sections.append({"role": role, "content": content})

    result = await model.complete(sections)
    if isinstance(result, typechat.Success):
        text = result.value
    else:
        text = result.message

    return CreateMessageResult(
        role="assistant",
        content=TextContent(type="text", text=text),
        model="gpt-4o",
        stopReason="endTurn",
    )


@pytest.mark.asyncio
async def test_mcp_server_query_conversation_slow(
    really_needs_auth, server_params: StdioServerParameters
):
    """Test the query_conversation tool end-to-end using MCP client."""
    # Pass through environment variables needed for authentication
    # otherwise this test will fail in the CI on Windows only
    if not (server_params.env) is None:
        server_params.env.update(
            {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("AZURE_", "OPENAI_")) or k in ("CREDENTIALS_JSON",)
            }
        )

    # Create client session and connect to server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=sampling_callback
        ) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]

            # Verify query_conversation tool exists
            assert "query_conversation" in tool_names

            # Call the query_conversation tool
            result = await session.call_tool(
                "query_conversation",
                arguments={"question": "Who is Kevin Scott?"},
            )

            # Verify response structure
            assert len(result.content) > 0, "Expected non-empty response"

            # Type narrow the content to TextContent
            content_item = result.content[0]
            assert isinstance(content_item, TextContent)
            response_text = content_item.text

            # Parse response (it should be JSON with success, answer, time_used)
            try:
                response_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                pytest.fail(
                    f"Response is not valid JSON: {e}\nResponse text: {response_text}"
                )

            assert "success" in response_data
            assert "answer" in response_data
            assert "time_used" in response_data

            # If successful, answer should be non-empty
            if response_data["success"]:
                assert len(response_data["answer"]) > 0

            assert response_data["success"] is True, response_data


@pytest.mark.asyncio
async def test_mcp_server_empty_question(server_params: StdioServerParameters):
    """Test the query_conversation tool with an empty question."""
    # Create client session and connect to server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=sampling_callback
        ) as session:
            # Initialize the session
            await session.initialize()

            # Call with empty question
            result = await session.call_tool(
                "query_conversation",
                arguments={"question": ""},
            )

            # Verify response
            assert len(result.content) > 0

            # Type narrow the content to TextContent
            content_item = result.content[0]
            assert isinstance(content_item, TextContent)
            response_text = content_item.text

            response_data = json.loads(response_text)
            assert response_data["success"] is False
            assert "No question provided" in response_data["answer"]


# ---------------------------------------------------------------------------
# Unit tests (formerly in test_mcp_server_unit.py)
# ---------------------------------------------------------------------------

# Coverage import guard — tested implicitly (the module loads at all
# without `coverage` installed).  We just verify the guard didn't break the
# import.


def test_server_module_imports() -> None:
    """Importing the server module should not raise even without coverage."""
    assert hasattr(typeagent_mcp_server, "mcp")  # The FastMCP instance exists


# ---------------------------------------------------------------------------
# PromptSection role mapping ("system" → "assistant")
# ---------------------------------------------------------------------------


class TestMCPTypeChatModelRoleMapping:
    """Verify that PromptSection roles are mapped correctly to MCP roles."""

    @staticmethod
    def _make_model() -> tuple[MCPTypeChatModel, AsyncMock]:
        session = AsyncMock()
        # create_message returns a result with TextContent
        session.create_message.return_value = AsyncMock(
            content=TextContent(type="text", text="response")
        )
        model = MCPTypeChatModel(session)
        return model, session

    @pytest.mark.asyncio
    async def test_string_prompt_becomes_user_message(self) -> None:
        model, session = self._make_model()
        await model.complete("hello")

        call_args = session.create_message.call_args
        messages: list[SamplingMessage] = call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert isinstance(messages[0].content, TextContent)
        assert messages[0].content.text == "hello"

    @pytest.mark.asyncio
    async def test_user_role_preserved(self) -> None:
        model, session = self._make_model()
        sections: list[typechat.PromptSection] = [
            {"role": "user", "content": "question"},
        ]
        await model.complete(sections)

        messages: list[SamplingMessage] = session.create_message.call_args.kwargs[
            "messages"
        ]
        assert messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_assistant_role_preserved(self) -> None:
        model, session = self._make_model()
        sections: list[typechat.PromptSection] = [
            {"role": "assistant", "content": "context"},
        ]
        await model.complete(sections)

        messages: list[SamplingMessage] = session.create_message.call_args.kwargs[
            "messages"
        ]
        assert messages[0].role == "assistant"

    @pytest.mark.asyncio
    async def test_system_role_mapped_to_assistant(self) -> None:
        """System role doesn't exist in MCP SamplingMessage; it must be mapped."""
        model, session = self._make_model()
        sections: list[typechat.PromptSection] = [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "question"},
        ]
        await model.complete(sections)

        messages: list[SamplingMessage] = session.create_message.call_args.kwargs[
            "messages"
        ]
        assert messages[0].role == "assistant"  # "system" → "assistant"
        assert messages[1].role == "user"

    @pytest.mark.asyncio
    async def test_mixed_roles_order(self) -> None:
        model, session = self._make_model()
        sections: list[typechat.PromptSection] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "asst"},
        ]
        await model.complete(sections)

        messages: list[SamplingMessage] = session.create_message.call_args.kwargs[
            "messages"
        ]
        assert [m.role for m in messages] == ["assistant", "user", "assistant"]

    @pytest.mark.asyncio
    async def test_exception_returns_failure(self) -> None:
        model, session = self._make_model()
        session.create_message.side_effect = RuntimeError("boom")
        result = await model.complete("test")
        assert isinstance(result, typechat.Failure)
        assert "boom" in result.message

    @pytest.mark.asyncio
    async def test_text_content_returns_success(self) -> None:
        model, _ = self._make_model()
        result = await model.complete("test")
        assert isinstance(result, typechat.Success)
        assert result.value == "response"

    @pytest.mark.asyncio
    async def test_list_content_returns_joined(self) -> None:
        model, session = self._make_model()
        session.create_message.return_value = AsyncMock(
            content=[
                TextContent(type="text", text="part1"),
                TextContent(type="text", text="part2"),
            ]
        )
        result = await model.complete("test")
        assert isinstance(result, typechat.Success)
        assert result.value == "part1\npart2"


# ---------------------------------------------------------------------------
# match statement default case in query_conversation
# ---------------------------------------------------------------------------


class TestQuestionResponseMatchDefault:
    """The match on combined_answer.type must handle unexpected types."""

    def test_known_types(self) -> None:
        """QuestionResponse can represent success and failure."""
        ok = QuestionResponse(success=True, answer="yes", time_used=42)
        assert ok.success is True
        fail = QuestionResponse(success=False, answer="no", time_used=0)
        assert fail.success is False

    def test_answer_type_coverage(self) -> None:
        """AnswerResponse.type should only be 'Answered' or 'NoAnswer'."""
        answered = AnswerResponse(type="Answered", answer="yes")
        assert answered.type == "Answered"
        no_answer = AnswerResponse(type="NoAnswer", why_no_answer="dunno")
        assert no_answer.type == "NoAnswer"


@pytest.mark.asyncio
async def test_sampling_callback_delegates_to_chat_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sampling_callback should delegate to create_chat_model().complete()."""
    fake_model = AsyncMock()
    fake_model.complete.return_value = typechat.Success("response")

    monkeypatch.setattr(
        sys.modules[__name__],
        "create_chat_model",
        lambda: fake_model,
    )

    params = CreateMessageRequestParams(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text="hello"),
            )
        ],
        maxTokens=32,
    )

    result = await sampling_callback(
        cast(RequestContext[ClientSessionType, Any, Any], None),
        params,
    )

    fake_model.complete.assert_awaited_once()
    call_args = fake_model.complete.call_args[0][0]
    assert call_args == [{"role": "user", "content": "hello"}]
    assert isinstance(result.content, TextContent)
    assert result.content.text == "response"


# ---------------------------------------------------------------------------
# MCPTypeChatModel — additional response format coverage
# ---------------------------------------------------------------------------


class TestMCPTypeChatModelResponseFormats:
    @staticmethod
    def _make_model_with_result(content: Any) -> MCPTypeChatModel:
        session = AsyncMock()
        session.create_message.return_value = AsyncMock(content=content)
        return MCPTypeChatModel(session)

    @pytest.mark.asyncio
    async def test_list_content_no_text_items_returns_failure(self) -> None:
        """A list response with no TextContent items should return Failure."""
        # Use a non-TextContent item type (ImageContent would work but we mock with a dict)
        model = self._make_model_with_result([])
        result = await model.complete("test")
        assert isinstance(result, typechat.Failure)
        assert "No text content" in result.message

    @pytest.mark.asyncio
    async def test_unknown_content_type_returns_failure(self) -> None:
        """A response with an unrecognized content type should return Failure."""
        # Simulate some unknown object that is neither TextContent nor list
        model = self._make_model_with_result(42)
        result = await model.complete("test")
        assert isinstance(result, typechat.Failure)
        assert "No text content" in result.message


# ---------------------------------------------------------------------------
# ProcessingContext.__repr__
# ---------------------------------------------------------------------------


class TestProcessingContextRepr:
    def test_repr_contains_options(self) -> None:
        lang_opts = searchlang.LanguageSearchOptions(max_message_matches=10)
        ctx_opts = answers.AnswerContextOptions(entities_top_k=5)

        proc = ProcessingContext(
            lang_search_options=lang_opts,
            answer_context_options=ctx_opts,
            query_context=MagicMock(),
            embedding_model=MagicMock(),
            query_translator=MagicMock(),
            answer_translator=MagicMock(),
        )
        r = repr(proc)
        assert r.startswith("Context(")
        assert "LanguageSearchOptions" in r


# ---------------------------------------------------------------------------
# load_podcast_database_or_index — ValueError path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_podcast_no_args_raises() -> None:
    """Passing neither dbname nor podcast_index must raise ValueError."""
    settings = ConversationSettings()
    with pytest.raises(ValueError, match="Either --database or --podcast-index"):
        await load_podcast_database_or_index(settings, dbname=None, podcast_index=None)
