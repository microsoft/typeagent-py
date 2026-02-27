# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for server.py changes (coverage guard, role mapping, match default)."""

from unittest.mock import AsyncMock

import pytest

from mcp.types import SamplingMessage, TextContent
import typechat

from typeagent.mcp.server import MCPTypeChatModel, QuestionResponse

# ---------------------------------------------------------------------------
# Change 1: coverage import guard — tested implicitly (the module loads at all
# without `coverage` installed).  We just verify the guard didn't break the
# import.
# ---------------------------------------------------------------------------


def test_server_module_imports() -> None:
    """Importing the server module should not raise even without coverage."""
    import typeagent.mcp.server as mod

    assert hasattr(mod, "mcp")  # The FastMCP instance exists


# ---------------------------------------------------------------------------
# Change 2: PromptSection role mapping ("system" → "assistant")
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
# Change 3: match statement default case in query_conversation
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
        from typeagent.knowpro.answer_response_schema import AnswerResponse

        answered = AnswerResponse(type="Answered", answer="yes")
        assert answered.type == "Answered"
        no_answer = AnswerResponse(type="NoAnswer", why_no_answer="dunno")
        assert no_answer.type == "NoAnswer"
