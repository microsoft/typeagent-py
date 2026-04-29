# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typeagent.knowpro.convutils import (
    get_time_range_for_conversation,
    get_time_range_prompt_section_for_conversation,
)

from conftest import FakeConversation, FakeMessage


class TestGetTimeRangeForConversation:
    @pytest.mark.asyncio
    async def test_empty_conversation_returns_none(self) -> None:
        conv = FakeConversation(messages=[])
        result = await get_time_range_for_conversation(conv)
        assert result is None

    @pytest.mark.asyncio
    async def test_message_without_timestamp_returns_none(self) -> None:
        msg = FakeMessage("hello")  # no message_ordinal → timestamp=None
        conv = FakeConversation(messages=[msg])
        result = await get_time_range_for_conversation(conv)
        assert result is None

    @pytest.mark.asyncio
    async def test_single_message_with_timestamp(self) -> None:
        msg = FakeMessage("hello", message_ordinal=0)
        conv = FakeConversation(messages=[msg])
        result = await get_time_range_for_conversation(conv)
        assert result is not None
        assert result.start.isoformat().startswith("2020-01-01T00")

    @pytest.mark.asyncio
    async def test_multiple_messages_range_start_end(self) -> None:
        msgs = [FakeMessage(f"msg{i}", message_ordinal=i) for i in range(3)]
        conv = FakeConversation(messages=msgs)
        result = await get_time_range_for_conversation(conv)
        assert result is not None
        assert result.start < result.end  # type: ignore[operator]


class TestGetTimeRangePromptSection:
    @pytest.mark.asyncio
    async def test_no_timestamps_returns_none(self) -> None:
        conv = FakeConversation(messages=[FakeMessage("hello")])
        result = await get_time_range_prompt_section_for_conversation(conv)
        assert result is None

    @pytest.mark.asyncio
    async def test_with_timestamps_returns_prompt_section(self) -> None:
        msgs = [FakeMessage(f"msg{i}", message_ordinal=i) for i in range(2)]
        conv = FakeConversation(messages=msgs)
        result = await get_time_range_prompt_section_for_conversation(conv)
        assert result is not None
        assert result["role"] == "system"
        assert "CONVERSATION TIME RANGE" in result["content"]
        assert "2020-01-01" in result["content"]
