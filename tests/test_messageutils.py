# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typeagent.knowpro.interfaces import TextLocation, TextRange
from typeagent.knowpro.messageutils import (
    get_message_chunk_batch,
    text_range_from_message_chunk,
)
from typeagent.storage.memory.collections import MemoryMessageCollection

from conftest import FakeMessage


class TestTextRangeFromMessageChunk:
    def test_default_chunk_ordinal(self) -> None:
        tr = text_range_from_message_chunk(message_ordinal=3)
        assert tr.start == TextLocation(3, 0)
        assert tr.end is None

    def test_explicit_chunk_ordinal(self) -> None:
        tr = text_range_from_message_chunk(message_ordinal=5, chunk_ordinal=2)
        assert tr.start == TextLocation(5, 2)
        assert tr.end is None

    def test_returns_text_range(self) -> None:
        tr = text_range_from_message_chunk(0)
        assert isinstance(tr, TextRange)


class TestGetMessageChunkBatch:
    @pytest.mark.asyncio
    async def test_empty_collection(self) -> None:
        messages: MemoryMessageCollection[FakeMessage] = MemoryMessageCollection()
        batches = await get_message_chunk_batch(messages, 0, 10)
        assert batches == []

    @pytest.mark.asyncio
    async def test_single_message_single_chunk(self) -> None:
        messages: MemoryMessageCollection[FakeMessage] = MemoryMessageCollection(
            [FakeMessage("hello")]
        )
        batches = await get_message_chunk_batch(messages, 0, 10)
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert batches[0][0] == TextLocation(0, 0)

    @pytest.mark.asyncio
    async def test_message_with_multiple_chunks(self) -> None:
        messages: MemoryMessageCollection[FakeMessage] = MemoryMessageCollection(
            [FakeMessage(["chunk0", "chunk1", "chunk2"])]
        )
        batches = await get_message_chunk_batch(messages, 0, 10)
        assert len(batches) == 1
        locs = batches[0]
        assert locs == [TextLocation(0, 0), TextLocation(0, 1), TextLocation(0, 2)]

    @pytest.mark.asyncio
    async def test_batch_size_splits_across_messages(self) -> None:
        messages: MemoryMessageCollection[FakeMessage] = MemoryMessageCollection(
            [FakeMessage("a"), FakeMessage("b"), FakeMessage("c")]
        )
        batches = await get_message_chunk_batch(messages, 0, batch_size=2)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1

    @pytest.mark.asyncio
    async def test_exact_batch_size(self) -> None:
        messages: MemoryMessageCollection[FakeMessage] = MemoryMessageCollection(
            [FakeMessage("a"), FakeMessage("b")]
        )
        batches = await get_message_chunk_batch(messages, 0, batch_size=2)
        assert len(batches) == 1
        assert len(batches[0]) == 2

    @pytest.mark.asyncio
    async def test_start_offset_skips_earlier_messages(self) -> None:
        messages: MemoryMessageCollection[FakeMessage] = MemoryMessageCollection(
            [FakeMessage("skip"), FakeMessage("include")]
        )
        batches = await get_message_chunk_batch(messages, 1, batch_size=10)
        assert len(batches) == 1
        assert batches[0][0] == TextLocation(1, 0)
