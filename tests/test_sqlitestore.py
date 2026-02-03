# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import AsyncGenerator
from dataclasses import field

import pytest
import pytest_asyncio

from pydantic.dataclasses import dataclass

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.convsettings import (
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.knowpro.interfaces import (
    IMessage,
    SemanticRef,
    TextLocation,
    TextRange,
    Topic,
)
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.storage import SqliteStorageProvider


# Dummy IMessage for testing
@dataclass
class DummyMessage(IMessage):
    text_chunks: list[str]
    tags: list[str] = field(default_factory=list)
    timestamp: str | None = None

    def get_knowledge(self) -> KnowledgeResponse:
        raise NotImplementedError("Should not be called")


@pytest_asyncio.fixture
async def dummy_sqlite_storage_provider(
    temp_db_path: str, embedding_model: AsyncEmbeddingModel
) -> AsyncGenerator[SqliteStorageProvider[DummyMessage], None]:
    """Create a SqliteStorageProvider for testing."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = SqliteStorageProvider(
        db_path=temp_db_path,
        message_type=DummyMessage,
        message_text_index_settings=message_text_settings,
        related_term_index_settings=related_terms_settings,
    )
    yield provider
    await provider.close()


def make_dummy_semantic_ref(ordinal: int = 0) -> SemanticRef:
    # Minimal valid Topic for knowledge
    topic = Topic(text="dummy_topic")
    # Minimal valid TextLocation and TextRange for range
    location = TextLocation(message_ordinal=0)
    text_range = TextRange(start=location)
    return SemanticRef(
        semantic_ref_ordinal=ordinal,
        range=text_range,
        knowledge=topic,
    )


@pytest.mark.asyncio
async def test_sqlite_message_collection_append_and_get(
    dummy_sqlite_storage_provider: SqliteStorageProvider[DummyMessage],
):
    store = await dummy_sqlite_storage_provider.get_message_collection()
    msg = DummyMessage(["foo"])
    await store.append(msg)
    assert await store.size() == 1
    loaded = await store.get_item(0)
    assert loaded.text_chunks == ["foo"]
    with pytest.raises(IndexError):
        _ = await store.get_item(999)
    with pytest.raises(TypeError):
        _ = await store.get_item("bad")  # type: ignore  # Tests runtime behavior


@pytest.mark.asyncio
async def test_sqlite_message_collection_iter(
    dummy_sqlite_storage_provider: SqliteStorageProvider[DummyMessage],
):
    collection = await dummy_sqlite_storage_provider.get_message_collection()
    msgs = [DummyMessage([f"msg{i}"]) for i in range(3)]
    for m in msgs:
        await collection.append(m)
    assert [m.text_chunks[0] async for m in collection] == ["msg0", "msg1", "msg2"]


@pytest.mark.asyncio
async def test_sqlite_semantic_ref_collection_append_and_get(
    dummy_sqlite_storage_provider: SqliteStorageProvider[DummyMessage],
):
    collection = await dummy_sqlite_storage_provider.get_semantic_ref_collection()
    ref = make_dummy_semantic_ref(123)
    await collection.append(ref)
    assert await collection.size() == 1
    loaded = await collection.get_item(123)
    assert loaded.semantic_ref_ordinal == 123
    with pytest.raises(IndexError):
        _ = await collection.get_item(999)
    with pytest.raises(TypeError):
        _ = await collection.get_item("bad")  # type: ignore  # Tests runtime behavior


@pytest.mark.asyncio
async def test_sqlite_semantic_ref_collection_iter(
    dummy_sqlite_storage_provider: SqliteStorageProvider[DummyMessage],
):
    collection = await dummy_sqlite_storage_provider.get_semantic_ref_collection()
    refs = [make_dummy_semantic_ref(i) for i in range(2)]
    for r in refs:
        await collection.append(r)
    assert [r.semantic_ref_ordinal async for r in collection] == [0, 1]


@pytest.mark.asyncio
async def test_sqlite_timestamp_index(
    dummy_sqlite_storage_provider: SqliteStorageProvider[DummyMessage],
):
    """Test SqliteTimestampToTextRangeIndex functionality."""
    from datetime import datetime

    from typeagent.knowpro.interfaces import DateRange

    # Set up database with some messages
    message_collection = await dummy_sqlite_storage_provider.get_message_collection()

    # Add test messages
    messages = [
        DummyMessage(text_chunks=["Hello world"], tags=["test"]),
        DummyMessage(text_chunks=["Goodbye world"], tags=["test"]),
        DummyMessage(text_chunks=["Another message"], tags=["test"]),
    ]

    for msg in messages:
        await message_collection.append(msg)

    # Create timestamp index
    timestamp_index = await dummy_sqlite_storage_provider.get_timestamp_index()

    # Test add_timestamp - use actual message ordinals from the database
    test_timestamps = [
        "2024-01-01T10:00:00Z",
        "2024-01-01T11:00:00Z",
        "2024-01-01T12:00:00Z",
    ]

    for i, timestamp in enumerate(test_timestamps):
        result = await timestamp_index.add_timestamp(i, timestamp)
        print(f"add_timestamp({i}, {timestamp}) = {result}")
        assert result is True

    # Test add_timestamps (will overwrite some of the above)
    more_timestamps = [(0, "2024-01-01T09:00:00Z"), (2, "2024-01-01T13:00:00Z")]
    await timestamp_index.add_timestamps(more_timestamps)

    # Test lookup_range - point query
    point_date = datetime.fromisoformat("2024-01-01T09:00:00+00:00")
    point_range = DateRange(start=point_date, end=None)
    results = await timestamp_index.lookup_range(point_range)
    assert len(results) == 1
    assert results[0].timestamp == "2024-01-01T09:00:00Z"  # Z format
    assert results[0].range.start.message_ordinal == 0  # Message ordinal 0

    # Test lookup_range - range query
    start_date = datetime.fromisoformat("2024-01-01T10:00:00Z")
    end_date = datetime.fromisoformat("2024-01-01T12:00:00Z")
    range_query = DateRange(start=start_date, end=end_date)
    results = await timestamp_index.lookup_range(range_query)

    # Should find messages with timestamps: 11:00 (09:00 is before range, 12:00 is excluded, 13:00 is after)
    assert len(results) == 1
    assert results[0].timestamp == "2024-01-01T11:00:00Z"  # Z format
    assert results[0].range.start.message_ordinal == 1

    # Test empty range
    empty_start = datetime.fromisoformat("2024-02-01T00:00:00Z")
    empty_end = datetime.fromisoformat("2024-02-01T23:59:59Z")
    empty_range = DateRange(start=empty_start, end=empty_end)
    empty_results = await timestamp_index.lookup_range(empty_range)
    assert len(empty_results) == 0


@pytest.mark.asyncio
async def test_sqlite_nested_transaction_error(
    dummy_sqlite_storage_provider: SqliteStorageProvider[DummyMessage],
):
    """Verify that nested transactions are handled gracefully with a clear error."""
    provider = dummy_sqlite_storage_provider

    # First transaction should work
    async with provider:
        # Try to start a nested transaction - should raise a clear RuntimeError
        with pytest.raises(RuntimeError, match="Cannot start a new transaction"):
            async with provider:
                pass
