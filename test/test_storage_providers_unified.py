# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unified tests for storage providers.

These tests run against both MemoryStorageProvider and SqliteStorageProvider
to ensure behavioral parity across implementations.
"""

import pytest
from dataclasses import field
from pydantic.dataclasses import dataclass

from fixtures import needs_auth, storage_provider_type, embedding_model, temp_db_path
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.knowpro.interfaces import (
    DateRange,
    Datetime,
    IMessage,
    IStorageProvider,
    SemanticRef,
    TextLocation,
    TextRange,
    Topic,
)
from typeagent.knowpro.messageindex import MessageTextIndexSettings
from typeagent.knowpro.reltermsindex import RelatedTermIndexSettings
from typeagent.storage.memorystore import MemoryStorageProvider
from typeagent.storage.sqlitestore import SqliteStorageProvider


# Test message for unified testing
@dataclass
class DummyTestMessage(IMessage):
    text_chunks: list[str]
    tags: list[str] = field(default_factory=list)

    def get_knowledge(self) -> KnowledgeResponse:
        raise NotImplementedError("Should not be called")


def make_test_semantic_ref(ordinal: int = 0) -> SemanticRef:
    """Create a minimal valid SemanticRef for testing."""
    topic = Topic(text=f"test_topic_{ordinal}")
    location = TextLocation(message_ordinal=0)
    text_range = TextRange(start=location)
    return SemanticRef(
        semantic_ref_ordinal=ordinal,
        range=text_range,
        knowledge=topic,
    )


@pytest.mark.asyncio
async def test_all_index_creation(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that all 6 index types are created and accessible in both providers."""
    storage_provider, provider_type = storage_provider_type

    # Test all index types are created and return proper interface objects
    conv_index = await storage_provider.get_semantic_ref_index()
    assert conv_index is not None
    assert hasattr(conv_index, "lookup_term")  # Basic interface check

    prop_index = await storage_provider.get_property_index()
    assert prop_index is not None
    assert hasattr(prop_index, "lookup_property")  # Basic interface check

    time_index = await storage_provider.get_timestamp_index()
    assert time_index is not None
    assert hasattr(time_index, "lookup_range")  # Basic interface check

    msg_index = await storage_provider.get_message_text_index()
    assert msg_index is not None
    assert hasattr(msg_index, "lookup_messages")  # Basic interface check

    rel_index = await storage_provider.get_related_terms_index()
    assert rel_index is not None
    assert hasattr(rel_index, "aliases")  # Basic interface check

    threads = await storage_provider.get_conversation_threads()
    assert threads is not None
    assert hasattr(threads, "threads")  # Basic interface check


@pytest.mark.asyncio
async def test_index_persistence(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that same index instance is returned across calls in both providers."""
    storage_provider, provider_type = storage_provider_type

    # All index types should return same instance across calls
    conv1 = await storage_provider.get_semantic_ref_index()
    conv2 = await storage_provider.get_semantic_ref_index()
    assert conv1 is conv2

    prop1 = await storage_provider.get_property_index()
    prop2 = await storage_provider.get_property_index()
    assert prop1 is prop2

    time1 = await storage_provider.get_timestamp_index()
    time2 = await storage_provider.get_timestamp_index()
    assert time1 is time2


@pytest.mark.asyncio
async def test_message_collection_basic_operations(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test basic message collection operations work identically in both providers."""
    storage_provider, provider_type = storage_provider_type

    # Create message collection
    collection = await storage_provider.get_message_collection(DummyTestMessage)

    # Test initial state
    assert await collection.size() == 0

    # Test adding messages
    msg1 = DummyTestMessage(["hello", "world"])
    msg2 = DummyTestMessage(["foo", "bar"])

    await collection.append(msg1)
    assert await collection.size() == 1

    await collection.append(msg2)
    assert await collection.size() == 2

    # Test retrieval
    retrieved_msg = await collection.get_item(0)
    assert isinstance(retrieved_msg, DummyTestMessage)
    assert retrieved_msg.text_chunks == ["hello", "world"]

    # Test slice
    slice_result = await collection.get_slice(0, 2)
    assert len(slice_result) == 2
    assert slice_result[0].text_chunks == ["hello", "world"]
    assert slice_result[1].text_chunks == ["foo", "bar"]

    # Test iteration
    collection_list = [item async for item in collection]
    assert len(collection_list) == 2
    assert collection_list[0].text_chunks == ["hello", "world"]


@pytest.mark.asyncio
async def test_semantic_ref_collection_basic_operations(
    storage_provider_type, needs_auth
):
    """Test basic semantic ref collection operations work identically in both providers."""
    storage_provider, provider_type = storage_provider_type

    # Create semantic ref collection
    collection = await storage_provider.get_semantic_ref_collection()

    # Test initial state
    assert await collection.size() == 0

    # Test adding semantic refs
    ref1 = make_test_semantic_ref(1)
    ref2 = make_test_semantic_ref(2)

    await collection.append(ref1)
    assert await collection.size() == 1

    await collection.append(ref2)
    assert await collection.size() == 2

    # Test retrieval - SQLite uses ordinal as ID, memory uses index
    # Try to get the first semantic ref we added
    if provider_type == "sqlite":
        # For SQLite, use the semantic_ref_ordinal as the ID
        retrieved_ref = await collection.get_item(1)  # ordinal 1
    else:
        # For memory, use index
        retrieved_ref = await collection.get_item(0)  # first item

    assert isinstance(retrieved_ref, SemanticRef)
    assert retrieved_ref.semantic_ref_ordinal == 1

    # Test iteration
    ref_list = [item async for item in collection]
    assert len(ref_list) == 2


@pytest.mark.asyncio
async def test_conversation_index_behavior_parity(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth
):
    """Test that conversation index behaves identically in both providers."""
    storage_provider, provider_type = storage_provider_type

    conv_index = await storage_provider.get_semantic_ref_index()

    # Test empty state
    empty_results = await conv_index.lookup_term("nonexistent")
    assert empty_results is None or len(empty_results) == 0

    # Test adding terms (this tests the interface is working)
    # Note: We can't test deep index behavior without full conversation setup,
    # but we can verify the interfaces work identically


@pytest.mark.asyncio
async def test_timestamp_index_behavior_parity(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that timestamp index behaves identically in both providers."""
    storage_provider, provider_type = storage_provider_type

    time_index = await storage_provider.get_timestamp_index()

    # Test empty lookup_range interface
    start_time = Datetime.fromisoformat("2024-01-01T00:00:00")
    end_time = Datetime.fromisoformat("2024-01-02T00:00:00")
    date_range = DateRange(start=start_time, end=end_time)

    empty_results = time_index.lookup_range(date_range)
    assert isinstance(empty_results, list)
    assert len(empty_results) == 0


@pytest.mark.asyncio
async def test_message_text_index_interface_parity(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that message text index interface works identically in both providers."""
    storage_provider, provider_type = storage_provider_type

    msg_index = await storage_provider.get_message_text_index()

    # Test empty lookup_messages
    empty_results = await msg_index.lookup_messages("nonexistent query", 10)
    assert isinstance(empty_results, list)
    assert len(empty_results) == 0


@pytest.mark.asyncio
async def test_related_terms_index_interface_parity(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that related terms index interface works identically in both providers."""
    storage_provider, provider_type = storage_provider_type

    rel_index = await storage_provider.get_related_terms_index()

    # Test interface properties
    aliases = rel_index.aliases
    assert aliases is not None

    # Test empty lookup via aliases
    empty_results = aliases.lookup_term("nonexistent")
    assert empty_results is None or len(empty_results) == 0


@pytest.mark.asyncio
async def test_conversation_threads_interface_parity(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that conversation threads interface works identically in both providers."""
    storage_provider, provider_type = storage_provider_type

    threads = await storage_provider.get_conversation_threads()

    # Test initial empty state
    assert len(threads.threads) == 0


# Cross-provider validation tests
@pytest.mark.asyncio
async def test_cross_provider_message_collection_equivalence(
    embedding_model, temp_db_path, needs_auth
):
    """Test that both providers handle message collections equivalently."""
    # Create both providers with identical settings
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    memory_provider = await MemoryStorageProvider.create(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )

    sqlite_provider = await SqliteStorageProvider.create(
        message_text_settings, related_terms_settings, temp_db_path
    )

    try:
        # Create collections in both
        memory_collection = await memory_provider.get_message_collection(
            DummyTestMessage
        )
        sqlite_collection = await sqlite_provider.get_message_collection(
            DummyTestMessage
        )

        # Add identical data to both
        test_messages = [
            DummyTestMessage(["hello", "world"]),
            DummyTestMessage(["foo", "bar", "baz"]),
            DummyTestMessage(["test", "message"]),
        ]

        for msg in test_messages:
            await memory_collection.append(msg)
            await sqlite_collection.append(msg)

        # Verify both have same size
        assert await memory_collection.size() == await sqlite_collection.size()

        # Verify both return equivalent data
        for i in range(len(test_messages)):
            memory_msg = await memory_collection.get_item(i)
            sqlite_msg = await sqlite_collection.get_item(i)
            assert memory_msg.text_chunks == sqlite_msg.text_chunks

        # Verify slices are equivalent
        memory_slice = await memory_collection.get_slice(0, 2)
        sqlite_slice = await sqlite_collection.get_slice(0, 2)
        assert len(memory_slice) == len(sqlite_slice)
        for mem_msg, sql_msg in zip(memory_slice, sqlite_slice):
            assert mem_msg.text_chunks == sql_msg.text_chunks

    finally:
        await sqlite_provider.close()
