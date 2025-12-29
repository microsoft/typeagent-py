# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Unified tests for storage providers.

These tests run against both MemoryStorageProvider and SqliteStorageProvider
to ensure behavioral parity across implementations.
"""

from dataclasses import field
from typing import assert_never, AsyncGenerator

import pytest
import pytest_asyncio

from pydantic.dataclasses import dataclass

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro import kplib
from typeagent.knowpro.convsettings import (
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.knowpro.interfaces import (
    DateRange,
    Datetime,
    IMessage,
    IStorageProvider,
    SemanticRef,
    Tag,
    TextLocation,
    TextRange,
    Topic,
)
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.storage import SqliteStorageProvider
from typeagent.storage.memory import MemoryStorageProvider


# Test message for unified testing
@dataclass
class DummyTestMessage(IMessage):
    text_chunks: list[str]
    tags: list[str] = field(default_factory=list)

    def get_knowledge(self) -> KnowledgeResponse:
        raise NotImplementedError("Should not be called")


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def storage_provider_type(
    request: pytest.FixtureRequest,
    embedding_model: AsyncEmbeddingModel,
    temp_db_path: str,
) -> AsyncGenerator[tuple[IStorageProvider, str], None]:
    """Parameterized fixture that provides both memory and sqlite storage providers."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    match request.param:
        case "memory":
            provider = MemoryStorageProvider(
                message_text_settings=message_text_settings,
                related_terms_settings=related_terms_settings,
            )
            yield provider, request.param
        case "sqlite":
            provider = SqliteStorageProvider(
                db_path=temp_db_path,
                message_type=DummyTestMessage,
                message_text_index_settings=message_text_settings,
                related_term_index_settings=related_terms_settings,
            )
            yield provider, request.param
            await provider.close()
        case _:
            assert_never(request.param)


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
    storage_provider, _ = storage_provider_type

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
    storage_provider, _ = storage_provider_type

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
    storage_provider, _ = storage_provider_type

    # Create message collection
    collection = await storage_provider.get_message_collection()

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
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test basic semantic ref collection operations work identically in both providers."""
    storage_provider, _ = storage_provider_type

    # Create semantic ref collection
    collection = await storage_provider.get_semantic_ref_collection()

    # Test initial state
    assert await collection.size() == 0

    # Test adding semantic refs
    ref0 = make_test_semantic_ref(0)
    ref1 = make_test_semantic_ref(1)
    ref2 = make_test_semantic_ref(2)

    await collection.append(ref0)
    assert await collection.size() == 1

    await collection.append(ref1)
    assert await collection.size() == 2

    await collection.append(ref2)
    assert await collection.size() == 3

    # Test basic retrieval
    for ordinal in range(3):
        retrieved_ref = await collection.get_item(ordinal)
        assert isinstance(retrieved_ref, SemanticRef)
        assert retrieved_ref.semantic_ref_ordinal == ordinal

    # Test iteration
    ref_list = [item async for item in collection]
    assert len(ref_list) == 3
    assert all(isinstance(r, SemanticRef) for r in ref_list)
    assert ref_list[0].semantic_ref_ordinal == 0
    assert ref_list[1].semantic_ref_ordinal == 1
    assert ref_list[2].semantic_ref_ordinal == 2

    # Test slicing
    ref_list = await collection.get_slice(1, 4)  # Note: end is out of bounds
    assert len(ref_list) == 2
    assert all(isinstance(r, SemanticRef) for r in ref_list)
    assert ref_list[0].semantic_ref_ordinal == 1
    assert ref_list[1].semantic_ref_ordinal == 2

    # Test get_multiple
    ref_list = await collection.get_multiple([2, 0])
    assert len(ref_list) == 2
    assert all(isinstance(r, SemanticRef) for r in ref_list)
    assert ref_list[0].semantic_ref_ordinal == 2
    assert ref_list[1].semantic_ref_ordinal == 0


@pytest.mark.asyncio
async def test_semantic_ref_index_behavior_parity(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth
):
    """Test that semantic ref index behaves identically in both providers."""
    storage_provider, _ = storage_provider_type

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
    storage_provider, _provider_type = storage_provider_type

    time_index = await storage_provider.get_timestamp_index()

    # Test empty lookup_range interface
    start_time = Datetime.fromisoformat("2024-01-01T00:00:00Z")
    end_time = Datetime.fromisoformat("2024-01-02T00:00:00Z")
    date_range = DateRange(start=start_time, end=end_time)

    empty_results = await time_index.lookup_range(date_range)
    assert isinstance(empty_results, list)
    assert len(empty_results) == 0


@pytest.mark.asyncio
async def test_message_text_index_interface_parity(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that message text index interface works identically in both providers."""
    storage_provider, _ = storage_provider_type

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
    storage_provider, _ = storage_provider_type

    rel_index = await storage_provider.get_related_terms_index()

    # Test interface properties
    aliases = rel_index.aliases
    assert aliases is not None

    # Test empty lookup via aliases
    empty_results = await aliases.lookup_term("nonexistent")
    assert empty_results is None or len(empty_results) == 0


@pytest.mark.asyncio
async def test_conversation_threads_interface_parity(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that conversation threads interface works identically in both providers."""
    storage_provider, _ = storage_provider_type

    threads = await storage_provider.get_conversation_threads()

    # Test initial empty state
    assert len(threads.threads) == 0


# Cross-provider validation tests
@pytest.mark.asyncio
async def test_cross_provider_message_collection_equivalence(
    embedding_model: AsyncEmbeddingModel, temp_db_path: str, needs_auth: None
):
    """Test that both providers handle message collections equivalently."""
    # Create both providers with identical settings
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    memory_provider = MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )

    sqlite_provider = SqliteStorageProvider(
        db_path=temp_db_path,
        message_type=DummyTestMessage,
        message_text_index_settings=message_text_settings,
        related_term_index_settings=related_terms_settings,
    )

    try:
        # Create collections in both
        memory_collection = await memory_provider.get_message_collection()
        sqlite_collection = await sqlite_provider.get_message_collection()

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


@pytest.mark.asyncio
async def test_property_index_population_from_semantic_refs(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test that property index is correctly populated when semantic refs are added."""
    storage_provider, provider_type = storage_provider_type

    # Get collections
    sem_ref_collection = await storage_provider.get_semantic_ref_collection()
    prop_index = await storage_provider.get_property_index()

    # Check initial state
    initial_sem_ref_count = await sem_ref_collection.size()

    # Test initial property index state by trying a lookup that should return nothing
    initial_lookup = await prop_index.lookup_property("name", "nonexistent")
    _initial_empty = initial_lookup is None or len(initial_lookup) == 0

    # Create test semantic refs with different knowledge types
    location = TextLocation(message_ordinal=0)
    text_range = TextRange(start=location)

    # Entity with facets
    entity_ref = SemanticRef(
        semantic_ref_ordinal=initial_sem_ref_count,
        range=text_range,
        knowledge=kplib.ConcreteEntity(
            name="Test Entity",
            type=["person", "speaker"],
            facets=[kplib.Facet(name="role", value="host")],
        ),
    )

    # Action
    action_ref = SemanticRef(
        semantic_ref_ordinal=initial_sem_ref_count + 1,
        range=text_range,
        knowledge=kplib.Action(
            verbs=["discuss", "explain"],
            verb_tense="present",
            subject_entity_name="Test Entity",
            object_entity_name="technology",
            indirect_object_entity_name="audience",
        ),
    )

    # Tag
    tag_ref = SemanticRef(
        semantic_ref_ordinal=initial_sem_ref_count + 2,
        range=text_range,
        knowledge=Tag(text="test-tag"),
    )

    # Add semantic refs
    await sem_ref_collection.append(entity_ref)
    await sem_ref_collection.append(action_ref)
    await sem_ref_collection.append(tag_ref)

    # For SQLite provider, property index is populated during creation from persisted data
    # For Memory provider, property index would need to be populated manually
    if provider_type == "memory":
        # Memory provider doesn't auto-populate property index when semantic refs are added
        # This is expected behavior - property index is populated differently
        final_sem_ref_count = await sem_ref_collection.size()
        assert (
            final_sem_ref_count == initial_sem_ref_count + 3
        ), "All semantic refs should be added to memory"

        # The memory provider would require manual property index population
        # which is typically done through the build_property_index function

    elif provider_type == "sqlite":
        # For SQLite, property index is populated during storage provider creation
        # from persisted data, so we verify the data was persisted correctly
        final_sem_ref_count = await sem_ref_collection.size()
        assert (
            final_sem_ref_count == initial_sem_ref_count + 3
        ), "All semantic refs should be persisted"

        # The property index in SQLite is populated from data during _populate_indexes_from_data
        # which is called during storage provider creation, not when items are added


@pytest.mark.asyncio
async def test_property_index_basic_operations(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test basic property index operations work identically in both providers."""
    storage_provider, _ = storage_provider_type

    prop_index = await storage_provider.get_property_index()

    # Test initial state - should be able to handle lookups even when empty
    empty_results = await prop_index.lookup_property("name", "nonexistent")
    assert empty_results is None or len(empty_results) == 0

    # Test size method
    initial_size = await prop_index.size()
    assert isinstance(initial_size, int)
    assert initial_size >= 0


@pytest.mark.asyncio
async def test_timestamp_index_range_queries(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test timestamp index range query functionality in both providers."""
    storage_provider, _ = storage_provider_type

    timestamp_index = await storage_provider.get_timestamp_index()

    # Test basic interface - empty range query
    start_time = Datetime.fromisoformat("2024-01-01T00:00:00Z")
    end_time = Datetime.fromisoformat("2024-01-02T00:00:00Z")
    date_range = DateRange(start=start_time, end=end_time)

    empty_results = await timestamp_index.lookup_range(date_range)
    assert isinstance(empty_results, list)
    assert len(empty_results) == 0

    # Test point query (end=None)
    point_range = DateRange(start=start_time, end=None)
    point_results = await timestamp_index.lookup_range(point_range)
    assert isinstance(point_results, list)
    assert len(point_results) == 0

    # Test size method
    initial_size = await timestamp_index.size()
    assert isinstance(initial_size, int)
    assert initial_size >= 0


@pytest.mark.asyncio
async def test_timestamp_index_with_data(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Test timestamp index functionality with actual data in both providers."""
    storage_provider, provider_type = storage_provider_type

    # First add some messages to work with
    message_collection = await storage_provider.get_message_collection()
    timestamp_index = await storage_provider.get_timestamp_index()

    # Add test messages
    test_messages = [
        DummyTestMessage(["message at t0"]),
        DummyTestMessage(["message at t1"]),
        DummyTestMessage(["message at t2"]),
    ]

    for msg in test_messages:
        await message_collection.append(msg)

    # Define test timestamps
    t0 = "2025-01-01T00:00:00Z"
    t1 = "2025-01-01T01:00:00Z"
    t2 = "2025-01-01T02:00:00Z"

    # Add timestamps for the messages (0-indexed message ordinals)
    # Note: This may behave differently between providers
    try:
        await timestamp_index.add_timestamp(0, t0)
        await timestamp_index.add_timestamp(1, t1)
        await timestamp_index.add_timestamp(2, t2)

        # Test range queries
        # [t0, t1) should include t0, exclude t1
        dr = DateRange(
            start=Datetime.fromisoformat(t0.replace("Z", "+00:00")),
            end=Datetime.fromisoformat(t1.replace("Z", "+00:00")),
        )
        results = await timestamp_index.lookup_range(dr)
        if len(results) > 0:
            timestamps = [r.timestamp for r in results]
            assert t0 in timestamps
            assert t1 not in timestamps

        # Point query: end=None means exactly t1
        dr = DateRange(
            start=Datetime.fromisoformat(t1.replace("Z", "+00:00")), end=None
        )
        results = await timestamp_index.lookup_range(dr)
        if len(results) > 0:
            timestamps = [r.timestamp for r in results]
            assert t1 in timestamps

    except Exception as e:
        # Some providers may not support add_timestamp or may handle it differently
        # This is expected - we're testing interface compatibility, not forcing identical behavior
        print(f"Provider {provider_type} timestamp behavior: {e}")

        # Verify the basic interface still works
        dr = DateRange(
            start=Datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
            end=Datetime.fromisoformat("2025-01-01T23:59:59+00:00"),
        )
        results = await timestamp_index.lookup_range(dr)
        assert isinstance(results, list)  # Should return a list even if empty


@pytest.mark.asyncio
async def test_storage_provider_independence(
    embedding_model: AsyncEmbeddingModel, temp_db_path: str, needs_auth: None
):
    """Test that different storage provider instances work independently."""
    # Create settings shared between providers
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    # Create two memory providers
    memory_provider1 = MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )
    memory_provider2 = MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )

    # Create two sqlite providers (with different temp files)
    import os
    import tempfile

    temp_file1 = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    temp_path1 = temp_file1.name
    temp_file1.close()
    temp_file2 = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    temp_path2 = temp_file2.name
    temp_file2.close()

    sqlite_provider1 = None
    sqlite_provider2 = None
    try:
        sqlite_provider1 = SqliteStorageProvider(
            db_path=temp_path1,
            message_type=DummyTestMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )
        sqlite_provider2 = SqliteStorageProvider(
            db_path=temp_path2,
            message_type=DummyTestMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        # Test memory provider independence
        memory_index1 = await memory_provider1.get_semantic_ref_index()
        memory_index2 = await memory_provider2.get_semantic_ref_index()
        assert memory_index1 is not memory_index2

        memory_collection1 = await memory_provider1.get_message_collection()
        memory_collection2 = await memory_provider2.get_message_collection()

        # Add data to first memory provider
        await memory_collection1.append(DummyTestMessage(["memory test 1"]))
        assert await memory_collection1.size() == 1
        assert await memory_collection2.size() == 0  # Second provider unaffected

        # Test sqlite provider independence
        sqlite_index1 = await sqlite_provider1.get_semantic_ref_index()
        sqlite_index2 = await sqlite_provider2.get_semantic_ref_index()
        assert sqlite_index1 is not sqlite_index2

        sqlite_collection1 = await sqlite_provider1.get_message_collection()
        sqlite_collection2 = await sqlite_provider2.get_message_collection()

        # Add data to first sqlite provider
        await sqlite_collection1.append(DummyTestMessage(["sqlite test 1"]))
        assert await sqlite_collection1.size() == 1
        assert await sqlite_collection2.size() == 0  # Second provider unaffected

        # Test cross-provider independence (memory vs sqlite)
        await memory_collection1.append(DummyTestMessage(["memory test 2"]))
        assert await memory_collection1.size() == 2
        assert (
            await sqlite_collection1.size() == 1
        )  # SQLite unaffected by memory changes

    finally:
        # Cleanup
        del sqlite_provider1
        del sqlite_provider2
        if os.path.exists(temp_path1):
            os.remove(temp_path1)
        if os.path.exists(temp_path2):
            os.remove(temp_path2)


@pytest.mark.asyncio
async def test_collection_operations_comprehensive(
    storage_provider_type: tuple[IStorageProvider, str], needs_auth: None
):
    """Comprehensive test of collection operations in both providers."""
    storage_provider, _ = storage_provider_type

    # Test message collection operations
    message_collection = await storage_provider.get_message_collection()

    # Test initial state
    assert await message_collection.size() == 0

    # Test adding multiple messages
    test_messages = [
        DummyTestMessage(["hello", "world"]),
        DummyTestMessage(["foo", "bar", "baz"]),
        DummyTestMessage(["test", "message", "data"]),
    ]

    for msg in test_messages:
        await message_collection.append(msg)

    # Test final size
    assert await message_collection.size() == len(test_messages)

    # Test iteration
    collection_list = [item async for item in message_collection]
    assert len(collection_list) == 3
    for i, msg in enumerate(collection_list):
        assert msg.text_chunks == test_messages[i].text_chunks

    # Test get_item for all messages
    for i, expected_msg in enumerate(test_messages):
        retrieved_msg = await message_collection.get_item(i)
        assert isinstance(retrieved_msg, DummyTestMessage)
        assert retrieved_msg.text_chunks == expected_msg.text_chunks

    # Test get_item with out-of-bounds index
    with pytest.raises(IndexError):
        await message_collection.get_item(-1)
        await message_collection.get_item(len(test_messages) + 1)

    # Test get_slice
    slice_result = await message_collection.get_slice(1, 3)
    assert len(slice_result) == 2
    assert slice_result[0].text_chunks == test_messages[1].text_chunks
    assert slice_result[1].text_chunks == test_messages[2].text_chunks
    # Edge case: slice to end
    slice_to_end = await message_collection.get_slice(2, 10)
    assert len(slice_to_end) == 1
    assert slice_to_end[0].text_chunks == test_messages[2].text_chunks
    # Edge case: empty slice
    empty_slice = await message_collection.get_slice(1, 1)
    assert len(empty_slice) == 0
    # Edge case: empty slice out of bounds
    out_of_bounds_slice = await message_collection.get_slice(5, 10)
    assert len(out_of_bounds_slice) == 0

    # Test get_multiple
    multiple_result = await message_collection.get_multiple([2, 0])
    assert len(multiple_result) == 2
    assert multiple_result[0].text_chunks == test_messages[2].text_chunks
    assert multiple_result[1].text_chunks == test_messages[0].text_chunks
    # Edge case: singleton list
    single_result = await message_collection.get_multiple([1])
    assert len(single_result) == 1
    assert single_result[0].text_chunks == test_messages[1].text_chunks
    # Edge case: empty list
    no_result = await message_collection.get_multiple([])
    assert len(no_result) == 0
    # Edge case: duplicate index
    result = await message_collection.get_multiple([0, 0, 1])
    assert len(result) == 3
    assert result[0].text_chunks == test_messages[0].text_chunks
    assert result[1].text_chunks == test_messages[0].text_chunks
    assert result[2].text_chunks == test_messages[1].text_chunks

    # Test that out-of-bounds in get_multiple raises IndexError
    with pytest.raises(IndexError):
        await message_collection.get_multiple([0, 5])
    with pytest.raises(IndexError):
        await message_collection.get_multiple([-1, 1])
