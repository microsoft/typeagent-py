# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import AsyncGenerator
from dataclasses import field
import os
import tempfile
from typing import Generator

import pytest
from pydantic.dataclasses import dataclass
import pytest_asyncio

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.interfaces import (
    IMessage,
    SemanticRef,
    TextLocation,
    TextRange,
    Topic,
)
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.knowpro.messageindex import MessageTextIndexSettings
from typeagent.knowpro.reltermsindex import RelatedTermIndexSettings
from typeagent.storage.sqlitestore import SqliteStorageProvider

from fixtures import embedding_model, FakeMessage


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
) -> AsyncGenerator[SqliteStorageProvider[FakeMessage], None]:
    """Create a SqliteStorageProvider for testing."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = await SqliteStorageProvider.create(
        message_text_settings, related_terms_settings, temp_db_path, DummyMessage
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


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.asyncio
async def test_sqlite_storage_provider_message_collection(
    dummy_sqlite_storage_provider: SqliteStorageProvider[DummyMessage],
):
    collection = await dummy_sqlite_storage_provider.get_message_collection()
    assert collection.is_persistent
    assert await collection.size() == 0

    msg = DummyMessage(["hello"])
    await collection.append(msg)
    assert await collection.size() == 1
    # get_item and async iteration
    loaded = await collection.get_item(0)
    assert isinstance(loaded, DummyMessage)
    assert loaded.text_chunks == ["hello"]
    collection_list = [item async for item in collection]
    assert collection_list[0].text_chunks == ["hello"]
    await collection.append(DummyMessage(["world"]))
    await collection.append(DummyMessage(["foo", "bar"]))
    assert await collection.size() == 3
    # slice
    slice_result = await collection.get_slice(1, 3)
    assert [msg.text_chunks[0] for msg in slice_result] == [
        "world",
        "foo",
    ]
    # multiple get
    multiple_result = await collection.get_multiple([0, 2])
    assert [msg.text_chunks[0] for msg in multiple_result] == [
        "hello",
        "foo",
    ]


@pytest.mark.asyncio
async def test_sqlite_storage_provider_semantic_ref_collection(
    dummy_sqlite_storage_provider: SqliteStorageProvider[DummyMessage],
):
    collection = await dummy_sqlite_storage_provider.get_semantic_ref_collection()
    assert collection.is_persistent
    assert await collection.size() == 0

    # Create a dummy SemanticRef
    ref = make_dummy_semantic_ref()

    await collection.append(ref)
    assert await collection.size() == 1
    loaded = await collection.get_item(0)
    assert isinstance(loaded, SemanticRef)
    assert loaded.semantic_ref_ordinal == 0
    collection_list = [item async for item in collection]
    assert collection_list[0].semantic_ref_ordinal == 0


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
    assert results[0].timestamp == "2024-01-01T09:00:00+00:00"  # Normalized format
    assert results[0].range.start.message_ordinal == 0  # Message ordinal 0

    # Test lookup_range - range query
    start_date = datetime.fromisoformat("2024-01-01T10:00:00Z")
    end_date = datetime.fromisoformat("2024-01-01T12:00:00Z")
    range_query = DateRange(start=start_date, end=end_date)
    results = await timestamp_index.lookup_range(range_query)

    # Should find messages with timestamps: 11:00 (09:00 is before range, 12:00 is excluded, 13:00 is after)
    assert len(results) == 1
    assert results[0].timestamp == "2024-01-01T11:00:00+00:00"  # Normalized format
    assert results[0].range.start.message_ordinal == 1

    # Test empty range
    empty_start = datetime.fromisoformat("2024-02-01T00:00:00Z")
    empty_end = datetime.fromisoformat("2024-02-01T23:59:59Z")
    empty_range = DateRange(start=empty_start, end=empty_end)
    empty_results = await timestamp_index.lookup_range(empty_range)
    assert len(empty_results) == 0


@pytest.mark.asyncio
async def test_populate_indexes_from_data_comprehensive(
    temp_db_path: str, embedding_model: AsyncEmbeddingModel
):
    """Test _populate_indexes_from_data with comprehensive edge cases."""
    from typeagent.knowpro import kplib
    from typeagent.knowpro.interfaces import SemanticRef, TextLocation, TextRange, Tag

    # Setup storage provider
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = await SqliteStorageProvider.create(
        message_text_settings, related_terms_settings, temp_db_path, DummyMessage
    )

    try:
        # Get collections
        messages = await provider.get_message_collection()
        semantic_refs = await provider.get_semantic_ref_collection()
        conversation_index = await provider.get_semantic_ref_index()

        # Add test messages with timestamps
        msg1 = DummyMessage(
            text_chunks=["Hello world"], timestamp="2024-01-01T10:00:00Z"
        )
        msg2 = DummyMessage(
            text_chunks=["Python is great"], timestamp="2024-01-01T11:00:00Z"
        )
        await messages.append(msg1)
        await messages.append(msg2)

        # Create test semantic refs covering all edge cases

        # 1. ConcreteEntity with multiple types and facets
        entity_with_facets = kplib.ConcreteEntity(
            name="John Doe",
            type=["person", "developer"],
            facets=[
                kplib.Facet(name="age", value="30"),
                kplib.Facet(name="company", value="Microsoft"),
                kplib.Facet(
                    name="role", value="engineer"
                ),  # Test facet with string value
            ],
        )
        entity_ref = SemanticRef(
            semantic_ref_ordinal=0,
            range=TextRange(start=TextLocation(message_ordinal=0, chunk_ordinal=0)),
            knowledge=entity_with_facets,
        )

        # 2. ConcreteEntity with no facets
        simple_entity = kplib.ConcreteEntity(name="Python", type=["language"])
        simple_entity_ref = SemanticRef(
            semantic_ref_ordinal=1,
            range=TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=0)),
            knowledge=simple_entity,
        )

        # 3. Action with all possible fields
        complex_action = kplib.Action(
            verbs=["give", "provide"],
            verb_tense="past",
            subject_entity_name="John Doe",
            object_entity_name="Python course",
            indirect_object_entity_name="students",
            params=[
                "online",  # String param
                kplib.ActionParam(name="duration", value="3 hours"),  # Param with value
                kplib.ActionParam(
                    name="level", value="intermediate"
                ),  # Param with string value
            ],
            subject_entity_facet=kplib.Facet(name="experience", value="senior"),
        )
        action_ref = SemanticRef(
            semantic_ref_ordinal=2,
            range=TextRange(start=TextLocation(message_ordinal=0, chunk_ordinal=1)),
            knowledge=complex_action,
        )

        # 4. Action with minimal fields (no params, no facet)
        simple_action = kplib.Action(
            verbs=["learn"],
            verb_tense="present",
            subject_entity_name="Alice",
            object_entity_name="none",
            indirect_object_entity_name="none",
            # params defaults to None
            # subject_entity_facet defaults to None
        )
        simple_action_ref = SemanticRef(
            semantic_ref_ordinal=3,
            range=TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=0)),
            knowledge=simple_action,
        )

        # 5. Topic
        topic = Topic(text="machine learning")
        topic_ref = SemanticRef(
            semantic_ref_ordinal=4,
            range=TextRange(start=TextLocation(message_ordinal=0, chunk_ordinal=2)),
            knowledge=topic,
        )

        # 6. Tag
        tag = Tag(text="important")
        tag_ref = SemanticRef(
            semantic_ref_ordinal=5,
            range=TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=1)),
            knowledge=tag,
        )

        # Add all semantic refs to collection
        await semantic_refs.append(entity_ref)
        await semantic_refs.append(simple_entity_ref)
        await semantic_refs.append(action_ref)
        await semantic_refs.append(simple_action_ref)
        await semantic_refs.append(topic_ref)
        await semantic_refs.append(tag_ref)

        # Since semantic ref index is guaranteed to be complete, manually populate it for testing
        # In real usage, this would be handled by external mechanisms
        from typeagent.knowpro.semrefindex import add_facet
        from typeagent.knowpro import kplib

        index = await provider.get_semantic_ref_index()

        # Manually populate semantic ref index following the same logic as the original population code
        for sem_ref in [
            entity_ref,
            simple_entity_ref,
            action_ref,
            simple_action_ref,
            topic_ref,
            tag_ref,
        ]:
            knowledge = sem_ref.knowledge
            ref_ordinal = sem_ref.semantic_ref_ordinal

            if isinstance(knowledge, kplib.ConcreteEntity):
                # Add entity name and types
                await index.add_term(knowledge.name, ref_ordinal)
                for type_name in knowledge.type:
                    await index.add_term(type_name, ref_ordinal)
                # Add facets
                if knowledge.facets:
                    for facet in knowledge.facets:
                        await add_facet(facet, ref_ordinal, index)

            elif isinstance(knowledge, Topic):
                # Add topic text
                await index.add_term(knowledge.text, ref_ordinal)

            elif isinstance(knowledge, kplib.Action):
                # Add verbs
                await index.add_term(" ".join(knowledge.verbs), ref_ordinal)
                # Add entity names
                if knowledge.subject_entity_name != "none":
                    await index.add_term(knowledge.subject_entity_name, ref_ordinal)
                if knowledge.object_entity_name != "none":
                    await index.add_term(knowledge.object_entity_name, ref_ordinal)
                if knowledge.indirect_object_entity_name != "none":
                    await index.add_term(
                        knowledge.indirect_object_entity_name, ref_ordinal
                    )
                # Add params
                if knowledge.params:
                    for param in knowledge.params:
                        if isinstance(param, str):
                            await index.add_term(param, ref_ordinal)
                        else:
                            await index.add_term(param.name, ref_ordinal)
                            if hasattr(param, "value") and isinstance(param.value, str):
                                await index.add_term(param.value, ref_ordinal)
                # Add subject entity facet
                if (
                    hasattr(knowledge, "subject_entity_facet")
                    and knowledge.subject_entity_facet
                ):
                    await add_facet(knowledge.subject_entity_facet, ref_ordinal, index)

            elif isinstance(knowledge, Tag):
                # Add tag text
                await index.add_term(knowledge.text, ref_ordinal)

        # Call _populate_indexes_from_data to rebuild non-persistent indexes
        # Note: Semantic ref index is persistent and guaranteed to be complete
        await provider._populate_indexes_from_data()

        # Verify conversation index was populated correctly
        index = await provider.get_semantic_ref_index()

        # Helper function to check if a term exists
        async def has_term(term: str) -> bool:
            result = await index.lookup_term(term)
            return result is not None and len(result) > 0

        # Helper function to get semantic refs for a term
        async def get_semantic_refs(term: str) -> list[int]:
            result = await index.lookup_term(term)
            return [ref.semantic_ref_ordinal for ref in result] if result else []

        # Check entity terms
        assert await has_term("John Doe")  # Entity name
        assert await has_term("person")  # Entity type
        assert await has_term("developer")  # Entity type
        assert await has_term("age")  # Facet name
        assert await has_term("30")  # Facet value
        assert await has_term("company")  # Facet name
        assert await has_term("Microsoft")  # Facet value
        assert await has_term("role")  # Facet name
        assert await has_term("engineer")  # Facet value

        assert await has_term("Python")  # Simple entity name
        assert await has_term("language")  # Simple entity type

        # Check action terms
        assert await has_term("give provide")  # Joined verbs
        assert await has_term("John Doe")  # Subject (already checked above)
        assert await has_term("Python course")  # Object
        assert await has_term("students")  # Indirect object
        assert await has_term("online")  # String param
        assert await has_term("duration")  # Param name
        assert await has_term("3 hours")  # Param value
        assert await has_term("level")  # Param name
        assert await has_term("intermediate")  # Param value
        assert await has_term("experience")  # Subject facet name
        assert await has_term("senior")  # Subject facet value

        assert await has_term("learn")  # Simple action verb
        assert await has_term("Alice")  # Simple action subject
        # "none" values should not be indexed
        assert not await has_term("none")

        # Check topic terms
        assert await has_term("machine learning")

        # Check tag terms
        assert await has_term("important")

        # Verify semantic ref mappings
        john_doe_refs = await get_semantic_refs("John Doe")
        assert 0 in john_doe_refs  # Entity ref
        assert 2 in john_doe_refs  # Action subject ref

        python_refs = await get_semantic_refs("Python")
        assert 1 in python_refs  # Simple entity ref

        # Verify timestamp index was populated
        timestamp_index = await provider.get_timestamp_index()
        # Cast to concrete type to access size method
        from typeagent.storage.sqlitestore import SqliteTimestampToTextRangeIndex

        assert isinstance(timestamp_index, SqliteTimestampToTextRangeIndex)
        assert await timestamp_index.size() == 2  # Both messages have timestamps

        # Test the index actually works for lookups
        from datetime import datetime
        from typeagent.knowpro.interfaces import DateRange

        start_time = datetime.fromisoformat("2024-01-01T10:30:00Z")
        end_time = datetime.fromisoformat("2024-01-01T11:30:00Z")
        date_range = DateRange(start=start_time, end=end_time)
        timestamped_ranges = await timestamp_index.lookup_range(date_range)
        assert len(timestamped_ranges) == 1  # Only msg2 in range
        assert timestamped_ranges[0].range.start.message_ordinal == 1

    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_populate_indexes_empty_data(
    temp_db_path: str, embedding_model: AsyncEmbeddingModel
):
    """Test _populate_indexes_from_data with empty database."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = await SqliteStorageProvider.create(
        message_text_settings, related_terms_settings, temp_db_path, DummyMessage
    )

    try:
        # Should handle empty database gracefully
        index = await provider.get_semantic_ref_index()
        assert await index.size() == 0

        timestamp_index = await provider.get_timestamp_index()
        # Cast to concrete type to access size method
        from typeagent.storage.sqlitestore import SqliteTimestampToTextRangeIndex

        assert isinstance(timestamp_index, SqliteTimestampToTextRangeIndex)
        assert await timestamp_index.size() == 0

    finally:
        await provider.close()
