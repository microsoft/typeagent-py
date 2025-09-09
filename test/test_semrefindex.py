# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Third-party imports
import pytest
import pytest_asyncio
from typing import cast, Dict, AsyncGenerator

# TypeAgent imports
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.storage.memory import MemorySemanticRefCollection
from typeagent.knowpro.interfaces import (
    Topic,
    IMessage,
    ITermToSemanticRefIndex,
    ISemanticRefCollection,
)
from typeagent.knowpro.kplib import ConcreteEntity, Facet, Action, KnowledgeResponse
from typeagent.knowpro.convsettings import (
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.storage.memory.semrefindex import (
    TermToSemanticRefIndex,
    add_entity_to_index,
    add_topic_to_index,
    add_action_to_index,
    add_knowledge_to_index,
)
from typeagent.storage.memory import MemoryStorageProvider
from typeagent.storage import SqliteStorageProvider

# Test fixtures
from fixtures import needs_auth, embedding_model, temp_db_path


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def semantic_ref_index(
    request: pytest.FixtureRequest,
    embedding_model: AsyncEmbeddingModel,
    temp_db_path: str,
) -> AsyncGenerator[ITermToSemanticRefIndex, None]:
    """Unified fixture to create a semantic ref index for both memory and SQLite providers."""

    class DummyTestMessage(IMessage):
        text_chunks: list[str]
        tags: list[str] = []

        def get_knowledge(self):
            return KnowledgeResponse(
                entities=[], actions=[], inverse_actions=[], topics=[]
            )

    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    if request.param == "memory":
        provider = MemoryStorageProvider(
            message_text_settings=message_text_settings,
            related_terms_settings=related_terms_settings,
        )
        index = await provider.get_semantic_ref_index()
        yield index
    else:
        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyTestMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        # For SQLite, we need to create semantic refs first due to foreign key constraints
        from typeagent.knowpro.interfaces import (
            SemanticRef,
            TextRange,
            TextLocation,
            Topic,
        )

        collection = await provider.get_semantic_ref_collection()

        # Create semantic refs with ordinals 1, 2, 3 that the tests expect
        for i in range(1, 4):
            ref = SemanticRef(
                semantic_ref_ordinal=i,
                range=TextRange(start=TextLocation(message_ordinal=0, chunk_ordinal=0)),
                knowledge=Topic(text=f"test_topic_{i}"),
            )
            await collection.append(ref)

        index = await provider.get_semantic_ref_index()
        yield index
        await provider.close()


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def semantic_ref_setup(
    request: pytest.FixtureRequest,
    embedding_model: AsyncEmbeddingModel,
    temp_db_path: str,
) -> AsyncGenerator[Dict[str, ITermToSemanticRefIndex | ISemanticRefCollection], None]:
    """Unified fixture that provides both semantic ref index and collection for testing helper functions."""

    class DummyTestMessage(IMessage):
        text_chunks: list[str]
        tags: list[str] = []

        def get_knowledge(self):
            return KnowledgeResponse(
                entities=[], actions=[], inverse_actions=[], topics=[]
            )

    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    if request.param == "memory":
        provider = MemoryStorageProvider(
            message_text_settings=message_text_settings,
            related_terms_settings=related_terms_settings,
        )
        index = await provider.get_semantic_ref_index()
        collection = await provider.get_semantic_ref_collection()
        yield {"index": index, "collection": collection}
    else:
        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyTestMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )
        index = await provider.get_semantic_ref_index()
        collection = await provider.get_semantic_ref_collection()
        yield {"index": index, "collection": collection}
        await provider.close()


@pytest.fixture
def legacy_semantic_ref_index() -> TermToSemanticRefIndex:
    """Legacy fixture for tests that specifically need TermToSemanticRefIndex instance."""
    return TermToSemanticRefIndex()


@pytest.mark.asyncio
async def test_semantic_ref_index_add_and_lookup(
    semantic_ref_index: ITermToSemanticRefIndex, needs_auth: None
) -> None:
    """Test adding and looking up terms in the TermToSemanticRefIndex."""
    await semantic_ref_index.add_term("example", 1)
    await semantic_ref_index.add_term("example", 2)
    await semantic_ref_index.add_term("test", 3)

    result = await semantic_ref_index.lookup_term("example")
    assert result is not None
    assert len(result) == 2
    assert result[0].semantic_ref_ordinal == 1
    assert result[1].semantic_ref_ordinal == 2

    result = await semantic_ref_index.lookup_term("test")
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 3

    result = await semantic_ref_index.lookup_term("nonexistent")
    assert result == []


@pytest.mark.asyncio
async def test_term_to_semantic_ref_index_remove_term(
    semantic_ref_index: ITermToSemanticRefIndex, needs_auth: None
) -> None:
    """Test removing terms from the TermToSemanticRefIndex."""
    await semantic_ref_index.add_term("example", 1)
    await semantic_ref_index.add_term("example", 2)

    await semantic_ref_index.remove_term("example", 1)
    result = await semantic_ref_index.lookup_term("example")
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 2


@pytest.mark.asyncio
async def test_semantic_ref_index_serialize_and_deserialize(
    legacy_semantic_ref_index: TermToSemanticRefIndex, needs_auth: None
) -> None:
    """Test serialization and deserialization of the TermToSemanticRefIndex."""
    await legacy_semantic_ref_index.add_term("example", 1)
    await legacy_semantic_ref_index.add_term("test", 2)

    serialized = await legacy_semantic_ref_index.serialize()
    assert "items" in serialized
    assert len(serialized["items"]) == 2

    new_index = TermToSemanticRefIndex()
    await new_index.deserialize(serialized)

    # Test that the new index has the correct size
    assert await new_index.size() == 2

    example = await new_index.lookup_term("example")
    assert example is not None
    assert len(example) >= 1
    assert example[0].semantic_ref_ordinal == 1

    test = await new_index.lookup_term("test")
    assert test is not None
    assert len(test) >= 1
    assert test[0].semantic_ref_ordinal == 2


@pytest.mark.asyncio
async def test_add_entity_to_index(
    semantic_ref_setup: Dict[str, ITermToSemanticRefIndex | ISemanticRefCollection],
    needs_auth: None,
) -> None:
    """Test adding an entity to the index."""
    semantic_ref_index: ITermToSemanticRefIndex = semantic_ref_setup["index"]  # type: ignore
    semantic_refs: ISemanticRefCollection = semantic_ref_setup["collection"]  # type: ignore

    entity = ConcreteEntity(
        name="ExampleEntity",
        type=["object", "example"],
        facets=[Facet(name="color", value="blue")],
    )
    await add_entity_to_index(entity, semantic_refs, semantic_ref_index, 0)

    assert await semantic_refs.size() == 1
    assert (await semantic_refs.get_item(0)).knowledge.knowledge_type == "entity"
    assert (
        cast(ConcreteEntity, (await semantic_refs.get_item(0)).knowledge).name
        == "ExampleEntity"
    )

    result = await semantic_ref_index.lookup_term("ExampleEntity")
    assert result is not None
    assert len(result) == 1

    result = await semantic_ref_index.lookup_term("object")
    assert result is not None
    assert len(result) == 1

    result = await semantic_ref_index.lookup_term("color")
    assert result is not None
    assert len(result) == 1


@pytest.mark.asyncio
async def test_add_topic_to_index(
    semantic_ref_setup: Dict[str, ITermToSemanticRefIndex | ISemanticRefCollection],
    needs_auth: None,
) -> None:
    """Test adding a topic to the index."""
    semantic_ref_index: ITermToSemanticRefIndex = semantic_ref_setup["index"]  # type: ignore
    semantic_refs: ISemanticRefCollection = semantic_ref_setup["collection"]  # type: ignore

    topic = "ExampleTopic"
    await add_topic_to_index(topic, semantic_refs, semantic_ref_index, 0)

    assert await semantic_refs.size() == 1
    assert (await semantic_refs.get_item(0)).knowledge.knowledge_type == "topic"
    assert (
        cast(Topic, (await semantic_refs.get_item(0)).knowledge).text == "ExampleTopic"
    )

    result = await semantic_ref_index.lookup_term("ExampleTopic")
    assert result is not None
    assert len(result) == 1


@pytest.mark.asyncio
async def test_add_action_to_index(
    semantic_ref_setup: Dict[str, ITermToSemanticRefIndex | ISemanticRefCollection],
    needs_auth: None,
) -> None:
    """Test adding an action to the index."""
    semantic_ref_index: ITermToSemanticRefIndex = semantic_ref_setup["index"]  # type: ignore
    semantic_refs: ISemanticRefCollection = semantic_ref_setup["collection"]  # type: ignore

    action = Action(
        verbs=["run", "jump"],
        verb_tense="present",
        subject_entity_name="John",
        object_entity_name="Ball",
        indirect_object_entity_name="none",
        params=None,
        subject_entity_facet=None,
    )
    await add_action_to_index(action, semantic_refs, semantic_ref_index, 0)

    assert await semantic_refs.size() == 1
    assert (await semantic_refs.get_item(0)).knowledge.knowledge_type == "action"
    assert cast(Action, (await semantic_refs.get_item(0)).knowledge).verbs == [
        "run",
        "jump",
    ]

    result = await semantic_ref_index.lookup_term("run jump")
    assert result is not None
    assert len(result) == 1

    result = await semantic_ref_index.lookup_term("John")
    assert result is not None
    assert len(result) == 1

    result = await semantic_ref_index.lookup_term("Ball")
    assert result
    assert len(result) == 1


@pytest.mark.asyncio
async def test_add_knowledge_to_index(
    semantic_ref_setup: Dict[str, ITermToSemanticRefIndex | ISemanticRefCollection],
    needs_auth: None,
) -> None:
    """Test adding knowledge to the index."""
    semantic_ref_index: ITermToSemanticRefIndex = semantic_ref_setup["index"]  # type: ignore
    semantic_refs: ISemanticRefCollection = semantic_ref_setup["collection"]  # type: ignore

    knowledge = KnowledgeResponse(
        entities=[
            ConcreteEntity(
                name="ExampleEntity",
                type=["object", "example"],
                facets=[Facet(name="color", value="blue")],
            )
        ],
        actions=[
            Action(
                verbs=["run", "jump"],
                verb_tense="present",
                subject_entity_name="John",
                object_entity_name="Ball",
                indirect_object_entity_name="none",
                params=None,
                subject_entity_facet=None,
            )
        ],
        inverse_actions=[],
        topics=["ExampleTopic"],
    )
    await add_knowledge_to_index(semantic_refs, semantic_ref_index, 0, knowledge)

    assert await semantic_refs.size() == 3  # 1 entity + 1 action + 1 topic

    result = await semantic_ref_index.lookup_term("ExampleEntity")
    assert result is not None
    assert len(result) == 1

    result = await semantic_ref_index.lookup_term("run jump")
    assert result is not None
    assert len(result) == 1

    result = await semantic_ref_index.lookup_term("ExampleTopic")
    assert result is not None
    assert len(result) == 1


@pytest.mark.asyncio
async def test_semantic_ref_index_size_and_get_terms(
    semantic_ref_index: ITermToSemanticRefIndex, needs_auth: None
) -> None:
    """Test size() and get_terms method."""
    assert await semantic_ref_index.size() == 0
    await semantic_ref_index.add_term("foo", 1)
    await semantic_ref_index.add_term("bar", 2)
    terms = await semantic_ref_index.get_terms()
    assert "foo" in terms
    assert "bar" in terms
    assert await semantic_ref_index.size() == 2


@pytest.mark.asyncio
async def test_semantic_ref_index_contains(
    semantic_ref_index: ITermToSemanticRefIndex, needs_auth: None
) -> None:
    """Test presence of a term using lookup_term."""
    await semantic_ref_index.add_term("foo", 1)
    assert await semantic_ref_index.lookup_term("foo") != []
    assert await semantic_ref_index.lookup_term("bar") == []


@pytest.mark.asyncio
async def test_semantic_ref_index_clear(
    semantic_ref_index: ITermToSemanticRefIndex, needs_auth: None
) -> None:
    """Test clear method."""
    await semantic_ref_index.add_term("foo", 1)
    await semantic_ref_index.add_term("bar", 2)
    await semantic_ref_index.clear()
    assert await semantic_ref_index.size() == 0
    assert await semantic_ref_index.lookup_term("foo") == []


@pytest.mark.asyncio
async def test_semantic_ref_index_remove_term_nonexistent(
    semantic_ref_index: ITermToSemanticRefIndex, needs_auth: None
) -> None:
    """Test removing a term that does not exist does not raise."""
    await semantic_ref_index.remove_term("nonexistent", 123)  # Should not raise


@pytest.mark.asyncio
async def test_semantic_ref_index_serialize_empty(
    legacy_semantic_ref_index: TermToSemanticRefIndex,
) -> None:
    """Test serialize on an empty index."""
    serialized = await legacy_semantic_ref_index.serialize()
    assert "items" in serialized
    assert serialized["items"] == []
