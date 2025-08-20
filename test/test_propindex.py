# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Iterator
from typing import Any

import pytest

from fixtures import needs_auth  # type: ignore  # Yes it is used!
from typeagent.knowpro.collections import TextRangeCollection, TextRangesInScope
from typeagent.knowpro.interfaces import (
    ICollection,
    IConversation,
    IMessage,
    IStorageProvider,
    ITermToSemanticRefIndex,
    SemanticRef,
    Tag,
    TextLocation,
    TextRange,
)
from typeagent.knowpro.kplib import Facet, ConcreteEntity, Action, KnowledgeResponse
from typeagent.knowpro.propindex import (
    PropertyIndex,
    PropertyNames,
    add_facet,
    add_entity_properties_to_index,
    add_action_properties_to_index,
    build_property_index,
    add_to_property_index,
    lookup_property_in_property_index,
    make_property_term_text,
    split_property_term_text,
    is_known_property,
)
from typeagent.knowpro.secindex import ConversationSecondaryIndexes
from typeagent.storage.memorystore import MemoryStorageProvider
from typeagent.knowpro.collections import (
    MemoryMessageCollection,
    MemorySemanticRefCollection,
)
from typeagent.knowpro.convsettings import ConversationSettings

from fixtures import needs_auth  # type: ignore  # Yes we use it!


class SimpleFakeConversation(IConversation):
    """Fake conversation for testing."""

    def __init__(self, semantic_refs):
        self.name_tag = "test"
        self.tags = []
        self.messages = MemoryMessageCollection()
        self.semantic_refs = MemorySemanticRefCollection(semantic_refs)
        self.semantic_ref_index = None
        # Create storage provider with test settings
        from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
        from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
        from typeagent.knowpro.messageindex import MessageTextIndexSettings
        from typeagent.knowpro.reltermsindex import RelatedTermIndexSettings

        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        embedding_settings = TextEmbeddingIndexSettings(test_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        storage_provider = MemoryStorageProvider()
        self.secondary_indexes = ConversationSecondaryIndexes(
            storage_provider, related_terms_settings
        )
        # Store settings with storage provider for access via conversation.settings.storage_provider
        self.settings = ConversationSettings(test_model, storage_provider)


@pytest.fixture
def property_index():
    """Fixture to create a PropertyIndex instance."""
    return PropertyIndex()


@pytest.mark.asyncio
async def test_add_facet(property_index):
    """Test adding a facet to the property index."""
    facet = Facet(name="color", value="blue")
    await add_facet(facet, property_index, 1)

    result = await property_index.lookup_property(
        PropertyNames.FacetName.value, "color"
    )
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(
        PropertyNames.FacetValue.value, "blue"
    )
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1


@pytest.mark.asyncio
async def test_add_entity_properties_to_index(property_index):
    """Test adding entity properties to the property index."""
    entity = ConcreteEntity(
        name="ExampleEntity",
        type=["object", "example"],
        facets=[Facet(name="color", value="blue")],
    )
    await add_entity_properties_to_index(entity, property_index, 1)

    result = await property_index.lookup_property(
        PropertyNames.EntityName.value, "ExampleEntity"
    )
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(
        PropertyNames.EntityType.value, "object"
    )
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(
        PropertyNames.FacetName.value, "color"
    )
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1


@pytest.mark.asyncio
async def test_add_action_properties_to_index(property_index):
    """Test adding action properties to the property index."""
    action = Action(
        verbs=["run", "jump"],
        verb_tense="present",
        subject_entity_name="John",
        object_entity_name="Ball",
        indirect_object_entity_name="none",
        params=None,
        subject_entity_facet=None,
    )
    await add_action_properties_to_index(action, property_index, 1)

    result = await property_index.lookup_property(PropertyNames.Verb.value, "run jump")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(PropertyNames.Subject.value, "John")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(PropertyNames.Object.value, "Ball")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1


def test_make_and_split_property_term_text():
    """Test creating and splitting property term text."""
    term_text = make_property_term_text("name", "value")
    assert term_text == "prop.name@@value"

    name, value = split_property_term_text(term_text)
    assert name == "prop.name"
    assert value == "value"


@pytest.mark.asyncio
async def test_is_known_property(property_index):
    """Test checking if a property is known."""
    await property_index.add_property("name", "value", 1)

    assert (
        await is_known_property(property_index, PropertyNames.EntityName, "value")
        is True
    )
    assert (
        await is_known_property(property_index, PropertyNames.EntityName, "unknown")
        is False
    )


@pytest.mark.asyncio
async def test_build_property_index(needs_auth):
    """Test the build_property_index function with a concrete conversation."""
    # Create a sample conversation with semantic references
    entity1 = ConcreteEntity(
        name="Entity1",
        type=["type1", "type2"],
        facets=[Facet(name="color", value="blue")],
    )
    action1 = Action(
        verbs=["run", "jump"],
        verb_tense="present",
        subject_entity_name="Subject1",
        object_entity_name="Object1",
        indirect_object_entity_name="IndirectObject1",
    )
    tag1 = Tag(text="Tag1")
    semantic_refs = [
        SemanticRef(
            semantic_ref_ordinal=0,
            knowledge=entity1,
            range=TextRange(start=TextLocation(0), end=TextLocation(10)),
        ),
        SemanticRef(
            semantic_ref_ordinal=1,
            knowledge=action1,
            range=TextRange(start=TextLocation(10), end=TextLocation(20)),
        ),
        SemanticRef(
            semantic_ref_ordinal=2,
            knowledge=tag1,
            range=TextRange(start=TextLocation(20), end=TextLocation(30)),
        ),
    ]

    conversation = SimpleFakeConversation(semantic_refs)

    # Call the function
    await build_property_index(conversation)

    # Assertions
    assert conversation.secondary_indexes is not None
    assert isinstance(
        conversation.secondary_indexes.property_to_semantic_ref_index, PropertyIndex
    )

    # Verify the property index contents
    property_index = conversation.secondary_indexes.property_to_semantic_ref_index
    assert await property_index.size() > 0
    assert await property_index.lookup_property("name", "Entity1") is not None
    assert await property_index.lookup_property("type", "Type1") is not None
    assert await property_index.lookup_property("verb", "run jump") is not None
    assert await property_index.lookup_property("tag", "Tag1") is not None


class FakeMessage(IMessage):
    """Concrete implementation of IMessage for testing."""

    def __init__(self, text_chunks: list[str]):
        self.text_chunks = text_chunks
        self.text_location = TextLocation(0, 0)
        self.tags = []

    def get_text(self):
        return " ".join(self.text_chunks)

    def get_text_location(self):
        return self.text_location

    def get_knowledge(self):
        return KnowledgeResponse(
            entities=[],
            actions=[],
            inverse_actions=[],
            topics=[],
        )


class FakeBaseCollection[T, TOrdinal: int](ICollection[T, int]):
    """Concrete implementation of IMessageCollection for testing."""

    def __init__(self, items: list[T] | None = None):
        self.items = items or []

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __getitem__(self, arg: Any) -> Any:
        if isinstance(arg, int):
            return self._get(arg)
        if isinstance(arg, slice):
            assert arg.step in (None, 1)
            return self._get_slice(arg.start, arg.stop)
        if isinstance(arg, list):
            return self._get_multiple(arg)
        raise TypeError(f"Invalid argument type for __getitem__: {type(arg)}")

    def _get(self, ordinal: int) -> T:
        return self.items[ordinal]

    def _get_multiple(self, ordinals: list[int]) -> list[T]:
        return [self.items[i] for i in ordinals]

    @property
    def is_persistent(self) -> bool:
        return False

    def _get_slice(self, start: int, end: int) -> list[T]:
        return self.items[start:end]

    async def append(self, item: T) -> None:
        self.items.append(item)


class FakeConversation[
    TMessage: IMessage, TTermToSemanticRefIndex: ITermToSemanticRefIndex
](IConversation[TMessage, TTermToSemanticRefIndex]):
    """Concrete implementation of IConversation for testing."""

    def __init__(self, semantic_refs: list[SemanticRef] | None = None):
        self.name_tag = "test_conversation"
        self.tags = []
        self.semantic_refs = MemorySemanticRefCollection(semantic_refs or [])
        self.semantic_ref_index = None
        self.messages: IMessageCollection[TMessage] = MemoryMessageCollection([FakeMessage(["Hello"])])  # type: ignore[assignment]
        # Create storage provider with test settings
        from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
        from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
        from typeagent.knowpro.messageindex import MessageTextIndexSettings
        from typeagent.knowpro.reltermsindex import RelatedTermIndexSettings

        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        embedding_settings = TextEmbeddingIndexSettings(test_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        self._storage_provider = MemoryStorageProvider()
        self.secondary_indexes = ConversationSecondaryIndexes(
            self._storage_provider, related_terms_settings
        )

    @property
    def storage_provider(self) -> "IStorageProvider":
        return self._storage_provider


@pytest.mark.asyncio
async def test_add_to_property_index(needs_auth, property_index):
    """Test adding semantic references to the property index."""
    entity = ConcreteEntity(
        name="ExampleEntity",
        type=["object"],
        facets=[Facet(name="color", value="blue")],
    )
    semantic_refs = [
        SemanticRef(
            semantic_ref_ordinal=0,
            range=TextRange(start=TextLocation(0), end=None),
            knowledge=entity,
        )
    ]
    conversation = FakeConversation(semantic_refs)
    await add_to_property_index(conversation, 0)

    assert conversation.secondary_indexes is not None
    assert conversation.secondary_indexes.property_to_semantic_ref_index is not None
    lookup_result = await conversation.secondary_indexes.property_to_semantic_ref_index.lookup_property(
        "name", "ExampleEntity"
    )
    assert lookup_result is not None
    assert len(lookup_result) == 1
    assert lookup_result[0].semantic_ref_ordinal == 0


@pytest.mark.asyncio
async def test_lookup_property_in_property_index(property_index):
    """Test filtering properties based on ranges_in_scope."""
    await property_index.add_property("name", "value1", 0)
    await property_index.add_property("name", "value2", 1)

    entity0 = ConcreteEntity("name0", ["type"])
    entity1 = ConcreteEntity("name1", ["type"])
    semantic_refs = [
        SemanticRef(
            semantic_ref_ordinal=0,
            range=TextRange(start=TextLocation(0), end=TextLocation(10)),
            knowledge=entity0,
        ),
        SemanticRef(
            semantic_ref_ordinal=1,
            range=TextRange(start=TextLocation(20), end=TextLocation(30)),
            knowledge=entity1,
        ),
    ]
    ranges_in_scope = TextRangesInScope(
        [TextRangeCollection([TextRange(TextLocation(0), TextLocation(15))])]
    )

    result = await lookup_property_in_property_index(
        property_index,
        "name",
        "value1",
        MemorySemanticRefCollection(semantic_refs),
        ranges_in_scope,
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 0

    result = await lookup_property_in_property_index(
        property_index,
        "name",
        "value2",
        MemorySemanticRefCollection(semantic_refs),
        ranges_in_scope,
    )
    assert result is None


@pytest.mark.asyncio
async def test_property_index_clear(property_index):
    """Test clearing all properties in the PropertyIndex."""
    await property_index.add_property("name", "value", 0)
    assert await property_index.size() > 0

    property_index.clear()
    assert await property_index.size() == 0


@pytest.mark.asyncio
async def test_property_index_get_values(property_index):
    """Test retrieving all property values from the PropertyIndex."""
    await property_index.add_property("name", "value1", 0)
    await property_index.add_property("name", "value2", 1)

    values = await property_index.get_values()
    assert len(values) == 2
    assert "value1" in values
    assert "value2" in values


def test_property_index_prepare_term_text(property_index):
    """Test preprocessing term text in PropertyIndex."""
    term_text = "Prop.Name@@Value"
    prepared_text = property_index._prepare_term_text(term_text)
    assert prepared_text == "prop.name@@value"  # Should be converted to lowercase


@pytest.mark.asyncio
async def test_property_index_lookup_property_edge_cases(property_index):
    """Test edge cases for lookup_property in PropertyIndex."""
    # Case: Property does not exist
    result = await property_index.lookup_property("name", "nonexistent")
    assert result is None

    # Case: Property exists
    await property_index.add_property("name", "value", 0)
    result = await property_index.lookup_property("name", "value")
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 0
