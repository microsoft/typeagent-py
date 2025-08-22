# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from typing import cast
from typeagent.knowpro.semrefindex import (
    TermToSemanticRefIndex,
    add_entity_to_index,
    add_topic_to_index,
    add_action_to_index,
    add_knowledge_to_index,
)
from typeagent.knowpro.kplib import ConcreteEntity, Facet, Action, KnowledgeResponse
from typeagent.knowpro.interfaces import Topic
from typeagent.knowpro.collections import MemorySemanticRefCollection


@pytest.fixture
def semantic_ref_index() -> TermToSemanticRefIndex:
    """Fixture to create a TermToSemanticRefIndex instance."""
    return TermToSemanticRefIndex()


@pytest.mark.asyncio
async def test_semantic_ref_index_add_and_lookup(
    semantic_ref_index: TermToSemanticRefIndex,
):
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
    semantic_ref_index: TermToSemanticRefIndex,
):
    """Test removing terms from the TermToSemanticRefIndex."""
    await semantic_ref_index.add_term("example", 1)
    await semantic_ref_index.add_term("example", 2)

    await semantic_ref_index.remove_term("example", 1)
    result = await semantic_ref_index.lookup_term("example")
    assert result is not None
    assert len(result) == 0


@pytest.mark.asyncio
async def test_conversation_index_remove_term_if_empty(
    semantic_ref_index: TermToSemanticRefIndex,
):
    """Test removing terms if they are empty."""
    await semantic_ref_index.add_term("example", 1)
    await semantic_ref_index.remove_term("example", 1)
    semantic_ref_index.remove_term_if_empty("example")

    assert await semantic_ref_index.size() == 0


@pytest.mark.asyncio
async def test_conversation_index_serialize_and_deserialize(
    semantic_ref_index: TermToSemanticRefIndex,
):
    """Test serialization and deserialization of the TermToSemanticRefIndex."""
    await semantic_ref_index.add_term("example", 1)
    await semantic_ref_index.add_term("test", 2)

    serialized = semantic_ref_index.serialize()
    assert "items" in serialized
    assert len(serialized["items"]) == 2

    new_index = TermToSemanticRefIndex()
    new_index.deserialize(serialized)

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
async def test_add_entity_to_index(semantic_ref_index: TermToSemanticRefIndex):
    """Test adding an entity to the index."""
    entity = ConcreteEntity(
        name="ExampleEntity",
        type=["object", "example"],
        facets=[Facet(name="color", value="blue")],
    )
    semantic_refs = MemorySemanticRefCollection()
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
async def test_add_topic_to_index(semantic_ref_index: TermToSemanticRefIndex):
    """Test adding a topic to the index."""
    topic = "ExampleTopic"
    semantic_refs = MemorySemanticRefCollection()
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
async def test_add_action_to_index(semantic_ref_index: TermToSemanticRefIndex):
    """Test adding an action to the index."""
    action = Action(
        verbs=["run", "jump"],
        verb_tense="present",
        subject_entity_name="John",
        object_entity_name="Ball",
        indirect_object_entity_name="none",
        params=None,
        subject_entity_facet=None,
    )
    semantic_refs = MemorySemanticRefCollection()
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
async def test_add_knowledge_to_index(semantic_ref_index: TermToSemanticRefIndex):
    """Test adding knowledge to the index."""
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
    semantic_refs = MemorySemanticRefCollection()
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
async def test_conversation_index_size_and_get_terms(
    semantic_ref_index: TermToSemanticRefIndex,
):
    """Test size() and get_terms method."""
    assert await semantic_ref_index.size() == 0
    await semantic_ref_index.add_term("foo", 1)
    await semantic_ref_index.add_term("bar", 2)
    terms = await semantic_ref_index.get_terms()
    assert "foo" in terms
    assert "bar" in terms
    assert await semantic_ref_index.size() == 2


@pytest.mark.asyncio
async def test_conversation_index_contains(semantic_ref_index: TermToSemanticRefIndex):
    """Test presence of a term using lookup_term."""
    await semantic_ref_index.add_term("foo", 1)
    assert await semantic_ref_index.lookup_term("foo") != []
    assert await semantic_ref_index.lookup_term("bar") == []


@pytest.mark.asyncio
async def test_conversation_index_clear(semantic_ref_index: TermToSemanticRefIndex):
    """Test clear method."""
    await semantic_ref_index.add_term("foo", 1)
    await semantic_ref_index.add_term("bar", 2)
    semantic_ref_index.clear()
    assert await semantic_ref_index.size() == 0
    assert await semantic_ref_index.lookup_term("foo") == []


@pytest.mark.asyncio
async def test_conversation_index_remove_term_nonexistent(
    semantic_ref_index: TermToSemanticRefIndex,
):
    """Test removing a term that does not exist does not raise."""
    await semantic_ref_index.remove_term("nonexistent", 123)  # Should not raise


def test_conversation_index_remove_term_if_empty_nonexistent(
    semantic_ref_index: TermToSemanticRefIndex,
):
    """Test remove_term_if_empty on a term that does not exist."""
    semantic_ref_index.remove_term_if_empty("nonexistent")  # Should not raise


def test_conversation_index_serialize_empty(semantic_ref_index: TermToSemanticRefIndex):
    """Test serialize on an empty index."""
    serialized = semantic_ref_index.serialize()
    assert "items" in serialized
    assert serialized["items"] == []
