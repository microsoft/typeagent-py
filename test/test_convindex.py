# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from typeagent.knowpro.convindex import (
    ConversationIndex,
    add_entity_to_index,
    add_topic_to_index,
    add_action_to_index,
    add_knowledge_to_index,
)
from typeagent.knowpro.kplib import ConcreteEntity, Facet, Action, KnowledgeResponse
from typeagent.knowpro.storage import SemanticRefCollection


@pytest.fixture
def conversation_index() -> ConversationIndex:
    """Fixture to create a ConversationIndex instance."""
    return ConversationIndex()


def test_conversation_index_add_and_lookup(conversation_index: ConversationIndex):
    """Test adding and looking up terms in the ConversationIndex."""
    conversation_index.add_term("example", 1)
    conversation_index.add_term("example", 2)
    conversation_index.add_term("test", 3)

    result = conversation_index.lookup_term("example")
    assert result is not None
    assert len(result) == 2
    assert result[0].semantic_ref_ordinal == 1
    assert result[1].semantic_ref_ordinal == 2

    result = conversation_index.lookup_term("test")
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 3

    result = conversation_index.lookup_term("nonexistent")
    assert result == []


def test_conversation_index_remove_term(conversation_index: ConversationIndex):
    """Test removing terms from the ConversationIndex."""
    conversation_index.add_term("example", 1)
    conversation_index.add_term("example", 2)

    conversation_index.remove_term("example", 1)
    result = conversation_index.lookup_term("example")
    assert result is not None
    assert len(result) == 0


def test_conversation_index_remove_term_if_empty(conversation_index: ConversationIndex):
    """Test removing terms if they are empty."""
    conversation_index.add_term("example", 1)
    conversation_index.remove_term("example", 1)
    conversation_index.remove_term_if_empty("example")

    assert len(conversation_index) == 0


def test_conversation_index_serialize_and_deserialize(
    conversation_index: ConversationIndex,
):
    """Test serialization and deserialization of the ConversationIndex."""
    conversation_index.add_term("example", 1)
    conversation_index.add_term("test", 2)

    serialized = conversation_index.serialize()
    assert "items" in serialized
    assert len(serialized["items"]) == 2

    new_index = ConversationIndex()
    new_index.deserialize(serialized)

    assert len(new_index) == 2

    example = new_index.lookup_term("example")
    assert example is not None
    assert len(example) >= 1
    assert example[0].semantic_ref_ordinal == 1

    test = new_index.lookup_term("test")
    assert test is not None
    assert len(test) >= 1
    assert test[0].semantic_ref_ordinal == 2


def test_add_entity_to_index(conversation_index: ConversationIndex):
    """Test adding an entity to the index."""
    entity = ConcreteEntity(
        name="ExampleEntity",
        type=["object", "example"],
        facets=[Facet(name="color", value="blue")],
    )
    semantic_refs = SemanticRefCollection()
    add_entity_to_index(entity, semantic_refs, conversation_index, 0)

    assert len(semantic_refs) == 1
    assert semantic_refs[0].knowledge_type == "entity"
    assert semantic_refs[0].knowledge.name == "ExampleEntity"

    result = conversation_index.lookup_term("ExampleEntity")
    assert result is not None
    assert len(result) == 1

    result = conversation_index.lookup_term("object")
    assert result is not None
    assert len(result) == 1

    result = conversation_index.lookup_term("color")
    assert result is not None
    assert len(result) == 1


def test_add_topic_to_index(conversation_index: ConversationIndex):
    """Test adding a topic to the index."""
    topic = "ExampleTopic"
    semantic_refs = SemanticRefCollection()
    add_topic_to_index(topic, semantic_refs, conversation_index, 0)

    assert len(semantic_refs) == 1
    assert semantic_refs[0].knowledge_type == "topic"
    assert semantic_refs[0].knowledge.text == "ExampleTopic"

    result = conversation_index.lookup_term("ExampleTopic")
    assert result is not None
    assert len(result) == 1


def test_add_action_to_index(conversation_index: ConversationIndex):
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
    semantic_refs = SemanticRefCollection()
    add_action_to_index(action, semantic_refs, conversation_index, 0)

    assert len(semantic_refs) == 1
    assert semantic_refs[0].knowledge_type == "action"
    assert semantic_refs[0].knowledge.verbs == ["run", "jump"]

    result = conversation_index.lookup_term("run jump")
    assert result is not None
    assert len(result) == 1

    result = conversation_index.lookup_term("John")
    assert result is not None
    assert len(result) == 1

    result = conversation_index.lookup_term("Ball")
    assert result
    assert len(result) == 1


def test_add_knowledge_to_index(conversation_index: ConversationIndex):
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
    semantic_refs = SemanticRefCollection()
    add_knowledge_to_index(semantic_refs, conversation_index, 0, knowledge)

    assert len(semantic_refs) == 3  # 1 entity + 1 action + 1 topic

    result = conversation_index.lookup_term("ExampleEntity")
    assert result is not None
    assert len(result) == 1

    result = conversation_index.lookup_term("run jump")
    assert result is not None
    assert len(result) == 1

    result = conversation_index.lookup_term("ExampleTopic")
    assert result is not None
    assert len(result) == 1


def test_conversation_index_len_and_get_terms(conversation_index: ConversationIndex):
    """Test __len__ and get_terms method."""
    assert len(conversation_index) == 0
    conversation_index.add_term("foo", 1)
    conversation_index.add_term("bar", 2)
    terms = conversation_index.get_terms()
    assert "foo" in terms
    assert "bar" in terms
    assert len(conversation_index) == 2


def test_conversation_index_contains(conversation_index: ConversationIndex):
    """Test presence of a term using lookup_term."""
    conversation_index.add_term("foo", 1)
    assert conversation_index.lookup_term("foo") != []
    assert conversation_index.lookup_term("bar") == []


def test_conversation_index_clear(conversation_index: ConversationIndex):
    """Test clear method."""
    conversation_index.add_term("foo", 1)
    conversation_index.add_term("bar", 2)
    conversation_index.clear()
    assert len(conversation_index) == 0
    assert conversation_index.lookup_term("foo") == []


def test_conversation_index_remove_term_nonexistent(
    conversation_index: ConversationIndex,
):
    """Test removing a term that does not exist does not raise."""
    conversation_index.remove_term("nonexistent", 123)  # Should not raise


def test_conversation_index_remove_term_if_empty_nonexistent(
    conversation_index: ConversationIndex,
):
    """Test remove_term_if_empty on a term that does not exist."""
    conversation_index.remove_term_if_empty("nonexistent")  # Should not raise


def test_conversation_index_serialize_empty(conversation_index: ConversationIndex):
    """Test serialize on an empty index."""
    serialized = conversation_index.serialize()
    assert "items" in serialized
    assert serialized["items"] == []
