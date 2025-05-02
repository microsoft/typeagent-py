# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typeagent.knowpro.propindex import (
    PropertyIndex,
    PropertyNames,
    add_facet,
    add_entity_properties_to_index,
    add_action_properties_to_index,
    build_property_index,
    add_to_property_index,
    make_property_term_text,
    split_property_term_text,
    is_known_property,
)
from typeagent.knowpro.kplib import Facet, ConcreteEntity, Action
from typeagent.knowpro.interfaces import (
    SemanticRef,
    ListIndexingResult,
    TextLocation,
    TextRange,
)


@pytest.fixture
def property_index():
    """Fixture to create a PropertyIndex instance."""
    return PropertyIndex()


def test_add_facet(property_index):
    """Test adding a facet to the property index."""
    facet = Facet(name="color", value="blue")
    add_facet(facet, property_index, 1)

    result = property_index.lookup_property(PropertyNames.FacetName.value, "color")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = property_index.lookup_property(PropertyNames.FacetValue.value, "blue")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1


def test_add_entity_properties_to_index(property_index):
    """Test adding entity properties to the property index."""
    entity = ConcreteEntity(
        name="ExampleEntity",
        type=["object", "example"],
        facets=[Facet(name="color", value="blue")],
    )
    add_entity_properties_to_index(entity, property_index, 1)

    result = property_index.lookup_property(
        PropertyNames.EntityName.value, "ExampleEntity"
    )
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = property_index.lookup_property(PropertyNames.EntityType.value, "object")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = property_index.lookup_property(PropertyNames.FacetName.value, "color")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1


def test_add_action_properties_to_index(property_index):
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
    add_action_properties_to_index(action, property_index, 1)

    result = property_index.lookup_property(PropertyNames.Verb.value, "run jump")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = property_index.lookup_property(PropertyNames.Subject.value, "John")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = property_index.lookup_property(PropertyNames.Object.value, "Ball")
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1


def test_make_and_split_property_term_text():
    """Test creating and splitting property term text."""
    term_text = make_property_term_text("name", "value")
    assert term_text == "prop.name@@value"

    name, value = split_property_term_text(term_text)
    assert name == "prop.name"
    assert value == "value"


def test_is_known_property(property_index):
    """Test checking if a property is known."""
    property_index.add_property("name", "value", 1)

    assert is_known_property(property_index, PropertyNames.EntityName, "value") is True
    assert (
        is_known_property(property_index, PropertyNames.EntityName, "unknown") is False
    )


def test_build_property_index(mocker):
    """Test building a property index from a conversation."""
    mock_conversation = mocker.MagicMock()
    mock_conversation.secondary_indexes = mocker.MagicMock()
    mock_conversation.semantic_refs = [
        SemanticRef(
            semantic_ref_ordinal=0,
            range=TextRange(start=TextLocation(0), end=None),
            knowledge_type="entity",
            knowledge=ConcreteEntity(
                name="ExampleEntity",
                type=["object"],
                facets=[Facet(name="color", value="blue")],
            ),
        )
    ]

    result = build_property_index(mock_conversation)
    assert isinstance(result, ListIndexingResult)
    assert result.number_completed == 1


def test_add_to_property_index(property_index):
    """Test adding semantic references to the property index."""
    semantic_refs = [
        SemanticRef(
            semantic_ref_ordinal=0,
            range=TextRange(start=TextLocation(0), end=None),
            knowledge_type="entity",
            knowledge=ConcreteEntity(
                name="ExampleEntity",
                type=["object"],
                facets=[Facet(name="color", value="blue")],
            ),
        )
    ]
    result = add_to_property_index(property_index, semantic_refs, 0)
    assert isinstance(result, ListIndexingResult)
    assert result.number_completed == 1

    lookup_result = property_index.lookup_property(
        PropertyNames.EntityName.value, "ExampleEntity"
    )
    assert len(lookup_result) == 1
    assert lookup_result[0].semantic_ref_ordinal == 0
