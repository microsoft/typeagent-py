# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typeagent.knowpro.collections import TextRangeCollection, TextRangesInScope
from typeagent.knowpro.interfaces import (
    SemanticRef,
    Tag,
    TextLocation,
    TextRange,
)
from typeagent.knowpro.kplib import Facet, ConcreteEntity, Action
from typeagent.storage.memory.propindex import (
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
from typeagent.storage.memory import MemorySemanticRefCollection

from fixtures import needs_auth, FakeConversation


@pytest.fixture
def property_index() -> PropertyIndex:
    """Fixture to create a PropertyIndex instance."""
    return PropertyIndex()


@pytest.mark.asyncio
async def test_add_facet(property_index: PropertyIndex):
    """Test adding a facet to the property index."""
    facet = Facet(name="color", value="blue")
    await add_facet(facet, property_index, 1)

    result = await property_index.lookup_property(
        PropertyNames.FacetName.value, "color"
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(
        PropertyNames.FacetValue.value, "blue"
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1


@pytest.mark.asyncio
async def test_add_entity_properties_to_index(property_index: PropertyIndex):
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
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(
        PropertyNames.EntityType.value, "object"
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(
        PropertyNames.FacetName.value, "color"
    )
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1


@pytest.mark.asyncio
async def test_add_action_properties_to_index(property_index: PropertyIndex):
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
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(PropertyNames.Subject.value, "John")
    assert result is not None
    assert len(result) == 1
    assert result[0].semantic_ref_ordinal == 1

    result = await property_index.lookup_property(PropertyNames.Object.value, "Ball")
    assert result is not None
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
async def test_is_known_property(property_index: PropertyIndex):
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
async def test_build_property_index(needs_auth: None):
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

    conversation = FakeConversation(
        semantic_refs=semantic_refs, has_secondary_indexes=True
    )
    await conversation.ensure_initialized()  # Ensure the secondary indexes are set up

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


@pytest.mark.asyncio
async def test_add_to_property_index(needs_auth: None, property_index: PropertyIndex):
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
    conversation = FakeConversation(
        semantic_refs=semantic_refs, has_secondary_indexes=True
    )
    await conversation.ensure_initialized()  # Ensure the secondary indexes are set up
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
async def test_lookup_property_in_property_index(property_index: PropertyIndex):
    """Test looking up properties in the property index."""
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
async def test_property_index_clear(property_index: PropertyIndex):
    """Test clearing all properties in the PropertyIndex."""
    await property_index.add_property("name", "value", 0)
    assert await property_index.size() > 0

    await property_index.clear()
    assert await property_index.size() == 0


@pytest.mark.asyncio
async def test_property_index_get_values(property_index: PropertyIndex):
    """Test retrieving all property values from the PropertyIndex."""
    await property_index.add_property("name", "value1", 0)
    await property_index.add_property("name", "value2", 1)

    values = await property_index.get_values()
    assert len(values) == 2
    assert "value1" in values
    assert "value2" in values


def test_property_index_prepare_term_text(property_index: PropertyIndex):
    """Test preprocessing term text in PropertyIndex."""
    term_text = "Prop.Name@@Value"
    prepared_text = property_index._prepare_term_text(term_text)
    assert prepared_text == "prop.name@@value"  # Should be converted to lowercase


@pytest.mark.asyncio
async def test_property_index_lookup_property_edge_cases(property_index: PropertyIndex):
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
