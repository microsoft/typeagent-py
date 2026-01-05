# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typechat import Failure, Result, Success

from typeagent.knowpro import convknowledge, kplib
from typeagent.knowpro.knowledge import (
    create_knowledge_extractor,
    extract_knowledge_from_text,
    extract_knowledge_from_text_batch,
    merge_concrete_entities,
    merge_topics,
)
from typeagent.knowpro.kplib import ConcreteEntity, Facet


class MockKnowledgeExtractor:
    async def extract(self, text: str) -> Result[kplib.KnowledgeResponse]:
        if text == "error":
            return Failure("Extraction failed")
        return Success(
            kplib.KnowledgeResponse(
                entities=[], actions=[], inverse_actions=[], topics=[text]
            )
        )


@pytest.fixture
def mock_knowledge_extractor() -> convknowledge.KnowledgeExtractor:
    """Fixture to create a mock KnowledgeExtractor."""
    return MockKnowledgeExtractor()  # type: ignore


def test_create_knowledge_extractor(really_needs_auth: None):
    """Test creating a knowledge extractor."""
    extractor = create_knowledge_extractor()
    assert isinstance(extractor, convknowledge.KnowledgeExtractor)


@pytest.mark.asyncio
async def test_extract_knowledge_from_text(
    mock_knowledge_extractor: convknowledge.KnowledgeExtractor,
):
    """Test extracting knowledge from a single text input."""
    result = await extract_knowledge_from_text(mock_knowledge_extractor, "test text", 3)
    assert isinstance(result, Success)
    assert result.value.topics[0] == "test text"

    failure_result = await extract_knowledge_from_text(
        mock_knowledge_extractor, "error", 3
    )
    assert isinstance(failure_result, Failure)
    assert failure_result.message == "Extraction failed"


@pytest.mark.asyncio
async def test_extract_knowledge_from_text_batch(
    mock_knowledge_extractor: convknowledge.KnowledgeExtractor,
):
    """Test extracting knowledge from a batch of text inputs."""
    text_batch = ["text 1", "text 2", "error"]
    results = await extract_knowledge_from_text_batch(
        mock_knowledge_extractor, text_batch, 2, 3
    )

    assert len(results) == 3
    assert isinstance(results[0], Success)
    assert results[0].value.topics[0] == "text 1"
    assert isinstance(results[1], Success)
    assert results[1].value.topics[0] == "text 2"
    assert isinstance(results[2], Failure)
    assert results[2].message == "Extraction failed"


def test_merge_topics():
    """Test merging a list of topics into a unique list."""
    topics = ["topic1", "topic2", "topic1", "topic3"]
    merged_topics = merge_topics(topics)

    assert len(merged_topics) == 3
    assert "topic1" in merged_topics
    assert "topic2" in merged_topics
    assert "topic3" in merged_topics


# Tests for merge_concrete_entities


def test_merge_concrete_entities_empty_list() -> None:
    """Test merging an empty list returns empty list."""
    result = merge_concrete_entities([])
    assert result == []


def test_merge_concrete_entities_single_entity() -> None:
    """Test merging a single entity preserves case."""
    entity = ConcreteEntity(name="Alice", type=["Person"])
    result = merge_concrete_entities([entity])

    assert len(result) == 1
    assert result[0].name == "Alice"
    assert result[0].type == ["Person"]


def test_merge_concrete_entities_distinct() -> None:
    """Test merging distinct entities keeps them separate."""
    entities = [
        ConcreteEntity(name="Alice", type=["Person"]),
        ConcreteEntity(name="Bob", type=["Person"]),
    ]
    result = merge_concrete_entities(entities)

    assert len(result) == 2
    names = {e.name for e in result}
    assert names == {"Alice", "Bob"}


def test_merge_concrete_entities_same_name_different_case() -> None:
    """Test that entities with different case names are NOT merged (case-sensitive)."""
    entities = [
        ConcreteEntity(name="Alice", type=["Person"]),
        ConcreteEntity(name="ALICE", type=["Employee"]),
        ConcreteEntity(name="alice", type=["Manager"]),
    ]
    result = merge_concrete_entities(entities)

    # Case-sensitive: all three are distinct
    assert len(result) == 3
    names = {e.name for e in result}
    assert names == {"Alice", "ALICE", "alice"}


def test_merge_concrete_entities_types_deduplicated_and_sorted() -> None:
    """Test that merged types are deduplicated and sorted."""
    entities = [
        ConcreteEntity(name="Alice", type=["Person", "Employee"]),
        ConcreteEntity(name="Alice", type=["Employee", "Manager"]),
    ]
    result = merge_concrete_entities(entities)

    assert len(result) == 1
    assert result[0].type == ["Employee", "Manager", "Person"]


def test_merge_concrete_entities_with_facets() -> None:
    """Test merging entities with facets."""
    entities = [
        ConcreteEntity(
            name="Alice",
            type=["Person"],
            facets=[Facet(name="age", value="30")],
        ),
        ConcreteEntity(
            name="Alice",
            type=["Employee"],
            facets=[Facet(name="department", value="Engineering")],
        ),
    ]
    result = merge_concrete_entities(entities)

    assert len(result) == 1
    assert result[0].facets is not None
    facet_names = {f.name for f in result[0].facets}
    assert facet_names == {"age", "department"}


def test_merge_concrete_entities_same_facet_combines_values() -> None:
    """Test that facets with the same name have values combined."""
    entities = [
        ConcreteEntity(
            name="Alice",
            type=["Person"],
            facets=[Facet(name="hobby", value="reading")],
        ),
        ConcreteEntity(
            name="Alice",
            type=["Person"],
            facets=[Facet(name="hobby", value="swimming")],
        ),
    ]
    result = merge_concrete_entities(entities)

    assert len(result) == 1
    assert result[0].facets is not None
    hobby_facet = next(f for f in result[0].facets if f.name == "hobby")
    assert hobby_facet.value == "reading; swimming"


def test_merge_concrete_entities_facets_deduplicated() -> None:
    """Test that duplicate facet values are deduplicated."""
    entities = [
        ConcreteEntity(
            name="Alice",
            type=["Person"],
            facets=[Facet(name="hobby", value="reading")],
        ),
        ConcreteEntity(
            name="Alice",
            type=["Person"],
            facets=[Facet(name="hobby", value="reading")],  # Duplicate
        ),
        ConcreteEntity(
            name="Alice",
            type=["Person"],
            facets=[Facet(name="hobby", value="swimming")],
        ),
    ]
    result = merge_concrete_entities(entities)

    assert len(result) == 1
    assert result[0].facets is not None
    hobby_facet = next(f for f in result[0].facets if f.name == "hobby")
    assert hobby_facet.value == "reading; swimming"


def test_merge_concrete_entities_without_facets_with_facets() -> None:
    """Test merging an entity without facets with one that has facets."""
    entities = [
        ConcreteEntity(name="Alice", type=["Person"]),
        ConcreteEntity(
            name="Alice",
            type=["Employee"],
            facets=[Facet(name="department", value="Engineering")],
        ),
    ]
    result = merge_concrete_entities(entities)

    assert len(result) == 1
    assert result[0].facets is not None
    assert len(result[0].facets) == 1
    assert result[0].facets[0].name == "department"
