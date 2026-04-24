# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for storage/memory/semrefindex.py helper functions."""

import pytest

from typeagent.knowpro import knowledge_schema as kplib
from typeagent.knowpro.interfaces import Topic
from typeagent.storage.memory import MemorySemanticRefCollection
from typeagent.storage.memory.semrefindex import (
    add_action,
    add_entity,
    add_facet,
    add_term_to_index,
    add_topic,
)

from conftest import FakeTermIndex


def make_semrefs() -> MemorySemanticRefCollection:
    return MemorySemanticRefCollection([])


def make_index() -> FakeTermIndex:
    return FakeTermIndex()


# ---------------------------------------------------------------------------
# add_term_to_index
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_term_to_index_basic() -> None:
    index = make_index()
    terms_added: set[str] = set()
    await add_term_to_index(index, "hello", 0, terms_added)
    assert "hello" in terms_added
    assert await index.size() == 1


@pytest.mark.asyncio
async def test_add_term_to_index_no_terms_added_set() -> None:
    index = make_index()
    await add_term_to_index(index, "world", 1)
    assert await index.size() == 1


@pytest.mark.asyncio
async def test_add_term_empty_string_is_stored() -> None:
    """The function does not filter empty terms — delegated to the index."""
    index = make_index()
    await add_term_to_index(index, "", 0)
    # FakeTermIndex stores empty strings too
    assert await index.size() == 1


# ---------------------------------------------------------------------------
# add_facet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_facet_none_does_nothing() -> None:
    index = make_index()
    await add_facet(None, 0, index)
    assert await index.size() == 0


@pytest.mark.asyncio
async def test_add_facet_string_value() -> None:
    index = make_index()
    facet = kplib.Facet(name="colour", value="red")
    await add_facet(facet, 0, index)
    terms = await index.get_terms()
    assert "colour" in terms
    assert "red" in terms


@pytest.mark.asyncio
async def test_add_facet_numeric_value() -> None:
    index = make_index()
    facet = kplib.Facet(name="count", value=42.0)
    await add_facet(facet, 0, index)
    terms = await index.get_terms()
    assert "count" in terms
    assert "42.0" in terms


# ---------------------------------------------------------------------------
# add_entity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_entity_registers_name_and_types() -> None:
    semrefs = make_semrefs()
    index = make_index()
    entity = kplib.ConcreteEntity(name="Alice", type=["person", "employee"])
    terms_added: set[str] = set()
    await add_entity(
        entity,
        semrefs,
        index,
        message_ordinal=0,
        chunk_ordinal=0,
        terms_added=terms_added,
    )
    assert "Alice" in terms_added
    assert "person" in terms_added
    assert "employee" in terms_added
    assert await semrefs.size() == 1


@pytest.mark.asyncio
async def test_add_entity_with_facets() -> None:
    semrefs = make_semrefs()
    index = make_index()
    entity = kplib.ConcreteEntity(
        name="Book",
        type=["item"],
        facets=[kplib.Facet(name="genre", value="fiction")],
    )
    await add_entity(entity, semrefs, index, message_ordinal=1, chunk_ordinal=0)
    terms = await index.get_terms()
    assert "genre" in terms
    assert "fiction" in terms


# ---------------------------------------------------------------------------
# add_topic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_topic_registers_text() -> None:
    semrefs = make_semrefs()
    index = make_index()
    topic = Topic(text="machine learning")
    terms_added: set[str] = set()
    await add_topic(
        topic,
        semrefs,
        index,
        message_ordinal=2,
        chunk_ordinal=0,
        terms_added=terms_added,
    )
    assert "machine learning" in terms_added
    assert await semrefs.size() == 1


# ---------------------------------------------------------------------------
# add_action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_action_registers_verbs() -> None:
    semrefs = make_semrefs()
    index = make_index()
    action = kplib.Action(
        verbs=["run", "execute"],
        verb_tense="present",
        subject_entity_name="Alice",
        object_entity_name="script",
        indirect_object_entity_name="none",
    )
    terms_added: set[str] = set()
    await add_action(
        action,
        semrefs,
        index,
        message_ordinal=0,
        chunk_ordinal=0,
        terms_added=terms_added,
    )
    terms = set(await index.get_terms())
    assert "run execute" in terms
    assert "Alice" in terms
    assert "script" in terms
    assert await semrefs.size() == 1


@pytest.mark.asyncio
async def test_add_action_none_entities_skipped() -> None:
    semrefs = make_semrefs()
    index = make_index()
    action = kplib.Action(
        verbs=["go"],
        verb_tense="present",
        subject_entity_name="none",
        object_entity_name="none",
        indirect_object_entity_name="none",
    )
    await add_action(action, semrefs, index, message_ordinal=0, chunk_ordinal=0)
    terms = await index.get_terms()
    assert "none" not in terms
    assert "go" in terms
