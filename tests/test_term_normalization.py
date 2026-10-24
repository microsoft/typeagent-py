# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Memory and SQLite term indexes must normalize terms identically (#322)."""

from collections.abc import Generator
import sqlite3
from typing import cast

import pytest

from typeagent.knowpro.common import normalize_term
from typeagent.knowpro.interfaces import (
    IConversation,
    IPropertyToSemanticRefIndex,
    ITermToRelatedTerms,
    ITermToSemanticRefIndex,
    SearchTerm,
    Term,
)
from typeagent.knowpro.knowledge_schema import ConcreteEntity
from typeagent.knowpro.query import (
    get_matching_term_for_text,
    match_entity_name_or_type,
)
from typeagent.knowpro.search import QueryCompiler
from typeagent.knowpro.searchlib import create_search_terms
from typeagent.storage.memory.property_index import PropertyIndex
from typeagent.storage.memory.related_terms_index import TermToRelatedTermsMap
from typeagent.storage.memory.semantic_ref_index import TermToSemanticRefIndex
from typeagent.storage.sqlite.property_index import SqlitePropertyIndex
from typeagent.storage.sqlite.related_terms_index import SqliteRelatedTermsAliases
from typeagent.storage.sqlite.schema import init_db_schema
from typeagent.storage.sqlite.semantic_ref_index import SqliteTermToSemanticRefIndex


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Space  Needle ", "space needle"),
        ("A ", "a"),
        ("\tfoo\n bar", "foo bar"),
        ("Café", "café"),  # NFD -> NFC
        (" ", ""),
        ("", ""),
    ],
)
def test_normalize_term(raw: str, expected: str) -> None:
    assert normalize_term(raw) == expected


# Terms whose lowercase form composes differently under NFC, so normalizing
# NFC-then-lowercase isn't idempotent.
NON_IDEMPOTENT_TERMS = [
    "T\u0308",  # Uppercase base + combining mark: lowercase composes to \u1e97
    "J\u030c",  # \u01f0 has no uppercase precomposed form
    "\u0391\u0342",  # Greek polytonic: Alpha + perispomeni lowercases to \u1fb6
]


@pytest.mark.parametrize("raw", NON_IDEMPOTENT_TERMS)
def test_normalize_term_is_idempotent(raw: str) -> None:
    once = normalize_term(raw)
    assert normalize_term(once) == once
    assert normalize_term(raw.lower()) == once


@pytest.fixture(params=["memory", "sqlite"])
def term_index(
    request: pytest.FixtureRequest,
) -> Generator[ITermToSemanticRefIndex, None, None]:
    if request.param == "memory":
        yield TermToSemanticRefIndex()
    else:
        db = sqlite3.connect(":memory:")
        init_db_schema(db)
        yield SqliteTermToSemanticRefIndex(db)
        db.close()


@pytest.fixture(params=["memory", "sqlite"])
def prop_index(
    request: pytest.FixtureRequest,
) -> Generator[IPropertyToSemanticRefIndex, None, None]:
    if request.param == "memory":
        yield PropertyIndex()
    else:
        db = sqlite3.connect(":memory:")
        init_db_schema(db)
        yield SqlitePropertyIndex(db)
        db.close()


@pytest.fixture(params=["memory", "sqlite"])
def alias_index(
    request: pytest.FixtureRequest,
) -> Generator[ITermToRelatedTerms, None, None]:
    if request.param == "memory":
        yield TermToRelatedTermsMap()
    else:
        db = sqlite3.connect(":memory:")
        init_db_schema(db)
        yield SqliteRelatedTermsAliases(db)
        db.close()


@pytest.mark.asyncio
async def test_term_index_normalizes_whitespace_and_case(
    term_index: ITermToSemanticRefIndex,
) -> None:
    assert await term_index.add_term("Space  Needle ", 1) == "space needle"
    assert await term_index.get_terms() == ["space needle"]
    found = await term_index.lookup_term("space needle")
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [1]
    found = await term_index.lookup_term("  SPACE\tNEEDLE")
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [1]


@pytest.mark.asyncio
async def test_term_index_lookup_ignores_trailing_space(
    term_index: ITermToSemanticRefIndex,
) -> None:
    await term_index.add_term("A", 0)
    found = await term_index.lookup_term("A ")
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [0]


@pytest.mark.asyncio
async def test_term_index_ignores_whitespace_only_terms(
    term_index: ITermToSemanticRefIndex,
) -> None:
    assert await term_index.add_term(" ", 0) == ""
    await term_index.add_term("\t", 0)
    await term_index.add_terms_batch([("\n", 1), (" Foo", 2)])
    assert await term_index.size() == 1
    assert await term_index.get_terms() == ["foo"]


@pytest.mark.asyncio
async def test_term_index_deserialize_merges_colliding_terms(
    term_index: ITermToSemanticRefIndex,
) -> None:
    data = {
        "items": [
            {
                "term": "Foo",
                "semanticRefOrdinals": [{"semanticRefOrdinal": 1, "score": 1.0}],
            },
            {
                "term": "foo ",
                "semanticRefOrdinals": [{"semanticRefOrdinal": 2, "score": 1.0}],
            },
        ]
    }
    await term_index.deserialize(data)  # type: ignore[arg-type]
    found = await term_index.lookup_term("foo")
    assert found is not None
    assert sorted(r.semantic_ref_ordinal for r in found) == [1, 2]


@pytest.mark.asyncio
async def test_property_index_normalizes_values(
    prop_index: IPropertyToSemanticRefIndex,
) -> None:
    await prop_index.add_property("name", "Space  Needle ", 1)
    found = await prop_index.lookup_property("name", "space needle")
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [1]
    found = await prop_index.lookup_property("NAME", " Space\tNeedle")
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [1]


@pytest.mark.asyncio
async def test_property_index_remove_property_normalizes_name(
    prop_index: IPropertyToSemanticRefIndex,
) -> None:
    await prop_index.add_property(" Name ", "x", 1)
    await prop_index.remove_property(" NAME", 1)
    assert not await prop_index.lookup_property("name", "x")


@pytest.mark.asyncio
async def test_term_index_deserialize_skips_empty_and_missing_terms(
    term_index: ITermToSemanticRefIndex,
) -> None:
    data = {
        "items": [
            {
                "term": "  ",
                "semanticRefOrdinals": [{"semanticRefOrdinal": 1, "score": 1.0}],
            },
            {
                "term": None,
                "semanticRefOrdinals": [{"semanticRefOrdinal": 2, "score": 1.0}],
            },
            {"semanticRefOrdinals": [{"semanticRefOrdinal": 3, "score": 1.0}]},
            {
                "term": "bar",
                "semanticRefOrdinals": [{"semanticRefOrdinal": 4, "score": 1.0}],
            },
        ]
    }
    await term_index.deserialize(data)  # type: ignore[arg-type]
    assert await term_index.get_terms() == ["bar"]


@pytest.mark.asyncio
async def test_term_index_deserialize_dedupes_merged_ordinals(
    term_index: ITermToSemanticRefIndex,
) -> None:
    data = {
        "items": [
            {
                "term": "Foo",
                "semanticRefOrdinals": [{"semanticRefOrdinal": 1, "score": 1.0}],
            },
            {
                "term": "foo ",
                "semanticRefOrdinals": [{"semanticRefOrdinal": 1, "score": 1.0}],
            },
        ]
    }
    await term_index.deserialize(data)  # type: ignore[arg-type]
    found = await term_index.lookup_term("foo")
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize("term", NON_IDEMPOTENT_TERMS)
async def test_term_index_normalization_round_trips(
    term_index: ITermToSemanticRefIndex, term: str
) -> None:
    """A stored term must normalize to itself, or it can't be looked up."""
    await term_index.add_term(term, 1)

    [stored] = await term_index.get_terms()
    found = await term_index.lookup_term(stored)
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [1]
    found = await term_index.lookup_term(term.lower())
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [1]

    # deserialize re-normalizes the already-normalized stored terms.
    data = await term_index.serialize()
    await term_index.clear()
    await term_index.deserialize(data)
    assert await term_index.get_terms() == [stored]
    found = await term_index.lookup_term(term)
    assert found is not None and [r.semantic_ref_ordinal for r in found] == [1]


# Search side and alias index (#346)


def test_create_search_terms_normalizes() -> None:
    [search_term] = create_search_terms([" Space\tNeedle ;Seattle  Tower"])
    assert search_term.term.text == "space needle"
    assert search_term.related_terms is not None
    assert [t.text for t in search_term.related_terms] == ["seattle tower"]


def test_validate_and_prepare_term_normalizes() -> None:
    # validate_and_prepare_term doesn't touch the conversation.
    compiler = QueryCompiler(cast(IConversation, None), None)
    term = Term(" Space\tNeedle ")
    assert compiler.validate_and_prepare_term(term)
    assert term.text == "space needle"


def test_get_matching_term_for_text_normalizes() -> None:
    related_term = Term("seattle tower")
    search_term = SearchTerm(Term("space needle"), related_terms=[related_term])
    assert get_matching_term_for_text(search_term, "Space  Needle") is (
        search_term.term
    )
    assert get_matching_term_for_text(search_term, " Seattle\tTower") is related_term
    assert get_matching_term_for_text(search_term, "space") is None


def test_match_entity_name_or_type_normalizes() -> None:
    entity = ConcreteEntity(name="Space  Needle", type=["Tourist  Attraction"])
    assert match_entity_name_or_type(SearchTerm(Term("space needle")), entity)
    assert match_entity_name_or_type(SearchTerm(Term("tourist attraction")), entity)
    entity = ConcreteEntity(name="Cafe\u0301", type=[])  # NFD
    assert match_entity_name_or_type(SearchTerm(Term("café")), entity)


@pytest.mark.asyncio
async def test_alias_index_normalizes_key_and_aliases(
    alias_index: ITermToRelatedTerms,
) -> None:
    await alias_index.add_related_term("Space  Needle", Term(" Seattle\tTower "))
    found = await alias_index.lookup_term(" SPACE\tNEEDLE")
    assert found is not None and [t.text for t in found] == ["seattle tower"]
    found = await alias_index.lookup_term("space needle")
    assert found is not None and [t.text for t in found] == ["seattle tower"]

    await alias_index.remove_term("Space Needle ")
    assert await alias_index.lookup_term("space needle") is None


@pytest.mark.asyncio
async def test_alias_index_normalizes_unicode_form(
    alias_index: ITermToRelatedTerms,
) -> None:
    await alias_index.add_related_term("Cafe\u0301", Term("coffee shop"))  # NFD
    found = await alias_index.lookup_term("café")  # NFC
    assert found is not None and [t.text for t in found] == ["coffee shop"]


@pytest.mark.asyncio
async def test_alias_index_merges_aliases_of_colliding_keys(
    alias_index: ITermToRelatedTerms,
) -> None:
    await alias_index.add_related_term("Space Needle", Term("tower"))
    await alias_index.add_related_term("space  needle", [Term("Tower"), Term("spire")])
    found = await alias_index.lookup_term("space needle")
    assert found is not None and sorted(t.text for t in found) == ["spire", "tower"]
    assert await alias_index.size() == 1


@pytest.mark.asyncio
async def test_alias_index_deserialize_normalizes(
    alias_index: ITermToRelatedTerms,
) -> None:
    await alias_index.deserialize(
        {
            "relatedTerms": [
                {
                    "termText": "Space  Needle",
                    "relatedTerms": [{"text": "Seattle Tower"}],
                },
                {"termText": "space needle", "relatedTerms": [{"text": "spire"}]},
            ]
        }
    )
    found = await alias_index.lookup_term("space needle")
    assert found is not None
    assert sorted(t.text for t in found) == ["seattle tower", "spire"]

    # A round trip keeps the normalized form.
    data = await alias_index.serialize()
    await alias_index.clear()
    await alias_index.deserialize(data)
    found = await alias_index.lookup_term(" Space\tNeedle")
    assert found is not None
    assert sorted(t.text for t in found) == ["seattle tower", "spire"]
