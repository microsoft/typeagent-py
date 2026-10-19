# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Memory and SQLite term indexes must normalize terms identically (#322)."""

from collections.abc import Generator
import sqlite3

import pytest

from typeagent.knowpro.common import normalize_term
from typeagent.knowpro.interfaces import (
    IPropertyToSemanticRefIndex,
    ITermToSemanticRefIndex,
)
from typeagent.storage.memory.property_index import PropertyIndex
from typeagent.storage.memory.semantic_ref_index import TermToSemanticRefIndex
from typeagent.storage.sqlite.property_index import SqlitePropertyIndex
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
