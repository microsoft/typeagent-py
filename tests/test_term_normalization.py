# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Memory and SQLite term indexes must normalize terms identically (#322)."""

import sqlite3

import pytest

from typeagent.knowpro.common import normalize_term
from typeagent.knowpro.interfaces import (
    IPropertyToSemanticRefIndex,
    ITermToSemanticRefIndex,
)
from typeagent.storage.memory.propindex import PropertyIndex
from typeagent.storage.memory.semrefindex import TermToSemanticRefIndex
from typeagent.storage.sqlite.propindex import SqlitePropertyIndex
from typeagent.storage.sqlite.schema import init_db_schema
from typeagent.storage.sqlite.semrefindex import SqliteTermToSemanticRefIndex


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


@pytest.fixture(params=["memory", "sqlite"])
def term_index(request: pytest.FixtureRequest):
    if request.param == "memory":
        yield TermToSemanticRefIndex()
    else:
        db = sqlite3.connect(":memory:")
        init_db_schema(db)
        yield SqliteTermToSemanticRefIndex(db)
        db.close()


@pytest.fixture(params=["memory", "sqlite"])
def prop_index(request: pytest.FixtureRequest):
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
async def test_term_index_whitespace_only_terms_agree(
    term_index: ITermToSemanticRefIndex,
) -> None:
    await term_index.add_term(" ", 0)
    await term_index.add_term("\t", 0)
    assert await term_index.size() == 1
    assert await term_index.get_terms() == [""]


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
