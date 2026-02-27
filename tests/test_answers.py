# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from typeagent.knowpro.answers import (
    facets_to_merged_facets,
    get_enclosing_date_range_for_text_range,
    get_enclosing_text_range,
    merged_facets_to_facets,
    text_range_from_message_range,
)
from typeagent.knowpro.interfaces import TextLocation, TextRange
from typeagent.knowpro.kplib import Facet

from conftest import FakeMessage, FakeMessageCollection

# ---------------------------------------------------------------------------
# Change 1: facets_to_merged_facets uses str(facet.value), not str(facet)
# ---------------------------------------------------------------------------


class TestFacetsToMergedFacets:
    """Verify that facet *values* (not the whole Facet object) are stringified."""

    def test_string_value(self) -> None:
        facets = [Facet(name="colour", value="red")]
        merged = facets_to_merged_facets(facets)
        assert merged == {"colour": ["red"]}

    def test_numeric_value(self) -> None:
        facets = [Facet(name="age", value=30.0)]
        merged = facets_to_merged_facets(facets)
        # Should be "30.0", NOT "Facet('age', 30.0)"
        assert merged == {"age": ["30.0"]}
        assert "Facet" not in merged["age"][0]

    def test_bool_value(self) -> None:
        facets = [Facet(name="active", value=True)]
        merged = facets_to_merged_facets(facets)
        assert merged == {"active": ["true"]}

    def test_multiple_facets_same_name(self) -> None:
        facets = [
            Facet(name="tag", value="a"),
            Facet(name="tag", value="b"),
        ]
        merged = facets_to_merged_facets(facets)
        assert merged == {"tag": ["a", "b"]}

    def test_lowercases_names_and_values(self) -> None:
        facets = [Facet(name="Colour", value="RED")]
        merged = facets_to_merged_facets(facets)
        assert "colour" in merged
        assert merged["colour"] == ["red"]

    def test_roundtrip_through_merged(self) -> None:
        """facets_to_merged_facets -> merged_facets_to_facets preserves semantics."""
        original = [
            Facet(name="colour", value="red"),
            Facet(name="colour", value="blue"),
            Facet(name="size", value="large"),
        ]
        merged = facets_to_merged_facets(original)
        restored = merged_facets_to_facets(merged)
        restored_by_name = {f.name: f.value for f in restored}
        assert restored_by_name["colour"] == "red; blue"
        assert restored_by_name["size"] == "large"


# ---------------------------------------------------------------------------
# Change 2: get_enclosing_date_range_for_text_range uses ordinal-1 for end
# ---------------------------------------------------------------------------


class TestGetEnclosingDateRangeForTextRange:
    """Verify the off-by-one fix: end is exclusive, so we subtract 1."""

    @pytest_asyncio.fixture()
    async def messages(self) -> AsyncGenerator[FakeMessageCollection, None]:
        """Three messages with ordinals 0, 1, 2 and timestamps derived from them."""
        coll = FakeMessageCollection()
        for i in range(3):
            msg = FakeMessage("text", message_ordinal=i)
            await coll.append(msg)
        yield coll

    @pytest.mark.asyncio
    async def test_single_message_range(self, messages: FakeMessageCollection) -> None:
        """Point range (end=None) should use only the start message's timestamp."""
        tr = TextRange(start=TextLocation(1))
        dr = await get_enclosing_date_range_for_text_range(messages, tr)
        assert dr is not None
        assert dr.start.hour == 1
        assert dr.end is None

    @pytest.mark.asyncio
    async def test_multi_message_range_uses_last_included(
        self, messages: FakeMessageCollection
    ) -> None:
        """Range [0, 2) should use message 1 for end (ordinal 2-1=1), not message 2."""
        tr = TextRange(
            start=TextLocation(0),
            end=TextLocation(2),  # exclusive end
        )
        dr = await get_enclosing_date_range_for_text_range(messages, tr)
        assert dr is not None
        assert dr.start.hour == 0
        # End timestamp comes from message ordinal 1 (= 2-1), NOT ordinal 2:
        assert dr.end is not None
        assert dr.end.hour == 1

    @pytest.mark.asyncio
    async def test_adjacent_messages(self, messages: FakeMessageCollection) -> None:
        """Range [1, 2) covers only message 1."""
        tr = TextRange(
            start=TextLocation(1),
            end=TextLocation(2),
        )
        dr = await get_enclosing_date_range_for_text_range(messages, tr)
        assert dr is not None
        assert dr.start.hour == 1
        assert dr.end is not None
        assert dr.end.hour == 1  # same message: end-1 == start

    @pytest.mark.asyncio
    async def test_no_timestamp_returns_none(self) -> None:
        """If start message has no timestamp, return None."""
        coll = FakeMessageCollection()
        msg = FakeMessage("text")  # no message_ordinal → no timestamp
        await coll.append(msg)
        tr = TextRange(start=TextLocation(0))
        dr = await get_enclosing_date_range_for_text_range(coll, tr)
        assert dr is None


# ---------------------------------------------------------------------------
# Helper functions (also exercised for completeness)
# ---------------------------------------------------------------------------


class TestGetEnclosingTextRange:
    def test_single_ordinal(self) -> None:
        tr = get_enclosing_text_range([5])
        assert tr is not None
        assert tr.start.message_ordinal == 5
        assert tr.end is None  # point range

    def test_multiple_ordinals(self) -> None:
        tr = get_enclosing_text_range([3, 1, 7])
        assert tr is not None
        assert tr.start.message_ordinal == 1
        assert tr.end is not None
        assert tr.end.message_ordinal == 7

    def test_empty_ordinals(self) -> None:
        tr = get_enclosing_text_range([])
        assert tr is None


class TestTextRangeFromMessageRange:
    def test_point(self) -> None:
        tr = text_range_from_message_range(3, 3)
        assert tr is not None
        assert tr.start.message_ordinal == 3
        assert tr.end is None

    def test_range(self) -> None:
        tr = text_range_from_message_range(2, 5)
        assert tr is not None
        assert tr.start.message_ordinal == 2
        assert tr.end is not None
        assert tr.end.message_ordinal == 5

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Expect message ordinal range"):
            text_range_from_message_range(5, 2)
