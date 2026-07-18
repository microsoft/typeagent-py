# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

import typechat

from typeagent.knowpro.answer_response_schema import AnswerResponse
from typeagent.knowpro.answers import (
    AnswerGeneratorSettings,
    facets_to_merged_facets,
    generate_answers,
    get_enclosing_date_range_for_text_range,
    get_enclosing_text_range,
    merged_facets_to_facets,
    text_range_from_message_range,
)
from typeagent.knowpro.interfaces import TextLocation, TextRange
from typeagent.knowpro.knowledge_schema import Facet
from typeagent.knowpro.search import ConversationSearchResult

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
    async def test_multi_message_range_uses_exclusive_end(
        self, messages: FakeMessageCollection
    ) -> None:
        """Range [0, 2) should use message 2 (the exclusive end) for end timestamp."""
        tr = TextRange(
            start=TextLocation(0),
            end=TextLocation(2),  # exclusive end
        )
        dr = await get_enclosing_date_range_for_text_range(messages, tr)
        assert dr is not None
        assert dr.start.hour == 0
        # End timestamp comes from the message at the exclusive end ordinal:
        assert dr.end is not None
        assert dr.end.hour == 2

    @pytest.mark.asyncio
    async def test_adjacent_messages(self, messages: FakeMessageCollection) -> None:
        """Range [1, 2) covers only message 1; end timestamp is message 2."""
        tr = TextRange(
            start=TextLocation(1),
            end=TextLocation(2),
        )
        dr = await get_enclosing_date_range_for_text_range(messages, tr)
        assert dr is not None
        assert dr.start.hour == 1
        assert dr.end is not None
        assert dr.end.hour == 2  # exclusive end: timestamp of the next message

    @pytest.mark.asyncio
    async def test_end_past_last_message(self, messages: FakeMessageCollection) -> None:
        """If the exclusive end ordinal is past the last message, end is None."""
        tr = TextRange(
            start=TextLocation(0),
            end=TextLocation(3),  # messages only have ordinals 0, 1, 2
        )
        dr = await get_enclosing_date_range_for_text_range(messages, tr)
        assert dr is not None
        assert dr.start.hour == 0
        assert dr.end is None

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


# ---------------------------------------------------------------------------
# generate_answers: concurrency and fast_stop
# ---------------------------------------------------------------------------


class FakeAnswerTranslator:
    """Records concurrency/call info; returns canned responses in call order."""

    def __init__(
        self,
        responses: list[AnswerResponse],
        delay: float = 0.0,
    ) -> None:
        self._responses = responses
        self._delay = delay
        self.calls = 0
        self.max_concurrent = 0
        self._concurrent = 0

    async def translate(self, request: str) -> typechat.Result[AnswerResponse]:
        self._concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self._concurrent)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            index = self.calls
            self.calls += 1
            return typechat.Success(self._responses[index])
        finally:
            self._concurrent -= 1


def make_search_results(count: int) -> list[ConversationSearchResult]:
    return [
        ConversationSearchResult(
            message_matches=[], knowledge_matches={}, raw_query_text=f"question {i}"
        )
        for i in range(count)
    ]


class TestGenerateAnswersConcurrency:
    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self) -> None:
        translator = FakeAnswerTranslator(
            responses=[
                AnswerResponse(type="NoAnswer", why_no_answer="none") for _ in range(5)
            ],
            delay=0.01,
        )
        search_results = make_search_results(5)

        all_answers, _combined = await generate_answers(
            translator,  # type: ignore[arg-type]
            search_results,
            None,  # type: ignore[arg-type]  # conversation is unused: matches are empty
            "orig question",
            settings=AnswerGeneratorSettings(concurrency=2, fast_stop=False),
        )

        assert translator.calls == 5
        assert len(all_answers) == 5
        assert translator.max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_fast_stop_skips_unstarted_results(self) -> None:
        responses = [AnswerResponse(type="Answered", answer="the answer")] + [
            AnswerResponse(type="NoAnswer", why_no_answer="none") for _ in range(4)
        ]
        translator = FakeAnswerTranslator(responses=responses)
        search_results = make_search_results(5)

        all_answers, combined = await generate_answers(
            translator,  # type: ignore[arg-type]
            search_results,
            None,  # type: ignore[arg-type]
            "orig question",
            # concurrency=1 makes execution deterministically sequential, so
            # the first result answers before any others are started.
            settings=AnswerGeneratorSettings(concurrency=1, fast_stop=True),
        )

        assert translator.calls == 1
        assert len(all_answers) == 1
        assert combined.type == "Answered"
        assert combined.answer == "the answer"

    @pytest.mark.asyncio
    async def test_fast_stop_false_processes_all_results(self) -> None:
        responses = [AnswerResponse(type="Answered", answer="the answer")] + [
            AnswerResponse(type="NoAnswer", why_no_answer="none") for _ in range(4)
        ]
        translator = FakeAnswerTranslator(responses=responses)
        search_results = make_search_results(5)

        all_answers, _combined = await generate_answers(
            translator,  # type: ignore[arg-type]
            search_results,
            None,  # type: ignore[arg-type]
            "orig question",
            settings=AnswerGeneratorSettings(concurrency=1, fast_stop=False),
        )

        assert translator.calls == 5
        assert len(all_answers) == 5
