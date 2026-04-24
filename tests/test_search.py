# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for knowpro/search.py — SearchOptions, ConversationSearchResult."""

import pytest

from typeagent.knowpro.interfaces import (
    SearchTerm,
    SearchTermGroup,
    Term,
)
from typeagent.knowpro.interfaces_core import ScoredMessageOrdinal
from typeagent.knowpro.query import is_conversation_searchable
from typeagent.knowpro.search import (
    ConversationSearchResult,
    search_conversation_knowledge,
    SearchOptions,
)

from conftest import FakeConversation, FakeMessage, FakeTermIndex

# ---------------------------------------------------------------------------
# SearchOptions
# ---------------------------------------------------------------------------


def test_search_options_defaults() -> None:
    opts = SearchOptions()
    assert opts.max_knowledge_matches is None
    assert opts.exact_match is False
    assert opts.max_message_matches is None
    assert opts.max_chars_in_budget is None
    assert opts.threshold_score is None


def test_search_options_repr_empty() -> None:
    opts = SearchOptions()
    # Only non-None values appear in repr; exact_match=False is still included.
    r = repr(opts)
    assert r.startswith("SearchOptions(")


def test_search_options_repr_with_fields() -> None:
    opts = SearchOptions(max_knowledge_matches=5, exact_match=True)
    r = repr(opts)
    assert "max_knowledge_matches=5" in r
    assert "exact_match=True" in r


# ---------------------------------------------------------------------------
# ConversationSearchResult
# ---------------------------------------------------------------------------


def test_conversation_search_result_basic() -> None:
    result = ConversationSearchResult(
        message_matches=[ScoredMessageOrdinal(0, 0.9)],
        knowledge_matches={},
        raw_query_text="test",
    )
    assert len(result.message_matches) == 1
    assert result.raw_query_text == "test"


def test_conversation_search_result_defaults() -> None:
    result = ConversationSearchResult(message_matches=[], knowledge_matches={})
    assert result.raw_query_text is None


# ---------------------------------------------------------------------------
# is_conversation_searchable (from query.py, used heavily in search.py)
# ---------------------------------------------------------------------------


def test_is_conversation_searchable_true() -> None:
    conv = FakeConversation(
        messages=[FakeMessage("hello", 0)],
        has_secondary_indexes=False,
    )
    conv.semantic_ref_index = FakeTermIndex()
    assert is_conversation_searchable(conv) is True


def test_is_conversation_searchable_no_index() -> None:
    conv = FakeConversation(has_secondary_indexes=False)
    conv.semantic_ref_index = None
    assert is_conversation_searchable(conv) is False


def test_is_conversation_searchable_no_semrefs() -> None:
    conv = FakeConversation(has_secondary_indexes=False)
    conv.semantic_refs = None  # type: ignore[assignment]
    assert is_conversation_searchable(conv) is False


# ---------------------------------------------------------------------------
# search_conversation_knowledge returns None when not searchable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_conversation_knowledge_non_searchable_returns_none() -> None:
    """When the conversation has no semantic ref index, result should be None."""
    conv = FakeConversation(has_secondary_indexes=False)
    conv.semantic_ref_index = None

    group = SearchTermGroup(
        boolean_op="or",
        terms=[SearchTerm(term=Term("hello"))],
    )
    result = await search_conversation_knowledge(conv, group)
    assert result is None
