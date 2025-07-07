# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typeagent.knowpro.collections import (
    MatchAccumulator,
    SemanticRefAccumulator,
    TermSet,
    PropertyTermSet,
    TextRangeCollection,
    TextRangesInScope,
)
from typeagent.knowpro.interfaces import (
    IConversation,
    IMessage,
    ITermToSemanticRefIndex,
    Term,
    SearchTerm,
    PropertySearchTerm,
    SemanticRef,
    ScoredSemanticRefOrdinal,
    TextRange,
    TextLocation,
    Topic,
)
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.knowpro.query import (
    TextRangeSelector,
    is_conversation_searchable,
    lookup_term_filtered,
    lookup_term,
    QueryEvalContext,
    QueryOpExpr,
    SelectTopNExpr,
    MatchTermsOrExpr,
    MatchTermsOrMaxExpr,
    MatchTermsAndExpr,
    MatchTermExpr,
    MatchSearchTermExpr,
    MatchPropertySearchTermExpr,
    GetScopeExpr,
    get_text_range_for_date_range,
    get_matching_term_for_text,
    match_search_term_to_text,
    match_search_term_to_one_of_text,
    match_entity_name_or_type,
    lookup_knowledge_type,
)
from typeagent.knowpro.propindex import PropertyIndex
from typeagent.knowpro.storage import MessageCollection, SemanticRefCollection


def downcast[T](cls: type[T], obj: object) -> T:
    """Downcast an object to a specific type."""
    assert isinstance(obj, cls), f"Expected type {cls}, but got {type(obj)}"
    return obj


class MockMessage(IMessage):
    """Mock message for testing."""

    def __init__(self, message_ordinal: int, text: str = ""):
        self.ordinal = message_ordinal
        self.text_chunks = [text]
        # Assign a valid ISO-format timestamp string
        # For variety, increment the hour by message_ordinal
        self.timestamp = f"2020-01-01T0{message_ordinal}:00:00"
        self.tags = []

    def get_knowledge(self) -> KnowledgeResponse:
        raise RuntimeError


class MockMessageCollection(MessageCollection[MockMessage]):
    pass


def make_semantic_ref(ordinal: int, text_range: TextRange):
    return SemanticRef(
        semantic_ref_ordinal=ordinal,
        range=text_range,
        knowledge_type="topic",
        knowledge=Topic("test_topic"),
    )


class MockTermIndex(ITermToSemanticRefIndex):
    """Mock term index for testing."""

    def __init__(self, term_to_refs: dict[str, list[ScoredSemanticRefOrdinal]]):
        self.term_to_refs = term_to_refs

    def get_terms(self) -> list[str]:
        return list(self.term_to_refs.keys())

    def add_term(self, term, semantic_ref_ordinal):
        raise RuntimeError

    def remove_term(self, term, semantic_ref_ordinal):
        raise RuntimeError

    def lookup_term(self, term: str) -> list[ScoredSemanticRefOrdinal]:
        return self.term_to_refs.get(term, [])


class MockConversation(IConversation[MockMessage, MockTermIndex]):
    """Mock conversation for testing."""

    def __init__(
        self,
        name_tag: str = "MockConversation",
        has_refs: bool = True,
        has_index: bool = True,
    ):
        self.name_tag = name_tag
        self.tags = []
        messages = [MockMessage(0, "First message"), MockMessage(1, "Second message")]
        self.messages = MockMessageCollection(messages)
        self.semantic_refs = None
        self.semantic_ref_index = None
        self.secondary_indexes = None

        # Create semantic refs
        refs = []
        if has_refs:
            refs = [
                make_semantic_ref(
                    0, TextRange(TextLocation(0, 0), TextLocation(0, 10))
                ),
                make_semantic_ref(
                    1, TextRange(TextLocation(1, 0), TextLocation(1, 10))
                ),
            ]
            self.semantic_refs = SemanticRefCollection(refs)
        else:
            self.semantic_refs = None

        # Create semantic ref index
        term_to_refs = {}
        if has_index:
            term_to_refs = {
                "first": [ScoredSemanticRefOrdinal(0, 1.0)],
                "second": [ScoredSemanticRefOrdinal(1, 0.8)],
                "test": [
                    ScoredSemanticRefOrdinal(0, 0.9),
                    ScoredSemanticRefOrdinal(1, 0.7),
                ],
            }
        self.semantic_ref_index = MockTermIndex(term_to_refs) if has_index else None


@pytest.fixture
def searchable_conversation():
    return MockConversation(has_refs=True, has_index=True)


@pytest.fixture
def non_searchable_conversation():
    return MockConversation(has_refs=False, has_index=False)


@pytest.fixture
def eval_context(searchable_conversation) -> QueryEvalContext:
    return QueryEvalContext(conversation=searchable_conversation)


class TestConversationSearchability:
    def test_is_conversation_searchable_true(
        self, searchable_conversation: MockConversation
    ):
        """Test is_conversation_searchable with a searchable conversation."""
        assert is_conversation_searchable(searchable_conversation) is True

    def test_is_conversation_searchable_false(
        self, non_searchable_conversation: MockConversation
    ):
        """Test is_conversation_searchable with a non-searchable conversation."""
        assert is_conversation_searchable(non_searchable_conversation) is False

    def test_is_conversation_searchable_partial(self):
        """Test is_conversation_searchable with partial initialization."""
        conv = MockConversation(has_refs=True, has_index=False)
        assert is_conversation_searchable(conv) is False

        conv = MockConversation(has_refs=False, has_index=True)
        assert is_conversation_searchable(conv) is False


class TestTermLookup:
    def test_lookup_term_filtered(self, searchable_conversation):
        """Test lookup_term_filtered function."""
        term = Term("test")

        # Filter to only high-scoring results
        def high_score_filter(semantic_ref, scored_ref):
            return scored_ref.score > 0.8

        results = lookup_term_filtered(
            searchable_conversation.semantic_ref_index,
            term,
            searchable_conversation.semantic_refs,
            high_score_filter,
        )

        assert results is not None
        assert len(results) == 1
        assert results[0].semantic_ref_ordinal == 0
        assert results[0].score == 0.9

    def test_lookup_term_filtered_no_results(self, searchable_conversation):
        """Test lookup_term_filtered with no matching results."""
        term = Term("nonexistent")

        def any_filter(semantic_ref, scored_ref):
            return True

        results = lookup_term_filtered(
            searchable_conversation.semantic_ref_index,
            term,
            searchable_conversation.semantic_refs,
            any_filter,
        )

        assert results is None

    def test_lookup_term(self, searchable_conversation):
        """Test lookup_term function with no scope."""
        term = Term("test")

        results = lookup_term(
            searchable_conversation.semantic_ref_index,
            term,
            searchable_conversation.semantic_refs,
        )

        assert results is not None
        assert len(results) == 2
        assert results[0].semantic_ref_ordinal == 0
        assert results[1].semantic_ref_ordinal == 1

    def test_lookup_term_with_scope(self, searchable_conversation):
        """Test lookup_term function with a scope."""
        term = Term("test")

        # Create a scope that only includes the first message
        range_collection = TextRangeCollection(
            [TextRange(TextLocation(0, 0), TextLocation(0, 20))]
        )
        ranges_in_scope = TextRangesInScope([range_collection])

        results = lookup_term(
            searchable_conversation.semantic_ref_index,
            term,
            searchable_conversation.semantic_refs,
            ranges_in_scope,
        )

        assert results is not None
        assert len(results) == 1
        assert results[0].semantic_ref_ordinal == 0


class TestQueryEvalContext:
    def test_initialization(self, searchable_conversation):
        """Test QueryEvalContext initialization."""
        context = QueryEvalContext(conversation=searchable_conversation)

        assert context.conversation == searchable_conversation
        assert context.property_index is None
        assert context.timestamp_index is None
        assert isinstance(context.matched_terms, TermSet)
        assert isinstance(context.matched_property_terms, PropertyTermSet)
        assert isinstance(context.text_ranges_in_scope, TextRangesInScope)

    def test_initialization_error(self, non_searchable_conversation):
        """Test QueryEvalContext initialization with non-searchable conversation."""
        with pytest.raises(ValueError):
            QueryEvalContext(conversation=non_searchable_conversation)

    def test_properties(self, eval_context: QueryEvalContext):
        """Test QueryEvalContext property accessors."""
        assert (
            eval_context.semantic_ref_index
            == eval_context.conversation.semantic_ref_index
        )
        assert eval_context.semantic_refs == eval_context.conversation.semantic_refs
        assert eval_context.messages == eval_context.conversation.messages

    def test_get_semantic_ref(self, eval_context: QueryEvalContext):
        """Test get_semantic_ref method."""
        ref = eval_context.get_semantic_ref(0)
        assert ref.semantic_ref_ordinal == 0

    def test_get_message_for_ref(self, eval_context: QueryEvalContext):
        """Test get_message_for_ref method."""
        ref = eval_context.get_semantic_ref(0)
        message = downcast(MockMessage, eval_context.get_message_for_ref(ref))
        assert message.ordinal == 0

    def test_get_message(self, eval_context: QueryEvalContext):
        """Test get_message method."""
        message = downcast(MockMessage, eval_context.get_message(1))
        assert message.ordinal == 1

    def test_clear_matched_terms(self, eval_context: QueryEvalContext):
        """Test clear_matched_terms method."""
        # Add some matched terms
        eval_context.matched_terms.add(Term("test"))
        eval_context.matched_property_terms.add("property", Term("value"))

        # Clear them
        eval_context.clear_matched_terms()

        # Verify they're gone
        assert len(eval_context.matched_terms) == 0
        assert len(eval_context.matched_property_terms.terms) == 0


class TestMatchSearchTermExpr:
    def test_initialization(self):
        """Test MatchSearchTermExpr initialization."""
        search_term = SearchTerm(term=Term("test"))
        expr = MatchSearchTermExpr(search_term)

        assert expr.search_term == search_term
        assert expr.score_booster is None

    def test_accumulate_matches(self, eval_context: QueryEvalContext):
        """Test accumulating matches for a search term."""
        search_term = SearchTerm(term=Term("test"))
        expr = MatchSearchTermExpr(search_term)

        matches = SemanticRefAccumulator()
        expr.accumulate_matches(eval_context, matches)

        assert len(matches) == 2

    def test_accumulate_matches_with_related_terms(
        self, eval_context: QueryEvalContext
    ):
        """Test accumulating matches for a search term with related terms."""
        search_term = SearchTerm(
            term=Term("test"), related_terms=[Term("first", weight=0.8)]
        )
        expr = MatchSearchTermExpr(search_term)

        matches = SemanticRefAccumulator()
        expr.accumulate_matches(eval_context, matches)

        # Should match 'test' (2 refs) and 'first' (1 ref)
        assert len(matches) == 2

    def test_score_booster(self, eval_context: QueryEvalContext):
        """Test score booster function."""

        def boost_score(search_term, semantic_ref, scored_ref):
            return ScoredSemanticRefOrdinal(
                semantic_ref_ordinal=scored_ref.semantic_ref_ordinal,
                score=scored_ref.score * 2.0,  # Double the score
            )

        search_term = SearchTerm(term=Term("test"))
        expr = MatchSearchTermExpr(search_term, score_booster=boost_score)

        matches = SemanticRefAccumulator()
        expr.accumulate_matches(eval_context, matches)

        sorted_matches = matches.get_sorted_by_score()
        assert sorted_matches[0].score == 1.8  # 0.9 * 2.0
        assert sorted_matches[1].score == 1.4  # 0.7 * 2.0


class MockPropertyIndex(PropertyIndex):
    """Mock property index for testing."""

    def __init__(self):
        super().__init__()
        properties = {
            "name": [
                ScoredSemanticRefOrdinal(0, 1.0),
                ScoredSemanticRefOrdinal(1, 0.8),
            ],
            "value": [
                ScoredSemanticRefOrdinal(0, 0.9),
                ScoredSemanticRefOrdinal(1, 0.7),
            ],
            "facet.name": [ScoredSemanticRefOrdinal(0, 0.7)],
            "facet.value": [ScoredSemanticRefOrdinal(1, 0.6)],
        }
        for name, values in properties.items():
            for value in values:
                self.add_property(name, "test", value)


class TestMatchPropertySearchTermExpr:
    """Tests for the MatchPropertySearchTermExpr class."""

    def test_accumulate_matches_known_prop(self, eval_context: QueryEvalContext):
        """Test accumulating matches for a property search term."""
        # property_name is a string (KnowledgePropertyName); calls accumulate_matches_for_property()
        eval_context.property_index = MockPropertyIndex()
        property_search_term = PropertySearchTerm("name", SearchTerm(term=Term("test")))
        expr = MatchPropertySearchTermExpr(property_search_term)
        matches = SemanticRefAccumulator()
        expr.accumulate_matches(eval_context, matches)
        assert len(matches) == 2

    def test_accumulate_matches_user_prop(self, eval_context: QueryEvalContext):
        """Test accumulating matches for a property search term with SearchTerm property name."""
        # property_name is a SearchTerm(Term()); calls accumulate_matches_for_facets()
        eval_context.property_index = MockPropertyIndex()
        property_search_term = PropertySearchTerm(
            SearchTerm(Term("name")),
            SearchTerm(term=Term("test")),
        )
        expr = MatchPropertySearchTermExpr(property_search_term)
        matches = SemanticRefAccumulator()
        expr.accumulate_matches(eval_context, matches)
        assert len(matches) == 1

    def test_accumulate_matches_for_property(self, eval_context: QueryEvalContext):
        """Test accumulate_matches_for_property method."""
        eval_context.property_index = MockPropertyIndex()
        dummy_search_term = PropertySearchTerm("name", SearchTerm(term=Term("test")))
        expr = MatchPropertySearchTermExpr(dummy_search_term)
        matches = SemanticRefAccumulator()
        expr.accumulate_matches_for_property(
            eval_context, "name", SearchTerm(Term("test")), matches
        )
        assert len(matches) == 2

    def test_accumulate_matches_for_facets(self, eval_context: QueryEvalContext):
        """Test accumulate_matches_for_facets method."""
        eval_context.property_index = MockPropertyIndex()
        dummy_search_term = PropertySearchTerm("name", SearchTerm(term=Term("test")))
        expr = MatchPropertySearchTermExpr(dummy_search_term)
        matches = SemanticRefAccumulator()
        st1, st2 = SearchTerm(Term("facet.name")), SearchTerm(Term("test"))
        expr.accumulate_matches_for_facets(eval_context, st1, st2, matches)
        assert len(matches) == 1

    def test_accumulate_matches_for_property_value(
        self, eval_context: QueryEvalContext
    ):
        """Test accumulate_matches_for_property_value method."""
        eval_context.property_index = MockPropertyIndex()
        dummy_search_term = PropertySearchTerm("name", SearchTerm(term=Term("test")))
        expr = MatchPropertySearchTermExpr(dummy_search_term)

        # First call has two matches
        matches = SemanticRefAccumulator()
        expr.accumulate_matches_for_property_value(
            eval_context, matches, "name", Term("test")
        )
        assert len(matches) == 2
        assert matches.get_match(0) is not None  # First semantic ref should be matched
        assert matches.get_match(1) is not None  # Second semantic ref should be matched
        assert eval_context.matched_property_terms.has("name", Term("test"))

        # Second call with same property should do nothing because it's already matched
        second_matches: SemanticRefAccumulator = SemanticRefAccumulator()
        expr.accumulate_matches_for_property_value(
            eval_context, second_matches, "name", Term("test")
        )
        assert len(second_matches) == 0


class TestGetScopeExpr:
    def test_eval(self, eval_context: QueryEvalContext):
        """Test evaluating a scope expression."""
        text_ranges = [TextRange(TextLocation(0, 0), TextLocation(0, 10))]
        selector = TextRangeSelector(text_ranges)
        expr = GetScopeExpr(range_selectors=[selector])

        result = expr.eval(eval_context)

        assert isinstance(result, TextRangesInScope)


class TestBooleanExpressions:
    def setup_method(self):
        """Set up test data."""

        # Create mock term expressions
        class MockTermExpr(MatchTermExpr):
            def __init__(self, term, matches_to_add):
                self.term = term
                self.matches_to_add = matches_to_add

            def accumulate_matches(self, context, matches):
                for ordinal, score in self.matches_to_add:
                    matches.add(ordinal, score)

        self.expr1 = MockTermExpr("term1", [(0, 1.0), (1, 0.8)])
        self.expr2 = MockTermExpr("term2", [(1, 0.9), (2, 0.7)])
        self.expr3 = MockTermExpr("term3", [])  # No matches

    def test_match_terms_or_expr(self, eval_context: QueryEvalContext):
        """Test OR expression."""
        expr = MatchTermsOrExpr(term_expressions=[self.expr1, self.expr2])

        result = expr.eval(eval_context)

        assert len(result) == 3  # Should include refs 0, 1, 2
        assert result.get_match(0) is not None
        assert result.get_match(1) is not None
        assert result.get_match(2) is not None

    def test_match_terms_or_expr_no_matches(self, eval_context: QueryEvalContext):
        """Test OR expression with no matches."""
        expr = MatchTermsOrExpr(term_expressions=[self.expr3])

        result = expr.eval(eval_context)

        assert len(result) == 0

    def test_match_terms_or_max_expr(self, eval_context: QueryEvalContext):
        """Test OR MAX expression."""
        expr = MatchTermsOrMaxExpr(term_expressions=[self.expr1, self.expr2])

        result = expr.eval(eval_context)

        # Should select only refs that match the max hit count
        assert len(result) == 1
        assert result.get_match(1) is not None  # Only ref 1 matches both expressions

    def test_match_terms_and_expr(self, eval_context):
        """Test AND expression."""
        expr = MatchTermsAndExpr(term_expressions=[self.expr1, self.expr2])

        result = expr.eval(eval_context)

        # Should include only ref 1 which appears in both expressions
        assert len(result) == 1
        assert result.get_match(1) is not None

    def test_match_terms_and_expr_no_matches(self, eval_context):
        """Test AND expression with a non-matching term."""
        expr = MatchTermsAndExpr(term_expressions=[self.expr1, self.expr3])

        result = expr.eval(eval_context)

        assert len(result) == 0


class TestSelectTopNExpr:
    def test_eval(self, eval_context):
        """Test selecting top N matches."""

        # Create a mock source expression
        class MockSourceExpr(QueryOpExpr[MatchAccumulator[int]]):
            def eval(self, context):
                matches = MatchAccumulator[int]()
                matches.add(1, score=1.0)
                matches.add(2, score=0.8)
                matches.add(3, score=0.6)
                return matches

        source_expr = MockSourceExpr()
        expr = SelectTopNExpr(source_expr=source_expr, max_matches=2)

        result = expr.eval(eval_context)

        assert len(result) == 2
        sorted_matches = result.get_sorted_by_score()
        assert sorted_matches[0].value == 1
        assert sorted_matches[1].value == 2


def test_get_text_range_for_date_range():
    from typeagent.knowpro.query import get_text_range_for_date_range
    from typeagent.knowpro.interfaces import (
        TextLocation,
        TextRange,
        DateRange,
        Datetime,
    )

    # Should return None for empty input and any date range
    empty_conv = MockConversation()
    empty_conv.messages = MockMessageCollection()
    date_range = DateRange(
        start=Datetime(2020, 1, 1, 0, 0, 0),
        end=Datetime(2020, 1, 2, 0, 0, 0),
    )
    assert get_text_range_for_date_range(empty_conv, date_range) is None

    # Should return a TextRange for a valid date range (simulate all messages in range)
    # (Assume all messages are in the date range for this mock)
    conv = MockConversation()
    result_with_range = get_text_range_for_date_range(conv, date_range)
    assert isinstance(result_with_range, TextRange)
    assert result_with_range.start == TextLocation(0, 0)
    assert result_with_range.end == TextLocation(2, 0)  # End is exclusive


def test_get_matching_term_for_text():
    from typeagent.knowpro.query import get_matching_term_for_text
    from typeagent.knowpro.interfaces import SearchTerm, Term

    # Should return None if no terms match
    assert get_matching_term_for_text(SearchTerm(term=Term("bar")), "foo") is None
    # Should return the matching term (case-insensitive)
    assert get_matching_term_for_text(SearchTerm(term=Term("Foo")), "foo") == Term(
        "Foo"
    )


def test_get_matching_term_for_text_multiple():
    from typeagent.knowpro.query import get_matching_term_for_text
    from typeagent.knowpro.interfaces import SearchTerm, Term

    terms = [SearchTerm(term=Term("bar")), SearchTerm(term=Term("baz"))]
    assert all(get_matching_term_for_text(term, "foo") is None for term in terms)


def test_match_search_term_to_text():
    from typeagent.knowpro.query import match_search_term_to_text
    from typeagent.knowpro.interfaces import SearchTerm, Term

    # Should return True if term is in text
    assert match_search_term_to_text(SearchTerm(term=Term("foo")), "foo")
    # Should return False if term is not in text
    assert not match_search_term_to_text(SearchTerm(term=Term("baz")), "foo bar")


def test_match_search_term_to_one_of_text():
    from typeagent.knowpro.query import match_search_term_to_one_of_text
    from typeagent.knowpro.interfaces import SearchTerm, Term

    # Should return True if term matches any text
    assert match_search_term_to_one_of_text(
        SearchTerm(term=Term("foo")), ["bar", "foo"]
    )
    # Should return False if term matches none
    assert not match_search_term_to_one_of_text(
        SearchTerm(term=Term("baz")), ["bar", "foo"]
    )


def test_match_entity_name_or_type():
    from typeagent.knowpro.query import match_entity_name_or_type, ConcreteEntity
    from typeagent.knowpro.interfaces import SearchTerm, Term

    entity = ConcreteEntity(name="foo", type=["bar"])
    # Should return True if name matches
    assert match_entity_name_or_type(SearchTerm(term=Term("foo")), entity)
    # Should return True if type matches
    assert match_entity_name_or_type(SearchTerm(term=Term("bar")), entity)
    # Should return False if neither matches
    assert not match_entity_name_or_type(SearchTerm(term=Term("baz")), entity)


def test_lookup_knowledge_type():
    from typeagent.knowpro.query import lookup_knowledge_type
    from typeagent.knowpro.interfaces import (
        SemanticRef,
        ScoredSemanticRefOrdinal,
        TextRange,
        TextLocation,
        Topic,
    )
    from typeagent.knowpro.storage import SemanticRefCollection

    # Create valid TextRange and Topic objects
    rng = TextRange(TextLocation(0, 0), TextLocation(0, 1))
    topic1 = Topic("foo")
    topic2 = Topic("bar")

    # Use only valid knowledge_type values: 'entity', 'action', 'topic', 'tag'
    refs = [
        SemanticRef(
            semantic_ref_ordinal=0, range=rng, knowledge_type="topic", knowledge=topic1
        ),
        SemanticRef(
            semantic_ref_ordinal=1, range=rng, knowledge_type="entity", knowledge=topic2
        ),
        SemanticRef(
            semantic_ref_ordinal=2, range=rng, knowledge_type="topic", knowledge=topic2
        ),
    ]
    collection = SemanticRefCollection(refs)
    result = lookup_knowledge_type(collection, "topic")
    assert isinstance(result, list)
    assert all(isinstance(r, ScoredSemanticRefOrdinal) for r in result)
    assert {r.semantic_ref_ordinal for r in result} == {0, 2}
    # Should return empty list if no match
    assert lookup_knowledge_type(collection, "action") == []
