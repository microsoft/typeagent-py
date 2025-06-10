# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime as Datetime

from typeagent.knowpro.interfaces import (
    DateRange,
    PropertySearchTerm,
    ScoredSemanticRefOrdinal,
    SearchSelectExpr,
    SearchTerm,
    SearchTermGroup,
    SemanticRefSearchResult,
    Term,
    TextLocation,
    TextRange,
    SemanticRef,
    Thread,
    IndexingEventHandlers,
    TextIndexingResult,
    ListIndexingResult,
    SecondaryIndexingResults,
    IndexingResults,
    WhenFilter,
)
from typeagent.knowpro.kplib import ConcreteEntity


def test_text_location_serialization():
    """Test serialization and deserialization of TextLocation."""
    location = TextLocation(message_ordinal=1, chunk_ordinal=2, char_ordinal=3)
    serialized = location.serialize()
    deserialized = TextLocation.deserialize(serialized)

    assert location == deserialized
    assert serialized == {
        "messageOrdinal": 1,
        "chunkOrdinal": 2,
        "charOrdinal": 3,
    }


def test_text_range_serialization():
    """Test serialization and deserialization of TextRange."""
    start = TextLocation(message_ordinal=1, chunk_ordinal=2, char_ordinal=3)
    end = TextLocation(message_ordinal=4, chunk_ordinal=5, char_ordinal=6)
    range_obj = TextRange(start=start, end=end)

    serialized = range_obj.serialize()
    deserialized = TextRange.deserialize(serialized)

    assert range_obj == deserialized
    assert serialized == {
        "start": {"messageOrdinal": 1, "chunkOrdinal": 2, "charOrdinal": 3},
        "end": {"messageOrdinal": 4, "chunkOrdinal": 5, "charOrdinal": 6},
    }


def test_text_range_equality():
    """Test equality of TextRange objects."""
    start1 = TextLocation(message_ordinal=1, chunk_ordinal=2, char_ordinal=3)
    end1 = TextLocation(message_ordinal=4, chunk_ordinal=5, char_ordinal=6)
    range1 = TextRange(start=start1, end=end1)
    range2 = TextRange(start=start1, end=end1)
    end3 = TextLocation(message_ordinal=4, chunk_ordinal=5, char_ordinal=7)

    assert range1 == range2
    assert range1 != TextRange(start=start1, end=end3)
    assert range1 != "not a TextRange"


def test_text_range_equality_end_none():
    """Test equality of TextRange objects with None end."""
    start1 = TextLocation(message_ordinal=1, chunk_ordinal=2, char_ordinal=3)
    range1 = TextRange(start=start1, end=None)

    start2 = TextLocation(message_ordinal=1, chunk_ordinal=2, char_ordinal=3)
    range2 = TextRange(start=start2, end=None)

    assert range1 == range2
    assert range1 != TextRange(start=start1, end=TextLocation(message_ordinal=4))
    assert range1 != "not a TextRange"


def test_text_range_ordering():
    """Test ordering of TextRange objects."""
    start1 = TextLocation(message_ordinal=1, chunk_ordinal=2, char_ordinal=3)
    end1 = TextLocation(message_ordinal=4, chunk_ordinal=5, char_ordinal=6)
    range1 = TextRange(start=start1, end=end1)

    start2 = TextLocation(message_ordinal=2, chunk_ordinal=3, char_ordinal=4)
    end2 = TextLocation(message_ordinal=5, chunk_ordinal=6, char_ordinal=7)
    range2 = TextRange(start=start2, end=end2)

    assert range1 < range2
    assert range2 > range1
    assert range1 <= range1
    assert range2 >= range2


def test_text_range_ordering_end_none():
    """Test ordering of TextRange objects with None end."""
    start1 = TextLocation(message_ordinal=1, chunk_ordinal=2, char_ordinal=3)
    start2 = TextLocation(message_ordinal=1, chunk_ordinal=2, char_ordinal=3)
    end1 = TextLocation(message_ordinal=4, chunk_ordinal=5, char_ordinal=6)

    range1 = TextRange(start=start1, end=None)
    range2 = TextRange(start=start2, end=end1)

    assert range1 < range2
    assert range2 > range1
    assert range1 <= range1
    assert range2 >= range2


def test_text_range_contains():
    """Test the __contains__ method of TextRange."""
    start = TextLocation(message_ordinal=1, chunk_ordinal=0, char_ordinal=0)
    end = TextLocation(message_ordinal=5, chunk_ordinal=0, char_ordinal=0)
    range_obj = TextRange(start=start, end=end)

    contained = TextRange(start=TextLocation(2), end=TextLocation(4))
    not_contained = TextRange(start=TextLocation(6))

    assert contained in range_obj
    assert not_contained not in range_obj


def test_semantic_ref_serialization():
    """Test serialization and deserialization of SemanticRef using ConcreteEntity."""
    # Create a concrete example of knowledge
    knowledge = ConcreteEntity(name="ExampleEntity", type=["ExampleType"])
    knowledge_type = "entity"

    # Define the range
    start = TextLocation(message_ordinal=1)
    end = TextLocation(message_ordinal=2)
    range_obj = TextRange(start=start, end=end)

    # Create the SemanticRef
    semantic_ref = SemanticRef(
        semantic_ref_ordinal=42,
        range=range_obj,
        knowledge_type=knowledge_type,
        knowledge=knowledge,
    )

    # Serialize and deserialize
    serialized = semantic_ref.serialize()
    deserialized = SemanticRef.deserialize(serialized)

    # Assertions
    assert semantic_ref.semantic_ref_ordinal == deserialized.semantic_ref_ordinal
    assert semantic_ref.range == deserialized.range
    assert semantic_ref.knowledge_type == deserialized.knowledge_type
    assert isinstance(deserialized.knowledge, ConcreteEntity)
    assert deserialized.knowledge.name == knowledge.name
    assert deserialized.knowledge.type == knowledge.type


def test_thread_serialization():
    """Test serialization and deserialization of Thread."""
    range1 = TextRange(
        start=TextLocation(message_ordinal=1),
        end=TextLocation(message_ordinal=2),
    )
    range2 = TextRange(
        start=TextLocation(message_ordinal=3),
        end=TextLocation(message_ordinal=4),
    )
    thread = Thread(description="Test Thread", ranges=[range1, range2])

    serialized = thread.serialize()
    deserialized = Thread.deserialize(serialized)

    assert thread == deserialized
    assert serialized == {
        "description": "Test Thread",
        "ranges": [
            {"start": {"messageOrdinal": 1}, "end": {"messageOrdinal": 2}},
            {"start": {"messageOrdinal": 3}, "end": {"messageOrdinal": 4}},
        ],
    }


def test_indexing_event_handlers():
    """Test that IndexingEventHandlers can be initialized and invoked."""

    def mock_handler(*args):
        return True

    handlers = IndexingEventHandlers(
        on_knowledge_extracted=mock_handler,
        on_embeddings_created=mock_handler,
        on_text_indexed=mock_handler,
        on_message_started=mock_handler,
    )

    assert handlers.on_knowledge_extracted is not None
    assert handlers.on_embeddings_created is not None
    assert handlers.on_text_indexed is not None
    assert handlers.on_message_started is not None


def test_text_indexing_result():
    """Test initialization of TextIndexingResult."""
    result = TextIndexingResult(
        completed_upto=TextLocation(message_ordinal=1),
        error="Test error",
    )

    assert result.completed_upto == TextLocation(message_ordinal=1)
    assert result.error == "Test error"


def test_list_indexing_result():
    """Test initialization of ListIndexingResult."""
    result = ListIndexingResult(number_completed=10, error=None)

    assert result.number_completed == 10
    assert result.error is None


def test_secondary_indexing_results():
    """Test initialization of SecondaryIndexingResults."""
    properties_result = ListIndexingResult(number_completed=5, error=None)
    timestamps_result = ListIndexingResult(number_completed=3, error="Test error")
    secondary_results = SecondaryIndexingResults(
        properties=properties_result,
        timestamps=timestamps_result,
    )

    assert secondary_results.properties == properties_result
    assert secondary_results.timestamps == timestamps_result
    assert secondary_results.related_terms is None
    assert secondary_results.message is None


def test_indexing_results():
    """Test initialization of IndexingResults."""
    semantic_result = TextIndexingResult(
        completed_upto=TextLocation(message_ordinal=1),
        error=None,
    )
    secondary_results = SecondaryIndexingResults(
        properties=ListIndexingResult(number_completed=5, error=None)
    )
    results = IndexingResults(
        semantic_refs=semantic_result,
        secondary_index_results=secondary_results,
    )

    assert results.semantic_refs == semantic_result
    assert results.secondary_index_results == secondary_results


def test_search_term():
    """Test the SearchTerm class."""
    term = Term(text="example", weight=0.5)
    related_term1 = Term(text="related1")
    related_term2 = Term(text="related2")
    search_term = SearchTerm(term=term, related_terms=[related_term1, related_term2])

    assert search_term.term.text == "example"
    assert search_term.term.weight == 0.5
    assert search_term.related_terms is not None
    assert len(search_term.related_terms) == 2
    assert search_term.related_terms[0].text == "related1"
    assert search_term.related_terms[1].text == "related2"


def test_property_search_term():
    """Test the PropertySearchTerm class."""
    property_name = "name"
    property_value = SearchTerm(term=Term(text="Bach"))
    property_search_term = PropertySearchTerm(property_name, property_value)

    assert property_search_term.property_name == "name"
    assert property_search_term.property_value.term.text == "Bach"

    # Test with a SearchTerm as property_name
    property_name_as_search_term = SearchTerm(term=Term(text="hue"))
    property_value_as_search_term = SearchTerm(term=Term(text="red"))
    property_search_term = PropertySearchTerm(
        property_name_as_search_term,
        property_value_as_search_term,
    )

    assert isinstance(property_search_term.property_name, SearchTerm)
    assert isinstance(property_search_term.property_value, SearchTerm)
    assert property_search_term.property_name.term.text == "hue"
    assert property_search_term.property_value.term.text == "red"


def test_search_term_group():
    """Test the SearchTermGroup class."""
    term1 = SearchTerm(term=Term(text="example1"))
    term2 = SearchTerm(term=Term(text="example2"))
    group = SearchTermGroup(boolean_op="and", terms=[term1, term2])
    empty_group = SearchTermGroup("or")

    assert group.boolean_op == "and"
    assert len(group.terms) == 2
    assert group.terms[0] == SearchTerm(Term("example1"))
    assert group.terms[1] == SearchTerm(Term("example2"))
    assert empty_group.boolean_op == "or"
    assert empty_group.terms == []


def test_when_filter():
    """Test the WhenFilter class."""
    knowledge_type = "entity"
    date_range = DateRange(start=Datetime(2025, 1, 1), end=Datetime(2025, 1, 10))
    thread_description = "Test thread"
    scope_defining_terms = SearchTermGroup(
        boolean_op="or", terms=[SearchTerm(term=Term(text="scope_term"))]
    )
    text_ranges_in_scope = [
        TextRange(start=TextLocation(0), end=TextLocation(10)),
        TextRange(start=TextLocation(20), end=TextLocation(30)),
    ]

    when_filter = WhenFilter(
        knowledge_type=knowledge_type,
        date_range=date_range,
        thread_description=thread_description,
        scope_defining_terms=scope_defining_terms,
        text_ranges_in_scope=text_ranges_in_scope,
    )

    assert when_filter.knowledge_type == "entity"
    assert when_filter.date_range is not None
    assert when_filter.date_range.start == Datetime(2025, 1, 1)
    assert when_filter.date_range.end == Datetime(2025, 1, 10)
    assert when_filter.thread_description == "Test thread"
    assert when_filter.scope_defining_terms is not None
    assert when_filter.scope_defining_terms.boolean_op == "or"
    assert when_filter.text_ranges_in_scope is not None
    assert len(when_filter.text_ranges_in_scope) == 2
    assert when_filter.text_ranges_in_scope[0].start.message_ordinal == 0
    assert when_filter.text_ranges_in_scope[1].end is not None
    assert when_filter.text_ranges_in_scope[1].end.message_ordinal == 30


def test_search_select_expr():
    """Test the SearchSelectExpr class."""
    search_term_group = SearchTermGroup(
        boolean_op="or", terms=[SearchTerm(term=Term(text="example"))]
    )
    when_filter = WhenFilter(
        knowledge_type="entity",
        date_range=DateRange(start=Datetime(2025, 1, 1), end=Datetime(2025, 1, 10)),
    )

    search_select_expr = SearchSelectExpr(
        search_term_group=search_term_group, when=when_filter
    )

    assert search_select_expr.search_term_group.boolean_op == "or"
    assert search_select_expr.search_term_group.terms[0] == SearchTerm(Term("example"))
    assert search_select_expr.when is not None
    assert search_select_expr.when.knowledge_type == "entity"
    assert search_select_expr.when.date_range is not None
    assert search_select_expr.when.date_range.start == Datetime(2025, 1, 1)
    assert search_select_expr.when.date_range.end == Datetime(2025, 1, 10)


def test_semantic_ref_search_result():
    """Test the SemanticRefSearchResult class."""
    term_matches = {"example1", "example2"}
    semantic_ref_matches = [
        ScoredSemanticRefOrdinal(semantic_ref_ordinal=1, score=0.9),
        ScoredSemanticRefOrdinal(semantic_ref_ordinal=2, score=0.8),
    ]

    search_result = SemanticRefSearchResult(
        term_matches=term_matches, semantic_ref_matches=semantic_ref_matches
    )

    assert len(search_result.term_matches) == 2
    assert "example1" in search_result.term_matches
    assert len(search_result.semantic_ref_matches) == 2
    assert search_result.semantic_ref_matches[0].semantic_ref_ordinal == 1
    assert search_result.semantic_ref_matches[0].score == 0.9
    assert search_result.semantic_ref_matches[1].semantic_ref_ordinal == 2
    assert search_result.semantic_ref_matches[1].score == 0.8
