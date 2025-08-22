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
    WhenFilter,
)
from typeagent.knowpro.kplib import ConcreteEntity


def test_text_location_serialization():
    """Test serialization and deserialization of TextLocation."""
    location = TextLocation(message_ordinal=1)
    serialized = location.serialize()
    deserialized = TextLocation.deserialize(serialized)

    assert location == deserialized
    assert serialized == {
        "messageOrdinal": 1,
        "chunkOrdinal": 0,
    }


def test_text_range_serialization():
    """Test serialization and deserialization of TextRange."""
    start = TextLocation(message_ordinal=1, chunk_ordinal=2)
    end = TextLocation(message_ordinal=4, chunk_ordinal=5)
    range_obj = TextRange(start=start, end=end)

    serialized = range_obj.serialize()
    deserialized = TextRange.deserialize(serialized)

    assert range_obj == deserialized
    assert serialized == {
        "start": {"messageOrdinal": 1, "chunkOrdinal": 2},
        "end": {"messageOrdinal": 4, "chunkOrdinal": 5},
    }


def test_text_range_equality():
    """Test equality of TextRange objects."""
    start1 = TextLocation(message_ordinal=1, chunk_ordinal=2)
    end1 = TextLocation(message_ordinal=4, chunk_ordinal=5)
    range1 = TextRange(start=start1, end=end1)
    range2 = TextRange(start=start1, end=end1)
    end3 = TextLocation(message_ordinal=4, chunk_ordinal=6)

    assert range1 == range2
    assert range1 != TextRange(start=start1, end=end3)
    assert range1 != "not a TextRange"


def test_text_range_equality_end_none():
    """Test equality of TextRange objects with None end."""
    start1 = TextLocation(message_ordinal=1, chunk_ordinal=2)
    range1 = TextRange(start=start1, end=None)

    start2 = TextLocation(message_ordinal=1, chunk_ordinal=2)
    range2 = TextRange(start=start2, end=None)

    assert range1 == range2
    assert range1 != TextRange(start=start1, end=TextLocation(message_ordinal=4))
    assert range1 != "not a TextRange"


def test_text_range_ordering():
    """Test ordering of TextRange objects."""
    start1 = TextLocation(message_ordinal=1, chunk_ordinal=2)
    end1 = TextLocation(message_ordinal=4, chunk_ordinal=5)
    range1 = TextRange(start=start1, end=end1)

    start2 = TextLocation(message_ordinal=2, chunk_ordinal=3)
    end2 = TextLocation(message_ordinal=5, chunk_ordinal=6)
    range2 = TextRange(start=start2, end=end2)

    assert range1 < range2
    assert range2 > range1
    assert range1 <= range1
    assert range2 >= range2


def test_text_range_ordering_end_none():
    """Test ordering of TextRange objects with None end."""
    start1 = TextLocation(message_ordinal=1, chunk_ordinal=2)
    start2 = TextLocation(message_ordinal=1, chunk_ordinal=2)
    end1 = TextLocation(message_ordinal=4, chunk_ordinal=5)

    range1 = TextRange(start=start1, end=None)
    range2 = TextRange(start=start2, end=end1)

    assert range1 < range2
    assert range2 > range1
    assert range1 <= range1
    assert range2 >= range2


def test_text_range_contains():
    """Test the __contains__ method of TextRange."""
    start = TextLocation(message_ordinal=1, chunk_ordinal=0)
    end = TextLocation(message_ordinal=5, chunk_ordinal=0)
    range_obj = TextRange(start=start, end=end)

    contained = TextRange(start=TextLocation(2), end=TextLocation(4))
    not_contained = TextRange(start=TextLocation(6))

    assert contained in range_obj
    assert not_contained not in range_obj


def test_text_range_contains_with_none_end():
    """Test the __contains__ method when end is None."""
    # Test range with None end - should default to start.chunk_ordinal + 1
    point_range = TextRange(start=TextLocation(message_ordinal=5, chunk_ordinal=3))

    # Range that fits exactly within the implied end
    exact_fit = TextRange(
        start=TextLocation(message_ordinal=5, chunk_ordinal=3),
        end=TextLocation(message_ordinal=5, chunk_ordinal=4),
    )

    # Range that starts at the same position but extends beyond
    too_long = TextRange(
        start=TextLocation(message_ordinal=5, chunk_ordinal=3),
        end=TextLocation(message_ordinal=5, chunk_ordinal=5),
    )

    # Range that starts before
    starts_before = TextRange(
        start=TextLocation(message_ordinal=5, chunk_ordinal=2),
        end=TextLocation(message_ordinal=5, chunk_ordinal=4),
    )

    # Point range at the same location
    same_point = TextRange(start=TextLocation(message_ordinal=5, chunk_ordinal=3))

    assert exact_fit in point_range
    assert too_long not in point_range
    assert starts_before not in point_range
    assert same_point in point_range


def test_text_range_contains_both_none_end():
    """Test __contains__ when both ranges have None end."""
    # Both ranges are point ranges
    point1 = TextRange(start=TextLocation(message_ordinal=5, chunk_ordinal=3))
    point2 = TextRange(start=TextLocation(message_ordinal=5, chunk_ordinal=3))
    point3 = TextRange(start=TextLocation(message_ordinal=5, chunk_ordinal=4))

    assert point2 in point1  # Same point
    assert point3 not in point1  # Different point


def test_text_range_ordering_with_none_end_detailed():
    """Test detailed ordering behavior when end is None."""
    # Point ranges at different locations
    point1 = TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=0))
    point2 = TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=1))
    point3 = TextRange(start=TextLocation(message_ordinal=2, chunk_ordinal=0))

    # Regular range that overlaps with point ranges
    regular_range = TextRange(
        start=TextLocation(message_ordinal=1, chunk_ordinal=0),
        end=TextLocation(message_ordinal=1, chunk_ordinal=5),
    )

    # Test point range ordering
    assert point1 < point2
    assert point2 < point3
    assert point1 < point3

    # Test point vs regular range
    # point1 has implied end at (1, 1), regular_range ends at (1, 5)
    assert point1 < regular_range

    # Test that point ranges with same start are ordered by implied end
    point_same_start = TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=0))
    assert point_same_start == point1  # Should be equal since they're the same


def test_text_range_comparison_operators_with_none():
    """Test all comparison operators when end is None."""
    # Create test ranges
    early_point = TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=0))
    later_point = TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=2))
    same_point = TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=0))

    # Test less than
    assert early_point < later_point
    assert not (later_point < early_point)
    assert not (early_point < same_point)

    # Test greater than
    assert later_point > early_point
    assert not (early_point > later_point)
    assert not (early_point > same_point)

    # Test less than or equal
    assert early_point <= later_point
    assert early_point <= same_point
    assert not (later_point <= early_point)

    # Test greater than or equal
    assert later_point >= early_point
    assert early_point >= same_point
    assert not (early_point >= later_point)

    # Test equality
    assert early_point == same_point
    assert not (early_point == later_point)


def test_text_range_mixed_none_and_explicit_end():
    """Test comparisons between ranges with None end and explicit end."""
    # Point range (None end) - implied end at (5, 4)
    point_range = TextRange(start=TextLocation(message_ordinal=5, chunk_ordinal=3))

    # Explicit range that should be equivalent for ordering purposes
    equivalent_range = TextRange(
        start=TextLocation(message_ordinal=5, chunk_ordinal=3),
        end=TextLocation(message_ordinal=5, chunk_ordinal=4),
    )

    # Explicit range that starts at same point but ends later
    longer_range = TextRange(
        start=TextLocation(message_ordinal=5, chunk_ordinal=3),
        end=TextLocation(message_ordinal=5, chunk_ordinal=6),
    )

    # Test equality/comparison - With new __eq__, these should now be equal
    assert point_range == equivalent_range  # Now equal due to logical equivalence
    assert (
        point_range < longer_range
    )  # Point range ends at implied (5, 4), longer ends at (5, 6)
    assert longer_range > point_range

    # They should be ordered the same relative to other ranges
    assert not (point_range < equivalent_range)  # They have same ordering
    assert not (equivalent_range < point_range)  # They have same ordering
    assert point_range <= equivalent_range
    assert equivalent_range <= point_range

    # Test containment
    assert point_range in longer_range
    assert equivalent_range in longer_range
    assert longer_range not in point_range


def test_text_range_equality_with_logical_equivalence():
    """Test that TextRange equality works with logical equivalence for None end."""
    # Point range with None end
    point_range = TextRange(start=TextLocation(message_ordinal=3, chunk_ordinal=7))

    # Equivalent explicit range (should have end at chunk_ordinal + 1)
    equivalent_range = TextRange(
        start=TextLocation(message_ordinal=3, chunk_ordinal=7),
        end=TextLocation(message_ordinal=3, chunk_ordinal=8),
    )

    # Different explicit range
    different_range = TextRange(
        start=TextLocation(message_ordinal=3, chunk_ordinal=7),
        end=TextLocation(message_ordinal=3, chunk_ordinal=9),
    )

    # Another point range at same location
    same_point = TextRange(start=TextLocation(message_ordinal=3, chunk_ordinal=7))

    # Point range at different location
    different_point = TextRange(start=TextLocation(message_ordinal=3, chunk_ordinal=8))

    # Test equality
    assert point_range == equivalent_range  # Logically equivalent
    assert point_range == same_point  # Same point
    assert equivalent_range == same_point  # All three are logically equivalent

    # Test inequality
    assert point_range != different_range  # Different end
    assert point_range != different_point  # Different start
    assert equivalent_range != different_range  # Different end

    # Test with non-TextRange object
    assert point_range != "not a TextRange"
    assert point_range != None


def test_text_range_equality_both_explicit_ends():
    """Test TextRange equality with both ranges having explicit ends."""
    range1 = TextRange(
        start=TextLocation(message_ordinal=1, chunk_ordinal=2),
        end=TextLocation(message_ordinal=1, chunk_ordinal=5),
    )
    range2 = TextRange(
        start=TextLocation(message_ordinal=1, chunk_ordinal=2),
        end=TextLocation(message_ordinal=1, chunk_ordinal=5),
    )
    range3 = TextRange(
        start=TextLocation(message_ordinal=1, chunk_ordinal=2),
        end=TextLocation(message_ordinal=1, chunk_ordinal=6),
    )

    assert range1 == range2  # Identical
    assert range1 != range3  # Different end


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
        knowledge=knowledge,
    )

    # Serialize and deserialize
    serialized = semantic_ref.serialize()
    deserialized = SemanticRef.deserialize(serialized)

    # Assertions
    assert semantic_ref.semantic_ref_ordinal == deserialized.semantic_ref_ordinal
    assert semantic_ref.range == deserialized.range
    assert (
        semantic_ref.knowledge.knowledge_type == deserialized.knowledge.knowledge_type
    )
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
            {
                "start": {"messageOrdinal": 1, "chunkOrdinal": 0},
                "end": {"messageOrdinal": 2, "chunkOrdinal": 0},
            },
            {
                "start": {"messageOrdinal": 3, "chunkOrdinal": 0},
                "end": {"messageOrdinal": 4, "chunkOrdinal": 0},
            },
        ],
    }


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
