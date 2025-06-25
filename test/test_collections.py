# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.knowpro.collections import (
    Match,
    MatchAccumulator,
    PropertyTermSet,
    Scored,
    SemanticRefAccumulator,
    TermSet,
    TextRangeCollection,
    TextRangesInScope,
    TopNCollection,
    add_smooth_related_score_to_match_score,
    add_to_set,
    get_smooth_score,
    get_top_k,
)
from typeagent.knowpro.interfaces import (
    TextRange,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    TextLocation,
    Term,
)
from typeagent.knowpro.kplib import Action, ConcreteEntity
from typeagent.knowpro.storage import SemanticRefCollection


def test_match_accumulator_add_and_get():
    """Test adding and retrieving matches in MatchAccumulator."""
    accumulator = MatchAccumulator[str]()
    accumulator.add("example", score=1.0)
    accumulator.add("example", score=0.5)

    match = accumulator.get_match("example")
    assert match is not None
    assert match.value == "example"
    assert match.score == 1.5
    assert match.hit_count == 2


def test_match_accumulator_get_sorted_by_score():
    """Test sorting matches by score in MatchAccumulator."""
    accumulator = MatchAccumulator[str]()
    accumulator.add("low", score=0.5)
    accumulator.add("high", score=2.0)
    accumulator.add("medium", score=1.0)

    sorted_matches = accumulator.get_sorted_by_score()
    assert len(sorted_matches) == 3
    assert sorted_matches[0].value == "high"
    assert sorted_matches[1].value == "medium"
    assert sorted_matches[2].value == "low"


def test_match_accumulator_get_matches_with_min_hit_count():
    """Test filtering matches by minimum hit count."""
    accumulator = MatchAccumulator[str]()
    accumulator.add("example1", score=1.0)
    accumulator.add("example1", score=0.5)
    accumulator.add("example2", score=2.0)

    matches = list(accumulator._matches_with_min_hit_count(min_hit_count=2))  # type: ignore  # Need an internal method.
    assert len(matches) == 1
    assert matches[0].value == "example1"


def test_semantic_ref_accumulator_add_term_matches():
    """Test adding term matches to SemanticRefAccumulator."""
    accumulator = SemanticRefAccumulator()
    search_term: Term = Term("example term")
    scored_refs = [
        ScoredSemanticRefOrdinal(semantic_ref_ordinal=1, score=0.8),
        ScoredSemanticRefOrdinal(semantic_ref_ordinal=2, score=0.6),
    ]

    accumulator.add_term_matches(
        search_term, scored_refs, is_exact_match=True, weight=1.0
    )

    match1 = accumulator.get_match(1)
    match2 = accumulator.get_match(2)

    assert match1 is not None
    assert match1.score == 0.8
    assert match1.hit_count == 1

    assert match2 is not None
    assert match2.score == 0.6
    assert match2.hit_count == 1

    # Convert Term to string for comparison
    assert search_term.text in accumulator.search_term_matches


def test_text_range_collection_add_and_check():
    """Test adding ranges to TextRangeCollection and checking containment."""
    range1 = TextRange(start=TextLocation(0), end=TextLocation(10))
    range2 = TextRange(start=TextLocation(20), end=TextLocation(30))
    range3 = TextRange(start=TextLocation(5), end=TextLocation(10))
    range4 = TextRange(start=TextLocation(5), end=TextLocation(25))
    range5 = TextRange(start=TextLocation(50), end=TextLocation(60))

    collection = TextRangeCollection()
    collection.add_range(range1)
    collection.add_range(range2)

    assert len(collection) == 2

    assert collection.is_in_range(range1) is True
    assert collection.is_in_range(range2) is True
    assert collection.is_in_range(range3) is False
    assert collection.is_in_range(range4) is False
    assert collection.is_in_range(range5) is False


def test_text_ranges_in_scope():
    """Test adding ranges to TextRangesInScope and checking scope."""
    range1 = TextRange(start=TextLocation(0), end=TextLocation(10))
    range2 = TextRange(start=TextLocation(0), end=TextLocation(20))
    range3 = TextRange(start=TextLocation(0), end=TextLocation(10))
    range4 = TextRange(start=TextLocation(5), end=TextLocation(15))

    collection1 = TextRangeCollection([range1])
    collection2 = TextRangeCollection([range2])

    ranges_in_scope = TextRangesInScope()
    ranges_in_scope.add_text_ranges(collection1)
    ranges_in_scope.add_text_ranges(collection2)

    assert ranges_in_scope.is_range_in_scope(range3)
    assert not ranges_in_scope.is_range_in_scope(range4)


def test_semantic_ref_accumulator_group_matches_by_type():
    """Test grouping matches by knowledge type in SemanticRefAccumulator."""
    accumulator = SemanticRefAccumulator()
    accumulator.add(0, score=0.8)
    accumulator.add(1, score=0.6)

    refs = [
        SemanticRef(
            0,
            range=TextRange(TextLocation(0)),
            knowledge_type="entity",
            knowledge=ConcreteEntity("ref1", ["ref"]),
        ),
        SemanticRef(
            1,
            range=TextRange(TextLocation(2)),
            knowledge_type="action",
            knowledge=Action(["go"], "past"),
        ),
    ]

    groups = accumulator.group_matches_by_type(SemanticRefCollection(refs))
    assert len(groups) == 2
    assert "entity" in groups
    assert "action" in groups
    assert groups["entity"].get_match(0) is not None
    assert groups["action"].get_match(1) is not None


def test_termset_add():
    """Test adding terms to the TermSet."""
    term1 = Term(text="example1", weight=1.0)
    term2 = Term(text="example2", weight=0.5)

    term_set = TermSet()
    assert term_set.add(term1) is True  # Term should be added
    assert term_set.add(term1) is False  # Duplicate term should not be added
    assert term_set.add(term2) is True  # Another term should be added

    assert len(term_set) == 2
    assert term1 in term_set
    assert term2 in term_set


def test_termset_add_or_union_single_term():
    """Test adding a single term using add_or_union."""
    term1 = Term(text="example1", weight=1.0)
    term2 = Term(text="example1", weight=2.0)  # Higher weight for the same term

    term_set = TermSet()
    term_set.add_or_union(term1)
    assert len(term_set) == 1
    term = term_set.get(term1)
    assert term is not None
    assert term.weight == 1.0

    term_set.add_or_union(term2)  # Should update the weight
    assert len(term_set) == 1
    term = term_set.get(term1)
    assert term is not None
    assert term.weight == 2.0


def test_termset_add_or_union_multiple_terms():
    """Test adding multiple terms using add_or_union."""
    term1 = Term(text="example1", weight=1.0)
    term2 = Term(text="example2", weight=0.5)
    term3 = Term(text="example3", weight=1.5)

    term_set = TermSet()
    term_set.add_or_union([term1, term2, term3])

    assert len(term_set) == 3
    assert term1 in term_set
    assert term2 in term_set
    assert term3 in term_set


def test_termset_get():
    """Test retrieving terms from the TermSet."""
    term1 = Term(text="example1", weight=1.0)
    term2 = Term(text="example2", weight=0.5)

    term_set = TermSet([term1, term2])
    assert term_set.get("example1") == term1
    assert term_set.get("example2") == term2
    assert term_set.get("nonexistent") is None


def test_termset_get_weight():
    """Test retrieving the weight of a term."""
    term1 = Term(text="example1", weight=1.0)
    term2 = Term(text="example2", weight=None)

    term_set = TermSet([term1, term2])
    assert term_set.get_weight(term1) == 1.0
    assert term_set.get_weight(term2) is None
    assert term_set.get_weight(Term(text="nonexistent")) is None


def test_termset_remove():
    """Test removing terms from the TermSet."""
    term1 = Term(text="example1", weight=1.0)
    term2 = Term(text="example2", weight=0.5)

    term_set = TermSet([term1, term2])
    assert term1 in term_set
    term_set.remove(term1)
    assert term1 not in term_set
    assert len(term_set) == 1


def test_termset_clear():
    """Test clearing all terms from the TermSet."""
    term1 = Term(text="example1", weight=1.0)
    term2 = Term(text="example2", weight=0.5)

    term_set = TermSet([term1, term2])
    assert len(term_set) == 2
    term_set.clear()
    assert len(term_set) == 0


def test_termset_values():
    """Test retrieving all terms as a list."""
    term1 = Term(text="example1", weight=1.0)
    term2 = Term(text="example2", weight=0.5)

    term_set = TermSet([term1, term2])
    values = term_set.values()

    assert len(values) == 2
    assert term1 in values
    assert term2 in values


def test_property_term_set_add():
    """Test adding property terms to the PropertyTermSet."""
    term1 = Term(text="value1", weight=1.0)
    term2 = Term(text="value2", weight=0.5)

    property_term_set = PropertyTermSet()
    property_term_set.add("property1", term1)
    property_term_set.add("property2", term2)

    assert len(property_term_set.terms) == 2
    assert property_term_set.has("property1", "value1") is True
    assert property_term_set.has("property2", term2) is True


def test_property_term_set_add_duplicate():
    """Test adding duplicate property terms."""
    term1 = Term(text="value1", weight=1.0)

    property_term_set = PropertyTermSet()
    property_term_set.add("property1", term1)
    property_term_set.add("property1", term1)  # Duplicate

    assert len(property_term_set.terms) == 1  # Should not add duplicate
    assert property_term_set.has("property1", "value1") is True


def test_property_term_set_has():
    """Test checking for the existence of property terms."""
    term1 = Term(text="value1", weight=1.0)
    term2 = Term(text="value2", weight=0.5)

    property_term_set = PropertyTermSet()
    property_term_set.add("property1", term1)

    assert property_term_set.has("property1", "value1") is True
    assert property_term_set.has("property1", term1) is True
    assert property_term_set.has("property2", term2) is False
    assert property_term_set.has("property3", "value3") is False


def test_property_term_set_clear():
    """Test clearing all property terms from the PropertyTermSet."""
    term1 = Term(text="value1", weight=1.0)
    term2 = Term(text="value2", weight=0.5)

    property_term_set = PropertyTermSet()
    property_term_set.add("property1", term1)
    property_term_set.add("property2", term2)

    assert len(property_term_set.terms) == 2
    property_term_set.clear()
    assert len(property_term_set.terms) == 0
    assert property_term_set.has("property1", "value1") is False
    assert property_term_set.has("property2", "value2") is False


def test_semantic_ref_accumulator_get_semantic_refs():
    """Test retrieving semantic references with a predicate."""
    accumulator = SemanticRefAccumulator()
    accumulator.add(0, score=1.0)
    accumulator.add(1, score=0.5)

    refs = [
        SemanticRef(
            0,
            range=TextRange(TextLocation(0)),
            knowledge_type="entity",
            knowledge=ConcreteEntity("ref1", ["ref"]),
        ),
        SemanticRef(
            1,
            range=TextRange(TextLocation(2)),
            knowledge_type="action",
            knowledge=Action(["go"], "past"),
        ),
    ]

    semantic_refs = SemanticRefCollection(refs)

    # Predicate to filter only "entity" knowledge type
    predicate = lambda ref: ref.knowledge_type == "entity"
    filtered_refs = list(accumulator.get_semantic_refs(semantic_refs, predicate))

    assert len(filtered_refs) == 1
    assert filtered_refs[0].knowledge_type == "entity"


def test_semantic_ref_accumulator_get_matches_in_scope():
    """Test filtering matches by scope in SemanticRefAccumulator."""
    accumulator = SemanticRefAccumulator()
    accumulator.add(0, score=1.0)
    accumulator.add(1, score=0.5)

    refs = [
        SemanticRef(
            0,
            range=TextRange(TextLocation(0), TextLocation(10)),
            knowledge_type="entity",
            knowledge=ConcreteEntity("ref1", ["ref"]),
        ),
        SemanticRef(
            1,
            range=TextRange(TextLocation(20), TextLocation(30)),
            knowledge_type="action",
            knowledge=Action(["go"], "past"),
        ),
    ]

    semantic_refs = SemanticRefCollection(refs)
    ranges_in_scope = TextRangesInScope(
        [TextRangeCollection([TextRange(TextLocation(0), TextLocation(15))])]
    )

    filtered_accumulator = accumulator.get_matches_in_scope(
        semantic_refs, ranges_in_scope
    )

    assert len(filtered_accumulator) == 1
    assert filtered_accumulator.get_match(0) is not None
    assert filtered_accumulator.get_match(1) is None


def test_match_accumulator_select_top_n_scoring():
    """Test retaining only the top N scoring matches."""
    accumulator = MatchAccumulator[str]()
    accumulator.add("low", score=0.5)
    accumulator.add("medium", score=1.0)
    accumulator.add("high", score=2.0)

    top_n_count = accumulator.select_top_n_scoring(max_matches=2)

    assert top_n_count == 2
    matches = accumulator.get_sorted_by_score()
    assert len(matches) == 2
    assert matches[0].value == "high"
    assert matches[1].value == "medium"


def test_get_smooth_score():
    """Test calculating smooth scores."""
    assert get_smooth_score(10.0, 1) == 10.0  # Single hit count, no smoothing
    assert get_smooth_score(0.0, 2) == 0.0  # Zero total score remains zero
    assert get_smooth_score(10.0, 0) == 0.0  # Zero hit count results in zero
    assert 0.0 < get_smooth_score(10.0, 2) < 10.0  # Smoothing increases score


def test_add_smooth_related_score_to_match_score():
    """Test adding smooth related scores to match scores."""
    match = Match(
        value="example",
        score=10.0,
        hit_count=2,
        related_score=5.0,
        related_hit_count=3,
    )

    add_smooth_related_score_to_match_score(match)

    assert match.score > 10.0  # Related score should increase total score


def test_top_n_collection():
    """Test maintaining the top N items in TopNCollection."""
    top_n = TopNCollection[str](max_count=3)
    top_n.push("low", score=0.5)
    top_n.push("medium", score=1.0)
    top_n.push("high", score=2.0)
    top_n.push("extra", score=0.2)  # Should not be included in top 3

    ranked = top_n.by_rank()
    assert len(ranked) == 3
    assert ranked[0].item == "high"
    assert ranked[1].item == "medium"
    assert ranked[2].item == "low"


def test_get_top_k():
    """Test retrieving the top K items from an unsorted list."""
    items = [
        Scored(item="low", score=0.5),
        Scored(item="medium", score=1.0),
        Scored(item="high", score=2.0),
    ]

    top_k = get_top_k(items, top_k=2)

    assert len(top_k) == 2
    assert top_k[0].item == "high"
    assert top_k[1].item == "medium"


def test_add_to_set():
    """Test adding values to a set."""
    my_set = set()
    add_to_set(my_set, ["a", "b", "c"])
    add_to_set(my_set, ["b", "c", "d"])

    assert len(my_set) == 4
    assert "a" in my_set
    assert "b" in my_set
    assert "c" in my_set
    assert "d" in my_set
