# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.knowpro.kplib import Action, ConcreteEntity
from typeagent.knowpro.collections import (
    MatchAccumulator,
    SemanticRefAccumulator,
    TextRangeCollection,
    TextRangesInScope,
)
from typeagent.knowpro.interfaces import (
    Knowledge,
    TextRange,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    KnowledgeType,
    TextLocation,
    Term,
    ISemanticRefCollection,
)


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

    matches = list(accumulator._matches_with_min_hit_count(min_hit_count=2))
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
    assert (
        collection.is_in_range(range5)
        is False
    )


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


class MockSemanticRefCollection(ISemanticRefCollection):
    """Mock implementation of ISemanticRefCollection."""

    def __init__(self):
        self.refs = {
            1: SemanticRef(
                1,
                range=TextRange(TextLocation(0)),
                knowledge_type="entity",
                knowledge=ConcreteEntity("ref1", ["ref"]),
            ),
            2: SemanticRef(
                1,
                range=TextRange(TextLocation(2)),
                knowledge_type="action",
                knowledge=Action(["go"], "past"),
            ),
        }

    def get(self, ordinal: int) -> SemanticRef:
        return self.refs[ordinal]

    def get_multiple(self, ordinals: list[int]) -> list[SemanticRef]:
        return [self.refs[o] for o in ordinals if o in self.refs]

    def get_slice(self, start: int, end: int) -> list[SemanticRef]:
        return [v for k, v in self.refs.items() if start <= k < end]

    def __len__(self):
        return len(self.refs)

    def __iter__(self):
        return iter(self.refs.values())

    @property
    def is_persistent(self) -> bool:
        return False

    def append(self, *items: SemanticRef) -> None:
        raise NotImplementedError


def test_semantic_ref_accumulator_group_matches_by_type():
    """Test grouping matches by knowledge type in SemanticRefAccumulator."""
    accumulator = SemanticRefAccumulator()
    accumulator.add(1, score=0.8)
    accumulator.add(2, score=0.6)

    groups = accumulator.group_matches_by_type(MockSemanticRefCollection())
    assert len(groups) == 2
    assert "entity" in groups
    assert "action" in groups
    assert groups["entity"].get_match(1) is not None
    assert groups["action"].get_match(2) is not None
