# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import bisect
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
import heapq
import math
import sys
from typing import Set, cast

from .interfaces import (
    ISemanticRefCollection,
    Knowledge,
    KnowledgeType,
    MessageOrdinal,
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    SemanticRefOrdinal,
    Term,
    TextRange,
)


@dataclass
class Match[T]:
    value: T
    score: float
    hit_count: int
    related_score: float
    related_hit_count: int


# TODO: sortMatchesByRelevance


class MatchAccumulator[T]:
    def __init__(self):
        self._matches: dict[T, Match[T]] = {}

    def __len__(self) -> int:
        return len(self._matches)

    def __iter__(self) -> Iterator[Match[T]]:
        return iter(self._matches.values())

    def __contains__(self, value: T) -> bool:
        return value in self._matches

    def get_match(self, value: T) -> Match[T] | None:
        return self._matches.get(value)

    def set_match(self, match: Match[T]) -> None:
        self._matches[match.value] = match

    # TODO: Maybe make the callers call clear_matches()?
    def set_matches(self, matches: Iterable[Match[T]], *, clear: bool = False) -> None:
        if clear:
            self.clear_matches()
        for match in matches:
            self.set_match(match)

    def get_max_hit_count(self) -> int:
        count = 0
        for match in self._matches.values():
            count = max(count, match.hit_count)
        return count

    # TODO: Rename to add_exact if we ever add add_related
    def add(self, value: T, score: float, is_exact_match: bool = True) -> None:
        existing_match = self.get_match(value)
        if existing_match is not None:
            if is_exact_match:
                existing_match.hit_count += 1
                existing_match.score += score
            else:
                existing_match.related_hit_count += 1
                existing_match.related_score += score
        else:
            if is_exact_match:
                self.set_match(
                    Match(
                        value,
                        hit_count=1,
                        score=score,
                        related_hit_count=0,
                        related_score=0.0,
                    )
                )
            else:
                self.set_match(
                    Match(
                        value,
                        hit_count=1,
                        score=0.0,
                        related_hit_count=1,
                        related_score=score,
                    )
                )

    def add_union(self, other: "MatchAccumulator[T]") -> None:
        """Add matches from another collection of matches."""
        for other_match in other:
            existing_match = self.get_match(other_match.value)
            if existing_match is None:
                self.set_match(other_match)
            else:
                self.combine_matches(existing_match, other_match)

    def intersect(
        self, other: "MatchAccumulator[T]", intersection: "MatchAccumulator[T]"
    ) -> "MatchAccumulator[T]":
        """Intersect with another collection of matches."""
        for self_match in self:
            other_match = other.get_match(self_match.value)
            if other_match is not None:
                self.combine_matches(self_match, other_match)
                intersection.set_match(self_match)
        return intersection

    def combine_matches(self, match: Match[T], other: Match[T]) -> None:
        """Combine the other match into the first."""
        match.hit_count += other.hit_count
        match.score += other.score
        match.related_hit_count += other.related_hit_count
        match.related_score += other.related_score

    def calculate_total_score(
        self, scorer: Callable[[Match[T]], None] | None = None
    ) -> None:
        if scorer is None:
            scorer = add_smooth_related_score_to_match_score
        for match in self:
            scorer(match)

    def get_sorted_by_score(self, min_hit_count: int | None = None) -> list[Match[T]]:
        """Get matches sorted by score"""
        if len(self._matches) == 0:
            return []
        matches = [*self._matches_with_min_hit_count(min_hit_count)]
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def get_top_n_scoring(
        self,
        max_matches: int | None = None,
        min_hit_count: int | None = None,
    ) -> list[Match[T]]:
        """Get the top N scoring matches."""
        if not self._matches:
            return []
        if max_matches and max_matches > 0:
            top_list = TopNList[T](max_matches)
            for match in self._matches_with_min_hit_count(min_hit_count):
                top_list.push(match.value, match.score)
            ranked = top_list.by_rank()
            return [self._matches[match.item] for match in ranked]
        else:
            return self.get_sorted_by_score(min_hit_count)

    def get_with_hit_count(self, min_hit_count: int) -> list[Match[T]]:
        """Get matches with a minimum hit count."""
        return list(self.matches_with_min_hit_count(min_hit_count))

    def get_matches(
        self, predicate: Callable[[Match[T]], bool] | None = None
    ) -> Iterator[Match[T]]:
        """Iterate over all matches."""
        if predicate is None:
            return iter(self._matches.values())
        else:
            return filter(predicate, self._matches.values())

    def get_matched_values(self) -> Iterator[T]:
        """Iterate over all matched values."""
        return iter(self._matches)

    def clear_matches(self):
        self._matches.clear()

    def select_top_n_scoring(
        self,
        max_matches: int | None = None,
        min_hit_count: int | None = None,
    ) -> int:
        """Retain only the top N matches sorted by score."""
        top_n = self.get_top_n_scoring(max_matches, min_hit_count)
        self.set_matches(top_n, clear=True)
        return len(top_n)

    def select_with_hit_count(self, min_hit_count: int) -> int:
        """Retain only matches with a minimum hit count."""
        matches = self.get_with_hit_count(min_hit_count)
        self.set_matches(matches, clear=True)
        return len(matches)

    def _matches_with_min_hit_count(
        self, min_hit_count: int | None
    ) -> Iterable[Match[T]]:
        """Get matches with a minimum hit count"""
        if min_hit_count is not None and min_hit_count > 0:
            return self.get_matches(lambda m: m.hit_count >= min_hit_count)
        else:
            return self._matches.values()

    def matches_with_min_hit_count(
        self, min_hit_count: int | None
    ) -> Iterable[Match[T]]:
        if min_hit_count is not None and min_hit_count > 0:
            return filter(lambda m: m.hit_count >= min_hit_count, self.get_matches())
        else:
            return self._matches.values()


def get_smooth_score(
    total_score: float,
    hit_count: int,
) -> float:
    """See the long comment in collections.ts for an explanation."""
    if hit_count > 0:
        if hit_count == 1:
            return total_score
        avg = total_score / hit_count
        smooth_avg = math.log(hit_count + 1) * avg
        return smooth_avg
    else:
        return 0.0


def add_smooth_related_score_to_match_score[T](match: Match[T]) -> None:
    """Add the smooth related score to the match score."""
    if match.related_hit_count > 0:
        # Related term matches can be noisy and duplicative.
        # See the comment on getSmoothScore  in collections.ts.
        smooth_related_score = get_smooth_score(
            match.related_score, match.related_hit_count
        )
        match.score += smooth_related_score


def smooth_match_score[T](match: Match[T]) -> None:
    if match.hit_count > 0:
        match.score = get_smooth_score(match.score, match.hit_count)


type KnowledgePredicate[T: Knowledge] = Callable[[T], bool]


class SemanticRefAccumulator(MatchAccumulator[SemanticRefOrdinal]):
    def __init__(self, search_term_matches: set[str] = set()):
        super().__init__()
        self.search_term_matches = search_term_matches

    def add_term_matches(
        self,
        search_term: Term,
        scored_refs: Iterable[ScoredSemanticRefOrdinal] | None,
        is_exact_match: bool,
        *,
        weight: float | None = None,
    ) -> None:
        """Add term matches to the accumulator"""
        if scored_refs is not None:
            if weight is None:
                weight = search_term.weight
                if weight is None:
                    weight = 1.0
            for scored_ref in scored_refs:
                self.add(
                    scored_ref.semantic_ref_ordinal,
                    scored_ref.score * weight,
                    is_exact_match,
                )
            self.search_term_matches.add(search_term.text)

    def add_term_matches_if_new(
        self,
        search_term: Term,
        scored_refs: Iterable[ScoredSemanticRefOrdinal] | None,
        is_exact_match: bool,
        weight: float | None = None,
    ) -> None:
        """Add term matches if they are new."""
        if scored_refs is not None:
            if weight is None:
                weight = search_term.weight
                if weight is None:
                    weight = 1.0
            for scored_ref in scored_refs:
                if scored_ref.semantic_ref_ordinal not in self:
                    self.add(
                        scored_ref.semantic_ref_ordinal,
                        scored_ref.score * weight,
                        is_exact_match,
                    )
            self.search_term_matches.add(search_term.text)

    def get_semantic_refs(
        self,
        semantic_refs: ISemanticRefCollection,
        predicate: Callable[[SemanticRef], bool],
    ) -> Iterable[SemanticRef]:
        for match in self:
            semantic_ref = semantic_refs[match.value]
            if predicate is None or predicate(semantic_ref):
                yield semantic_ref

    def get_matches_of_type[T: Knowledge](
        self,
        semantic_refs: list[SemanticRef],
        knowledgeType: KnowledgeType,
        predicate: KnowledgePredicate[T] | None = None,
    ) -> Iterable[Match[SemanticRefOrdinal]]:
        for match in self:
            semantic_ref = semantic_refs[match.value]
            if predicate is None or predicate(cast(T, semantic_ref.knowledge)):
                yield match

    def group_matches_by_type(
        self,
        semantic_refs: ISemanticRefCollection,
    ) -> dict[KnowledgeType, "SemanticRefAccumulator"]:
        groups: dict[KnowledgeType, SemanticRefAccumulator] = {}
        for match in self:
            semantic_ref = semantic_refs[match.value]
            group = groups.get(semantic_ref.knowledge_type)
            if group is None:
                group = SemanticRefAccumulator()
                group.search_term_matches = self.search_term_matches
                groups[semantic_ref.knowledge_type] = group
            group.set_match(match)
        return groups

    def get_matches_in_scope(
        self,
        semantic_refs: ISemanticRefCollection,
        ranges_in_scope: "TextRangesInScope",
    ) -> "SemanticRefAccumulator":
        accumulator = SemanticRefAccumulator(self.search_term_matches)
        for match in self:
            if ranges_in_scope.is_range_in_scope(semantic_refs[match.value].range):
                accumulator.set_match(match)
        return accumulator

    def add_union(self, other: "MatchAccumulator[SemanticRefOrdinal]") -> None:
        """Add matches from another SemanticRefAccumulator."""
        assert isinstance(other, SemanticRefAccumulator)
        super().add_union(other)
        self.search_term_matches.update(other.search_term_matches)

    def intersect(
        self,
        other: MatchAccumulator[SemanticRefOrdinal],
        intersection: MatchAccumulator[SemanticRefOrdinal] | None = None,
    ) -> "SemanticRefAccumulator":
        """Intersect with another SemanticRefAccumulator."""
        assert isinstance(other, SemanticRefAccumulator)
        if intersection is None:
            intersection = SemanticRefAccumulator()
        else:
            assert isinstance(intersection, SemanticRefAccumulator)
        super().intersect(other, intersection)
        if len(intersection) > 0:
            intersection.search_term_matches.update(self.search_term_matches)
            intersection.search_term_matches.update(other.search_term_matches)
        return intersection

    def to_scored_semantic_refs(self) -> list[ScoredSemanticRefOrdinal]:
        """Convert the accumulator to a list of scored semantic references."""
        return [
            ScoredSemanticRefOrdinal(
                semantic_ref_ordinal=match.value,
                score=match.score,
            )
            for match in self.get_sorted_by_score()
        ]


class MessageAccumulator(MatchAccumulator[MessageOrdinal]):
    def __init__(self, matches: list[Match[MessageOrdinal]] | None = None):
        super().__init__()
        if matches:
            self.set_matches(matches)

    def add(
        self, value: MessageOrdinal, score: float, is_exact_match: bool = True
    ) -> None:
        match = self.get_match(value)
        if match is None:
            match = Match(value, score, 1, 0.0, 0)
            self.set_match(match)
        elif score > match.score:
            match.score = score
            # TODO: Question(Guido->Umesh): Why not increment hit_count always?
            match.hit_count += 1

    # TODO: add_messages_from_locations

    def add_messages_for_semantic_ref(
        self,
        semantic_ref: SemanticRef,
        score: float,
    ) -> None:
        message_ordinal_start = semantic_ref.range.start.message_ordinal
        if semantic_ref.range.end is not None:
            message_ordinal_end = semantic_ref.range.end.message_ordinal
            for message_ordinal in range(message_ordinal_start, message_ordinal_end):
                self.add(message_ordinal, score)
        else:
            self.add(message_ordinal_start, score)

    # TODO: add_range, add_scored_matches

    def intersect(
        self,
        other: MatchAccumulator[MessageOrdinal],
        intersection: MatchAccumulator[MessageOrdinal] | None = None,
    ) -> "MessageAccumulator":
        if intersection is None:
            intersection = MessageAccumulator()
        else:
            assert isinstance(intersection, MessageAccumulator)
        super().intersect(other, intersection)
        return intersection

    def smooth_scores(self) -> None:
        for match in self:
            smooth_match_score(match)

    def to_scored_message_ordinals(self) -> list[ScoredMessageOrdinal]:
        sorted_matches = self.get_sorted_by_score()
        return [ScoredMessageOrdinal(m.value, m.score) for m in sorted_matches]

    # TODO: select_messages_in_budget
    # TODO: from_scored_ordinals


# TODO: intersectScoredMessageOrdinals


@dataclass
class TextRangeCollection(Iterable[TextRange]):
    _ranges: list[TextRange]

    def __init__(
        self,
        ranges: list[TextRange] | None = None,
        ensure_sorted: bool = False,
    ) -> None:
        if ensure_sorted:
            self._ranges = []
            if ranges:
                self.add_ranges(ranges)
        else:
            self._ranges = ranges if ranges is not None else []

    def __len__(self) -> int:
        return len(self._ranges)

    def __iter__(self) -> Iterator[TextRange]:
        return iter(self._ranges)

    def get_ranges(self) -> list[TextRange]:
        return self._ranges  # TODO: Maybe return a copy?

    def add_range(self, text_range: TextRange) -> bool:
        # This assumes TextRanges are totally ordered.
        pos = bisect.bisect_left(self._ranges, text_range)
        if pos < len(self._ranges) and self._ranges[pos] == text_range:
            return False
        self._ranges.insert(pos, text_range)
        return True

    def add_ranges(self, text_ranges: "list[TextRange] | TextRangeCollection") -> None:
        if isinstance(text_ranges, list):
            for text_range in text_ranges:
                self.add_range(text_range)
        else:
            assert isinstance(text_ranges, TextRangeCollection)
            for text_range in text_ranges._ranges:
                self.add_range(text_range)

    def is_in_range(self, inner_range: TextRange) -> bool:
        if len(self._ranges) == 0:
            return False
        i = bisect.bisect_left(self._ranges, inner_range)
        for outer_range in self._ranges[i:]:
            if outer_range.start > inner_range.start:
                break
            if inner_range in outer_range:
                return True
        return False


@dataclass
class TextRangesInScope:
    text_ranges: list[TextRangeCollection] | None = None

    def add_text_ranges(
        self,
        ranges: TextRangeCollection,
    ) -> None:
        if self.text_ranges is None:
            self.text_ranges = []
        self.text_ranges.append(ranges)

    def is_range_in_scope(self, inner_range: TextRange) -> bool:
        if self.text_ranges is not None:
            # Since outer ranges come from a set of range selectors, they may overlap, or may not agree.
            # Outer ranges allowed by say a date range selector... may not be allowed by a tag selector.
            # We have a very simple impl: we don't intersect/union ranges yet.
            # Instead, we ensure that the inner range is not rejected by any outer ranges.
            for outer_ranges in self.text_ranges:
                if not outer_ranges.is_in_range(inner_range):
                    return False
        return True


@dataclass
class TermSet:
    """A collection of terms with support for adding, updating, and retrieving terms."""

    terms: dict[str, Term]

    def __init__(self, terms: list[Term] | None = None):
        self.terms = {}
        self.add_or_union(terms)

    def __len__(self) -> int:
        """Return the number of terms in the set."""
        return len(self.terms)

    def add(self, term: Term) -> bool:
        """Add a term to the set if it doesn't already exist."""
        if term.text in self.terms:
            return False
        self.terms[term.text] = term
        return True

    def add_or_union(self, terms: Term | list[Term] | None) -> None:
        """Add a term or merge a list of terms into the set."""
        if terms is None:
            return
        if isinstance(terms, list):
            for term in terms:
                self.add_or_union(term)
        else:
            existing_term = self.terms.get(terms.text)
            if existing_term:
                existing_score = existing_term.weight or 0
                new_score = terms.weight or 0
                if new_score > existing_score:
                    existing_term.weight = new_score
            else:
                self.terms[terms.text] = terms

    def get(self, term: str | Term) -> Term | None:
        """Retrieve a term by its text."""
        return self.terms.get(term if isinstance(term, str) else term.text)

    def get_weight(self, term: Term) -> float | None:
        """Retrieve the weight of a term."""
        t = self.terms.get(term.text)
        return t.weight if t is not None else None

    def __contains__(self, term: Term) -> bool:
        """Check if a term exists in the set."""
        return term.text in self.terms

    def remove(self, term: Term):
        """Remove a term from the set, if present."""
        self.terms.pop(term.text, None)

    def clear(self):
        """Clear all terms from the set."""
        self.terms.clear()

    def values(self) -> list[Term]:
        """Retrieve all terms in the set."""
        return list(self.terms.values())


@dataclass
class PropertyTermSet:
    """A collection of property terms with support for adding, checking, and clearing."""

    terms: dict[str, Term] = field(default_factory=dict[str, Term])

    def add(self, property_name: str, property_value: Term) -> None:
        """Add a property term to the set."""
        key = self._make_key(property_name, property_value)
        if key not in self.terms:
            self.terms[key] = property_value

    def has(self, property_name: str, property_value: Term | str) -> bool:
        """Check if a property term exists in the set."""
        key = self._make_key(property_name, property_value)
        return key in self.terms

    def clear(self) -> None:
        """Clear all property terms from the set."""
        self.terms.clear()

    def _make_key(self, property_name: str, property_value: Term | str) -> str:
        """Create a unique key for a property term."""
        value = (
            property_value if isinstance(property_value, str) else property_value.text
        )
        return f"{property_name}:{value}"


# TODO: unionArrays
# TODO: union
# TODO: addToSet
# TODO: setUnion
# TODO: setIntersect
# TODO: getBatches


@dataclass
class Scored[T]:
    item: T
    score: float

    def __lt__(self, other: "Scored[T]") -> bool:
        return self.score < other.score

    def __gt__(self, other: "Scored[T]") -> bool:
        return self.score > other.score

    def __le__(self, other: "Scored[T]") -> bool:
        return self.score <= other.score

    def __ge__(self, other: "Scored[T]") -> bool:
        return self.score >= other.score


# Implementation change compared to TS version: Use heapq; no sentinel.
# API change: pop/top are not properties.
class TopNCollection[T]:
    """A collection that maintains the top N items based on their scores."""

    def __init__(self, max_count: int):
        self._max_count = max_count
        self._heap: list[Scored[T]] = []

    def __len__(self) -> int:
        return len(self._heap)

    def reset(self) -> None:
        self._heap = []

    def pop(self) -> Scored[T]:
        return heapq.heappop(self._heap)

    def top(self) -> Scored[T]:
        return self._heap[0]

    def push(self, item: T, score: float) -> None:
        if len(self._heap) < self._max_count:
            heapq.heappush(self._heap, Scored(item, score))
        else:
            heapq.heappushpop(self._heap, Scored(item, score))

    def by_rank(self) -> list[Scored[T]]:
        return sorted(self._heap, reverse=True)

    def values_by_rank(self) -> list[T]:
        return [item.item for item in self.by_rank()]


class TopNList[T](TopNCollection[T]):
    """Alias for TopNCollection."""


class TopNListAll[T](TopNList[T]):
    """A Top N list for N = infinity (approximated by sys.maxsize)."""

    def __init__(self):
        super().__init__(sys.maxsize)


def get_top_k[T](
    scored_items: Iterable[Scored[T]],
    top_k: int,
) -> list[Scored[T]]:
    """A function to get the top K of an unsorted list of scored items."""
    top_n_list = TopNCollection[T](top_k)
    for scored_item in scored_items:
        top_n_list.push(scored_item.item, scored_item.score)
    return top_n_list.by_rank()


def add_to_set[T](
    set: Set[T],
    values: Iterable[T],
) -> None:
    """Add values to a set."""
    set.update(values)
