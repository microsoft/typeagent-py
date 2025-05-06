# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import bisect
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Iterator, cast

from .interfaces import (
    ISemanticRefCollection,
    Knowledge,
    KnowledgeType,
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


# TODO: sortMatchesByRelevance,


class MatchAccumulator[T]:
    def __init__(self):
        self._matches: dict[T, Match[T]] = {}

    def __len__(self) -> int:
        return len(self._matches)

    def __bool__(self) -> bool:
        return True

    def __contains__(self, value: T) -> bool:
        return value in self._matches

    def get_match(self, value: T) -> Match[T] | None:
        return self._matches.get(value)

    def set_match(self, match: Match[T]) -> None:
        self._matches[match.value] = match

    # TODO: Maybe make the callers call clear_matches()?
    def set_matches(self, matches: Iterable[Match[T]], clear=False) -> None:
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
        assert is_exact_match, "Only exact matches are supported"
        existing = self.get_match(value)
        if existing is not None:
            existing.hit_count += 1
            existing.score += score
        else:
            self.set_match(
                Match(
                    value=value,
                    hit_count=1,
                    score=score,
                    related_hit_count=0,
                    related_score=0,
                )
            )

    def add_union(self, other: Iterable["MatchAccumulator"]) -> None:
        raise NotImplementedError  # TODO

    def intersect(
        self, other: "MatchAccumulator", intersection: "MatchAccumulator | None"
    ) -> None:
        raise NotImplementedError  # TODO

    # def combine_matches...

    # def calculate_total_score...

    def get_sorted_by_score(self, min_hit_count: int | None = None) -> list[Match[T]]:
        """Get matches sorted by score"""
        if len(self._matches) == 0:
            return []
        matches = [*self._matches_with_min_hit_count(min_hit_count)]
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    # def get_top_n_scoring...

    # def get_with_hit_count...

    def get_matches(
        self, predicate: Callable[[Match[T]], bool] | None = None
    ) -> Iterable[Match[T]]:
        """Iterate over all matches"""
        for match in self._matches.values():
            if predicate is None or predicate(match):
                yield match

    def get_matched_values(self) -> Iterable[T]:
        """Iterate over all matched values"""
        for value in self._matches:
            yield value

    def clear_matches(self):
        self._matches.clear()

    # de select_top_n_scoring...

    # def select_with_hit_count...

    def _matches_with_min_hit_count(
        self, min_hit_count: int | None
    ) -> Iterable[Match[T]]:
        """Get matches with a minimum hit count"""
        if min_hit_count is not None and min_hit_count > 0:
            return self.get_matches(lambda m: m.hit_count >= min_hit_count)
        else:
            return self._matches.values()


# TODO: getSmoothScore, addSmoothRelatedScoreToMatchScore


type KnowledgePredicate[T: Knowledge] = Callable[[T], bool]


class SemanticRefAccumulator(MatchAccumulator[SemanticRefOrdinal]):
    def __init__(self, search_term_matches=set[str]()):
        super().__init__()
        self.search_term_matches = search_term_matches

    def add_term_matches(
        self,
        search_term: Term,
        scored_refs: Iterable[ScoredSemanticRefOrdinal] | None,
        is_exact_match: bool,  # TODO: May disappear
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

    def add_term_matches_if_new(self, *_, **__) -> None:
        """Add term matches if they are new"""
        raise NotImplementedError("TODO: add_term_matches_if_new")

    # TODO: Do we need this? And why the `| None` in the return type?
    # def get_sorted_by_score(
    #         self, min_hit_count: int | None
    # ) -> list[Match[SemanticRefOrdinal]] | None:
    #     return super().get_sorted_by_score(min_hit_count)

    # TODO: Do we need get_top_n_scoring if it just passes on to super?

    def get_semantic_refs(
        self,
        semantic_refs: ISemanticRefCollection,
        predicate: Callable[[SemanticRef], bool],
    ) -> Iterable[SemanticRef]:
        for match in self.get_matches():
            semantic_ref = semantic_refs.get(match.value)
            if predicate is None or predicate(semantic_ref):
                yield semantic_ref

    def get_matches_of_type[T: Knowledge](
        self,
        semantic_refs: list[SemanticRef],
        knowledgeType: KnowledgeType,
        predicate: KnowledgePredicate[T] | None = None,
    ) -> Iterable[Match[SemanticRefOrdinal]]:
        for match in self.get_matches():
            semantic_ref = semantic_refs[match.value]
            if predicate is None or predicate(cast(T, semantic_ref.knowledge)):
                yield match

    def group_matches_by_type(
        self,
        semantic_refs: ISemanticRefCollection,
    ) -> dict[KnowledgeType, "SemanticRefAccumulator"]:
        groups: dict[KnowledgeType, SemanticRefAccumulator] = {}
        for match in self.get_matches():
            semantic_ref = semantic_refs.get(match.value)
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
        for match in self.get_matches():
            if ranges_in_scope.is_range_in_scope(semantic_refs.get(match.value).range):
                accumulator.set_match(match)
        return accumulator


# TODO: MessageAccumulator, intersectScoredMessageOrdinals


@dataclass
class TextRangeCollection(Iterable[TextRange]):
    _ranges: list[TextRange]

    def __init__(
        self,
        ranges: list[TextRange] | None = None,
    ) -> None:
        if ranges is None:
            ranges = []
        self._ranges = ranges  # TODO: Maybe make a copy?
        # TODO: Maybe sort? Or assert it's sorted?

    def __len__(self) -> int:
        return len(self._ranges)

    def __iter__(self) -> Iterator[TextRange]:
        return iter(self._ranges)

    def get_ranges(self) -> list[TextRange]:
        return self._ranges  # TODO: Maybe return a copy?

    def add_range(self, text_range: TextRange) -> bool:
        # TODO: Are TextRanges total-ordered?
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
    def __init__(self, text_ranges: list[TextRangeCollection] | None = None):
        self.text_ranges = text_ranges

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
class TermSet: ...  # TODO


@dataclass
class PropertyTermSet: ...  # TODO


# TODO: unionArrays, union, addToSet, setUnion, setIntersect, getBatches,
