# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable

from .interfaces import ScoredSemanticRefOrdinal, SemanticRefOrdinal, Term


@dataclass
class Match[T]:
    value: T
    score: float
    hit_count: int
    related_score: float
    related_hit_count: int


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

    # def get_sorted_by_score...

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

    # def matches_with_min_hit_count...


class SemanticRefAccumulator(MatchAccumulator[SemanticRefOrdinal]):
    def __init__(self, search_term_matches=set[str]()):
        super().__init__()
        self.search_term_matches = search_term_matches

    def add_term_matches(
        self,
        search_term: Term,
        scored_refs: Iterable[ScoredSemanticRefOrdinal] | None,
        *,
        is_exact_match: bool = True,  # TODO: May disappear
        weight: float | None = None,
    ) -> None:
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


@dataclass
class TermSet: ...  # TODO


@dataclass
class PropertyTermSet: ...  # TODO


@dataclass
class TextRangesInScope: ...  # TODO
