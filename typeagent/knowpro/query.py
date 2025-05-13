# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .collections import (
    MatchAccumulator,
    PropertyTermSet,
    SemanticRefAccumulator,
    TermSet,
    TextRangeCollection,
    TextRangesInScope,
)
from .interfaces import (
    IConversation,
    IMessage,
    IMessageCollection,
    IPropertyToSemanticRefIndex,
    ISemanticRefCollection,
    ITermToSemanticRefIndex,
    ITimestampToTextRangeIndex,
    MessageOrdinal,
    ScoredSemanticRefOrdinal,
    SearchTerm,
    SemanticRef,
    SemanticRefOrdinal,
    Term,
)


def is_conversation_searchable(conversation: IConversation) -> bool:
    """Determine if a conversation is searchable.

    A conversation is searchable if it has a semantic reference index
    and semantic references initialized.
    """
    return (
        conversation.semantic_ref_index is not None
        and conversation.semantic_refs is not None
    )


# TODO: getTextRangeForDateRange ... matchPropertySearchTermToSemanticRef


def lookup_term_filtered(
    semantic_ref_index: ITermToSemanticRefIndex,
    term: Term,
    semantic_refs: ISemanticRefCollection,
    filter: Callable[[SemanticRef, ScoredSemanticRefOrdinal], bool],
) -> list[ScoredSemanticRefOrdinal] | None:
    """Look up a term in the semantic reference index and filter the results."""
    scored_refs = semantic_ref_index.lookup_term(term.text)
    if scored_refs:
        filtered = [
            sr
            for sr in scored_refs
            if filter(semantic_refs[sr.semantic_ref_ordinal], sr)
        ]
        return filtered
    return None


def lookup_term(
    semantic_ref_index: ITermToSemanticRefIndex,
    term: Term,
    semantic_refs: ISemanticRefCollection,
    ranges_in_scope: TextRangesInScope | None = None,
) -> list[ScoredSemanticRefOrdinal] | None:
    """Look up a term in the semantic reference index, optionally filtering by ranges in scope."""
    if ranges_in_scope:
        # If ranges_in_scope has no actual text ranges, lookups can't possibly match
        return lookup_term_filtered(
            semantic_ref_index,
            term,
            semantic_refs,
            lambda sr, _: ranges_in_scope.is_range_in_scope(sr.range),
        )
    return semantic_ref_index.lookup_term(term.text)


# TODO: lookupProperty, lookupTermFiltered, lookupKnowledgeType


@dataclass
class QueryEvalContext:
    """Context for evaluating a query within a conversation.

    This class provides the necessary context for query evaluation, including
    the conversation being queried, optional indexes for properties and timestamps,
    and structures for tracking matched terms and text ranges in scope.
    """

    # TODO: Make property and timestamp indexes NON-OPTIONAL
    # TODO: Move non-index based code to test

    conversation: IConversation[IMessage, ITermToSemanticRefIndex]
    # If a property secondary index is available, the query processor will use it
    property_index: IPropertyToSemanticRefIndex | None = None
    # If a timestamp secondary index is available, the query processor will use it
    timestamp_index: ITimestampToTextRangeIndex | None = None
    matched_terms: TermSet = field(init=False, default_factory=TermSet)
    matched_property_terms: PropertyTermSet = field(
        init=False, default_factory=PropertyTermSet
    )
    text_ranges_in_scope: TextRangesInScope | None = field(
        init=False, default_factory=TextRangesInScope
    )

    def __post_init__(self):
        if not is_conversation_searchable(self.conversation):
            raise ValueError(
                f"{self.conversation.name_tag} "
                + "is not initialized and cannot be searched."
            )

    @property
    def semantic_ref_index(self) -> ITermToSemanticRefIndex:
        """Return the semantic reference index."""
        assert self.conversation.semantic_ref_index is not None
        return self.conversation.semantic_ref_index

    @property
    def semantic_refs(self) -> ISemanticRefCollection:
        """Return the semantic references."""
        assert self.conversation.semantic_refs is not None
        return self.conversation.semantic_refs

    @property
    def messages(self) -> IMessageCollection:
        """Return the messages in the conversation."""
        return self.conversation.messages

    def get_semantic_ref(self, semantic_ref_ordinal: SemanticRefOrdinal) -> SemanticRef:
        """Retrieve a semantic reference by its ordinal."""
        assert self.conversation.semantic_refs is not None
        return self.conversation.semantic_refs[semantic_ref_ordinal]

    def get_message_for_ref(self, semantic_ref: SemanticRef) -> IMessage:
        """Retrieve the message associated with a semantic reference."""
        message_ordinal = semantic_ref.range.start.message_ordinal
        return self.conversation.messages[message_ordinal]

    def get_message(self, message_ordinal: MessageOrdinal) -> IMessage:
        """Retrieve a message by its ordinal."""
        return self.messages[message_ordinal]

    def clear_matched_terms(self) -> None:
        """Clear all matched terms and property terms."""
        self.matched_terms.clear()
        self.matched_property_terms.clear()


class IQueryOpExpr[T](Protocol):
    """Protocol for query operation expressions that can be evaluated in a context."""

    def eval(self, context: QueryEvalContext) -> T:
        raise NotImplementedError


class QueryOpExpr[T](IQueryOpExpr[T]):
    """Base class for query operation expressions."""

    def eval(self, context: QueryEvalContext) -> T:
        raise NotImplementedError


@dataclass
class SelectTopNExpr[T: MatchAccumulator](QueryOpExpr[T]):
    """Expression for selecting the top N matches from a query."""

    source_expr: IQueryOpExpr[T]
    max_matches: int | None = None
    min_hit_count: int | None = None

    def eval(self, context: QueryEvalContext) -> T:
        """Evaluate the expression and return the top N matches."""
        matches = self.source_expr.eval(context)
        matches.select_top_n_scoring(self.max_matches, self.min_hit_count)
        return matches


@dataclass
class MatchTermsBooleanExpr(QueryOpExpr[SemanticRefAccumulator]):
    """Expression for matching terms in a boolean query."""

    get_scope_expr: "GetScopeExpr | None" = None

    def begin_match(self, context: QueryEvalContext) -> None:
        """Prepare for matching terms in the context by resetting some things."""
        if self.get_scope_expr is not None:
            context.text_ranges_in_scope = self.get_scope_expr.eval(context)
        context.clear_matched_terms()


@dataclass
class MatchTermsOrExpr(MatchTermsBooleanExpr):
    """Expression for matching terms with an OR condition."""

    term_expressions: list[IQueryOpExpr[SemanticRefAccumulator | None]] = []

    def eval(self, context: QueryEvalContext) -> SemanticRefAccumulator:
        self.begin_match(context)
        all_matches: SemanticRefAccumulator | None = None
        for match_expr in self.term_expressions:
            term_matches = match_expr.eval(context)
            if term_matches:
                if all_matches is None:
                    all_matches = term_matches
                else:
                    all_matches.add_union(term_matches)
        if all_matches is not None:
            all_matches.calculate_total_score()
        return all_matches or SemanticRefAccumulator()


class MatchTermExpr(QueryOpExpr[SemanticRefAccumulator | None], ABC):
    """Expression for matching terms in a query.

    This class evaluates a query expression to accumulate matches
    in the form of a SemanticRefAccumulator. If no matches are found,
    it returns None.
    """

    def eval(self, context: QueryEvalContext) -> SemanticRefAccumulator | None:
        matches = SemanticRefAccumulator()
        self.accumulate_matches(context, matches)
        if len(matches) > 0:
            return matches
        return None

    @abstractmethod
    def accumulate_matches(
        self, context: QueryEvalContext, matches: SemanticRefAccumulator
    ) -> None:
        raise NotImplementedError("Subclass must implement accumulate_matches")


type ScoreBoosterType = Callable[
    [SearchTerm, SemanticRef, ScoredSemanticRefOrdinal],
    ScoredSemanticRefOrdinal,
]


class MatchSearchTermExpr(MatchTermExpr):
    def __init__(
        self,
        search_term: SearchTerm,
        score_booster: ScoreBoosterType | None = None,
    ) -> None:
        super().__init__()
        self.search_term = search_term
        self.score_booster = score_booster

    def accumulate_matches(
        self, context: QueryEvalContext, matches: SemanticRefAccumulator
    ) -> None:
        """Accumulate matches for the search term and its related terms."""
        # Match the search term
        self.accumulate_matches_for_term(context, matches, self.search_term.term)

        # And any related terms
        if self.search_term.related_terms is not None:
            for related_term in self.search_term.related_terms:
                self.accumulate_matches_for_term(
                    context, matches, self.search_term.term, related_term
                )

    def lookup_term(
        self, context: QueryEvalContext, term: Term
    ) -> list[ScoredSemanticRefOrdinal] | None:
        """Look up a term in the semantic reference index."""
        matches = lookup_term(
            context.semantic_ref_index,
            term,
            context.semantic_refs,
            context.text_ranges_in_scope,
        )
        if matches and self.score_booster:
            for i in range(len(matches)):
                matches[i] = self.score_booster(
                    self.search_term,
                    context.get_semantic_ref(matches[i].semantic_ref_ordinal),
                    matches[i],
                )
        return matches

    def accumulate_matches_for_term(
        self,
        context: QueryEvalContext,
        matches: SemanticRefAccumulator,
        term: Term,
        related_term: Term | None = None,
    ) -> None:
        """Accumulate matches for a term or a related term."""
        if related_term is None:
            if term not in context.matched_terms:
                semantic_refs = self.lookup_term(context, term)
                matches.add_term_matches(term, semantic_refs, True)
                context.matched_terms.add(term)
        else:
            if related_term not in context.matched_terms:
                # If this related term had not already matched as a related term for some other term
                # Minimize over counting
                semantic_refs = self.lookup_term(context, related_term)
                # This will only consider semantic refs that have not already matched this expression.
                # In other words, if a semantic ref already matched due to the term 'novel',
                # don't also match it because it matched the related term 'book'
                matches.add_term_matches_if_new(
                    term, semantic_refs, False, related_term.weight
                )
                context.matched_terms.add(related_term)


class IQueryTextRangeSelector(Protocol):
    """Protocol for a selector that can evaluate to a text range."""

    def eval(
        self,
        context: QueryEvalContext,
        semantic_refs: SemanticRefAccumulator | None = None,
    ) -> TextRangeCollection | None:
        """Evaluate the selector and return the text range."""
        raise NotImplementedError


@dataclass
class GetScopeExpr(QueryOpExpr[TextRangesInScope]):
    """Expression for getting the scope of a query."""

    range_selectors: list[IQueryTextRangeSelector]

    def eval(self, context: QueryEvalContext) -> TextRangesInScope:
        """Evaluate the expression and return the text ranges in scope."""
        ranges_in_scope = TextRangesInScope()
        for selector in self.range_selectors:
            range_collection = selector.eval(context)
            if range_collection is not None:
                ranges_in_scope.add_text_ranges(range_collection)
        return ranges_in_scope
