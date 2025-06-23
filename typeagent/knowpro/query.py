# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from ast import Not
from collections.abc import Iterator
from dataclasses import dataclass, field
from re import search
from typing import Callable, Literal, Protocol

from ..aitools.embeddings import NormalizedEmbedding

from .collections import (
    Match,
    MatchAccumulator,
    MessageAccumulator,
    PropertyTermSet,
    SemanticRefAccumulator,
    TermSet,
    TextRangeCollection,
    TextRangesInScope,
)
from .common import is_search_term_wildcard
from .interfaces import (
    Datetime,
    DateRange,
    IConversation,
    IMessage,
    IMessageCollection,
    IPropertyToSemanticRefIndex,
    ISemanticRefCollection,
    ITermToSemanticRefIndex,
    ITimestampToTextRangeIndex,
    KnowledgeType,
    MessageOrdinal,
    PropertySearchTerm,
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
    SearchTerm,
    SemanticRef,
    SemanticRefOrdinal,
    SemanticRefSearchResult,
    Term,
    TextLocation,
    TextRange,
)
from .kplib import ConcreteEntity
from .messageindex import IMessageTextEmbeddingIndex
from .propindex import PropertyNames, lookup_property_in_property_index


# TODO: Move to compilelib.py
type BooleanOp = Literal["and", "or", "or_max"]


# TODO: Move to compilelib.py
@dataclass
class CompiledTermGroup:
    boolean_op: BooleanOp
    terms: list[SearchTerm]


def is_conversation_searchable(conversation: IConversation) -> bool:
    """Determine if a conversation is searchable.

    A conversation is searchable if it has a semantic reference index
    and semantic references initialized.
    """
    # TODO: also require secondary indices, once we have removed non-index based retrieval to test.
    return (
        conversation.semantic_ref_index is not None
        and conversation.semantic_refs is not None
    )


def get_text_range_for_date_range(
    conversation: IConversation,
    date_range: DateRange,
) -> TextRange | None:
    messages = conversation.messages
    message_count = len(messages)
    range_start_ordinal: MessageOrdinal = -1
    range_end_ordinal = range_start_ordinal
    for message in messages:
        if Datetime.fromisoformat(message.timestamp) in date_range:
            if range_start_ordinal < 0:
                range_start_ordinal = message.ordinal
            range_end_ordinal = message.ordinal
        else:
            if range_start_ordinal >= 0:
                # We have a range, so break.
                break
    if range_start_ordinal >= 0:
        return TextRange(
            start=TextLocation(range_start_ordinal),
            end=TextLocation(range_end_ordinal + 1),
        )
    return None


def get_matching_term_for_text(search_term: SearchTerm, text: str) -> Term | None:
    # Do case-INSENSITIVE comparisons, since stored entities may have different case.
    if text.lower() == search_term.term.text.lower():
        return search_term.term
    if search_term.related_terms:
        for related_term in search_term.related_terms:
            if text.lower() == related_term.text.lower():
                return related_term
    return None


def match_search_term_to_text(search_term: SearchTerm, text: str | None) -> bool:
    if text:
        return get_matching_term_for_text(search_term, text) is not None
    return False


def match_search_term_to_one_of_text(
    search_term: SearchTerm, texts: list[str] | None
) -> bool:
    if texts:
        for text in texts:
            if match_search_term_to_text(search_term, text):
                return True
    return False


# TODO: match_search_term_to_entity
# TODO: match_property_search_term_to_entity
# TODO: match_concrete_entity


def match_entity_name_or_type(
    property_value: SearchTerm,
    entity: ConcreteEntity,
) -> bool:
    return match_search_term_to_text(
        property_value, entity.name
    ) or match_search_term_to_one_of_text(property_value, entity.type)


# TODO: match_property_name_to_facet_name
# TODO: match_property_name_to_facet_value
# TODO: match_property_search_term_to_action
# TODO: match_property_search_term_to_tag
# TODO: match_property_search_term_to_semantic_ref


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
    ktype: KnowledgeType | None = None,
) -> list[ScoredSemanticRefOrdinal] | None:
    """Look up a term in the semantic reference index, optionally filtering by ranges in scope."""
    if ranges_in_scope is not None:
        # If ranges_in_scope has no actual text ranges, lookups can't possibly match.
        return lookup_term_filtered(
            semantic_ref_index,
            term,
            semantic_refs,
            lambda sr, _: (not ktype or sr.knowledge_type == ktype)
            and ranges_in_scope.is_range_in_scope(sr.range),
        )
    return semantic_ref_index.lookup_term(term.text)


# TODO: lookup_property
# TODO: lookup_knowledge_type


@dataclass
class QueryEvalContext[TMessage: IMessage, TIndex: ITermToSemanticRefIndex]:
    """Context for evaluating a query within a conversation.

    This class provides the necessary context for query evaluation, including
    the conversation being queried, optional indexes for properties and timestamps,
    and structures for tracking matched terms and text ranges in scope.
    """

    # TODO: Make property and timestamp indexes NON-OPTIONAL
    # TODO: Move non-index based code to test

    conversation: IConversation[TMessage, TIndex]
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
        assert self.conversation.semantic_ref_index is not None
        return self.conversation.semantic_ref_index

    @property
    def semantic_refs(self) -> ISemanticRefCollection:
        assert self.conversation.semantic_refs is not None
        return self.conversation.semantic_refs

    @property
    def messages(self) -> IMessageCollection:
        return self.conversation.messages

    def get_semantic_ref(self, semantic_ref_ordinal: SemanticRefOrdinal) -> SemanticRef:
        """Retrieve a semantic reference by its ordinal."""
        assert self.conversation.semantic_refs is not None
        return self.conversation.semantic_refs[semantic_ref_ordinal]

    def get_message_for_ref(self, semantic_ref: SemanticRef) -> TMessage:
        """Retrieve the message associated with a semantic reference."""
        message_ordinal = semantic_ref.range.start.message_ordinal
        return self.conversation.messages[message_ordinal]

    def get_message(self, message_ordinal: MessageOrdinal) -> TMessage:
        """Retrieve a message by its ordinal."""
        return self.messages[message_ordinal]

    def clear_matched_terms(self) -> None:
        """Clear all matched terms and property terms."""
        self.matched_terms.clear()
        self.matched_property_terms.clear()


def lookup_knowledge_type(
    semantic_refs: ISemanticRefCollection, ktype: KnowledgeType
) -> list[ScoredSemanticRefOrdinal]:
    return [
        ScoredSemanticRefOrdinal(sr.semantic_ref_ordinal, 1.0)
        for sr in semantic_refs
        if sr.knowledge_type == ktype
    ]


class IQueryOpExpr[T](Protocol):
    """Protocol for query operation expressions that can be evaluated in a context."""

    def eval(self, context: QueryEvalContext) -> T:
        raise NotImplementedError


class QueryOpExpr[T](IQueryOpExpr[T]):
    """Base class for query operation expressions."""


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


# Abstract base class.
class MatchTermsBooleanExpr(QueryOpExpr[SemanticRefAccumulator]):
    """Expression for matching terms in a boolean query.

    Subclasses implement 'OR', 'OR MAX' and 'AND' logic.
    """

    get_scope_expr: "GetScopeExpr | None" = None

    def begin_match(self, context: QueryEvalContext) -> None:
        """Prepare for matching terms in the context by resetting some things."""
        if self.get_scope_expr is not None:
            context.text_ranges_in_scope = self.get_scope_expr.eval(context)
        context.clear_matched_terms()


@dataclass
class MatchTermsOrExpr(MatchTermsBooleanExpr):
    """Expression for matching terms with an OR condition."""

    term_expressions: list[IQueryOpExpr[SemanticRefAccumulator | None]] = field(
        default_factory=list
    )
    get_scope_expr: "GetScopeExpr | None" = None

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


@dataclass
class MatchTermsOrMaxExpr(MatchTermsOrExpr):
    """OR-MAX returns the union if there are no common matches, else the maximum scoring match."""

    term_expressions: list[IQueryOpExpr[SemanticRefAccumulator | None]] = field(
        default_factory=list
    )
    get_scope_expr: "GetScopeExpr | None" = None

    def eval(self, context: QueryEvalContext) -> SemanticRefAccumulator:
        matches = super().eval(context)
        max_hit_count = matches.get_max_hit_count()
        if max_hit_count > 1:
            matches.select_with_hit_count(max_hit_count)
        return matches


@dataclass
class MatchTermsAndExpr(MatchTermsBooleanExpr):
    term_expressions: list[IQueryOpExpr[SemanticRefAccumulator | None]] = field(
        default_factory=list
    )
    get_scope_expr: "GetScopeExpr | None" = None

    def eval(self, context: QueryEvalContext) -> SemanticRefAccumulator:
        self.begin_match(context)
        all_matches: SemanticRefAccumulator | None = None
        for match_expr in self.term_expressions:
            term_matches = match_expr.eval(context)
            if not term_matches:
                if all_matches is not None:
                    all_matches.clear_matches()
                break
            if all_matches is None:
                all_matches = term_matches
            else:
                all_matches.intersect(term_matches)
        if all_matches is not None:
            all_matches.calculate_total_score()
            all_matches.select_with_hit_count(len(self.term_expressions))
        else:
            all_matches = SemanticRefAccumulator()
        return all_matches


class MatchTermExpr(QueryOpExpr[SemanticRefAccumulator | None], ABC):
    """Expression for matching terms in a query.

    Subclasses need to define accumulate_matches(), which must add
    matches to its SemanticRefAccumulator argument.
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


@dataclass
class MatchSearchTermExpr(MatchTermExpr):
    search_term: SearchTerm
    score_booster: ScoreBoosterType | None = None

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


@dataclass
class MatchPropertySearchTermExpr(MatchTermExpr):
    property_search_term: PropertySearchTerm

    def accumulate_matches(
        self, context: QueryEvalContext, matches: SemanticRefAccumulator
    ) -> None:
        if isinstance(self.property_search_term.property_name, str):
            self.accumulate_matches_for_property(
                context,
                self.property_search_term.property_name,
                self.property_search_term.property_value,
                matches,
            )
        else:
            self.accumulate_matches_for_facets(
                context,
                self.property_search_term.property_name,
                self.property_search_term.property_value,
                matches,
            )

    def accumulate_matches_for_facets(
        self,
        context: QueryEvalContext,
        property_name: SearchTerm,
        property_value: SearchTerm,
        matches: SemanticRefAccumulator,
    ):
        self.accumulate_matches_for_property(
            context,
            PropertyNames.FacetName.value,
            property_name,
            matches,
        )
        if not is_search_term_wildcard(property_value):
            self.accumulate_matches_for_property(
                context,
                PropertyNames.FacetValue.value,
                property_value,
                matches,
            )

    def accumulate_matches_for_property(
        self,
        context: QueryEvalContext,
        property_name: str,
        property_value: SearchTerm,
        matches: SemanticRefAccumulator,
    ):
        self.accumulate_matches_for_property_value(
            context,
            matches,
            property_name,
            property_value.term,
        )
        if property_value.related_terms:
            for related_property_value in property_value.related_terms:
                self.accumulate_matches_for_property_value(
                    context,
                    matches,
                    property_name,
                    property_value.term,
                    related_property_value,
                )

    def accumulate_matches_for_property_value(
        self,
        context: QueryEvalContext,
        matches: SemanticRefAccumulator,
        property_name: str,
        property_value: Term,
        related_prop_val: Term | None = None,
    ) -> None:
        if related_prop_val is None:
            if not context.matched_property_terms.has(property_name, property_value):
                semantic_refs = self.lookup_property(
                    context,
                    property_name,
                    property_value.text,
                )
                if semantic_refs:
                    matches.add_term_matches(property_value, semantic_refs, True)
                    context.matched_property_terms.add(property_name, property_value)
        else:
            # To prevent over-counting, ensure this related_prop_val was not already used to match terms earlier
            if not context.matched_property_terms.has(property_name, related_prop_val):
                semantic_refs = self.lookup_property(
                    context,
                    property_name,
                    related_prop_val.text,
                )
                if semantic_refs:
                    # This will only consider semantic refs that were not already matched by this expression.
                    # In other words, if a semantic ref already matched due to the term 'novel',
                    # don't also match it because it matched the related term 'book'
                    matches.add_term_matches_if_new(
                        property_value,
                        semantic_refs,
                        False,
                        related_prop_val.weight,
                    )
                    context.matched_property_terms.add(property_name, related_prop_val)

    def lookup_property(
        self,
        context: QueryEvalContext,
        property_name: str,
        property_value: str,
    ) -> list[ScoredSemanticRefOrdinal] | None:
        if context.property_index is not None:
            return lookup_property_in_property_index(
                context.property_index,
                property_name,
                property_value,
                context.semantic_refs,
                context.text_ranges_in_scope,
            )


class MatchTagExpr(MatchSearchTermExpr):
    def __init__(self, tag_term: SearchTerm):
        self.tag_term = tag_term
        super().__init__(tag_term)

    def lookup_term(
        self, context: QueryEvalContext, term: Term
    ) -> list[ScoredSemanticRefOrdinal] | None:
        if self.tag_term.term.text == "*":
            return lookup_knowledge_type(context.semantic_refs, "tag")
        else:
            return lookup_term(
                context.semantic_ref_index,
                term,
                context.semantic_refs,
                context.text_ranges_in_scope,
                "tag",
            )


class MatchTopicExpr(MatchSearchTermExpr):
    def __init__(self, topic: SearchTerm):
        self.topic = topic
        super().__init__(topic)

    def lookup_term(
        self, context: QueryEvalContext, term: Term
    ) -> list[ScoredSemanticRefOrdinal] | None:
        if self.topic.term.text == "*":
            return lookup_knowledge_type(context.semantic_refs, "topic")
        else:
            return lookup_term(
                context.semantic_ref_index,
                term,
                context.semantic_refs,
                context.text_ranges_in_scope,
                "topic",
            )


@dataclass
class GroupByKnowledgeTypeExpr(
    QueryOpExpr[dict[KnowledgeType, SemanticRefAccumulator]]
):
    matches: IQueryOpExpr[SemanticRefAccumulator]

    def eval(
        self, context: QueryEvalContext
    ) -> dict[KnowledgeType, SemanticRefAccumulator]:
        semantic_ref_matches = self.matches.eval(context)
        return semantic_ref_matches.group_matches_by_type(context.semantic_refs)


@dataclass
class SelectTopNKnowledgeGroupExpr(
    QueryOpExpr[dict[KnowledgeType, SemanticRefAccumulator]]
):
    source_expr: IQueryOpExpr[dict[KnowledgeType, SemanticRefAccumulator]]
    max_matches: int | None = None
    min_hit_count: int | None = None

    def eval(
        self, context: QueryEvalContext
    ) -> dict[KnowledgeType, SemanticRefAccumulator]:
        groups_accumulators = self.source_expr.eval(context)
        for accumulator in groups_accumulators.values():
            accumulator.select_top_n_scoring(self.max_matches, self.min_hit_count)
        return groups_accumulators


@dataclass
class GroupSearchResultsExpr(QueryOpExpr[dict[KnowledgeType, SemanticRefSearchResult]]):
    src_expr: IQueryOpExpr[dict[KnowledgeType, SemanticRefAccumulator]]

    def eval(
        self, context: QueryEvalContext
    ) -> dict[KnowledgeType, SemanticRefSearchResult]:
        return to_grouped_search_results(self.src_expr.eval(context))


@dataclass
class WhereSemanticRefExpr(QueryOpExpr[SemanticRefAccumulator]):
    source_expr: IQueryOpExpr[SemanticRefAccumulator]
    predicates: list["IQuerySemanticRefPredicate"]

    def eval(self, context: QueryEvalContext) -> SemanticRefAccumulator:
        accumulator = self.source_expr.eval(context)
        filtered = SemanticRefAccumulator(accumulator.search_term_matches)
        filtered.set_matches(
            accumulator.get_matches(
                lambda match: self._eval_predicates(context, self.predicates, match)
            )
        )
        return filtered

    def _eval_predicates(
        self,
        context: QueryEvalContext,
        predicates: list["IQuerySemanticRefPredicate"],
        match: Match[SemanticRefOrdinal],
    ) -> bool:
        for predicate in predicates:
            semantic_ref = context.get_semantic_ref(match.value)
            if not predicate.eval(context, semantic_ref):
                return False

        return True


class IQuerySemanticRefPredicate(Protocol):
    def eval(self, context: QueryEvalContext, semantic_ref: SemanticRef) -> bool:
        raise NotImplementedError


# TODO: match_predicates
# TODO: knowledge_type_predicate
# TODO: property_match_predicate


# NOTE: GetScopeExpr is moved after TextRangeSelector to avoid circular references.


class IQueryTextRangeSelector(Protocol):
    """Protocol for a selector that can evaluate to a text range."""

    def eval(
        self,
        context: QueryEvalContext,
        semantic_refs: SemanticRefAccumulator | None = None,
    ) -> TextRangeCollection | None:
        """Evaluate the selector and return the text range."""
        raise NotImplementedError("Subclass must implement eval")


class TextRangeSelector(IQueryTextRangeSelector):
    """A selector that evaluates to a pre-computed TextRangeCollection."""

    text_ranges_in_scope: TextRangeCollection

    def __init__(self, ranges_in_scope: list[TextRange]) -> None:
        self.text_ranges_in_scope = TextRangeCollection(ranges_in_scope, True)

    def eval(
        self,
        context: QueryEvalContext,
        semantic_refs: SemanticRefAccumulator | None = None,
    ) -> TextRangeCollection | None:
        return self.text_ranges_in_scope


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


# TODO: SelectInScopeExpr


@dataclass
class TextRangesInDateRangeSelector(IQueryTextRangeSelector):
    date_range_in_scope: DateRange

    def eval(
        self,
        context: QueryEvalContext,
        semantic_refs: SemanticRefAccumulator | None = None,
    ) -> TextRangeCollection | None:
        """Evaluate the selector and return text ranges in the specified date range."""
        text_ranges_in_scope = TextRangeCollection()

        if context.timestamp_index is not None:
            text_ranges = context.timestamp_index.lookup_range(
                self.date_range_in_scope,
            )
            for time_range in text_ranges:
                text_ranges_in_scope.add_range(time_range.range)
        else:
            text_range = get_text_range_for_date_range(
                context.conversation,
                self.date_range_in_scope,
            )
            if text_range is not None:
                text_ranges_in_scope.add_range(text_range)

        return text_ranges_in_scope


# TODO: TextRangesPredicateSelector
# TODO: TextRangesWithTagSelector
# TODO: TextRangesFromSemanticRefsSelector


@dataclass
class TextRangesFromMessagesSelector(IQueryTextRangeSelector):
    source_expr: IQueryOpExpr[MessageAccumulator]

    def eval(
        self,
        context: QueryEvalContext,
        semantic_refs: SemanticRefAccumulator | None = None,
    ) -> TextRangeCollection | None:
        matches = self.source_expr.eval(context)
        ranges_in_scope: list[TextRange] | None = None
        if matches:
            all_ordinals = sorted(matches.get_matched_values())
            ranges_in_scope = text_ranges_from_message_ordinals(all_ordinals)
        return TextRangeCollection(ranges_in_scope)


# TODO: Move to messagelib.py
def text_ranges_from_message_ordinals(
    message_ordinals: list[MessageOrdinal],
) -> list[TextRange]:
    return [text_range_from_message(ordinal) for ordinal in message_ordinals]


# TODO: Move to messagelib.py
def text_range_from_message(message_ordinal: MessageOrdinal) -> TextRange:
    return TextRange(start=TextLocation(message_ordinal))


# TODO: ThreadSelector
# TODO: to_grouped_search_results


def to_grouped_search_results(
    eval_results: dict[KnowledgeType, SemanticRefAccumulator],
) -> dict[KnowledgeType, SemanticRefSearchResult]:
    semantic_ref_matches: dict[KnowledgeType, SemanticRefSearchResult] = {}
    for typ, accumulator in eval_results.items():
        if len(accumulator) > 0:
            semantic_ref_matches[typ] = SemanticRefSearchResult(
                term_matches=accumulator.search_term_matches,
                semantic_ref_matches=accumulator.to_scored_semantic_refs(),
            )
    return semantic_ref_matches


@dataclass
class MessagesFromKnowledgeExpr(QueryOpExpr[MessageAccumulator]):
    src_expr: (
        IQueryOpExpr[dict[KnowledgeType, SemanticRefSearchResult]]
        | dict[KnowledgeType, SemanticRefSearchResult]
    )

    def eval(self, context: QueryEvalContext) -> MessageAccumulator:
        knowledge = (
            self.src_expr
            if isinstance(self.src_expr, dict)
            else self.src_expr.eval(context)
        )
        return message_matches_from_knowledge_matches(context.semantic_refs, knowledge)


# TODO: SelectMessagesInCharBudget


@dataclass
class RankMessagesBySimilarityExpr(QueryOpExpr[MessageAccumulator]):
    src_expr: IQueryOpExpr[MessageAccumulator]
    embedding: NormalizedEmbedding
    max_messages: int | None = None
    threshold_score: float | None = None

    def eval(self, context: QueryEvalContext) -> MessageAccumulator:
        matches = self.src_expr.eval(context)
        if self.max_messages is not None and len(matches) <= self.max_messages:
            return matches

        # Try to use the message embedding index for re-ranking if available.
        message_index = (
            None
            if context.conversation.secondary_indexes is None
            else context.conversation.secondary_indexes.message_index
        )
        if isinstance(message_index, IMessageTextEmbeddingIndex):
            message_ordinals = self._get_message_ordinals_in_index(
                message_index, matches
            )
            if len(message_ordinals) == len(matches):
                matches.clear_matches()
                ranked_messages = message_index.lookup_in_subset_by_embedding(
                    self.embedding,
                    message_ordinals,
                    self.max_messages,
                    self.threshold_score,
                )
                for match in ranked_messages:
                    matches.add(match.message_ordinal, match.score)
                return matches

        if self.max_messages is not None:
            # Can't re rank, so just take the top K from what we already have.
            matches.select_top_n_scoring(self.max_messages)
        return matches

    def _get_message_ordinals_in_index(
        self, message_index, matches: MessageAccumulator
    ):
        message_ordinals: list[MessageOrdinal] = []
        index_size = len(message_index)
        for message_ordinal in matches.get_matched_values():
            if message_ordinal >= index_size:
                break
            message_ordinals.append(message_ordinal)
        return message_ordinals


@dataclass
class GetScoredMessagesExpr(QueryOpExpr[list[ScoredMessageOrdinal]]):
    src_expr: IQueryOpExpr[MessageAccumulator]

    def eval(self, context: QueryEvalContext) -> list[ScoredMessageOrdinal]:
        matches = self.src_expr.eval(context)
        return matches.to_scored_message_ordinals()


@dataclass
class MatchMessagesBooleanExpr(IQueryOpExpr[MessageAccumulator]):
    term_expressions: list[
        IQueryOpExpr[SemanticRefAccumulator | MessageAccumulator | None]
    ]

    def _begin_match(self, context: QueryEvalContext) -> None:
        context.clear_matched_terms()

    def _accumulate_messages(
        self,
        context: QueryEvalContext,
        semantic_ref_matches: SemanticRefAccumulator,
    ) -> MessageAccumulator:
        message_matches = MessageAccumulator()
        for semantic_ref_match in semantic_ref_matches:
            semantic_ref = context.get_semantic_ref(semantic_ref_match.value)
            message_matches.add_messages_for_semantic_ref(
                context.get_semantic_ref(semantic_ref_match.value),
                semantic_ref_match.score,
            )
        return message_matches


@dataclass
class MatchMessagesOrExpr(MatchMessagesBooleanExpr):

    def eval(self, context: QueryEvalContext) -> MessageAccumulator:
        self._begin_match(context)

        all_matches: MessageAccumulator | None = None
        for match_expr in self.term_expressions:
            matches = match_expr.eval(context)
            if not matches:
                continue
            if isinstance(matches, SemanticRefAccumulator):
                message_matches = self._accumulate_messages(context, matches)
            else:
                message_matches = matches
            if all_matches is not None:
                all_matches.add_union(message_matches)
            else:
                all_matches = message_matches
        if all_matches is not None:
            all_matches.calculate_total_score()
        else:
            all_matches = MessageAccumulator()
        return all_matches


@dataclass
class MatchMessagesAndExpr(MatchMessagesBooleanExpr):

    def eval(self, context: QueryEvalContext) -> MessageAccumulator:
        self._begin_match(context)

        all_matches: MessageAccumulator | None = None
        all_done = False
        for match_expr in self.term_expressions:
            matches = match_expr.eval(context)
            if not matches:
                # If any expr does not match, the AND fails.
                break
            if isinstance(matches, SemanticRefAccumulator):
                message_matches = self._accumulate_messages(context, matches)
            else:
                message_matches = matches
            if all_matches is None:
                all_matches = message_matches
            else:
                # Intersect the message matches
                all_matches.intersect(message_matches)
                if not all_matches:
                    # If the intersection is empty, we can stop early.
                    break
        else:
            # If we did not break, all terms matched.
            all_done = True

        if all_matches is not None:
            if all_done:
                all_matches.calculate_total_score()
                all_matches.select_with_hit_count(len(self.term_expressions))
            else:
                all_matches.clear_matches()
        else:
            all_matches = MessageAccumulator()
        return all_matches


@dataclass
class MatchMessagesOrMaxExpr(MatchMessagesOrExpr):

    def eval(self, context: QueryEvalContext) -> MessageAccumulator:
        matches = super().eval(context)
        max_hit_count = matches.get_max_hit_count()
        if max_hit_count > 1:
            matches.select_with_hit_count(max_hit_count)
        return matches


# TODO: class MatchMessagesBySimilarityExpr(QueryOpExpr[list[ScoredMessageOrdinal]]):


class NoOpExpr[T](QueryOpExpr[T]):
    def __init__(self, src_expr: IQueryOpExpr[T]) -> None:
        self.src_expr = src_expr
        super().__init__()

    def eval(self, context: QueryEvalContext) -> T:
        return self.src_expr.eval(context)


def message_matches_from_knowledge_matches(
    semantic_refs: ISemanticRefCollection,
    knowledge_matches: dict[KnowledgeType, SemanticRefSearchResult],
    intersect_across_knowledge_types: bool = True,
) -> MessageAccumulator:
    message_matches = MessageAccumulator()
    knowledge_type_hit_count = 0  # How many types of knowledge matched?
    for knowledge_type, matches_by_type in knowledge_matches.items():
        if matches_by_type and matches_by_type.semantic_ref_matches:
            knowledge_type_hit_count += 1
            for match in matches_by_type.semantic_ref_matches:
                message_matches.add_messages_for_semantic_ref(
                    semantic_refs[match.semantic_ref_ordinal],
                    match.score,
                )
    if intersect_across_knowledge_types and knowledge_type_hit_count > 0:
        # Intersect the sets of messages that matched each knowledge type
        relevant_messages = message_matches.get_with_hit_count(knowledge_type_hit_count)
        if relevant_messages:
            message_matches = MessageAccumulator(relevant_messages)
    message_matches.smooth_scores()
    return message_matches
