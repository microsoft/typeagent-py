# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Callable, TypeGuard, cast
from xmlrpc.client import boolean

from .collections import MessageAccumulator, SemanticRefAccumulator
from .interfaces import (
    IConversation,
    IConversationSecondaryIndexes,
    KnowledgeType,
    PropertySearchTerm,
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
    SearchSelectExpr,
    SearchTerm,
    SearchTermGroup,
    SemanticRef,
    SemanticRefSearchResult,
    Term,
    WhenFilter,
)
from .kplib import ConcreteEntity
from .messageindex import IMessageTextEmbeddingIndex
from .query import (
    BooleanOp,
    CompiledTermGroup,
    GetScopeExpr,
    GetScoredMessagesExpr,
    GroupByKnowledgeTypeExpr,
    GroupSearchResultsExpr,
    IQueryOpExpr,
    IQuerySemanticRefPredicate,
    IQueryTextRangeSelector,
    MatchMessagesAndExpr,
    MatchMessagesBooleanExpr,
    MatchMessagesOrExpr,
    MatchMessagesOrMaxExpr,
    MatchPropertySearchTermExpr,
    MatchSearchTermExpr,
    MatchTagExpr,
    MatchTermsAndExpr,
    MatchTermsBooleanExpr,
    MatchTermsOrExpr,
    MatchTermsOrMaxExpr,
    MatchTopicExpr,
    MessagesFromKnowledgeExpr,
    NoOpExpr,
    QueryEvalContext,
    RankMessagesBySimilarityExpr,
    SelectTopNExpr,
    SelectTopNKnowledgeGroupExpr,
    TextRangeSelector,
    TextRangesFromMessagesSelector,
    TextRangesInDateRangeSelector,
    WhereSemanticRefExpr,
    is_conversation_searchable,
    match_entity_name_or_type,
)
from .reltermsindex import resolve_related_terms
from .secindex import ConversationSecondaryIndexes


@dataclass
class SearchQueryExpr:
    select_expressions: list[SearchSelectExpr]
    raw_query: str | None = None


@dataclass
class SearchOptions:
    max_knowledge_matches: int | None = None
    exact_match: bool = False
    max_message_matches: int | None = None
    # The maximum # of total message characters to select
    # The query processor will ensure that the cumulative character count of message matches
    # is less than this number
    max_chars_in_budget: int | None = None
    threshold_score: float | None = None

    def __repr__(self):
        parts = []
        for key in dir(self):
            if not key.startswith("_"):
                value = getattr(self, key)
                if value is not None:
                    parts.append(f"{key}={value!r}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


@dataclass
class ConversationSearchResult:
    message_matches: list[ScoredMessageOrdinal]
    knowledge_matches: dict[KnowledgeType, SemanticRefSearchResult]
    raw_query_text: str | None = None


async def search_conversation(
    conversation: IConversation,
    search_term_group: SearchTermGroup,
    when_filter: WhenFilter | None = None,
    options: SearchOptions | None = None,
    raw_search_query: str | None = None,
) -> ConversationSearchResult | None:
    options = options or SearchOptions()
    knowledge_matches = await search_conversation_knowledge(
        conversation, search_term_group, when_filter, options
    )
    if knowledge_matches is None:
        return None
    # return ConversationSearchResult([], knowledge_matches, raw_search_query)
    # Future: Combine knowledge and message queries into single query tree.
    compiler = QueryCompiler(conversation, conversation.secondary_indexes)
    message_query = await compiler.compile_message_query(
        knowledge_matches, options, raw_search_query
    )
    message_matches: list[ScoredMessageOrdinal] = run_query(
        conversation, options, message_query
    )
    return ConversationSearchResult(
        message_matches, knowledge_matches, raw_search_query
    )


async def search_conversation_knowledge(
    conversation: IConversation,
    search_term_group: SearchTermGroup,
    when_filter: WhenFilter | None = None,
    options: SearchOptions | None = None,
) -> dict[KnowledgeType, SemanticRefSearchResult] | None:
    """Search a conversation for knowledge that matches the given search terms and filter."""
    options = options or SearchOptions()
    if not is_conversation_searchable(conversation):
        return None
    compiler = QueryCompiler(
        conversation, conversation.secondary_indexes or ConversationSecondaryIndexes()
    )
    knowledge_query = await compiler.compile_knowledge_query(
        search_term_group, when_filter, options
    )
    return run_query(conversation, options, knowledge_query)


# TODO: search_conversation_by_text_similarity


async def run_search_query(
    conversation: IConversation,
    query: SearchQueryExpr,
    options: SearchOptions | None = None,
    original_query_text: str | None = None,
) -> list[ConversationSearchResult]:
    options = options or SearchOptions()
    results: list[ConversationSearchResult] = []
    for expr in query.select_expressions:
        search_results = await search_conversation(
            conversation,
            expr.search_term_group,
            expr.when,
            options,
            original_query_text or query.raw_query,
        )
        if search_results is not None:
            results.append(search_results)
    return results


# TODO: run_search_query_by_text_similarity


def run_query[T](
    conversation: IConversation,
    options: SearchOptions | None,
    query: IQueryOpExpr[T],
) -> T:
    secondary_indexes = conversation.secondary_indexes
    if secondary_indexes is None:
        secondary_indexes = ConversationSecondaryIndexes()
    return query.eval(
        QueryEvalContext(
            conversation,
            secondary_indexes.property_to_semantic_ref_index,
            secondary_indexes.timestamp_index,
        )
    )


# NOTE: QueryCompiler instances are stateful, and not thread-safe.
#       Create a new one for each query.
class QueryCompiler:
    def __init__(
        self,
        conversation: IConversation,
        secondary_indexes: IConversationSecondaryIndexes | None,
        entity_term_match_weight: float = 100.0,
        default_term_match_weight: float = 10.0,
        related_is_exact_threshold: float = 0.95,
    ):
        self.conversation = conversation
        self.secondary_indexes = secondary_indexes
        self.entity_term_match_weight = entity_term_match_weight
        self.default_term_match_weight = default_term_match_weight
        self.related_is_exact_threshold = related_is_exact_threshold
        # All SearchTerms used which compiling the 'select' portion of the query.
        self.all_search_terms: list[CompiledTermGroup] = []
        # All search terms used while compiling predicates in the query.
        self.all_predicate_search_terms: list[CompiledTermGroup] = []
        self.all_scope_search_terms: list[CompiledTermGroup] = []

    # NOTE: Everything is async because we sometimes use embeddings.

    async def compile_knowledge_query(
        self,
        terms: SearchTermGroup,
        filter: WhenFilter | None = None,
        options: SearchOptions | None = None,
    ) -> GroupSearchResultsExpr:
        query = await self.compile_query(terms, filter, options)

        exact_match = (
            options.exact_match
            if options is not None and options.exact_match is not None
            else False
        )
        if not exact_match:
            await self.resolve_related_terms(self.all_search_terms, True)
            await self.resolve_related_terms(self.all_predicate_search_terms, False)
            await self.resolve_related_terms(self.all_scope_search_terms, False)

        return GroupSearchResultsExpr(query)

    async def compile_message_query(
        self,
        knowledge: (
            IQueryOpExpr[dict[KnowledgeType, SemanticRefSearchResult]]
            | dict[KnowledgeType, SemanticRefSearchResult]
        ),
        options: SearchOptions | None = None,
        raw_query_text: str | None = None,
    ) -> GetScoredMessagesExpr:
        query: IQueryOpExpr = MessagesFromKnowledgeExpr(knowledge)
        if options is not None:
            query = await self.compile_message_re_rank(
                query,
                raw_query_text,
                options,
            )
            if options.max_chars_in_budget and options.max_chars_in_budget > 0:
                query = SelectMessagesInCharBudget(
                    query,
                    options.max_chars_in_budget,
                )

        return GetScoredMessagesExpr(query)

    # TODO: compile_message_similarity_query

    async def compile_query(
        self,
        search_term_group: SearchTermGroup,
        filter: WhenFilter | None = None,
        options: SearchOptions | None = None,
    ) -> IQueryOpExpr[dict[KnowledgeType, SemanticRefAccumulator]]:
        select_expr = self.compile_select(
            search_term_group,
            await self.compile_scope(search_term_group, filter),
            options,
        )
        # Constrain the select with scopes and 'where'.
        if filter:
            select_expr = WhereSemanticRefExpr(
                select_expr,
                self.compile_where(filter),
            )
        # And lastly, select 'TopN' and group knowledge by type.
        tmp = GroupByKnowledgeTypeExpr(select_expr)
        return SelectTopNKnowledgeGroupExpr(
            tmp,
            (
                options.max_knowledge_matches
                if options
                and hasattr(options, "max_knowledge_matches")
                and options.max_knowledge_matches
                else None
            ),
        )

    def compile_select(
        self,
        term_group: SearchTermGroup,
        scope_expr: GetScopeExpr | None = None,
        options: SearchOptions | None = None,
    ) -> IQueryOpExpr[SemanticRefAccumulator]:
        search_terms_used, select_expr = self.compile_search_group_terms(
            term_group, scope_expr
        )
        self.all_search_terms.extend(search_terms_used)
        return select_expr

    def compile_search_group_terms(
        self,
        search_group: SearchTermGroup,
        scope_expr: GetScopeExpr | None = None,
    ) -> tuple[list[CompiledTermGroup], IQueryOpExpr[SemanticRefAccumulator]]:
        return self.compile_search_group(
            search_group,
            create_match_terms_boolean_expr,
            scope_expr,
        )

    def compile_search_group_messages(
        self,
        search_group: SearchTermGroup,
    ) -> tuple[list[CompiledTermGroup], IQueryOpExpr[MessageAccumulator]]:
        return self.compile_search_group(
            search_group, create_match_messages_boolean_expr
        )

    def compile_search_group(
        self,
        search_group: SearchTermGroup,
        create_op: Callable[
            [list[IQueryOpExpr], BooleanOp, GetScopeExpr | None],
            IQueryOpExpr[SemanticRefAccumulator | MessageAccumulator],
        ],
        scope_expr: GetScopeExpr | None = None,
    ) -> tuple[list[CompiledTermGroup], IQueryOpExpr]:
        t0_terms: list[SearchTerm] = []
        compiled_terms: list[CompiledTermGroup] = [
            CompiledTermGroup(boolean_op="and", terms=t0_terms)
        ]
        term_expressions: list[IQueryOpExpr[SemanticRefAccumulator | None]] = []
        for term in search_group.terms:
            if isinstance(term, PropertySearchTerm):
                term_expressions.append(self.compile_property_term(term))
                if not isinstance(term.property_name, str):
                    t0_terms.append(term.property_name)
                t0_terms.append(term.property_value)
            elif isinstance(term, SearchTermGroup):
                nested_terms, group_expr = self.compile_search_group(term, create_op)
                compiled_terms.extend(nested_terms)
                term_expressions.append(group_expr)
            else:
                assert isinstance(
                    term, SearchTerm
                ), f"Unexpected term type: {type(term)}"
                term_expressions.append(self.compile_search_term(term))
                t0_terms.append(term)
        bool_expr = create_op(term_expressions, search_group.boolean_op, scope_expr)
        return (compiled_terms, bool_expr)

    def compile_search_term(
        self,
        term: SearchTerm,
    ) -> IQueryOpExpr[SemanticRefAccumulator | None]:
        boost_weight = self.entity_term_match_weight / self.default_term_match_weight
        return MatchSearchTermExpr(
            term,
            lambda term, sr, scored: self.boost_entities(
                term, sr, scored, boost_weight
            ),
        )

    def compile_property_term(
        self,
        term: PropertySearchTerm,
    ) -> IQueryOpExpr[SemanticRefAccumulator | None]:
        match term.property_name:
            case "tag":
                return MatchTagExpr(term.property_value)
            case "topic":
                return MatchTopicExpr(term.property_value)
            case _:
                if term.property_name in ("name", "type"):
                    tpvt = term.property_value.term
                    if tpvt.weight is None:
                        tpvt.weight = self.entity_term_match_weight
                return MatchPropertySearchTermExpr(term)

    async def compile_scope(
        self,
        term_group: SearchTermGroup | None = None,
        filter: WhenFilter | None = None,
    ) -> GetScopeExpr | None:
        scope_selectors: list[IQueryTextRangeSelector] = []

        # First, use any provided date ranges to select scope
        if filter and filter.date_range:
            scope_selectors.append(TextRangesInDateRangeSelector(filter.date_range))

        # Apply 'OUTER' scope
        # If specific scoping terms were provided
        if filter and filter.scope_defining_terms is not None:
            self.add_terms_scope_selector(filter.scope_defining_terms, scope_selectors)
        elif term_group is not None:
            # Treat any actions as inherently scope selecting
            action_terms_group = self.get_action_terms_from_search_group(term_group)
            if action_terms_group is not None:
                self.add_terms_scope_selector(action_terms_group, scope_selectors)

        # Include any ranges directly provided by the caller
        if filter and filter.text_ranges_in_scope:
            scope_selectors.append(TextRangeSelector(filter.text_ranges_in_scope))

        # Tags...
        if filter and filter.tags:
            self.add_terms_scope_selector(
                create_tag_search_term_group(filter.tags), scope_selectors
            )

        # If a thread index is available...
        threads = None
        if self.secondary_indexes:
            threads = self.secondary_indexes.threads
        if filter and filter.thread_description and threads:
            threads_in_scope = await threads.lookup_thread(filter.thread_description)
            if threads_in_scope:
                scope_selectors.append(
                    ThreadSelector(
                        [threads.threads[t.thread_ordinal] for t in threads_in_scope]
                    )
                )

        return GetScopeExpr(scope_selectors) if scope_selectors else None

    def add_terms_scope_selector(
        self,
        term_group: SearchTermGroup,
        scope_selectors: list[IQueryTextRangeSelector],
    ) -> None:
        if term_group.terms:
            search_terms_used, select_expr = self.compile_search_group_messages(
                term_group
            )
            scope_selectors.append(TextRangesFromMessagesSelector(select_expr))
            self.all_scope_search_terms.extend(search_terms_used)

    def compile_where(self, filter: WhenFilter) -> list[IQuerySemanticRefPredicate]:
        predicates: list[IQuerySemanticRefPredicate] = []
        if filter.knowledge_type:
            predicates.append(KnowledgeTypePredicate(filter.knowledge_type))
        return predicates

    async def compile_message_re_rank(
        self,
        src_expr: IQueryOpExpr,
        raw_query_text: str | None = None,
        options: SearchOptions | None = None,
    ) -> IQueryOpExpr:
        message_index = (
            self.conversation.secondary_indexes.message_index
            if self.conversation.secondary_indexes
            else None
        )
        if (
            raw_query_text is not None
            and isinstance(message_index, IMessageTextEmbeddingIndex)
            and len(message_index) > 0
        ):
            embedding = await message_index.generate_embedding(raw_query_text)
            return RankMessagesBySimilarityExpr(
                src_expr,
                embedding,
                options.max_message_matches if options else None,
                options.threshold_score if options else None,
            )
        elif (
            options
            and options.max_message_matches is not None
            and options.max_message_matches > 0
        ):
            return SelectTopNExpr(src_expr, options.max_message_matches)
        else:
            return NoOpExpr(src_expr)

    # TODO: compile_message_similarity

    def get_action_terms_from_search_group(
        self,
        search_group: SearchTermGroup,
    ) -> SearchTermGroup | None:
        action_group: SearchTermGroup | None = None
        for term in search_group.terms:
            if isinstance(term, PropertySearchTerm) and is_action_property_term(term):
                action_group = action_group or SearchTermGroup(boolean_op="and")
                action_group.terms.append(term)
        return action_group

    async def resolve_related_terms(
        self,
        compiled_terms: list[CompiledTermGroup],
        dedupe: bool,
        filter: WhenFilter | None = None,
    ) -> None:
        if not compiled_terms:
            return
        for ct in compiled_terms:
            self.validate_and_prepare_search_terms(ct.terms)
        if (
            self.secondary_indexes is not None
            and self.secondary_indexes.term_to_related_terms_index is not None
        ):
            await resolve_related_terms(
                self.secondary_indexes.term_to_related_terms_index,
                compiled_terms,
                dedupe,
            )
            # Ensure that the resolved terms are valid etc.
            for ct in compiled_terms:
                self.validate_and_prepare_search_terms(ct.terms)

    def validate_and_prepare_search_terms(self, terms: list[SearchTerm]) -> None:
        for term in terms:
            self.validate_and_prepare_search_term(term)

    def validate_and_prepare_search_term(self, search_term: SearchTerm) -> bool:
        if not self.validate_and_prepare_term(search_term.term):
            return False
        # Matching the term - exact match - counts for more than matching related terms
        # Therefore, we boost any matches where the term matches directly...
        if search_term.term.weight is None:
            search_term.term.weight = self.default_term_match_weight
        if search_term.related_terms is not None:
            for related_term in search_term.related_terms:
                if not self.validate_and_prepare_term(related_term):
                    return False
                # If related term is *really* similar to the main term, score it the same
                if (
                    related_term.weight is not None
                    and related_term.weight >= self.related_is_exact_threshold
                ):
                    related_term.weight = self.default_term_match_weight
        return True

    # Currently, just changes the case of a term
    #  But here, we may do other things like:
    # - Check for noise terms
    # - Do additional rewriting
    # - Additional checks that *reject* certain search terms
    # Return false if the term should be rejected
    def validate_and_prepare_term(self, term: Term | None) -> bool:
        if term:
            term.text = term.text.lower()
        return True

    def boost_entities(
        self,
        search_term: SearchTerm,
        sr: SemanticRef,
        scored_ref: ScoredSemanticRefOrdinal,
        boost_weight: float,
    ) -> ScoredSemanticRefOrdinal:
        if sr.knowledge_type == "entity" and match_entity_name_or_type(
            search_term, cast(ConcreteEntity, sr.knowledge)
        ):
            return ScoredSemanticRefOrdinal(
                scored_ref.semantic_ref_ordinal,
                scored_ref.score * boost_weight,
            )
        else:
            return scored_ref


# TODO: Move to compilelib.py
def create_match_terms_boolean_expr(
    term_expressions: list[IQueryOpExpr[SemanticRefAccumulator | None]],
    boolean_op: BooleanOp,
    scope_expr: GetScopeExpr | None = None,
) -> MatchTermsBooleanExpr:
    match boolean_op:
        case "and":
            return MatchTermsAndExpr(term_expressions, scope_expr)
        case "or":
            return MatchTermsOrExpr(term_expressions, scope_expr)
        case "or_max":
            return MatchTermsOrMaxExpr(term_expressions, scope_expr)
        case _:
            raise ValueError(f"Unknown boolean op: {boolean_op}")


# TODO: Move to compilelib.py
def create_match_messages_boolean_expr(
    term_expressions: list[
        IQueryOpExpr[SemanticRefAccumulator | MessageAccumulator | None]
    ],
    boolean_op: BooleanOp,
    scope_expr: GetScopeExpr | None = None,
) -> MatchMessagesBooleanExpr:
    match boolean_op:
        case "and":
            return MatchMessagesAndExpr(term_expressions)
        case "or":
            return MatchMessagesOrExpr(term_expressions)
        case "or_max":
            return MatchMessagesOrMaxExpr(term_expressions)
        case _:
            raise ValueError(f"Unknown boolean op: {boolean_op}")


# TODO: Move to compilelib.py
# TODO: Just call isinstance!
def is_property_term(term: SearchTerm) -> TypeGuard[PropertySearchTerm]:
    return isinstance(term, PropertySearchTerm)


def is_action_property_term(term: PropertySearchTerm) -> bool:
    return term.property_name in ("subject", "verb", "object", "indirectObject")
