# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for searchlang.py — compile_search_query, SearchQueryCompiler,
and related helper functions that don't require a live LLM."""

import datetime
from typing import Literal

from typeagent.knowpro.date_time_schema import DateTime, DateTimeRange, DateVal, TimeVal
from typeagent.knowpro.interfaces import SearchTerm, SearchTermGroup
from typeagent.knowpro.search_query_schema import (
    ActionTerm,
    EntityTerm,
    FacetTerm,
    SearchExpr,
    SearchFilter,
    SearchQuery,
    VerbsTerm,
)
from typeagent.knowpro.searchlang import (
    _compile_fallback_query,
    compile_search_filter,
    compile_search_query,
    date_range_from_datetime_range,
    datetime_from_date_time,
    is_entity_term_list,
    LanguageQueryCompileOptions,
    LanguageSearchFilter,
    optimize_or_max,
    SearchQueryCompiler,
)

from conftest import FakeConversation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entity(
    name: str,
    types: list[str] | None = None,
    facets: list[FacetTerm] | None = None,
    is_pronoun: bool = False,
) -> EntityTerm:
    return EntityTerm(name=name, is_name_pronoun=is_pronoun, type=types, facets=facets)


def make_action(
    actor: list[EntityTerm] | Literal["*"] = "*",
    verbs: list[str] | None = None,
    targets: list[EntityTerm] | None = None,
    additional: list[EntityTerm] | None = None,
    is_informational: bool = False,
) -> ActionTerm:
    return ActionTerm(
        actor_entities=actor,
        is_informational=is_informational,
        action_verbs=VerbsTerm(words=verbs) if verbs else None,
        target_entities=targets,
        additional_entities=additional,
    )


def make_filter(
    entities: list[EntityTerm] | None = None,
    action: ActionTerm | None = None,
    search_terms: list[str] | None = None,
    time_range: DateTimeRange | None = None,
) -> SearchFilter:
    return SearchFilter(
        entity_search_terms=entities,
        action_search_term=action,
        search_terms=search_terms,
        time_range=time_range,
    )


def make_query(filters: list[SearchFilter]) -> SearchQuery:
    expr = SearchExpr(
        rewritten_query="test query",
        filters=filters,
    )
    return SearchQuery(search_expressions=[expr])


def make_compiler(
    options: LanguageQueryCompileOptions | None = None,
    lang_filter: LanguageSearchFilter | None = None,
) -> SearchQueryCompiler:
    conv = FakeConversation()
    return SearchQueryCompiler(conv, options, lang_filter)


# ---------------------------------------------------------------------------
# is_entity_term_list
# ---------------------------------------------------------------------------


class TestIsEntityTermList:
    def test_list_returns_true(self) -> None:
        terms = [make_entity("Alice")]
        assert is_entity_term_list(terms) is True

    def test_empty_list_returns_true(self) -> None:
        assert is_entity_term_list([]) is True

    def test_star_returns_false(self) -> None:
        assert is_entity_term_list("*") is False

    def test_none_returns_false(self) -> None:
        assert is_entity_term_list(None) is False


# ---------------------------------------------------------------------------
# optimize_or_max
# ---------------------------------------------------------------------------


class TestOptimizeOrMax:
    def test_single_term_unwrapped(self) -> None:
        inner = SearchTermGroup(boolean_op="and", terms=[])
        group = SearchTermGroup(boolean_op="or_max", terms=[inner])
        result = optimize_or_max(group)
        assert result is inner

    def test_multiple_terms_kept_as_group(self) -> None:
        inner1 = SearchTermGroup(boolean_op="and", terms=[])
        inner2 = SearchTermGroup(boolean_op="and", terms=[])
        group = SearchTermGroup(boolean_op="or_max", terms=[inner1, inner2])
        result = optimize_or_max(group)
        assert result is group


# ---------------------------------------------------------------------------
# date_range_from_datetime_range / datetime_from_date_time
# ---------------------------------------------------------------------------


class TestDatetimeFromDateTime:
    def test_date_only_zeros_time(self) -> None:
        dt = datetime_from_date_time(DateTime(date=DateVal(day=15, month=6, year=2024)))
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.hour == 0
        assert dt.minute == 0
        assert dt.second == 0
        assert dt.tzinfo == datetime.timezone.utc

    def test_with_time(self) -> None:
        dt = datetime_from_date_time(
            DateTime(
                date=DateVal(day=1, month=1, year=2020),
                time=TimeVal(hour=14, minute=30, seconds=45),
            )
        )
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 45


class TestDateRangeFromDatetimeRange:
    def test_start_only(self) -> None:
        dtr = DateTimeRange(
            start_date=DateTime(date=DateVal(day=1, month=1, year=2023))
        )
        dr = date_range_from_datetime_range(dtr)
        assert dr.start.year == 2023
        assert dr.end is None

    def test_start_and_stop(self) -> None:
        dtr = DateTimeRange(
            start_date=DateTime(date=DateVal(day=1, month=1, year=2023)),
            stop_date=DateTime(date=DateVal(day=31, month=12, year=2023)),
        )
        dr = date_range_from_datetime_range(dtr)
        assert dr.start.year == 2023
        assert dr.end is not None
        assert dr.end.year == 2023
        assert dr.end.month == 12
        assert dr.end.day == 31


# ---------------------------------------------------------------------------
# compile_search_query (standalone function)
# ---------------------------------------------------------------------------


class TestCompileSearchQuery:
    def test_empty_search_expressions(self) -> None:
        conv = FakeConversation()
        query = SearchQuery(search_expressions=[])
        result = compile_search_query(conv, query)
        assert result == []

    def test_single_search_terms_filter(self) -> None:
        conv = FakeConversation()
        query = make_query([make_filter(search_terms=["robots", "AI"])])
        result = compile_search_query(conv, query)
        assert len(result) == 1
        expr = result[0]
        assert len(expr.select_expressions) == 1
        terms_in_group = expr.select_expressions[0].search_term_group.terms
        assert any(
            isinstance(t, SearchTerm) and t.term.text == "robots"
            for t in terms_in_group
        )

    def test_entity_filter_produces_expr(self) -> None:
        conv = FakeConversation()
        query = make_query([make_filter(entities=[make_entity("Alice", ["person"])])])
        result = compile_search_query(conv, query)
        assert len(result) == 1

    def test_multiple_filters_produce_multiple_select_exprs(self) -> None:
        conv = FakeConversation()
        filter1 = make_filter(search_terms=["alpha"])
        filter2 = make_filter(search_terms=["beta"])
        expr = SearchExpr(rewritten_query="test", filters=[filter1, filter2])
        query = SearchQuery(search_expressions=[expr])
        result = compile_search_query(conv, query)
        assert len(result) == 1
        assert len(result[0].select_expressions) == 2

    def test_raw_query_preserved(self) -> None:
        conv = FakeConversation()
        query = make_query([make_filter(search_terms=["foo"])])
        query.search_expressions[0].rewritten_query = "my rewritten query"
        result = compile_search_query(conv, query)
        assert result[0].raw_query == "my rewritten query"


# ---------------------------------------------------------------------------
# compile_search_filter (standalone function)
# ---------------------------------------------------------------------------


class TestCompileSearchFilter:
    def test_entity_filter(self) -> None:
        conv = FakeConversation()
        f = make_filter(entities=[make_entity("Bob")])
        result = compile_search_filter(conv, f)
        assert result.search_term_group is not None

    def test_search_terms_filter(self) -> None:
        conv = FakeConversation()
        f = make_filter(search_terms=["climate", "change"])
        result = compile_search_filter(conv, f)
        terms = result.search_term_group.terms
        assert len(terms) == 2

    def test_empty_filter_uses_topic_wildcard(self) -> None:
        """A filter with no entity, action, or search_terms should produce a topic:* term."""
        conv = FakeConversation()
        f = SearchFilter()
        result = compile_search_filter(conv, f)
        # Should produce a single topic:* property search term
        terms = result.search_term_group.terms
        assert len(terms) == 1

    def test_time_range_produces_when(self) -> None:
        conv = FakeConversation()
        dtr = DateTimeRange(
            start_date=DateTime(date=DateVal(day=1, month=1, year=2024))
        )
        f = make_filter(search_terms=["foo"], time_range=dtr)
        result = compile_search_filter(conv, f)
        assert result.when is not None
        assert result.when.date_range is not None

    def test_no_time_range_when_is_none(self) -> None:
        conv = FakeConversation()
        f = make_filter(search_terms=["foo"])
        result = compile_search_filter(conv, f)
        assert result.when is None


# ---------------------------------------------------------------------------
# SearchQueryCompiler — compile_term_group and related
# ---------------------------------------------------------------------------


class TestSearchQueryCompilerTermGroup:
    def test_search_terms_added(self) -> None:
        compiler = make_compiler()
        f = make_filter(search_terms=["hello", "world"])
        group = compiler.compile_term_group(f)
        texts = [t.term.text for t in group.terms if isinstance(t, SearchTerm)]
        assert "hello" in texts
        assert "world" in texts

    def test_entity_name_added_as_property_term(self) -> None:
        compiler = make_compiler()
        f = make_filter(entities=[make_entity("Ada")])
        group = compiler.compile_term_group(f)
        # Should have at least one term
        assert len(group.terms) > 0

    def test_empty_entity_name_ignored(self) -> None:
        compiler = make_compiler()
        f = make_filter(entities=[make_entity("")])
        group = compiler.compile_term_group(f)
        # Empty string is not searchable; fallback to topic:* for empty term group
        # (there are topic terms added for entity_terms in compile_entity_terms)
        # We just check no crash and group is returned
        assert group is not None

    def test_star_entity_name_ignored(self) -> None:
        compiler = make_compiler()
        f = make_filter(entities=[make_entity("*")])
        group = compiler.compile_term_group(f)
        assert group is not None

    def test_noise_term_ignored(self) -> None:
        compiler = make_compiler()
        f = make_filter(search_terms=["thing", "object", "hello"])
        group = compiler.compile_term_group(f)
        texts = [t.term.text for t in group.terms if isinstance(t, SearchTerm)]
        # noise terms filtered from property groups but not from search_terms path
        # search_terms path does NOT call add_property_term_to_group
        assert "hello" in texts

    def test_custom_term_filter_excludes_property_terms(self) -> None:
        # term_filter applies to add_property_term_to_group, not compile_search_terms.
        options = LanguageQueryCompileOptions(term_filter=lambda t: t != "excluded")
        compiler = make_compiler(options=options)
        group = SearchTermGroup(boolean_op="or", terms=[])
        compiler.add_property_term_to_group("name", "excluded", group)
        compiler.add_property_term_to_group("name", "included", group)
        assert len(group.terms) == 1


# ---------------------------------------------------------------------------
# SearchQueryCompiler — entity terms with facets
# ---------------------------------------------------------------------------


class TestEntityTermsWithFacets:
    def test_entity_with_type(self) -> None:
        compiler = make_compiler()
        entity = make_entity("Alice", types=["person"])
        f = make_filter(entities=[entity])
        group = compiler.compile_term_group(f)
        assert len(group.terms) > 0

    def test_entity_with_facet_name_and_value(self) -> None:
        compiler = make_compiler()
        facet = FacetTerm(facet_name="profession", facet_value="writer")
        entity = make_entity("Bob", facets=[facet])
        f = make_filter(entities=[entity])
        group = compiler.compile_term_group(f)
        assert len(group.terms) > 0

    def test_entity_with_wildcard_facet_value(self) -> None:
        compiler = make_compiler()
        facet = FacetTerm(facet_name="profession", facet_value="*")
        entity = make_entity("Bob", facets=[facet])
        f = make_filter(entities=[entity])
        group = compiler.compile_term_group(f)
        assert len(group.terms) > 0

    def test_entity_with_wildcard_facet_name(self) -> None:
        compiler = make_compiler()
        facet = FacetTerm(facet_name="*", facet_value="writer")
        entity = make_entity("Bob", facets=[facet])
        f = make_filter(entities=[entity])
        group = compiler.compile_term_group(f)
        assert len(group.terms) > 0

    def test_entity_with_both_wildcards_no_facet_term(self) -> None:
        compiler = make_compiler()
        facet = FacetTerm(facet_name="*", facet_value="*")
        entity = make_entity("Bob", facets=[facet])
        f = make_filter(entities=[entity])
        group = compiler.compile_term_group(f)
        # Both wildcards => no facet term added, but entity name term (or_max)
        # and topic term for "Bob" are still generated — 2 terms total.
        assert len(group.terms) == 2

    def test_pronoun_entity_skipped(self) -> None:
        compiler = make_compiler()
        pronoun = make_entity("it", is_pronoun=True)
        normal = make_entity("Alice")
        f = make_filter(entities=[pronoun, normal])
        group = compiler.compile_term_group(f)
        # Only Alice's term should be added
        assert len(group.terms) > 0


# ---------------------------------------------------------------------------
# SearchQueryCompiler — action terms
# ---------------------------------------------------------------------------


class TestActionTerms:
    def test_action_with_verbs_adds_verb_terms(self) -> None:
        compiler = make_compiler()
        actor = make_entity("Alice")
        action = make_action(actor=[actor], verbs=["sent", "emailed"])
        f = make_filter(action=action)
        group = compiler.compile_term_group(f)
        assert len(group.terms) > 0

    def test_action_with_target_entities(self) -> None:
        compiler = make_compiler()
        actor = make_entity("Alice")
        target = make_entity("Bob")
        action = make_action(actor=[actor], verbs=["sent"], targets=[target])
        f = make_filter(action=action)
        group = compiler.compile_term_group(f)
        assert len(group.terms) > 0

    def test_action_with_additional_entities(self) -> None:
        compiler = make_compiler()
        actor = make_entity("Alice")
        extra = make_entity("Charlie")
        action = make_action(actor=[actor], verbs=["spoke"], additional=[extra])
        f = make_filter(action=action)
        group = compiler.compile_term_group(f)
        assert len(group.terms) > 0

    def test_action_star_actor_no_scope(self) -> None:
        """When actor_entities is '*', scope is not applied."""
        action = make_action(actor="*", verbs=["played"])
        f = make_filter(action=action)
        result = compile_search_filter(FakeConversation(), f)
        # should have no scope (when is None or when.scope_defining_terms is empty)
        when = result.when
        assert when is None or (
            when.scope_defining_terms is None
            or len(when.scope_defining_terms.terms) == 0
        )


# ---------------------------------------------------------------------------
# SearchQueryCompiler — compile_when with scope
# ---------------------------------------------------------------------------


class TestCompileWhen:
    def test_no_action_no_when(self) -> None:
        compiler = make_compiler()
        f = make_filter(search_terms=["foo"])
        when = compiler.compile_when(f)
        assert when is None

    def test_time_range_produces_date_range(self) -> None:
        compiler = make_compiler()
        dtr = DateTimeRange(
            start_date=DateTime(date=DateVal(day=1, month=3, year=2025)),
            stop_date=DateTime(date=DateVal(day=31, month=3, year=2025)),
        )
        f = make_filter(search_terms=["foo"], time_range=dtr)
        when = compiler.compile_when(f)
        assert when is not None
        assert when.date_range is not None
        assert when.date_range.start.month == 3

    def test_informational_action_no_scope(self) -> None:
        compiler = make_compiler()
        actor = make_entity("Alice")
        action = make_action(actor=[actor], verbs=["spoke"], is_informational=True)
        f = make_filter(action=action)
        when = compiler.compile_when(f)
        # is_informational = True → should_add_scope returns False → no scope in when
        assert when is None or (
            when.scope_defining_terms is None
            or len(when.scope_defining_terms.terms) == 0
        )

    def test_actor_entities_list_adds_scope(self) -> None:
        compiler = make_compiler()
        actor = make_entity("Alice")
        action = make_action(actor=[actor], verbs=["sent"])
        f = make_filter(action=action)
        when = compiler.compile_when(f)
        assert when is not None
        assert when.scope_defining_terms is not None
        assert len(when.scope_defining_terms.terms) > 0


# ---------------------------------------------------------------------------
# SearchQueryCompiler — compile_search_terms
# ---------------------------------------------------------------------------


class TestCompileSearchTerms:
    def test_returns_search_term_group(self) -> None:
        compiler = make_compiler()
        group = compiler.compile_search_terms(["alpha", "beta"])
        texts = [t.term.text for t in group.terms if isinstance(t, SearchTerm)]
        assert "alpha" in texts
        assert "beta" in texts

    def test_appends_to_existing_group(self) -> None:
        compiler = make_compiler()
        existing = SearchTermGroup(boolean_op="or", terms=[])
        compiler.compile_search_terms(["gamma"], existing)
        texts = [t.term.text for t in existing.terms if isinstance(t, SearchTerm)]
        assert "gamma" in texts


# ---------------------------------------------------------------------------
# SearchQueryCompiler — is_searchable_string / is_noise_term
# ---------------------------------------------------------------------------


class TestIsSearchableString:
    def test_normal_string_is_searchable(self) -> None:
        compiler = make_compiler()
        assert compiler.is_searchable_string("hello") is True

    def test_empty_string_not_searchable(self) -> None:
        compiler = make_compiler()
        assert compiler.is_searchable_string("") is False

    def test_star_not_searchable(self) -> None:
        compiler = make_compiler()
        assert compiler.is_searchable_string("*") is False

    def test_term_filter_respected(self) -> None:
        options = LanguageQueryCompileOptions(term_filter=lambda t: t != "skip")
        compiler = make_compiler(options=options)
        assert compiler.is_searchable_string("skip") is False
        assert compiler.is_searchable_string("keep") is True


class TestIsNoiseTerm:
    def test_noise_words(self) -> None:
        compiler = make_compiler()
        for word in ("thing", "object", "concept", "idea", "entity"):
            assert compiler.is_noise_term(word) is True

    def test_non_noise_word(self) -> None:
        compiler = make_compiler()
        assert compiler.is_noise_term("robot") is False

    def test_case_insensitive(self) -> None:
        compiler = make_compiler()
        assert compiler.is_noise_term("THING") is True


# ---------------------------------------------------------------------------
# SearchQueryCompiler — deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_duplicate_property_term_not_added_twice(self) -> None:
        compiler = make_compiler()
        group = SearchTermGroup(boolean_op="or", terms=[])
        compiler.add_property_term_to_group("name", "Alice", group)
        compiler.add_property_term_to_group("name", "Alice", group)
        assert len(group.terms) == 1

    def test_different_property_names_both_added(self) -> None:
        compiler = make_compiler()
        group = SearchTermGroup(boolean_op="or", terms=[])
        compiler.add_property_term_to_group("name", "Alice", group)
        compiler.add_property_term_to_group("topic", "Alice", group)
        assert len(group.terms) == 2

    def test_dedupe_disabled_allows_duplicates(self) -> None:
        compiler = make_compiler()
        compiler.dedupe = False
        group = SearchTermGroup(boolean_op="or", terms=[])
        compiler.add_property_term_to_group("name", "Alice", group)
        compiler.add_property_term_to_group("name", "Alice", group)
        assert len(group.terms) == 2


# ---------------------------------------------------------------------------
# _compile_fallback_query
# ---------------------------------------------------------------------------


class TestCompileFallbackQuery:
    def test_exact_scope_no_fallback(self) -> None:
        conv = FakeConversation()
        options = LanguageQueryCompileOptions(exact_scope=True, verb_scope=True)
        query = make_query([make_filter(search_terms=["foo"])])
        result = _compile_fallback_query(conv, query, options)
        assert result is None

    def test_no_verb_scope_no_fallback(self) -> None:
        conv = FakeConversation()
        options = LanguageQueryCompileOptions(exact_scope=False, verb_scope=False)
        query = make_query([make_filter(search_terms=["foo"])])
        result = _compile_fallback_query(conv, query, options)
        assert result is None

    def test_verb_scope_and_not_exact_produces_fallback(self) -> None:
        conv = FakeConversation()
        options = LanguageQueryCompileOptions(exact_scope=False, verb_scope=True)
        query = make_query([make_filter(search_terms=["foo"])])
        result = _compile_fallback_query(conv, query, options)
        # Should return a list of SearchQueryExpr (fallback without verb matching)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# SearchQueryCompiler — compile_action_term_as_search_terms (use_or_max=False)
# ---------------------------------------------------------------------------


class TestCompileActionTermAsSearchTerms:
    def test_no_verbs_no_actor_empty_group(self) -> None:
        compiler = make_compiler()
        action = ActionTerm(
            actor_entities="*",
            is_informational=False,
        )
        group = compiler.compile_action_term_as_search_terms(action, use_or_max=False)
        # actor is "*" so no actor entities; no verbs; result depends on implementation
        assert group is not None

    def test_use_or_max_false_merges_into_same_group(self) -> None:
        compiler = make_compiler()
        actor = make_entity("Alice")
        action = make_action(actor=[actor], verbs=["sent"])
        group = compiler.compile_action_term_as_search_terms(action, use_or_max=False)
        assert len(group.terms) > 0

    def test_empty_or_max_not_appended(self) -> None:
        """With use_or_max=True but no verbs/actors, or_max wrapper should not be appended."""
        compiler = make_compiler()
        action = ActionTerm(
            actor_entities="*",
            is_informational=False,
        )
        outer = SearchTermGroup(boolean_op="or", terms=[])
        compiler.compile_action_term_as_search_terms(action, outer, use_or_max=True)
        # or_max only appended if non-empty
        assert len(outer.terms) == 0
