# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: Move this file into knowpro.

import copy
from pyexpat.errors import XML_ERROR_RESERVED_PREFIX_XML
from typing import Callable, Literal, TypeGuard, cast


from ..knowpro.collections import PropertyTermSet
from ..knowpro.interfaces import (
    DateRange,
    Datetime,
    KnowledgePropertyName,
    PropertySearchTerm,
    SearchSelectExpr,
    SearchTerm,
    SearchTermGroup,
    SearchTermGroupTypes,
    Term,
    WhenFilter,
)
from ..knowpro.propindex import PropertyNames
from ..knowpro.search import SearchQueryExpr

from .date_time_schema import DateTime, DateTimeRange
from .search_query_schema import (
    ActionTerm,
    EntityTerm,
    SearchExpr,
    SearchFilter,
    SearchQuery,
    VerbsTerm,
)


class SearchQueryCompiler:

    def __init__(
        self,
        *,
        exact_scope: bool = False,
        verb_scope: bool = True,
        # Use to ignore noise terms etc.
        term_filter: Callable[[str], bool] | None = None,
        # Debug flags
        apply_scope: bool = True,  # False to turn off scope matching entirely
    ):
        self.exact_scope = exact_scope
        self.verb_scope = verb_scope
        self.term_filter = term_filter
        self.apply_scope = apply_scope

        self.entity_terms_added = PropertyTermSet()
        self.dedupe = True

    def compile_query(self, query: SearchQuery) -> list[SearchQueryExpr]:
        query = copy.copy(query)  # Shallow copy so we can modify it
        query_expressions: list[SearchQueryExpr] = []
        for search_expr in query.search_expressions:
            query_expressions.append(self.compile_search_expr(search_expr))
        return query_expressions

    def compile_search_expr(self, search_expr: SearchExpr) -> SearchQueryExpr:
        query_expr = SearchQueryExpr(select_expressions=[])
        if search_expr.filters:
            for filter in search_expr.filters:
                query_expr.select_expressions.append(self.compile_search_filter(filter))
        query_expr.raw_query = search_expr.rewritten_query
        return query_expr

    def compile_search_filter(self, filter: SearchFilter) -> SearchSelectExpr:
        search_term_group = self.compile_term_group(filter)
        when = self.compile_when(filter)
        return SearchSelectExpr(
            search_term_group,
            when,
        )

    def compile_term_group(self, filter: SearchFilter) -> SearchTermGroup:
        term_group = SearchTermGroup(boolean_op="or", terms=[])
        self.entity_terms_added.clear()
        terms = filter.entity_search_terms
        if is_entity_term_list(terms):
            self.compile_entity_terms(terms, term_group)
        if filter.action_search_term:
            # term_group.terms.append("filter.actionSearchTerm, false, true")
            self.compile_action_term_as_search_terms(
                filter.action_search_term, term_group, False
            )
        if filter.search_terms is not None:
            self.compile_search_terms(filter.search_terms, term_group)
        elif len(term_group.terms) == 0:
            # Summary
            term_group.terms.append(create_property_search_term("topic", "*"))
        return term_group

    def compile_when(self, filter: SearchFilter) -> WhenFilter | None:
        when: WhenFilter | None = None
        action_term = filter.action_search_term
        if (
            self.apply_scope
            and action_term is not None
            and self.should_add_scope(action_term)
        ):
            scope_defining_terms = self.compile_scope(
                action_term,
                include_additional_entities=False,
                include_verbs=self.verb_scope if self.verb_scope is not None else True,
            )
            if scope_defining_terms.terms:
                if when is None:
                    when = WhenFilter()
                when.scope_defining_terms = scope_defining_terms
        if filter.time_range is not None:
            if when is None:
                when = WhenFilter()
            when.date_range = date_range_from_datetime_range(filter.time_range)

    def compile_action_term_as_search_terms(
        self,
        action_term: ActionTerm,
        term_group: SearchTermGroup | None = None,
        use_or_max: bool = True,
    ) -> SearchTermGroup:
        if term_group is None:
            term_group = SearchTermGroup("or")
        action_group = SearchTermGroup("or_max") if use_or_max else term_group
        if action_term.action_verbs is not None:
            for verb in action_term.action_verbs.words:
                self.add_property_term_to_group("topic", verb, action_group)
        if is_entity_term_list(action_term.actor_entities):
            self.compile_entity_terms_as_search_terms(
                action_term.actor_entities, action_group
            )
        if is_entity_term_list(action_term.target_entities):
            self.compile_entity_terms_as_search_terms(
                action_term.target_entities, action_group
            )
        if is_entity_term_list(action_term.additional_entities):
            self.compile_entity_terms_as_search_terms(
                action_term.additional_entities, action_group
            )
        return term_group

    def compile_search_terms(
        self, search_terms: list[str], term_group: SearchTermGroup | None = None
    ) -> SearchTermGroup:
        if term_group is None:
            term_group = SearchTermGroup(boolean_op="or", terms=[])
        for search_term in search_terms:
            term_group.terms.append(SearchTerm(Term(search_term)))
        return term_group

    def compile_entity_terms(
        self,
        entity_terms: list[EntityTerm],
        term_group: SearchTermGroup,
        use_or_max: bool = True,
    ) -> None:
        if use_or_max:
            save_dedupe = self.dedupe
            self.dedupe = False
            for term in entity_terms:
                or_max = SearchTermGroup(
                    boolean_op="or_max",
                    terms=[],
                )
                self.add_entity_term_to_group(term, or_max)
                term_group.terms.append(optimize_or_max(or_max))
            self.dedupe = save_dedupe
        else:
            for term in entity_terms:
                self.add_entity_term_to_group(term, term_group)
        # Also search for topics.
        for term in entity_terms:
            self.add_entity_name_to_group(term, PropertyNames.Topic, term_group)
            if term.facets is not None:
                for facet in term.facets:
                    if facet.facet_value not in (None, "*"):
                        self.add_property_term_to_group(
                            facet.facet_value, "topic", term_group
                        )

    def compile_entity_terms_as_search_terms(
        self,
        entity_terms: list[EntityTerm],
        term_group: SearchTermGroup,
    ) -> None:
        for term in entity_terms:
            self.add_entity_term_as_search_terms_to_group(term, term_group)

    def compile_scope(
        self,
        action_term: ActionTerm,
        include_additional_entities: bool = True,
        include_verbs: bool = True,
    ) -> SearchTermGroup:
        save_dedupe = self.dedupe
        self.dedupe = False

        term_group = self.compile_action_term(action_term, True, include_verbs)
        if include_additional_entities and is_entity_term_list(
            action_term.additional_entities
        ):
            self.add_entity_names_to_group(
                action_term.additional_entities,
                PropertyNames.EntityName,
                term_group,
                self.exact_scope,
            )

        self.dedupe = save_dedupe
        return term_group

    def compile_action_term(
        self,
        action_term: ActionTerm,
        use_and: bool,
        include_verbs: bool,
    ) -> SearchTermGroup:
        save_dedupe = self.dedupe
        self.dedupe = False

        term_group: SearchTermGroup
        if is_entity_term_list(action_term.target_entities):
            term_group = SearchTermGroup("and" if use_and else "or")
            for entity in action_term.target_entities:
                # S.V.O. == Subject, Verb, Object
                svo_term_group = (
                    self.compile_subject_and_verb(action_term)
                    if include_verbs
                    else self.compile_subject(action_term)
                )
                # A target can be the name of an object of an action OR the name of an entity.
                object_term_group = self.compile_object(entity)
                if object_term_group.terms:
                    svo_term_group.terms.append(object_term_group)
                term_group.terms.append(svo_term_group)
            if len(term_group.terms) == 1:
                term_group = cast(SearchTermGroup, term_group.terms[0])
        else:
            term_group = self.compile_subject_and_verb(action_term)

        self.dedupe = save_dedupe
        return term_group

    def compile_subject_and_verb(self, action_term: ActionTerm) -> SearchTermGroup:
        term_group = SearchTermGroup("and")
        self.add_subject_to_group(action_term, term_group)
        term_group = SearchTermGroup("and")
        self.add_subject_to_group(action_term, term_group)
        if action_term.action_verbs is not None:
            self.add_verbs_to_group(action_term.action_verbs, term_group)
        return term_group

    def compile_subject(self, action_term: ActionTerm) -> SearchTermGroup:
        term_group = SearchTermGroup("and")
        self.add_subject_to_group(action_term, term_group)
        return term_group

    def add_subject_to_group(
        self,
        action_term: ActionTerm,
        term_group: SearchTermGroup,
    ) -> None:
        if is_entity_term_list(action_term.actor_entities):
            self.add_entity_names_to_group(
                action_term.actor_entities, PropertyNames.Subject, term_group
            )

    def compile_object(self, entity: EntityTerm) -> SearchTermGroup:
        # A target can be the name of an object of an action OR the name of an entity.
        term_group = SearchTermGroup("or")
        self.add_entity_name_to_group(entity, PropertyNames.Object, term_group)
        self.add_entity_name_to_group(
            entity, PropertyNames.EntityName, term_group, self.exact_scope
        )
        self.add_entity_name_to_group(
            entity, PropertyNames.Topic, term_group, self.exact_scope
        )
        return term_group

    def add_verbs_to_group(
        self,
        verbs: VerbsTerm,
        term_group: SearchTermGroup,
    ) -> None:
        for verb in verbs.words:
            self.add_property_term_to_group("verb", verb, term_group)

    def add_entity_term_as_search_terms_to_group(
        self, entity_term: EntityTerm, term_group: SearchTermGroup
    ) -> None:
        if entity_term.is_name_pronoun:
            return
        self.add_search_term_to_group(entity_term.name, term_group)
        if entity_term.type:
            for type in entity_term.type:
                self.add_search_term_to_group(type, term_group)
        if entity_term.facets:
            for facet in entity_term.facets:
                self.add_search_term_to_group(facet.facet_name, term_group)
                self.add_search_term_to_group(facet.facet_value, term_group)

    def add_search_term_to_group(
        self,
        term: str,
        term_group: SearchTermGroup,
    ) -> None:
        if self.is_searchable_string(term):
            term_group.terms.append(SearchTerm(Term(term)))

    def add_entity_term_to_group(
        self,
        entity_term: EntityTerm,
        term_group: SearchTermGroup,
        exact_match_name=False,
    ) -> None:
        self.add_property_term_to_group(
            PropertyNames.EntityName.value,
            entity_term.name,
            term_group,
            exact_match_name,
        )
        if entity_term.type:
            for type in entity_term.type:
                self.add_property_term_to_group(
                    PropertyNames.EntityType.value, type, term_group
                )
        if entity_term.facets:
            for facet in entity_term.facets:
                name_is_wildcard = facet.facet_name == "*"
                value_is_wildcard = facet.facet_value == "*"
                match name_is_wildcard, value_is_wildcard:
                    case False, False:
                        self.add_property_term_to_group(
                            facet.facet_name,
                            facet.facet_value,
                            term_group,
                        )
                    case False, True:
                        self.add_property_term_to_group(
                            PropertyNames.FacetName.value,
                            facet.facet_name,
                            term_group,
                        )
                    case True, False:
                        self.add_property_term_to_group(
                            PropertyNames.FacetValue.value,
                            facet.facet_value,
                            term_group,
                        )
                    case True, True:
                        pass

    def add_entity_names_to_group(
        self,
        entity_terms: list[EntityTerm],
        property_name: PropertyNames,
        term_group: SearchTermGroup,
        exact_match_value: bool = False,
    ) -> None:
        for entity_term in entity_terms:
            self.add_entity_name_to_group(
                entity_term, property_name, term_group, exact_match_value
            )

    def add_entity_name_to_group(
        self,
        entity_term: EntityTerm,
        property_name: PropertyNames,
        term_group: SearchTermGroup,
        exact_match_value: bool = False,
    ) -> None:
        if not entity_term.is_name_pronoun:
            self.add_property_term_to_group(
                property_name.value,
                entity_term.name,
                term_group,
                exact_match_value,
            )

    def add_search_term_to_groupadd_entity_name_to_group(
        self,
        entity_term: EntityTerm,
        property_name: PropertyNames,
        term_group: SearchTermGroup,
        exact_match_value: bool = False,
    ) -> None:
        if not entity_term.is_name_pronoun:
            self.add_property_term_to_group(
                property_name.value,
                entity_term.name,
                term_group,
                exact_match_value,
            )

    def add_property_term_to_group(
        self,
        property_name: str,
        property_value: str,
        term_group: SearchTermGroup,
        exact_match_value=False,
    ) -> None:
        if not self.is_searchable_string(property_name):
            return
        if not self.is_searchable_string(property_value):
            return
        if self.is_noise_term(property_value):
            return
        # Dedupe any terms already added to the group earlier.
        if not self.dedupe or not self.entity_terms_added.has(
            property_name, property_value
        ):
            search_term = create_property_search_term(
                property_name, property_value, exact_match_value
            )
            term_group.terms.append(search_term)
            self.entity_terms_added.add(property_name, search_term.property_value.term)

    def is_searchable_string(self, value: str) -> bool:
        if not value or value == "*":
            return False
        return self.term_filter is None or self.term_filter(value)

    def is_noise_term(self, value: str) -> bool:
        return value.lower() in ("thing", "object", "concept", "idea", "entity")

    def should_add_scope(self, action_term: ActionTerm) -> bool:
        if not action_term or action_term.is_informational:
            return False
        if self.exact_scope:
            return True
        # If the action has no subject, disable scope.
        return is_entity_term_list(action_term.actor_entities)


# Miscellaneous helper functions.


# A type guard makes a promise to the type checker when it returns True.
def is_entity_term_list(
    terms: list[EntityTerm] | Literal["*"] | None,
) -> TypeGuard[list[EntityTerm]]:
    return isinstance(terms, list)


def optimize_or_max(term_group: SearchTermGroup) -> SearchTermGroupTypes:
    if len(term_group.terms) == 1:
        return term_group.terms[0]
    return term_group


def date_range_from_datetime_range(date_time_range: DateTimeRange) -> DateRange:
    return DateRange(
        start=datetime_from_date_time(date_time_range.start_date),
        end=(
            datetime_from_date_time(date_time_range.stop_date)
            if date_time_range.stop_date
            else None
        ),
    )


def datetime_from_date_time(date_time: DateTime) -> Datetime:
    return Datetime(
        year=date_time.date.year,
        month=date_time.date.month,
        day=date_time.date.day,
        hour=date_time.time.hour if date_time.time else 0,
        minute=date_time.time.minute if date_time.time else 0,
        second=date_time.time.seconds if date_time.time else 0,
    )


def create_property_search_term(
    name: str,
    value: str,
    exact_match_value: bool = False,
) -> PropertySearchTerm:
    property_name: KnowledgePropertyName | SearchTerm
    if name in KnowledgePropertyName.__value__.__args__:
        property_name = cast(KnowledgePropertyName, name)
    else:
        property_name = SearchTerm(Term(name))
    property_value = SearchTerm(Term(value))
    if exact_match_value:
        property_value.related_terms = []
    return PropertySearchTerm(property_name, property_value)
