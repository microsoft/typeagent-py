# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
INTERNAL LIBRARY
Functions that help with creating search and property terms
"""

from typing import cast

from .interfaces import (
    ISemanticRefCollection,
    KnowledgePropertyName,
    PropertySearchTerm,
    ScoredSemanticRefOrdinal,
    SearchTerm,
    SearchTermGroup,
    SearchTermGroupTypes,
    SemanticRef,
    Term,
)
from ..storage.memory.propindex import PropertyNames


def create_search_term(
    text: str,
    weight: float | None = None,
    exact_match_value: bool = False,
) -> SearchTerm:
    """
    Create a search term with an optional weight

    Args:
        text: term text
        weight: optional weight for the term
        exact_match_value: if True, configures term to only match exactly

    Returns:
        SearchTerm
    """
    term = Term(text=text, weight=weight)
    related_terms = [] if exact_match_value else None
    return SearchTerm(term=term, related_terms=related_terms)


def create_property_search_term(
    name: str,
    value: str,
    exact_match_value: bool = False,
) -> PropertySearchTerm:
    """
    Create a new property search term from the given name and value

    Args:
        name: property name
        value: property value
        exact_match_value: if True, configures propertyValue to only match exactly

    Returns:
        PropertySearchTerm
    """
    # Check if this is one of our well known predefined values
    if name in (
        "name",
        "type",
        "verb",
        "subject",
        "object",
        "indirectObject",
        "tag",
        "topic",
    ):
        property_name: KnowledgePropertyName | SearchTerm = cast(
            KnowledgePropertyName, name
        )
    else:
        property_name = create_search_term(name)

    property_value = create_search_term(value)
    if exact_match_value:
        # No related terms should be matched for this term
        property_value.related_terms = []

    return PropertySearchTerm(
        property_name=property_name, property_value=property_value
    )


def create_and_term_group(*terms: SearchTermGroupTypes) -> SearchTermGroup:
    """
    Create a term group whose matches are intersected

    Args:
        terms: search terms to group

    Returns:
        SearchTermGroup with "and" boolean operation
    """
    return SearchTermGroup(boolean_op="and", terms=list(terms))


def create_or_term_group(*terms: SearchTermGroupTypes) -> SearchTermGroup:
    """
    Create a term group whose matches are union-ed

    Args:
        terms: search terms to group

    Returns:
        SearchTermGroup with "or" boolean operation
    """
    return SearchTermGroup(boolean_op="or", terms=list(terms))


def create_or_max_term_group(*terms: SearchTermGroupTypes) -> SearchTermGroup:
    """
    Create an or_max search group

    Args:
        terms: search terms to group

    Returns:
        SearchTermGroup with "or_max" boolean operation
    """
    return SearchTermGroup(boolean_op="or_max", terms=list(terms))


def create_search_terms(terms: list[str]) -> list[SearchTerm]:
    """
    Create an array of SearchTerms from the given term strings.
    You can also provide related terms for each term string by using the following syntax:
     'novel;book;bestseller': Here, 'book' and 'bestseller' become related terms for 'novel'

    Args:
        terms: term text, with optional embedded related terms

    Returns:
        list of SearchTerm
    """
    search_terms: list[SearchTerm] = []
    for term in terms:
        search_term = _parse_search_term(term)
        if search_term:
            search_terms.append(search_term)
    return search_terms


def _parse_search_term(text: str) -> SearchTerm | None:
    """Parse a search term string with optional related terms separated by ';'."""
    term_strings = _split_term_values(text, ";")
    if len(term_strings) > 0:
        term_strings = [t.lower() for t in term_strings]
        search_term = SearchTerm(term=Term(text=term_strings[0]))
        if len(term_strings) > 1:
            search_term.related_terms = []
            for i in range(1, len(term_strings)):
                search_term.related_terms.append(Term(text=term_strings[i]))
        return search_term
    return None


def create_property_search_terms(
    property_name_values: dict[str, str],
) -> list[PropertySearchTerm]:
    """
    Create property search from the given record of name, value pairs
    To search for multiple values for same property name, the value should be a ',' separated list of sub values

    Args:
        property_name_values: dictionary of property name to value mappings

    Returns:
        list of PropertySearchTerm
    """
    property_search_terms: list[PropertySearchTerm] = []
    for property_name, property_value in property_name_values.items():
        all_values = _split_term_values(property_value, ",")
        for value in all_values:
            property_search_terms.append(
                create_property_search_term(property_name, value)
            )
    return property_search_terms


def create_topic_search_term_group(
    topic_terms: str | list[str],
    exact_match: bool = False,
) -> SearchTermGroup:
    """
    Create a search term group for topic searches.

    Args:
        topic_terms: single topic term or list of topic terms
        exact_match: if True, force exact matching

    Returns:
        SearchTermGroup for topic search
    """
    term_group = create_or_max_term_group()
    if isinstance(topic_terms, str):
        term_group.terms.append(
            create_property_search_term(
                PropertyNames.Topic.value,
                topic_terms,
                exact_match,
            )
        )
    else:
        for term in topic_terms:
            term_group.terms.append(
                create_property_search_term(
                    PropertyNames.Topic.value, term, exact_match
                )
            )
    return term_group


def create_entity_search_term_group(
    name: str | None = None,
    type_: str | None = None,
    facet_name: str | None = None,
    facet_value: str | None = None,
    exact_match: bool = False,
) -> SearchTermGroup:
    """
    Create a search term group for entity searches.

    Args:
        name: entity name to search for
        type_: entity type to search for
        facet_name: facet name to search for
        facet_value: facet value to search for
        exact_match: if True, force exact matching

    Returns:
        SearchTermGroup for entity search
    """
    term_group = create_or_max_term_group()
    if name:
        term_group.terms.append(
            create_property_search_term(
                PropertyNames.EntityName.value,
                name,
                exact_match,
            )
        )
    if type_:
        term_group.terms.append(
            create_property_search_term(
                PropertyNames.EntityType.value,
                type_,
                exact_match,
            )
        )
    if facet_name:
        term_group.terms.append(
            create_property_search_term(
                PropertyNames.FacetName.value,
                facet_name,
                exact_match,
            )
        )
    if facet_value:
        term_group.terms.append(
            create_property_search_term(
                PropertyNames.FacetValue.value,
                facet_value,
                exact_match,
            )
        )
    return term_group


def create_tag_search_term_group(
    tags: list[str],
    exact_match: bool = True,
) -> SearchTermGroup:
    """
    Create a search term group for tag searches.

    Args:
        tags: list of tags to search for
        exact_match: if True, force exact matching (default True)

    Returns:
        SearchTermGroup for tag search
    """
    term_group = create_or_max_term_group()
    for tag in tags:
        term_group.terms.append(
            create_property_search_term(PropertyNames.Tag.value, tag, exact_match)
        )
    return term_group


def _split_term_values(term: str, split_char: str) -> list[str]:
    """Split term values by the given character, trimming and removing empty strings."""
    # Simple implementation - in TS this uses kpLib.split with trim and removeEmpty options
    parts = [part.strip() for part in term.split(split_char)]
    return [part for part in parts if part]


def create_multiple_choice_question(
    question: str,
    choices: list[str],
    add_none: bool = True,
) -> str:
    """
    Create a multiple choice question string.

    Args:
        question: the question text
        choices: list of choices
        add_none: if True, add "None of the above" option

    Returns:
        formatted multiple choice question string
    """
    text = question
    if len(choices) > 0:
        text = f"Multiple choice question:\n{question}\n"
        text += "Answer using *one or more* of the following choices *only*:\n"
        for choice in choices:
            text += f"- {choice.strip()}\n"
        if add_none:
            text += "- None of the above\n"
    return text


async def get_semantic_refs_from_scored_ordinals(
    semantic_refs: ISemanticRefCollection,
    scored_ordinals: list[ScoredSemanticRefOrdinal],
) -> list[SemanticRef]:
    """
    Get semantic references from scored ordinals.

    Args:
        semantic_refs: collection of semantic references
        scored_ordinals: list of scored semantic reference ordinals

    Returns:
        list of SemanticRef objects
    """
    ordinals = [sr.semantic_ref_ordinal for sr in scored_ordinals]
    return await semantic_refs.get_multiple(ordinals)
