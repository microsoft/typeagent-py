# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Search-related interfaces for knowpro."""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic.dataclasses import dataclass
from .field_helpers import CamelCaseField
from .interfaces_core import (
    DateRange,
    KnowledgeType,
    ScoredSemanticRefOrdinal,
    Term,
    TextRange,
)


if TYPE_CHECKING:
    from ._dataclass_typing import dataclass


@dataclass
class SearchTerm:
    """Represents a term being searched for.

    Attributes:
        term: The term being searched for.
        related_terms: Additional terms related to the term. These can be supplied
            from synonym tables and so on.
            - An empty list indicates no related matches for this term.
            - `None` indicates that the search processor may try to resolve related
              terms from any available secondary indexes (e.g., ITermToRelatedTermsIndex).
    """

    term: Term
    related_terms: list[Term] | None = CamelCaseField(
        "Additional terms related to the term. These can be supplied from synonym tables and so on",
        default=None,
    )


# Well-known knowledge properties.
type KnowledgePropertyName = Literal[
    "name",  # the name of an entity
    "type",  # the type of an entity
    "verb",  # the verb of an action
    "subject",  # the subject of an action
    "object",  # the object of an action
    "indirectObject",  # the indirect object of an action
    "tag",  # tag
    "topic",  # topic
]


@dataclass
class PropertySearchTerm:
    """PropertySearch terms let you match named property values.

    - You can match a well-known property name (e.g., name("Bach"), type("book")).
    - Or you can provide a SearchTerm as a propertyName.
      For example, to match hue(red):
        - propertyName as SearchTerm, set to 'hue'
        - propertyValue as SearchTerm, set to 'red'
      We also want hue(red) to match any facets called color(red).

    SearchTerms can include related terms:
    - For example, you could include "color" as a related term for the
      propertyName "hue", or 'crimson' for red.

    The query processor can also resolve related terms using a
    related terms secondary index, if one is available.
    """

    property_name: KnowledgePropertyName | SearchTerm = CamelCaseField(
        "The property name to search for"
    )
    property_value: SearchTerm = CamelCaseField("The property value to search for")


@dataclass
class SearchTermGroup:
    """A group of search terms."""

    boolean_op: Literal["and", "or", "or_max"] = CamelCaseField(
        "The boolean operation to apply to the terms"
    )
    terms: list["SearchTermGroupTypes"] = CamelCaseField(
        "The list of search terms in this group", default_factory=list
    )


type SearchTermGroupTypes = SearchTerm | PropertySearchTerm | SearchTermGroup


@dataclass
class WhenFilter:
    """Additional constraints on when a SemanticRef is considered a match.

    A SemanticRef matching a term is actually considered a match
    when the following optional conditions are met (if present, must match):
      knowledgeType matches, e.g. knowledgeType == 'entity'
      dateRange matches, e.g. (Jan 3rd to Jan 10th)
      Semantic Refs are within supplied SCOPE,
        i.e. only Semantic Refs from a 'scoping' set of text ranges will match
    """

    knowledge_type: KnowledgeType | None = None
    date_range: DateRange | None = None
    thread_description: str | None = None
    tags: list[str] | None = None

    # SCOPE DEFINITION

    # Search terms whose matching text ranges supply the scope for this query
    scope_defining_terms: SearchTermGroup | None = None
    # Additional scoping ranges separately computed by caller
    text_ranges_in_scope: list[TextRange] | None = None


@dataclass
class SearchSelectExpr:
    """An expression used to select structured contents of a conversation."""

    search_term_group: SearchTermGroup = CamelCaseField(
        "Term group that matches information"
    )  # Term group that matches information
    when: WhenFilter | None = None  # Filter that scopes what information to match


@dataclass
class SemanticRefSearchResult:
    """Result of a semantic reference search."""

    term_matches: set[str]
    semantic_ref_matches: list[ScoredSemanticRefOrdinal]


__all__ = [
    "KnowledgePropertyName",
    "PropertySearchTerm",
    "SearchSelectExpr",
    "SearchTerm",
    "SearchTermGroup",
    "SearchTermGroupTypes",
    "SemanticRefSearchResult",
    "WhenFilter",
]
