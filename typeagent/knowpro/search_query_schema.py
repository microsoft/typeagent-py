# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: Move this file into knowpro.

from pydantic.dataclasses import dataclass
from pydantic import Field
from typing import Annotated, Literal
from typing_extensions import Doc

from .field_helpers import CamelCaseField
from .date_time_schema import DateTimeRange


@dataclass
class FacetTerm:
    facet_name: Annotated[
        str,
        Doc(
            "The name of the facet, such as 'color', 'profession', 'patent number'; "
            "'*' means match any facet name."
        ),
    ]
    facet_value: Annotated[
        str,
        Doc(
            "The value of the facet, such as 'red', 'writer'; "
            "'*' means match any facet value."
        ),
    ]


@dataclass
class EntityTerm:
    """
    Use to find information about specific, tangible people, places, institutions or things only.
    This includes entities with particular facets.
    Abstract concepts or topics are not entityTerms. Use string for them.
    Any terms will match fuzzily.
    """

    name: Annotated[
        str,
        Doc(
            "The name of the entity or thing such as 'Bach', 'Great Gatsby', 'frog' or 'piano' or 'we', 'I'; "
            "'*' means match any entity name."
        ),
    ]
    is_name_pronoun: bool
    type: Annotated[
        list[str] | None,
        Doc(
            "The specific types of the entity such as 'book', 'movie', 'song', 'speaker', "
            "'person', 'artist', 'animal', 'instrument', 'school', 'room', 'museum', 'food' etc.\n"
            "Generic types like 'object', 'thing' etc. are NOT allowed.\n"
            "An entity can have multiple types; entity types should be single words."
        ),
    ] = None
    facets: Annotated[
        list[FacetTerm] | None,
        Doc(
            "Facet terms search for properties or attributes of the entity.\n"
            "E.g.: color(blue), profession(writer), author(*), aunt(Agatha), weight(4kg), phoneNumber(...), etc."
        ),
    ] = None


@dataclass
class VerbsTerm:
    words: Annotated[list[str], Doc("Individual words in single or compound verb.")]
    tense: Literal["Past", "Present", "Future"] = "Present"


@dataclass
class ActionTerm:
    actor_entities: Annotated[
        list[EntityTerm] | Literal["*"],
        Doc(
            "The origin of the action or information, typically the entity performing the action."
        ),
    ]
    is_informational: Annotated[
        bool,
        Doc(
            "Is the intent of the phrase translated to this ActionTerm "
            "to actually get information about specific entities?\n"
            "Examples:\n"
            "True: if asking for specific information about an entity, "
            "such as 'What is Mia's phone number?' or 'Where did Jane study?\n"
            "False: if involves actions and interactions between entities, "
            "such as 'What phone number did Mia mention in her note to Jane?'"
        ),
    ]
    action_verbs: Annotated[
        VerbsTerm | None, Doc("Action verbs describing the interaction.")
    ] = None
    target_entities: Annotated[
        list[EntityTerm] | None,
        Doc(
            "The recipient or target of the action or information.\n"
            "Action verbs can imply relevant facet names on the targetEntity. "
            "E.g. write -> writer, sing -> singer etc."
        ),
    ] = None
    additional_entities: Annotated[
        list[EntityTerm] | None,
        Doc(
            "Additional entities participating in the action.\n"
            "E.g. in the phrase 'Jane ate the spaghetti with the fork', "
            "'the fork' would be an additional entity.\n"
            "E.g. in the phrase 'Did Jane speak about Bach with Nina', "
            "'Bach' would be the additional entity."
        ),
    ] = None


@dataclass
class SearchFilter:
    """
    Specifies the search terms for a search expression.
    Make sure at least one field below is present and not None nor empty.
    entity_search_terms cannot contain entities already in action_search_terms.
    """

    action_search_term: ActionTerm | None = None
    entity_search_terms: list[EntityTerm] | None = None
    search_terms: Annotated[
        list[str] | None,
        Doc(
            "search_terms:\n"
            "Concepts, topics or other terms that don't fit ActionTerms or EntityTerms.\n"
            "- Do not use noisy searchTerms like 'topic', 'topics', 'subject', "
            "'discussion' etc. even if they are mentioned in the user request.\n"
            "- Phrases like 'email address' or 'first name' are a single term.\n"
            "- Use empty searchTerms array when use asks for summaries."
        ),
    ] = None
    time_range: Annotated[
        DateTimeRange | None,
        Doc(
            "Use only if request explicitly asks for time range, particular year, month etc.\n"
            "in this time range."
        ),
    ] = None


@dataclass
class SearchExpr:
    rewritten_query: Annotated[
        str,
        Doc("The rewritten search query"),
        CamelCaseField(field_name="rewritten_query"),
    ]
    filters: Annotated[list[SearchFilter], Doc("List of search filters")]


@dataclass
class SearchQuery:
    search_expressions: Annotated[
        list[SearchExpr],
        Doc(
            "One expression for each search required by user request. Each SearchExpr runs independently, so make them standalone by resolving references like 'it', 'that', 'them' etc."
        ),
        CamelCaseField(field_name="search_expressions"),
    ]
