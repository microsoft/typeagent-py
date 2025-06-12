# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: Move this file into knowpro.

from pydantic.dataclasses import dataclass
from typing import Literal, Annotated
from typing_extensions import Doc

from .date_time_schema import DateTimeRange


@dataclass
class FacetTerm:
    facet_name: Annotated[
        str,
        Doc(
            "The name of the facet, such as 'color', 'profession', 'patent number'; "
            "'*' means match any facet name"
        ),
    ]
    facet_value: Annotated[
        str,
        Doc(
            "The value of the facet, such as 'red', 'writer'; "
            "'*' means match any facet value"
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
            "'*' means match any entity name"
        ),
    ]
    is_name_pronoun: bool | None
    type: Annotated[
        list[str] | None,
        Doc(
            "The specific types of the entity such as 'book', 'movie', 'song', 'speaker', "
            "'person', 'artist', 'animal', 'instrument', 'school', 'room', 'museum', 'food' etc. "
            "Generic types like 'object', 'thing' etc. are NOT allowed. "
            "An entity can have multiple types; entity types should be single words"
        ),
    ]
    facets: Annotated[
        list[FacetTerm] | None,
        Doc(
            "Facet terms search for properties or attributes of the entity. Eg: "
            "color(blue), profession(writer), author(*), aunt(Agatha), weight(4kg), phoneNumber(...), etc."
        ),
    ]


@dataclass
class VerbsTerm:
    words: Annotated[list[str], Doc("Individual words in single or compound verb")]
    tense: Literal["Past", "Present", "Future"]


@dataclass
class ActionTerm:
    action_verbs: Annotated[
        VerbsTerm | None, Doc("Action verbs describing the interaction.")
    ]
    actor_entities: Annotated[
        list[EntityTerm] | Literal["*"],
        Doc(
            "The origin of the action or information, typically the entity performing the action"
        ),
    ]
    target_entities: Annotated[
        list[EntityTerm] | None,
        Doc(
            "The recipient or target of the action or information. "
            "Action verbs can imply relevant facet names on the targetEntity. "
            "E.g. write -> writer, sing -> singer etc."
        ),
    ]
    additional_entities: Annotated[
        list[EntityTerm] | None,
        Doc(
            "Additional entities participating in the action. "
            "E.g. in the phrase 'Jane ate the spaghetti with the fork', "
            "'the fork' would be an additional entity. "
            "E.g. in the phrase 'Did Jane speak about Bach with Nina', "
            "'Bach' would be the additional entity"
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


@dataclass
class SearchFilter:
    """
    Search a search engine for:
    entitySearchTerms cannot contain entities already in actionSearchTerms.
    """

    action_search_term: ActionTerm | None
    entity_search_terms: list[EntityTerm] | None
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
    ]
    time_range: Annotated[
        DateTimeRange | None,
        Doc(
            "Use only if request explicitly asks for time range, particular year, month etc.\n"
            "In this time range."
        ),
    ]


@dataclass
class SearchExpr:
    rewritten_query: str
    filters: list[SearchFilter]


@dataclass
class SearchQuery:
    search_expressions: Annotated[
        list[SearchExpr],
        Doc(
            "One expression for each search required by user request.\n"
            "Each SearchExpr runs independently, so make them standalone by resolving "
            "references like 'it', 'that', 'them' etc."
        ),
    ]
