# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: Are we sure this isn't used as a translator schema class?

from dataclasses import dataclass
from typing import Annotated, Any, Union
from typing_extensions import Doc

from ..knowpro.interfaces import DateRange

EntityNames = Union[str, list[str]]


@dataclass
class RelevantKnowledge:
    knowledge: Annotated[Any, Doc("The actual knowledge")]
    origin: Annotated[
        EntityNames | None, Doc("Entity or entities who mentioned the knowledge")
    ] = None
    audience: Annotated[
        EntityNames | None,
        Doc("Entity or entities who received or consumed this knowledge"),
    ] = None
    timeRange: Annotated[
        DateRange | None, Doc("Time period during which this knowledge was gathered")
    ] = None


@dataclass
class RelevantMessage:
    from_: Annotated[EntityNames | None, Doc("Sender(s) of the message")]
    to: Annotated[EntityNames | None, Doc("Recipient(s) of the message")]
    timestamp: Annotated[str | None, Doc("Timestamp of the message in ISO format")]
    messageText: Annotated[str | list[str] | None, Doc("Text chunks in this message")]


@dataclass
class AnswerContext:
    """Use empty lists for unneeded properties."""

    entities: Annotated[
        list[RelevantKnowledge],
        Doc(
            "Relevant entities. Use the 'name' and 'type' properties of entities to PRECISELY identify those that answer the user question."
        ),
    ]
    topics: Annotated[list[RelevantKnowledge], Doc("Relevant topics")]
    messages: Annotated[list[RelevantMessage], Doc("Relevant messages")]
