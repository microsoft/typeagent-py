# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Annotated, Any, Union

from pydantic.dataclasses import dataclass

from ..knowpro.interfaces import DateRange

Doc = lambda s: s  # Placeholder for Doc metadata

EntityNames = Union[str, list[str]]


@dataclass
class RelevantKnowledge:
    knowledge: Annotated[Any, Doc("The actual knowledge")]
    origin: Annotated[
        EntityNames | None, Doc("Entity or entities who mentioned the knowledge")
    ]
    audience: Annotated[
        EntityNames | None,
        Doc("Entity or entities who received or consumed this knowledge"),
    ]
    timeRange: Annotated[
        DateRange | None, Doc("Time period during which this knowledge was gathered")
    ]


@dataclass
class RelevantMessage:
    from_: Annotated[EntityNames | None, Doc("Sender(s) of the message")]
    to: Annotated[EntityNames | None, Doc("Recipient(s) of the message")]
    timestamp: Annotated[str | None, Doc("Timestamp of the message in ISO format")]
    messageText: Annotated[str | list[str] | None, Doc("Text chunks in this message")]


@dataclass
class AnswerContext:
    entities: Annotated[
        list[RelevantKnowledge] | None,
        Doc(
            "Relevant entities. Use the 'name' and 'type' properties of entities to PRECISELY identify those that answer the user question."
        ),
    ]
    topics: Annotated[list[RelevantKnowledge] | None, Doc("Relevant topics")]
    messages: Annotated[list[RelevantMessage] | None, Doc("Relevant messages")]
