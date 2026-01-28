# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# TODO: Are we sure this isn't used as a translator schema class?

from typing import Annotated, Any

from typing_extensions import Doc

from ..knowpro.interfaces import DateRange
from .dataclasses import dataclass
from .field_helpers import CamelCaseField

type EntityNames = str | list[str]


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
    time_range: Annotated[
        DateRange | None,
        Doc("Time period during which this knowledge was gathered"),
        CamelCaseField(field_name="time_range"),
    ] = None


@dataclass
class RelevantMessage:
    from_: Annotated[
        EntityNames | None,
        Doc("Sender(s) of the message"),
        CamelCaseField(field_name="from_"),
    ] = None
    to: Annotated[EntityNames | None, Doc("Recipient(s) of the message")] = None
    timestamp: Annotated[
        str | None,
        Doc("Timestamp of the message in ISO format"),
    ] = None
    message_text: Annotated[
        str | list[str] | None,
        Doc("Text chunks in this message"),
        CamelCaseField(field_name="message_text"),
    ] = None


@dataclass
class AnswerContext:
    """Use empty lists for unneeded properties."""

    entities: Annotated[
        list[RelevantKnowledge] | None,
        Doc(
            "Relevant entities. Use the 'name' and 'type' properties of entities to PRECISELY identify those that answer the user question."
        ),
    ] = None
    topics: Annotated[list[RelevantKnowledge] | None, Doc("Relevant topics")] = None
    messages: Annotated[list[RelevantMessage] | None, Doc("Relevant messages")] = None
