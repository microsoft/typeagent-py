# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Universal message types for all conversation types (transcripts, podcasts, emails, chats, etc.)."""

from datetime import datetime, timezone
from typing import TypedDict

from pydantic import AliasChoices, Field

from . import kplib
from .dataclasses import dataclass as pydantic_dataclass
from .field_helpers import CamelCaseField
from .interfaces import IKnowledgeSource, IMessage, IMessageMetadata

# Unix epoch sentinel for unknown dates (Easter egg!)
UNIX_EPOCH = datetime(1970, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def format_timestamp_utc(dt: datetime) -> str:
    """
    Format a datetime as ISO 8601 with explicit Z suffix for UTC.

    Args:
        dt: Datetime to format (must be timezone-aware; will be converted to UTC if needed)

    Returns:
        ISO 8601 string with Z suffix, e.g., "2024-01-01T12:34:56Z"

    Raises:
        ValueError: If datetime is timezone-naive
    """
    if dt.tzinfo is None:
        raise ValueError(
            f"Datetime must be timezone-aware, got naive datetime: {dt}. "
            f"Use dt.replace(tzinfo=timezone.utc) or parse with 'Z' suffix."
        )

    # Convert non-UTC to UTC if needed
    if dt.tzinfo != timezone.utc:
        dt = dt.astimezone(timezone.utc)

    # Format as ISO 8601 and replace +00:00 with Z
    iso_str = dt.isoformat()
    if iso_str.endswith("+00:00"):
        return iso_str[:-6] + "Z"
    return iso_str


@pydantic_dataclass
class ConversationMessageMeta(IKnowledgeSource, IMessageMetadata):
    """
    Universal metadata for conversation messages.

    Supports transcripts, podcasts, chats, emails, forums, and other conversation types.
    All fields are optional to maximize flexibility across different source types.
    """

    speaker: str | None = None
    """
    The primary source/sender of the message.

    Examples:
    - Transcript: Speaker name from WebVTT voice tag or text pattern
    - Podcast: Participant speaking this turn
    - Email: Email sender address or name
    - Chat: Username or display name
    - Forum: Post author
    """

    recipients: list[str] = Field(
        default_factory=list,
        serialization_alias="listeners",
        validation_alias=AliasChoices("recipients", "listeners"),
    )
    """
    Intended recipients/listeners of the message.

    Examples:
    - Transcript: Empty (broadcast medium, no specific recipients)
    - Podcast: Other participants in conversation
    - Email: To/CC recipients
    - Chat: @-mentioned users, or all room members
    - Forum: Empty (public post) or @-mentioned users
    """

    @property
    def source(self) -> str | None:  # type: ignore[reportIncompatibleVariableOverride]
        """IMessageMetadata.source property - returns the sender."""
        return self.speaker

    @property
    def dest(self) -> list[str] | None:  # type: ignore[reportIncompatibleVariableOverride]
        """IMessageMetadata.dest property - returns recipients if any."""
        return self.recipients if self.recipients else None

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        """
        Extract structured knowledge from metadata.

        Creates:
        - Person entities for speaker and recipients
        - "Say/speak" actions linking speaker to each recipient
        """
        if not self.speaker:
            return kplib.KnowledgeResponse(
                entities=[],
                actions=[],
                inverse_actions=[],
                topics=[],
            )

        # Create speaker entity
        entities: list[kplib.ConcreteEntity] = [
            kplib.ConcreteEntity(
                name=self.speaker,
                type=["person"],
            )
        ]

        # Create recipient entities
        entities.extend(
            [
                kplib.ConcreteEntity(
                    name=recipient,
                    type=["person"],
                )
                for recipient in self.recipients
            ]
        )

        # Create communication actions
        if self.recipients:
            # Podcast style: speaker says to each recipient
            actions = [
                kplib.Action(
                    verbs=["say"],
                    verb_tense="past",
                    subject_entity_name=self.speaker,
                    object_entity_name=recipient,
                    indirect_object_entity_name="none",
                )
                for recipient in self.recipients
            ]
        else:
            # Transcript style: speaker speaks (no specific audience)
            actions = [
                kplib.Action(
                    verbs=["say", "speak"],
                    verb_tense="past",
                    subject_entity_name=self.speaker,
                    object_entity_name="none",
                    indirect_object_entity_name="none",
                )
            ]

        return kplib.KnowledgeResponse(
            entities=entities,
            actions=actions,
            inverse_actions=[],
            topics=[],
        )


class ConversationMessageMetaData(TypedDict):
    """Serialization format for ConversationMessageMeta."""

    speaker: str | None
    listeners: list[str]


class ConversationMessageData(TypedDict):
    """Serialization format for ConversationMessage."""

    metadata: ConversationMessageMetaData
    textChunks: list[str]
    tags: list[str]
    timestamp: str | None


@pydantic_dataclass
class ConversationMessage(IMessage):
    """Universal message for any conversation type."""

    text_chunks: list[str] = CamelCaseField("The text chunks of the message")
    metadata: ConversationMessageMeta = CamelCaseField(
        "Metadata associated with the message"
    )
    tags: list[str] = CamelCaseField(
        "Tags associated with the message", default_factory=list
    )
    timestamp: str | None = None
    """
    ISO 8601 datetime when message occurred/was sent (UTC timezone).

    Set during ingestion (not after):
    - Transcripts: base_date + webvtt_offset
    - Podcasts: proportional allocation
    - Email/Chat: from message headers

    If no base_date provided during ingestion, uses Unix epoch (1970-01-01 00:00:00 UTC)
    as a sentinel value (Easter egg for Unix folks: "timestamp left at zero").

    Format: "2024-01-01T12:34:56Z" or "1970-01-01T00:01:23Z" (epoch-based)
    MUST include "Z" suffix to explicitly indicate UTC timezone.
    """

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        return self.metadata.get_knowledge()

    def add_timestamp(self, timestamp: str) -> None:
        self.timestamp = timestamp

    def add_content(self, content: str) -> None:
        self.text_chunks[0] += content

    def serialize(self) -> ConversationMessageData:
        # pydantic's __pydantic_serializer__ is not in type stubs
        return self.__pydantic_serializer__.to_python(self, by_alias=True)  # type: ignore[attr-defined]

    @staticmethod
    def deserialize(message_data: ConversationMessageData) -> "ConversationMessage":
        # pydantic's __pydantic_validator__ is not in type stubs
        return ConversationMessage.__pydantic_validator__.validate_python(message_data)  # type: ignore[attr-defined]
