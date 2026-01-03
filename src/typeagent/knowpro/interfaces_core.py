# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Core conversation and knowledge interfaces for knowpro."""

from __future__ import annotations

from dataclasses import dataclass as stdlib_dataclass
from datetime import datetime as Datetime
from typing import (
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TYPE_CHECKING,
    TypedDict,
)

if TYPE_CHECKING:
    # Use stdlib dataclass for type checking (pyright understands this)
    dataclass = stdlib_dataclass
else:
    # Use pydantic dataclass at runtime for serialization support
    from pydantic.dataclasses import dataclass

import typechat

from . import kplib
from .field_helpers import CamelCaseField

__all__ = [
    "AddMessagesResult",
    "DateRange",
    "DeletionInfo",
    "ITermToSemanticRefIndex",
    "Datetime",
    "IKnowledgeExtractor",
    "IKnowledgeSource",
    "IMessage",
    "IMessageMetadata",
    "IndexingStartPoints",
    "Knowledge",
    "KnowledgeData",
    "KnowledgeType",
    "MessageOrdinal",
    "ScoredMessageOrdinal",
    "ScoredSemanticRefOrdinal",
    "SemanticRef",
    "SemanticRefData",
    "SemanticRefOrdinal",
    "Tag",
    "Term",
    "TextLocation",
    "TextLocationData",
    "TextRange",
    "TextRangeData",
    "Topic",
]

if TYPE_CHECKING:
    from .interfaces_serialization import ScoredSemanticRefOrdinalData, TermData


class IKnowledgeSource(Protocol):
    """A Knowledge Source is any object that returns knowledge."""

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        """Retrieves knowledge from the source."""
        ...


class IKnowledgeExtractor(Protocol):
    """Interface for extracting knowledge from messages."""

    async def extract(self, message: str) -> typechat.Result[kplib.KnowledgeResponse]:
        """Extract knowledge from a message."""
        ...


@dataclass
class DeletionInfo:
    timestamp: str
    reason: str | None = None


@dataclass
class IndexingStartPoints:
    """Track collection sizes before adding new items."""

    message_count: int
    semref_count: int


@dataclass
class AddMessagesResult:
    """Result of add_messages_with_indexing operation."""

    messages_added: int
    semrefs_added: int


# Messages are referenced by their sequential ordinal numbers.
type MessageOrdinal = int


class IMessageMetadata(Protocol):
    """Metadata associated with a message."""

    # The source ("senders") of the message
    source: str | list[str] | None = None

    # The dest ("recipients") of the message
    dest: str | list[str] | None = None


class IMessage[TMetadata: IMessageMetadata](IKnowledgeSource, Protocol):
    """A message in a conversation.

    A Message contains one or more text chunks.
    """

    # The text of the message, split into chunks.
    text_chunks: list[str]

    # (Optional) tags associated with the message.
    tags: list[str]

    # The (optional) timestamp of the message.
    timestamp: str | None = None

    # (Future) Information about the deletion of the message.
    deletion_info: DeletionInfo | None = None

    # Metadata associated with the message such as its source.
    metadata: TMetadata | None = None


# Semantic references are also ordinal.
type SemanticRefOrdinal = int


@dataclass
class ScoredSemanticRefOrdinal:
    semantic_ref_ordinal: SemanticRefOrdinal = CamelCaseField(
        "The ordinal of the semantic reference"
    )
    score: float = CamelCaseField("The relevance score")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.semantic_ref_ordinal}, {self.score})"

    def serialize(self) -> ScoredSemanticRefOrdinalData:
        return self.__pydantic_serializer__.to_python(self, by_alias=True)  # type: ignore

    @staticmethod
    def deserialize(data: ScoredSemanticRefOrdinalData) -> "ScoredSemanticRefOrdinal":
        return ScoredSemanticRefOrdinal.__pydantic_validator__.validate_python(data)  # type: ignore


@dataclass
class ScoredMessageOrdinal:
    message_ordinal: MessageOrdinal
    score: float


class ITermToSemanticRefIndex(Protocol):
    async def size(self) -> int: ...

    async def get_terms(self) -> list[str]: ...

    async def add_term(
        self,
        term: str,
        semantic_ref_ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
    ) -> str: ...

    async def remove_term(
        self, term: str, semantic_ref_ordinal: SemanticRefOrdinal
    ) -> None: ...

    async def lookup_term(self, term: str) -> list[ScoredSemanticRefOrdinal] | None: ...

    async def clear(self) -> None: ...

    async def serialize(self) -> Any: ...

    async def deserialize(self, data: Any) -> None: ...


# Knowledge modeling ---------------------------------------------------------

type KnowledgeType = Literal["entity", "action", "topic", "tag"]


@dataclass
class Topic:
    knowledge_type: ClassVar[Literal["topic"]] = "topic"
    text: str


@dataclass
class Tag:
    knowledge_type: ClassVar[Literal["tag"]] = "tag"
    text: str


type Knowledge = kplib.ConcreteEntity | kplib.Action | Topic | Tag


class TextLocationData(TypedDict):
    messageOrdinal: MessageOrdinal
    chunkOrdinal: int


@dataclass(order=True)
class TextLocation:
    # The ordinal of the message.
    message_ordinal: MessageOrdinal = CamelCaseField("The ordinal of the message")
    # The ordinal of the chunk.
    # In the end of a TextRange, 1 + ordinal of the last chunk in the range.
    chunk_ordinal: int = CamelCaseField(
        "The ordinal of the chunk; in the end of a TextRange, 1 + ordinal of the last chunk in the range",
        default=0,
    )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.message_ordinal}, {self.chunk_ordinal})"
        )

    def serialize(self) -> TextLocationData:
        return self.__pydantic_serializer__.to_python(self, by_alias=True)  # type: ignore

    @staticmethod
    def deserialize(data: TextLocationData) -> "TextLocation":
        return TextLocation.__pydantic_validator__.validate_python(data)  # type: ignore


class TextRangeData(TypedDict):
    start: TextLocationData
    end: NotRequired[TextLocationData | None]


# TODO: Are TextRanges totally ordered?
@dataclass
class TextRange:
    """A text range within a session."""

    start: TextLocation  # The start of the range.
    end: TextLocation | None = None  # exclusive end; None indicates a single point

    def __repr__(self) -> str:
        if self.end is None:
            return f"{self.__class__.__name__}({self.start})"
        else:
            return f"{self.__class__.__name__}({self.start}, {self.end})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TextRange):
            return NotImplemented

        if self.start != other.start:
            return False

        # Get the effective end for both ranges
        self_end = self.end or TextLocation(
            self.start.message_ordinal, self.start.chunk_ordinal + 1
        )
        other_end = other.end or TextLocation(
            other.start.message_ordinal, other.start.chunk_ordinal + 1
        )
        return self_end == other_end

    def __lt__(self, other: Self) -> bool:
        if self.start != other.start:
            return self.start < other.start
        self_end = self.end or TextLocation(
            self.start.message_ordinal, self.start.chunk_ordinal + 1
        )
        other_end = other.end or TextLocation(
            other.start.message_ordinal, other.start.chunk_ordinal + 1
        )
        return self_end < other_end

    def __gt__(self, other: Self) -> bool:
        return other.__lt__(self)

    def __ge__(self, other: Self) -> bool:
        return not self.__lt__(other)

    def __le__(self, other: Self) -> bool:
        return not other.__lt__(self)

    def __contains__(self, other: Self) -> bool:
        other_end = other.end or TextLocation(
            other.start.message_ordinal, other.start.chunk_ordinal + 1
        )
        self_end = self.end or TextLocation(
            self.start.message_ordinal, self.start.chunk_ordinal + 1
        )
        return self.start <= other.start and other_end <= self_end

    def serialize(self) -> TextRangeData:
        return self.__pydantic_serializer__.to_python(  # type: ignore
            self, by_alias=True, exclude_none=True
        )

    @staticmethod
    def deserialize(data: TextRangeData) -> "TextRange":
        return TextRange.__pydantic_validator__.validate_python(data)  # type: ignore


# TODO: Implement serializing KnowledgeData (or import from kplib).
class KnowledgeData(TypedDict):
    pass


class SemanticRefData(TypedDict):
    semanticRefOrdinal: SemanticRefOrdinal
    range: TextRangeData
    knowledgeType: KnowledgeType
    knowledge: KnowledgeData


@dataclass
class SemanticRef:
    semantic_ref_ordinal: SemanticRefOrdinal = CamelCaseField(
        "The ordinal of the semantic reference"
    )
    range: TextRange = CamelCaseField("The text range of the semantic reference")
    knowledge: Knowledge = CamelCaseField(
        "The knowledge associated with this semantic reference"
    )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.semantic_ref_ordinal}, {self.range}, "
            f"{self.knowledge.knowledge_type!r}, {self.knowledge})"
        )

    def serialize(self) -> SemanticRefData:
        from . import serialization

        return SemanticRefData(
            semanticRefOrdinal=self.semantic_ref_ordinal,
            range=self.range.serialize(),
            knowledgeType=self.knowledge.knowledge_type,
            knowledge=serialization.serialize_object(self.knowledge),
        )

    @staticmethod
    def deserialize(data: SemanticRefData) -> "SemanticRef":
        from . import serialization

        knowledge = serialization.deserialize_knowledge(
            data["knowledgeType"], data["knowledge"]
        )
        return SemanticRef(
            semantic_ref_ordinal=data["semanticRefOrdinal"],
            range=TextRange.deserialize(data["range"]),
            knowledge=knowledge,
        )


@dataclass
class DateRange:
    start: Datetime
    end: Datetime | None = None  # inclusive; None means unbounded

    def __repr__(self) -> str:
        if self.end is None:
            return f"{self.__class__.__name__}({self.start!r})"
        else:
            return f"{self.__class__.__name__}({self.start!r}, {self.end!r})"

    def __contains__(self, datetime: Datetime) -> bool:
        if self.end is None:
            return self.start <= datetime
        return self.start <= datetime <= self.end


@dataclass(unsafe_hash=True)
class Term:
    """A # Term must be hashable to allow using it as a dict key or set member."""

    text: str
    weight: float | None = None  # Optional weighting for these matches.

    def __repr__(self) -> str:
        if self.weight is None:
            return f"{self.__class__.__name__}({self.text!r})"
        else:
            return f"{self.__class__.__name__}({self.text!r}, {self.weight:.4g})"

    def serialize(self) -> TermData:
        return self.__pydantic_serializer__.to_python(  # type: ignore
            self, by_alias=True, exclude_none=True
        )


__all__ = [
    "AddMessagesResult",
    "DateRange",
    "DeletionInfo",
    "ITermToSemanticRefIndex",
    "Datetime",
    "IKnowledgeExtractor",
    "IKnowledgeSource",
    "IMessage",
    "IMessageMetadata",
    "IndexingStartPoints",
    "Knowledge",
    "KnowledgeData",
    "KnowledgeType",
    "MessageOrdinal",
    "ScoredMessageOrdinal",
    "ScoredSemanticRefOrdinal",
    "SemanticRef",
    "SemanticRefData",
    "SemanticRefOrdinal",
    "Tag",
    "Term",
    "TextLocation",
    "TextLocationData",
    "TextRange",
    "TextRangeData",
    "Topic",
]
