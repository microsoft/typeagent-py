# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Iterable, Sequence
from datetime import (
    datetime as Datetime,  # For export.
    timedelta as Timedelta,  # For export.
)
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    runtime_checkable,
)

from pydantic.dataclasses import dataclass
from pydantic import Field, AliasChoices
import typechat

from ..aitools.embeddings import NormalizedEmbeddings
from . import kplib
from .field_helpers import CamelCaseField


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
    """A message in a conversation

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


type SemanticRefOrdinal = int


@dataclass
class ScoredSemanticRefOrdinal:
    semantic_ref_ordinal: SemanticRefOrdinal = CamelCaseField(
        "The ordinal of the semantic reference"
    )
    score: float = CamelCaseField("The relevance score")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.semantic_ref_ordinal}, {self.score})"

    def serialize(self) -> "ScoredSemanticRefOrdinalData":
        return self.__pydantic_serializer__.to_python(self, by_alias=True)  # type: ignore

    @staticmethod
    def deserialize(data: "ScoredSemanticRefOrdinalData") -> "ScoredSemanticRefOrdinal":
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


# A text range within a session.
# TODO: Are TextRanges totally ordered?
@dataclass
class TextRange:
    # The start of the range.
    start: TextLocation
    # The end of the range (exclusive). If None, the range is a single point.
    end: TextLocation | None = None

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
        return self.__pydantic_serializer__.to_python(self, by_alias=True, exclude_none=True)  # type: ignore

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
        return f"{self.__class__.__name__}({self.semantic_ref_ordinal}, {self.range}, {self.knowledge.knowledge_type!r}, {self.knowledge})"

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
    # Inclusive. If None, the range is unbounded.
    end: Datetime | None = None

    def __repr__(self) -> str:
        if self.end is None:
            return f"{self.__class__.__name__}({self.start!r})"
        else:
            return f"{self.__class__.__name__}({self.start!r}, {self.end!r})"

    def __contains__(self, datetime: Datetime) -> bool:
        if self.end is None:
            return self.start <= datetime
        return self.start <= datetime <= self.end


# Term must be hashable to allow using it as a dict key or set member.
@dataclass(unsafe_hash=True)
class Term:
    text: str
    # Optional weighting for these matches.
    weight: float | None = None

    def __repr__(self) -> str:
        if self.weight is None:
            return f"{self.__class__.__name__}({self.text!r})"
        else:
            return f"{self.__class__.__name__}({self.text!r}, {self.weight:.4g})"

    def serialize(self) -> "TermData":
        return self.__pydantic_serializer__.to_python(self, by_alias=True, exclude_none=True)  # type: ignore


# Allows for faster retrieval of name, value properties
@runtime_checkable
class IPropertyToSemanticRefIndex(Protocol):
    async def size(self) -> int: ...

    async def get_values(self) -> list[str]: ...

    async def add_property(
        self,
        property_name: str,
        value: str,
        semantic_ref_ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
    ) -> None: ...

    async def lookup_property(
        self, property_name: str, value: str
    ) -> list[ScoredSemanticRefOrdinal] | None: ...

    async def clear(self) -> None: ...

    async def remove_property(self, prop_name: str, semref_id: int) -> None: ...

    async def remove_all_for_semref(self, semref_id: int) -> None: ...


@dataclass
class TimestampedTextRange:
    timestamp: str
    range: TextRange


# Return text ranges in the given date range.
class ITimestampToTextRangeIndex(Protocol):
    # Contract (stable across providers):
    # - Timestamps must be ISO-8601 strings sortable lexicographically.
    # - lookup_range(DateRange) returns items with start <= t < end (end exclusive).
    #   If end is None, treat as a point query with end = start + epsilon.
    async def size(self) -> int: ...

    async def add_timestamp(
        self, message_ordinal: MessageOrdinal, timestamp: str
    ) -> bool: ...

    async def add_timestamps(
        self, message_timestamps: list[tuple[MessageOrdinal, str]]
    ) -> None: ...

    async def lookup_range(
        self, date_range: DateRange
    ) -> list[TimestampedTextRange]: ...


class ITermToRelatedTerms(Protocol):
    async def lookup_term(self, text: str) -> list[Term] | None: ...

    async def size(self) -> int: ...

    async def is_empty(self) -> bool: ...

    async def clear(self) -> None: ...

    async def add_related_term(
        self, text: str, related_terms: Term | list[Term]
    ) -> None: ...

    async def remove_term(self, text: str) -> None: ...

    async def serialize(self) -> "TermToRelatedTermsData": ...

    async def deserialize(self, data: "TermToRelatedTermsData | None") -> None: ...


class ITermToRelatedTermsFuzzy(Protocol):
    async def size(self) -> int: ...

    async def add_terms(self, texts: list[str]) -> None: ...

    async def lookup_term(
        self,
        text: str,
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[Term]: ...

    async def lookup_terms(
        self,
        texts: list[str],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[list[Term]]: ...


class ITermToRelatedTermsIndex(Protocol):
    # Providers may implement aliases and fuzzy via separate tables, but must
    # expose them through these properties.
    @property
    def aliases(self) -> ITermToRelatedTerms: ...

    @property
    def fuzzy_index(self) -> ITermToRelatedTermsFuzzy | None: ...

    async def serialize(self) -> "TermsToRelatedTermsIndexData": ...

    async def deserialize(self, data: "TermsToRelatedTermsIndexData") -> None: ...


class ThreadData(TypedDict):
    description: str
    ranges: list[TextRangeData]


# A Thread is a set of text ranges in a conversation.
@dataclass
class Thread:
    description: str
    ranges: Sequence[TextRange]

    def serialize(self) -> ThreadData:
        return self.__pydantic_serializer__.to_python(self, by_alias=True)  # type: ignore

    @staticmethod
    def deserialize(data: ThreadData) -> "Thread":
        return Thread.__pydantic_validator__.validate_python(data)  # type: ignore


type ThreadOrdinal = int


@dataclass
class ScoredThreadOrdinal:
    thread_ordinal: ThreadOrdinal
    score: float


class IConversationThreads(Protocol):
    threads: list[Thread]

    async def add_thread(self, thread: Thread) -> None: ...

    async def lookup_thread(
        self,
        thread_description: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredThreadOrdinal] | None: ...

    def serialize(self) -> "ConversationThreadData[ThreadDataItem]": ...

    def deserialize(self, data: "ConversationThreadData[ThreadDataItem]") -> None: ...


@runtime_checkable
class IMessageTextIndex[TMessage: IMessage](Protocol):

    async def add_messages(
        self,
        messages: Iterable[TMessage],
    ) -> None: ...

    async def add_messages_starting_at(
        self,
        start_message_ordinal: int,
        messages: list[TMessage],
    ) -> None: ...

    async def lookup_messages(
        self,
        message_text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]: ...

    async def lookup_messages_in_subset(
        self,
        message_text: str,
        ordinals_to_search: list[MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]: ...

    # Async alternatives to __len__ and __bool__
    async def size(self) -> int: ...

    async def is_empty(self) -> bool: ...

    # TODO: Others?

    async def serialize(self) -> "MessageTextIndexData": ...

    async def deserialize(self, data: "MessageTextIndexData") -> None: ...


class IConversationSecondaryIndexes[TMessage: IMessage](Protocol):
    property_to_semantic_ref_index: IPropertyToSemanticRefIndex | None
    timestamp_index: ITimestampToTextRangeIndex | None
    term_to_related_terms_index: ITermToRelatedTermsIndex | None
    threads: IConversationThreads | None = None
    message_index: IMessageTextIndex[TMessage] | None = None


class IConversation[
    TMessage: IMessage,
    TTermToSemanticRefIndex: ITermToSemanticRefIndex,
](Protocol):
    name_tag: str
    tags: list[str]
    messages: "IMessageCollection[TMessage]"
    semantic_refs: "ISemanticRefCollection"
    semantic_ref_index: TTermToSemanticRefIndex
    secondary_indexes: IConversationSecondaryIndexes[TMessage] | None


# -------------
# Search Types
# -------------


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


# --------------------------------------------------
# Serialization formats use TypedDict and camelCase
# --------------------------------------------------


class ThreadDataItem(TypedDict):
    thread: ThreadData
    embedding: list[float] | None  # TODO: Why not NormalizedEmbedding?


class ConversationThreadData[TThreadDataItem: ThreadDataItem](TypedDict):
    threads: list[TThreadDataItem] | None


class TermData(TypedDict):
    text: str
    weight: NotRequired[float | None]


class TermsToRelatedTermsDataItem(TypedDict):
    termText: str
    relatedTerms: list[TermData]


class TermToRelatedTermsData(TypedDict):
    relatedTerms: NotRequired[list[TermsToRelatedTermsDataItem] | None]


class TextEmbeddingIndexData(TypedDict):
    textItems: list[str]
    embeddings: NormalizedEmbeddings | None


class TermsToRelatedTermsIndexData(TypedDict):
    aliasData: NotRequired[TermToRelatedTermsData]
    textEmbeddingData: NotRequired[TextEmbeddingIndexData]


class ScoredSemanticRefOrdinalData(TypedDict):
    semanticRefOrdinal: SemanticRefOrdinal
    score: float


class TermToSemanticRefIndexItemData(TypedDict):
    term: str
    semanticRefOrdinals: list[ScoredSemanticRefOrdinalData]


# Persistent form of a term index.
class TermToSemanticRefIndexData(TypedDict):
    items: list[TermToSemanticRefIndexItemData]


class ConversationData[TMessageData](TypedDict):
    nameTag: str
    messages: list[TMessageData]
    tags: list[str]
    semanticRefs: list[SemanticRefData] | None
    semanticIndexData: NotRequired[TermToSemanticRefIndexData | None]


class TextToTextLocationIndexData(TypedDict):
    textLocations: list[TextLocationData]
    embeddings: NormalizedEmbeddings | None


class MessageTextIndexData(TypedDict):
    indexData: NotRequired[TextToTextLocationIndexData | None]


class ConversationDataWithIndexes[TMessageData](ConversationData[TMessageData]):
    relatedTermsIndexData: NotRequired[TermsToRelatedTermsIndexData | None]
    threadData: NotRequired[ConversationThreadData[ThreadDataItem] | None]
    messageIndexData: NotRequired[MessageTextIndexData | None]


# --------------------------------
# Indexing helper data structures
# --------------------------------


# --------
# Storage
# --------


class IReadonlyCollection[T, TOrdinal](AsyncIterable[T], Protocol):
    async def size(self) -> int: ...

    async def get_item(self, arg: TOrdinal) -> T: ...

    async def get_slice(self, start: int, stop: int) -> list[T]: ...

    async def get_multiple(self, arg: list[TOrdinal]) -> list[T]: ...


class ICollection[T, TOrdinal](IReadonlyCollection[T, TOrdinal], Protocol):
    """An APPEND-ONLY collection."""

    @property
    def is_persistent(self) -> bool: ...

    async def append(self, item: T) -> None: ...

    async def extend(self, items: Iterable[T]) -> None:
        """Append multiple items to the collection."""
        # The default implementation just calls append for each item.
        for item in items:
            await self.append(item)


class IMessageCollection[TMessage: IMessage](
    ICollection[TMessage, MessageOrdinal], Protocol
):
    """A collection of Messages."""


class ISemanticRefCollection(ICollection[SemanticRef, SemanticRefOrdinal], Protocol):
    """A collection of SemanticRefs."""


class IStorageProvider[TMessage: IMessage](Protocol):
    """API spec for storage providers -- maybe in-memory or persistent."""

    async def get_message_collection(self) -> IMessageCollection[TMessage]: ...

    async def get_semantic_ref_collection(self) -> ISemanticRefCollection: ...

    # Index getters - ALL 6 index types for this conversation
    async def get_semantic_ref_index(self) -> ITermToSemanticRefIndex: ...

    async def get_property_index(self) -> IPropertyToSemanticRefIndex: ...

    async def get_timestamp_index(self) -> ITimestampToTextRangeIndex: ...

    async def get_message_text_index(self) -> IMessageTextIndex[TMessage]: ...

    async def get_related_terms_index(self) -> ITermToRelatedTermsIndex: ...

    async def get_conversation_threads(self) -> IConversationThreads: ...

    # Transaction management
    async def __aenter__(self) -> Self:
        """Enter transaction context. Calls begin_transaction()."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit transaction context. Commits on success, rolls back on exception."""
        ...

    async def close(self) -> None: ...
