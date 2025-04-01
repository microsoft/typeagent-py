# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime as Datetime, timedelta as Timedelta
from typing import (
    Any,
    Callable,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
)

from ..aitools.embeddings import NormalizedEmbedding, NormalizedEmbeddings
from ..aitools.vectorbase import VectorBase
from . import kplib


# An object that can provide a KnowledgeResponse structure.
class IKnowledgeSource(Protocol):
    def get_knowledge(self) -> kplib.KnowledgeResponse:
        raise NotImplementedError


@dataclass
class DeletionInfo:
    timestamp: str
    reason: str | None = None


type MessageOrdinal = int


class IMessage(IKnowledgeSource, Protocol):
    # The text of the message, split into chunks.
    text_chunks: list[str]
    timestamp: str | None = None
    tags: list[str]
    deletion_info: DeletionInfo | None = None


type SemanticRefOrdinal = int


@dataclass
class ScoredSemanticRefOrdinal:
    semantic_ref_ordinal: SemanticRefOrdinal
    score: float

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.semantic_ref_ordinal}, {self.score})"

    def serialize(self) -> "ScoredSemanticRefOrdinalData":
        return ScoredSemanticRefOrdinalData(
            semanticRefOrdinal=self.semantic_ref_ordinal, score=self.score
        )

    @staticmethod
    def deserialize(data: "ScoredSemanticRefOrdinalData") -> "ScoredSemanticRefOrdinal":
        return ScoredSemanticRefOrdinal(
            semantic_ref_ordinal=data["semanticRefOrdinal"],
            score=data["score"],
        )


@dataclass
class ScoredMessageOrdinal:
    message_ordinal: MessageOrdinal
    score: float


class ITermToSemanticRefIndex(Protocol):
    def get_terms(self) -> Sequence[str]:
        raise NotImplementedError

    def add_term(
        self,
        term: str,
        semantic_ref_ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
    ) -> None:
        raise NotImplementedError

    def remove_term(self, term: str, semantic_ref_ordinal: SemanticRefOrdinal) -> None:
        raise NotImplementedError

    def lookup_term(self, term: str) -> Sequence[ScoredSemanticRefOrdinal] | None:
        raise NotImplementedError


type KnowledgeType = Literal["entity", "action", "topic", "tag"]


@dataclass
class Topic:
    text: str


@dataclass
class Tag:
    text: str


type Knowledge = kplib.ConcreteEntity | kplib.Action | Topic | Tag


class TextLocationData(TypedDict):
    messageOrdinal: MessageOrdinal
    chunkOrdinal: int
    charOrdinal: int


@dataclass(order=True)
class TextLocation:
    # The ordinal of the message.
    message_ordinal: MessageOrdinal
    # The ordinal of the chunk.
    chunk_ordinal: int = 0
    # The ordinal of the character within the chunk.
    char_ordinal: int = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message_ordinal}, {self.chunk_ordinal}, {self.char_ordinal})"

    def serialize(self) -> TextLocationData:
        return TextLocationData(
            messageOrdinal=self.message_ordinal,
            chunkOrdinal=self.chunk_ordinal,
            charOrdinal=self.char_ordinal,
        )

    @staticmethod
    def deserialize(data: TextLocationData) -> "TextLocation":
        return TextLocation(
            message_ordinal=data["messageOrdinal"],
            chunk_ordinal=data["chunkOrdinal"],
            char_ordinal=data["charOrdinal"],
        )


class TextRangeData(TypedDict):
    start: TextLocationData
    end: NotRequired[TextLocationData | None]


# A text range within a session.
@dataclass(order=True)
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

    def __contains__(self, other: Self) -> bool:
        otherend = other.end or other.start
        selfend = self.end or self.start
        return self.start <= other.start and otherend <= selfend

    def serialize(self) -> TextRangeData:
        if self.end is None:
            return TextRangeData(start=self.start.serialize())
        else:
            return TextRangeData(
                start=self.start.serialize(),
                end=self.end.serialize(),
            )

    @staticmethod
    def deserialize(data: TextRangeData) -> "TextRange":
        start = TextLocation.deserialize(data["start"])
        end_data = data.get("end")
        if end_data is None:
            return TextRange(start)
        else:
            end = TextLocation.deserialize(end_data)
            return TextRange(start, end)


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
    semantic_ref_ordinal: SemanticRefOrdinal
    range: TextRange
    knowledge_type: KnowledgeType
    knowledge: Knowledge

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.semantic_ref_ordinal}, {self.range}, {self.knowledge_type!r}, {self.knowledge})"

    def serialize(self) -> SemanticRefData:
        from . import serialization

        return SemanticRefData(
            semanticRefOrdinal=self.semantic_ref_ordinal,
            range=self.range.serialize(),
            knowledgeType=self.knowledge_type,
            knowledge=serialization.serialize_object(self.knowledge),
        )

    @staticmethod
    def deserialize(data: SemanticRefData) -> "SemanticRef":
        from . import serialization

        return SemanticRef(
            semantic_ref_ordinal=data["semanticRefOrdinal"],
            range=TextRange.deserialize(data["range"]),
            knowledge_type=data["knowledgeType"],
            knowledge=serialization.deserialize_knowledge(
                data["knowledgeType"], data["knowledge"]
            ),
        )


@dataclass
class DateRange:
    start: Datetime
    # Inclusive. If None, the range is unbounded.
    end: Datetime | None = None

    def __repr__(self) -> str:
        if self.end is None:
            return f"{self.__class__.__name__}({self.start})"
        else:
            return f"{self.__class__.__name__}({self.start}, {self.end})"

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

    def serialize(self) -> "TermData":
        if self.weight is None:
            return TermData(text=self.text)
        else:
            return TermData(text=self.text, weight=self.weight)


@dataclass
class ScoredKnowledge:
    knowledge_type: KnowledgeType
    knowledge: Knowledge
    score: float


# Allows for faster retrieval of name, value properties
class IPropertyToSemanticRefIndex(Protocol):
    def get_values(self) -> list[str]:
        raise NotImplementedError

    def add_property(
        self,
        property_name: str,
        value: str,
        semantic_ref_ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
    ) -> None:
        raise NotImplementedError

    def lookup_property(
        self, property_name: str, value: str
    ) -> Sequence[ScoredSemanticRefOrdinal] | None:
        raise NotImplementedError


@dataclass
class TimestampedTextRange:
    timestamp: str
    range: TextRange


# Return text ranges in the given date range.
class ITimestampToTextRangeIndex(Protocol):
    def add_timestamp(self, message_ordinal: MessageOrdinal, timestamp: str) -> bool:
        raise NotImplementedError

    def add_timestamps(
        self, message_timestamps: list[tuple[MessageOrdinal, str]]
    ) -> "ListIndexingResult":
        raise NotImplementedError

    def lookup_range(self, date_range: DateRange) -> list[TimestampedTextRange]:
        raise NotImplementedError


class ITermToRelatedTerms(Protocol):
    def lookup_term(self, text: str) -> Sequence[Term] | None:
        raise NotImplementedError


class ITermToRelatedTermsFuzzy(Protocol):
    async def add_terms(
        self, terms: Sequence[str], event_handler: "IndexingEventHandlers | None" = None
    ) -> "ListIndexingResult":
        raise NotImplementedError

    async def lookup_term(
        self,
        text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> Sequence[Term]:
        raise NotImplementedError

    async def lookup_terms(
        self,
        text_array: Sequence[str],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> Sequence[Sequence[Term]]:
        raise NotImplementedError


class ITermToRelatedTermsIndex(Protocol):
    @property
    def aliases(self) -> ITermToRelatedTerms | None:
        raise NotImplementedError

    @property
    def fuzzy_index(self) -> VectorBase | None:
        raise NotImplementedError

    def serialize(self) -> "ITermsToRelatedTermsIndexData":
        raise NotImplementedError

    def deserialize(self, data: "ITermsToRelatedTermsIndexData") -> None:
        raise NotImplementedError


class ThreadData(TypedDict):
    description: str
    ranges: list[TextRangeData]


# A Thread is a set of text ranges in a conversation.
@dataclass
class Thread:
    description: str
    ranges: Sequence[TextRange]

    def serialize(self) -> ThreadData:
        return ThreadData(
            description=self.description,
            ranges=[range.serialize() for range in self.ranges],
        )

    @staticmethod
    def deserialize(data: ThreadData) -> "Thread":
        description = data["description"]
        ranges = [TextRange.deserialize(range_data) for range_data in data["ranges"]]
        return Thread(description, ranges)


type ThreadOrdinal = int


@dataclass
class ScoredThreadOrdinal:
    thread_ordinal: ThreadOrdinal
    score: float


class IConversationThreads(Protocol):
    threads: list[Thread]

    async def add_thread(self, thread: Thread) -> None:
        raise NotImplementedError

    async def lookup_thread(
        self,
        thread_description: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> Sequence[ScoredThreadOrdinal] | None:
        raise NotImplementedError

    def serialize(self) -> "IConversationThreadData":
        raise NotImplementedError

    def deserialize(self, data: "IConversationThreadData") -> None:
        raise NotImplementedError


class IMessageTextIndex[TMessage: IMessage](Protocol):

    async def add_messages(
        self,
        messages: list[TMessage],
        event_handler: "IndexingEventHandlers | None" = None,
    ) -> "ListIndexingResult":
        raise NotImplementedError

    async def lookup_messages(
        self,
        message_text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]:
        raise NotImplementedError

    async def lookup_messages_in_subset(
        self,
        message_text: str,
        ordinals_to_search: list[MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]:
        raise NotImplementedError

    # TODO: Others?

    def serialize(self) -> "IMessageTextIndexData":
        raise NotImplementedError

    def deserialize(self, data: "IMessageTextIndexData") -> None:
        raise NotImplementedError


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
    messages: list[TMessage]
    semantic_refs: list[SemanticRef] | None
    semantic_ref_index: TTermToSemanticRefIndex | None
    secondary_indexes: IConversationSecondaryIndexes[TMessage] | None


# --------------------------------------------------
# Serialization formats use TypedDict and camelCase
# --------------------------------------------------


class IThreadDataItem(TypedDict):
    thread: ThreadData
    embedding: NormalizedEmbedding | None


class IConversationThreadData[TThreadDataItem: IThreadDataItem](TypedDict):
    threads: list[TThreadDataItem] | None


class TermData(TypedDict):
    text: str
    weight: NotRequired[float | None]


class ITermsToRelatedTermsDataItem(TypedDict):
    termText: str
    relatedTerms: list[TermData]


class ITermToRelatedTermsData(TypedDict):
    relatedTerms: NotRequired[list[ITermsToRelatedTermsDataItem] | None]


class ITextEmbeddingIndexData(TypedDict):
    textItems: list[str]
    embeddings: NormalizedEmbeddings | None


class ITermsToRelatedTermsIndexData(TypedDict):
    aliasData: NotRequired[ITermToRelatedTermsData]
    textEmbeddingData: NotRequired[ITextEmbeddingIndexData]


class ScoredSemanticRefOrdinalData(TypedDict):
    semanticRefOrdinal: SemanticRefOrdinal
    score: float


class TermToSemanticRefIndexItemData(TypedDict):
    term: str
    scoredSemanticRefOrdinals: list[ScoredSemanticRefOrdinalData]


# Persistent form of a term index.
class TermToSemanticRefIndexData(TypedDict):
    items: list[TermToSemanticRefIndexItemData]


class IConversationData[TMessageData](TypedDict):
    nameTag: str
    messages: list[TMessageData]
    tags: list[str]
    semanticRefs: list[SemanticRefData] | None
    semanticIndexData: NotRequired[TermToSemanticRefIndexData | None]


class ITextToTextLocationIndexData(TypedDict):
    textLocations: list[TextLocationData]
    embeddings: ITextEmbeddingIndexData


class IMessageTextIndexData(TypedDict):
    indexData: NotRequired[ITextToTextLocationIndexData | None]


class IConversationDataWithIndexes[TMessageData](IConversationData[TMessageData]):
    relatedTermsIndexData: NotRequired[ITermsToRelatedTermsIndexData | None]
    threadData: NotRequired[IConversationThreadData | None]
    messageIndexData: NotRequired[IMessageTextIndexData | None]


# --------------------------------
# Indexing helper data structures
# --------------------------------


# TODO: Should the callables become methods with a default implementation?
@dataclass
class IndexingEventHandlers:
    on_knowledge_extracted: (
        Callable[
            [
                TextLocation,  # chunk
                kplib.KnowledgeResponse,  # knowledge_result
            ],
            bool,
        ]
        | None
    ) = None
    on_embeddings_created: (
        Callable[
            [
                Sequence[str],  # source_texts
                Sequence[str],  # batch
                int,  # batch_start_at
            ],
            bool,
        ]
        | None
    ) = None
    on_text_indexed: (
        Callable[
            [
                list[tuple[str, TextLocation]],  # text_and_locations
                list[tuple[str, TextLocation]],  # batch
                int,  # batch_start_at
            ],
            bool,
        ]
        | None
    ) = None
    on_message_started: Callable[[MessageOrdinal], bool] | None = None


@dataclass
class TextIndexingResult:
    completed_upto: TextLocation | None = None  # Last message and chunk indexed
    error: str | None = None


@dataclass
class ListIndexingResult:
    number_completed: int
    error: str | None = None


@dataclass
class SecondaryIndexingResults:
    properties: ListIndexingResult | None = None
    timestamps: ListIndexingResult | None = None
    related_terms: ListIndexingResult | None = None
    message: TextIndexingResult | None = None


@dataclass
class IndexingResults:
    semantic_refs: TextIndexingResult | None = None
    secondary_index_results: SecondaryIndexingResults | None = None
