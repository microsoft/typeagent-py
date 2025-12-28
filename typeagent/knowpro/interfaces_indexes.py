# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Index-related protocols and helpers for knowpro conversations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic.dataclasses import dataclass
from .interfaces_core import (
    DateRange,
    IMessage,
    MessageOrdinal,
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
    SemanticRefOrdinal,
    Term,
    TextRange,
)
from .interfaces_serialization import (
    ConversationThreadData,
    MessageTextIndexData,
    TermToRelatedTermsData,
    TermsToRelatedTermsIndexData,
    ThreadData,
    ThreadDataItem,
)

__all__ = [
    "IConversationSecondaryIndexes",
    "IConversationThreads",
    "IMessageTextIndex",
    "IPropertyToSemanticRefIndex",
    "ITermToRelatedTerms",
    "ITermToRelatedTermsFuzzy",
    "ITermToRelatedTermsIndex",
    "ITimestampToTextRangeIndex",
    "ScoredThreadOrdinal",
    "Thread",
    "ThreadOrdinal",
    "TimestampedTextRange",
]


@runtime_checkable
class IPropertyToSemanticRefIndex(Protocol):
    """Allows for faster retrieval of name, value properties."""

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


class ITimestampToTextRangeIndex(Protocol):
    """Return text ranges over a date range."""

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

    async def serialize(self) -> TermToRelatedTermsData: ...

    async def deserialize(self, data: TermToRelatedTermsData | None) -> None: ...


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

    async def serialize(self) -> TermsToRelatedTermsIndexData: ...

    async def deserialize(self, data: TermsToRelatedTermsIndexData) -> None: ...


@dataclass
class Thread:
    """A conversation thread consisting of a description and associated text ranges."""

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

    def serialize(self) -> ConversationThreadData[ThreadDataItem]: ...

    def deserialize(self, data: ConversationThreadData[ThreadDataItem]) -> None: ...


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

    async def serialize(self) -> MessageTextIndexData: ...

    async def deserialize(self, data: MessageTextIndexData) -> None: ...


class IConversationSecondaryIndexes[TMessage: IMessage](Protocol):
    property_to_semantic_ref_index: IPropertyToSemanticRefIndex | None
    timestamp_index: ITimestampToTextRangeIndex | None
    term_to_related_terms_index: ITermToRelatedTermsIndex | None
    threads: IConversationThreads | None = None
    message_index: IMessageTextIndex[TMessage] | None = None


__all__ = [
    "IConversationSecondaryIndexes",
    "IConversationThreads",
    "IMessageTextIndex",
    "IPropertyToSemanticRefIndex",
    "ITermToRelatedTerms",
    "ITermToRelatedTermsFuzzy",
    "ITermToRelatedTermsIndex",
    "ITimestampToTextRangeIndex",
    "ScoredThreadOrdinal",
    "Thread",
    "ThreadOrdinal",
    "TimestampedTextRange",
]
