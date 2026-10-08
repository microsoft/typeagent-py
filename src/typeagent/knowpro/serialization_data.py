# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""TypedDict helpers for serializing knowpro conversations and indexes."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from ..aitools.embeddings import NormalizedEmbeddings
from .interfaces_core import SemanticRefData, TextLocationData, TextRangeData

__all__ = [
    "ConversationData",
    "ConversationDataWithIndexes",
    "ConversationThreadData",
    "MessageTextIndexData",
    "ScoredSemanticRefOrdinalData",
    "TermData",
    "TermToRelatedTermsData",
    "TermsToRelatedTermsDataItem",
    "TermToSemanticRefIndexData",
    "TermToSemanticRefIndexItemData",
    "TermsToRelatedTermsIndexData",
    "TextEmbeddingIndexData",
    "TextToTextLocationIndexData",
    "ThreadData",
    "ThreadDataItem",
]


class ThreadData(TypedDict):
    description: str
    ranges: list[TextRangeData]


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
    semanticRefOrdinal: int
    score: float


class TermToSemanticRefIndexItemData(TypedDict):
    term: str
    semanticRefOrdinals: list[ScoredSemanticRefOrdinalData]


class TermToSemanticRefIndexData(TypedDict):
    """Persistent form of a term index."""

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


__all__ = [
    "ConversationData",
    "ConversationDataWithIndexes",
    "ConversationThreadData",
    "MessageTextIndexData",
    "ScoredSemanticRefOrdinalData",
    "TermData",
    "TermToRelatedTermsData",
    "TermsToRelatedTermsDataItem",
    "TermToSemanticRefIndexData",
    "TermToSemanticRefIndexItemData",
    "TermsToRelatedTermsIndexData",
    "TextEmbeddingIndexData",
    "TextToTextLocationIndexData",
    "ThreadData",
    "ThreadDataItem",
]
