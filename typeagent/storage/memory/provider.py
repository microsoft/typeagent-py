# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Memory storage provider implementation."""

from .collections import MemoryMessageCollection, MemorySemanticRefCollection
from .semrefindex import TermToSemanticRefIndex
from .convthreads import ConversationThreads
from .messageindex import MessageTextIndex
from .reltermsindex import RelatedTermsIndex
from .propindex import PropertyIndex
from .timestampindex import TimestampToTextRangeIndex
from ...knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from ...knowpro.interfaces import (
    IConversationThreads,
    IMessage,
    IMessageTextIndex,
    IPropertyToSemanticRefIndex,
    IStorageProvider,
    ITermToRelatedTermsIndex,
    ITermToSemanticRefIndex,
    ITimestampToTextRangeIndex,
)


class MemoryStorageProvider[TMessage: IMessage](IStorageProvider[TMessage]):
    """A storage provider that operates in memory."""

    _message_collection: MemoryMessageCollection[TMessage]
    _semantic_ref_collection: MemorySemanticRefCollection

    _conversation_index: TermToSemanticRefIndex
    _property_index: PropertyIndex
    _timestamp_index: TimestampToTextRangeIndex
    _message_text_index: MessageTextIndex
    _related_terms_index: RelatedTermsIndex
    _conversation_threads: ConversationThreads

    def __init__(
        self,
        message_text_settings: MessageTextIndexSettings,
        related_terms_settings: RelatedTermIndexSettings,
    ) -> None:
        """Create and initialize a MemoryStorageProvider with all indexes."""
        self._message_collection = MemoryMessageCollection[TMessage]()
        self._semantic_ref_collection = MemorySemanticRefCollection()

        self._conversation_index = TermToSemanticRefIndex()
        self._property_index = PropertyIndex()
        self._timestamp_index = TimestampToTextRangeIndex()
        self._message_text_index = MessageTextIndex(message_text_settings)
        self._related_terms_index = RelatedTermsIndex(related_terms_settings)
        thread_settings = message_text_settings.embedding_index_settings
        self._conversation_threads = ConversationThreads(thread_settings)

    async def __aenter__(self) -> "MemoryStorageProvider[TMessage]":
        """Enter transaction context. No-op for in-memory storage."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit transaction context. No-op for in-memory storage."""
        pass

    async def get_semantic_ref_index(self) -> ITermToSemanticRefIndex:
        return self._conversation_index

    async def get_property_index(self) -> IPropertyToSemanticRefIndex:
        return self._property_index

    async def get_timestamp_index(self) -> ITimestampToTextRangeIndex:
        return self._timestamp_index

    async def get_message_text_index(self) -> IMessageTextIndex[TMessage]:
        return self._message_text_index

    async def get_related_terms_index(self) -> ITermToRelatedTermsIndex:
        return self._related_terms_index

    async def get_conversation_threads(self) -> IConversationThreads:
        return self._conversation_threads

    async def get_message_collection(
        self, message_type: type[TMessage] | None = None
    ) -> MemoryMessageCollection[TMessage]:
        return self._message_collection

    async def get_semantic_ref_collection(self) -> MemorySemanticRefCollection:
        return self._semantic_ref_collection

    async def close(self) -> None:
        """Close the storage provider."""
        pass
