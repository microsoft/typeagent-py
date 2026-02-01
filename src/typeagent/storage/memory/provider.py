# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""In-memory storage provider implementation."""

from datetime import datetime

from ...knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from ...knowpro.interfaces import (
    ConversationMetadata,
    IConversationThreads,
    IMessage,
    IMessageTextIndex,
    IPropertyToSemanticRefIndex,
    IStorageProvider,
    ITermToRelatedTermsIndex,
    ITermToSemanticRefIndex,
    ITimestampToTextRangeIndex,
    STATUS_INGESTED,
)
from .collections import MemoryMessageCollection, MemorySemanticRefCollection
from .convthreads import ConversationThreads
from .messageindex import MessageTextIndex
from .propindex import PropertyIndex
from .reltermsindex import RelatedTermsIndex
from .semrefindex import TermToSemanticRefIndex
from .timestampindex import TimestampToTextRangeIndex


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
    _ingested_sources: set[str]

    def __init__(
        self,
        message_text_settings: MessageTextIndexSettings,
        related_terms_settings: RelatedTermIndexSettings,
        metadata: ConversationMetadata | None = None,
    ) -> None:
        """Create and initialize a MemoryStorageProvider with all indexes."""
        self._metadata = metadata or ConversationMetadata()
        self._message_collection = MemoryMessageCollection[TMessage]()
        self._semantic_ref_collection = MemorySemanticRefCollection()

        self._conversation_index = TermToSemanticRefIndex()
        self._property_index = PropertyIndex()
        self._timestamp_index = TimestampToTextRangeIndex()
        self._message_text_index = MessageTextIndex(message_text_settings)
        self._related_terms_index = RelatedTermsIndex(related_terms_settings)
        thread_settings = message_text_settings.embedding_index_settings
        self._conversation_threads = ConversationThreads(thread_settings)
        self._ingested_sources = set()

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

    def get_conversation_metadata(self) -> ConversationMetadata:
        """Get conversation metadata.

        For in-memory storage, returns the metadata provided during initialization
        or an empty ConversationMetadata instance if none was provided.
        """
        return self._metadata

    def set_conversation_metadata(self, **kwds: str | list[str] | None) -> None:
        """Set conversation metadata (no-op for in-memory storage).

        This method exists for API compatibility with SqliteStorageProvider
        but does nothing since in-memory storage doesn't persist metadata.

        Args:
            **kwds: Metadata keys and values (ignored)
        """
        pass

    def update_conversation_timestamps(
        self,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        """Update conversation timestamps (no-op for in-memory storage).

        This method exists for API compatibility with SqliteStorageProvider
        but does nothing since in-memory storage doesn't persist metadata.

        Args:
            created_at: Optional creation timestamp (ignored)
            updated_at: Optional last updated timestamp (ignored)
        """
        pass

    def is_source_ingested(self, source_id: str) -> bool:
        """Check if a source has already been ingested.

        Args:
            source_id: External source identifier (email ID, file path, etc.)

        Returns:
            True if the source has been ingested, False otherwise.
        """
        return source_id in self._ingested_sources

    def get_source_status(self, source_id: str) -> str | None:
        """Get the ingestion status of a source.

        Args:
            source_id: External source identifier (email ID, file path, etc.)

        Returns:
            The ingestion status if the source has been ingested, None otherwise.
        """
        if source_id in self._ingested_sources:
            return STATUS_INGESTED
        return None

    def mark_source_ingested(
        self, source_id: str, status: str = STATUS_INGESTED
    ) -> None:
        """Mark a source as ingested.

        Args:
            source_id: External source identifier (email ID, file path, etc.)
        """
        self._ingested_sources.add(source_id)
