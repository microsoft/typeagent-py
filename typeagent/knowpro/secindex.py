# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..aitools.embeddings import AsyncEmbeddingModel
from ..aitools.vectorbase import TextEmbeddingIndexSettings
from .convsettings import ConversationSettings
from .interfaces import (
    IConversation,
    IConversationSecondaryIndexes,
    IMessage,
    IStorageProvider,
    ITermToSemanticRefIndex,
    TextLocation,
)
from ..storage.memory.messageindex import build_message_index
from ..storage.memory.propindex import PropertyIndex, build_property_index
from ..storage.memory.reltermsindex import (
    RelatedTermsIndex,
    build_related_terms_index,
)
from .convsettings import RelatedTermIndexSettings
from ..storage.memory.timestampindex import (
    TimestampToTextRangeIndex,
    build_timestamp_index,
)


class ConversationSecondaryIndexes(IConversationSecondaryIndexes):
    def __init__(
        self,
        storage_provider: IStorageProvider,
        settings: RelatedTermIndexSettings,
    ):
        self._storage_provider = storage_provider
        # Initialize all indexes through storage provider immediately
        self.property_to_semantic_ref_index = None
        self.timestamp_index = None
        self.term_to_related_terms_index = None
        self.threads = None
        self.message_index = None

    @classmethod
    async def create(
        cls,
        storage_provider: IStorageProvider,
        settings: RelatedTermIndexSettings,
    ) -> "ConversationSecondaryIndexes":
        """Create and initialize a ConversationSecondaryIndexes with all indexes."""
        self = cls(storage_provider, settings)
        # Initialize all indexes from storage provider
        self.property_to_semantic_ref_index = (
            await storage_provider.get_property_index()
        )
        self.timestamp_index = await storage_provider.get_timestamp_index()
        self.term_to_related_terms_index = (
            await storage_provider.get_related_terms_index()
        )
        self.threads = await storage_provider.get_conversation_threads()
        self.message_index = await storage_provider.get_message_text_index()
        return self


async def build_secondary_indexes[
    TMessage: IMessage,
    TTermToSemanticRefIndex: ITermToSemanticRefIndex,
](
    conversation: IConversation[TMessage, TTermToSemanticRefIndex],
    conversation_settings: ConversationSettings,
) -> None:
    if conversation.secondary_indexes is None:
        storage_provider = await conversation_settings.get_storage_provider()
        conversation.secondary_indexes = await ConversationSecondaryIndexes.create(
            storage_provider, conversation_settings.related_term_index_settings
        )
    else:
        storage_provider = await conversation_settings.get_storage_provider()
    await build_transient_secondary_indexes(conversation, conversation_settings)
    await build_related_terms_index(
        conversation, conversation_settings.related_term_index_settings
    )
    if conversation.secondary_indexes is not None:
        await build_message_index(
            conversation,
            storage_provider,
        )


async def build_transient_secondary_indexes[
    TMessage: IMessage, TTermToSemanticRefIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TTermToSemanticRefIndex],
    settings: ConversationSettings,
) -> None:
    if conversation.secondary_indexes is None:
        conversation.secondary_indexes = await ConversationSecondaryIndexes.create(
            await settings.get_storage_provider(),
            (settings.related_term_index_settings),
        )
    await build_property_index(conversation)
    await build_timestamp_index(conversation)
