# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ..storage.memory.messageindex import build_message_index
from ..storage.memory.propindex import build_property_index
from ..storage.memory.reltermsindex import build_related_terms_index
from ..storage.memory.timestampindex import build_timestamp_index
from .convsettings import ConversationSettings, RelatedTermIndexSettings
from .interfaces import (
    IConversation,
    IConversationSecondaryIndexes,
    IMessage,
    IStorageProvider,
    ITermToSemanticRefIndex,
)


class ConversationSecondaryIndexes(IConversationSecondaryIndexes):
    def __init__(
        self,
        storage_provider: IStorageProvider,
        settings: RelatedTermIndexSettings,
    ):
        self._storage_provider = storage_provider
        # Initialize all indexes from storage provider
        self.property_to_semantic_ref_index = storage_provider.property_index
        self.timestamp_index = storage_provider.timestamp_index
        self.term_to_related_terms_index = storage_provider.related_terms_index
        self.threads = storage_provider.conversation_threads
        self.message_index = storage_provider.message_text_index


async def build_secondary_indexes[
    TMessage: IMessage,
    TTermToSemanticRefIndex: ITermToSemanticRefIndex,
](
    conversation: IConversation[TMessage, TTermToSemanticRefIndex],
    conversation_settings: ConversationSettings,
) -> None:
    if conversation.secondary_indexes is None:
        storage_provider = await conversation_settings.get_storage_provider()
        conversation.secondary_indexes = ConversationSecondaryIndexes(
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
        conversation.secondary_indexes = ConversationSecondaryIndexes(
            await settings.get_storage_provider(),
            settings.related_term_index_settings,
        )
    await build_property_index(conversation)
    await build_timestamp_index(conversation)
