# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory functions for creating conversation objects."""

from ..storage.utils import create_storage_provider
from . import secindex
from .conversation_base import ConversationBase
from .convsettings import ConversationSettings
from .interfaces import IMessage


async def create_conversation[TMessage: IMessage](
    dbname: str | None,
    message_type: type[TMessage],
    name: str = "",
    tags: list[str] | None = None,
    settings: ConversationSettings | None = None,
) -> ConversationBase[TMessage]:
    """
    Create a conversation with the given message type and settings.

    Args:
        dbname: Database name for storage (None for in-memory storage)
        message_type: The type of messages this conversation will contain
        name: Optional name for the conversation
        tags: Optional list of tags for the conversation
        settings: Optional conversation settings (creates default if None)

    Returns:
        A fully initialized conversation ready to accept messages
    """
    if settings is None:
        settings = ConversationSettings()
        # Enable knowledge extraction by default for new conversations
        settings.semantic_ref_index_settings.auto_extract_knowledge = True

    storage_provider = await create_storage_provider(
        message_text_settings=settings.message_text_index_settings,
        related_terms_settings=settings.related_term_index_settings,
        dbname=dbname,
        message_type=message_type,
    )

    settings.storage_provider = storage_provider

    conversation = ConversationBase(
        settings=settings,
        name=name,
        tags=tags if tags is not None else [],
    )
    conversation.storage_provider = storage_provider
    conversation.messages = await storage_provider.get_message_collection()
    conversation.semantic_refs = await storage_provider.get_semantic_ref_collection()
    conversation.semantic_ref_index = await storage_provider.get_semantic_ref_index()
    conversation.secondary_indexes = await secindex.ConversationSecondaryIndexes.create(
        storage_provider, settings.related_term_index_settings
    )
    return conversation
