# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Factory functions for creating conversation objects."""

from . import secindex
from .conversation_base import ConversationBase
from .convsettings import ConversationSettings
from .interfaces import IMessage


async def create_conversation[TMessage: IMessage](
    message_type: type[TMessage],
    name: str = "",
    tags: list[str] | None = None,
    settings: ConversationSettings | None = None,
) -> ConversationBase[TMessage]:
    """
    Create a conversation with the given message type and settings.

    Args:
        message_type: The type of messages this conversation will contain
        name: Optional name for the conversation
        tags: Optional list of tags for the conversation
        settings: Optional conversation settings (creates default if None)

    Returns:
        A fully initialized conversation ready to accept messages
    """
    if settings is None:
        settings = ConversationSettings()

    storage_provider = await settings.get_storage_provider()

    return ConversationBase(
        settings=settings,
        name_tag=name,
        messages=await storage_provider.get_message_collection(),
        semantic_refs=await storage_provider.get_semantic_ref_collection(),
        tags=tags if tags is not None else [],
        semantic_ref_index=await storage_provider.get_semantic_ref_index(),
        secondary_indexes=await secindex.ConversationSecondaryIndexes.create(
            storage_provider, settings.related_term_index_settings
        ),
    )
