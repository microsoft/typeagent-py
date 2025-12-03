# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Storage provider utilities.

This module provides utility functions for creating storage providers
without circular import issues.
"""

from ..knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from ..knowpro.interfaces import ConversationMetadata, IMessage, IStorageProvider
from .memory import MemoryStorageProvider
from .sqlite import SqliteStorageProvider


async def create_storage_provider[TMessage: IMessage](
    message_text_settings: MessageTextIndexSettings,
    related_terms_settings: RelatedTermIndexSettings,
    dbname: str | None = None,
    message_type: type[TMessage] | None = None,
    metadata: ConversationMetadata | None = None,
) -> IStorageProvider[TMessage]:
    """Create a storage provider.

    MemoryStorageProvider if dbname is None, SqliteStorageProvider otherwise.
    """
    if dbname is None:
        return MemoryStorageProvider(
            message_text_settings, related_terms_settings, metadata=metadata
        )
    else:
        if message_type is None:
            raise ValueError("Message type must be specified for SQLite storage")

        # Create the new provider directly (constructor is now synchronous)
        provider = SqliteStorageProvider(
            db_path=dbname,
            message_type=message_type,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
            metadata=metadata,
        )
        return provider
