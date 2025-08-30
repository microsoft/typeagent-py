# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Storage provider utilities.

This module provides utility functions for creating storage providers
without circular import issues.
"""

from ..knowpro.interfaces import IMessage, IStorageProvider
from ..knowpro.messageindex import MessageTextIndexSettings
from ..knowpro.reltermsindex import RelatedTermIndexSettings


async def create_storage_provider[TMessage: IMessage](
    message_text_settings: MessageTextIndexSettings,
    related_terms_settings: RelatedTermIndexSettings,
    dbname: str | None = None,
    message_type: type[TMessage] | None = None,
) -> IStorageProvider[TMessage]:
    """Create a storage provider.

    MemoryStorageProvider if dbname is None, SqliteStorageProvider otherwise.
    """
    if dbname is None:
        from .memorystore import MemoryStorageProvider

        return MemoryStorageProvider(message_text_settings, related_terms_settings)
    else:
        from .sqlitestore import SqliteStorageProvider

        if message_type is None:
            raise ValueError("Message type must be specified for SQLite storage")
        return await SqliteStorageProvider.create(
            message_text_settings, related_terms_settings, dbname, message_type
        )
