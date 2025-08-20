# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Storage provider utilities.

This module provides utility functions for creating storage providers
without circular import issues.
"""

from ..knowpro.interfaces import IStorageProvider
from ..knowpro.messageindex import MessageTextIndexSettings
from ..knowpro.reltermsindex import RelatedTermIndexSettings


async def create_storage_provider(
    message_text_settings: MessageTextIndexSettings,
    related_terms_settings: RelatedTermIndexSettings,
    dbname: str | None = None,
) -> IStorageProvider:
    """
    Create a storage provider - MemoryStorageProvider if dbname is None,
    SqliteStorageProvider otherwise.
    """
    if dbname is None:
        from .memorystore import MemoryStorageProvider

        return await MemoryStorageProvider.create(
            message_text_settings, related_terms_settings
        )
    else:
        from .sqlitestore import SqliteStorageProvider

        return await SqliteStorageProvider.create(
            message_text_settings, related_terms_settings, dbname
        )
