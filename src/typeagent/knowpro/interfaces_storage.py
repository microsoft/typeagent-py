# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Storage provider and collection interfaces for knowpro."""

from __future__ import annotations

from collections.abc import AsyncIterable, Iterable
from datetime import datetime as Datetime
from typing import Any, Protocol, Self

from pydantic.dataclasses import dataclass

from typeagent.storage.utils import STATUS_INGESTED

from .interfaces_core import (
    IMessage,
    ITermToSemanticRefIndex,
    MessageOrdinal,
    SemanticRef,
    SemanticRefOrdinal,
)
from .interfaces_indexes import (
    IConversationSecondaryIndexes,
    IConversationThreads,
    IMessageTextIndex,
    IPropertyToSemanticRefIndex,
    ITermToRelatedTermsIndex,
    ITimestampToTextRangeIndex,
)

__all__ = [
    "ConversationMetadata",
    "IReadonlyCollection",
    "ICollection",
    "IMessageCollection",
    "ISemanticRefCollection",
    "IStorageProvider",
]


@dataclass
class ConversationMetadata:
    """Storage-provider-agnostic metadata for a conversation.

    This dataclass represents metadata that can be read from and written to
    any storage provider (SQLite, in-memory, etc.). Providers may store this
    internally in different formats (e.g., key-value pairs), but this provides
    a uniform interface for accessing conversation metadata.

    When passed to a storage provider during initialization:
    - None values indicate the provider should auto-generate/use defaults
    - Non-None values are used as-is

    When returned from get_conversation_metadata():
    - None values indicate the field was not found in the database
    - Non-None values are the actual stored values
    - If the database has no metadata rows, returns an instance with all fields None
    """

    name_tag: str | None = None
    schema_version: int | None = None
    created_at: Datetime | None = None
    updated_at: Datetime | None = None
    embedding_size: int | None = None
    embedding_model: str | None = None
    tags: list[str] | None = None
    extra: dict[str, str] | None = None


class IReadonlyCollection[T, TOrdinal](AsyncIterable[T], Protocol):
    async def size(self) -> int: ...

    async def get_item(self, arg: TOrdinal) -> T: ...

    async def get_slice(self, start: int, stop: int) -> list[T]: ...

    async def get_multiple(self, arg: list[TOrdinal]) -> list[T]: ...


class ICollection[T, TOrdinal](IReadonlyCollection[T, TOrdinal], Protocol):
    """An APPEND-ONLY collection."""

    @property
    def is_persistent(self) -> bool: ...

    async def append(self, item: T) -> None: ...

    async def extend(self, items: Iterable[T]) -> None:
        """Append multiple items to the collection."""
        # The default implementation just calls append for each item.
        for item in items:
            await self.append(item)


class IMessageCollection[TMessage: IMessage](
    ICollection[TMessage, MessageOrdinal], Protocol
):
    """A collection of Messages."""


class ISemanticRefCollection(ICollection[SemanticRef, SemanticRefOrdinal], Protocol):
    """A collection of SemanticRefs."""


class IStorageProvider[TMessage: IMessage](Protocol):
    """API spec for storage providers -- maybe in-memory or persistent."""

    async def get_message_collection(self) -> IMessageCollection[TMessage]: ...

    async def get_semantic_ref_collection(self) -> ISemanticRefCollection: ...

    # Index getters - ALL 6 index types for this conversation

    async def get_semantic_ref_index(self) -> ITermToSemanticRefIndex: ...

    async def get_property_index(self) -> IPropertyToSemanticRefIndex: ...

    async def get_timestamp_index(self) -> ITimestampToTextRangeIndex: ...

    async def get_message_text_index(self) -> IMessageTextIndex[TMessage]: ...

    async def get_related_terms_index(self) -> ITermToRelatedTermsIndex: ...

    async def get_conversation_threads(self) -> IConversationThreads: ...

    # Metadata management

    def get_conversation_metadata(self) -> ConversationMetadata:
        """Get conversation metadata (missing fields set to None)."""
        ...

    def set_conversation_metadata(self, **kwds: str | list[str] | None) -> None:
        """Set or update conversation metadata key-value pairs.
        Args:
            **kwds: Metadata keys and values where:
                - str value: Sets a single key-value pair (replaces existing)
                - list[str] value: Sets multiple values for the same key
                - None value: Deletes all rows for the given key
        """
        ...

    def update_conversation_timestamps(
        self,
        created_at: Datetime | None = None,
        updated_at: Datetime | None = None,
    ) -> None:
        """Update conversation timestamps."""
        ...

    # Ingested source tracking
    def is_source_ingested(self, source_id: str) -> bool:
        """Check if a source has already been ingested."""
        ...

    def get_source_status(self, source_id: str) -> str | None:
        """Get the ingestion status of a source."""
        ...

    def mark_source_ingested(
        self, source_id: str, status: str = STATUS_INGESTED
    ) -> None:
        """Mark a source as ingested (no commit; call within transaction context)."""
        ...

    # Transaction management
    async def __aenter__(self) -> Self:
        """Enter transaction context. Calls begin_transaction()."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit transaction context. Commits on success, rolls back on exception."""
        ...

    async def close(self) -> None: ...


class IConversation[
    TMessage: IMessage,
    TTermToSemanticRefIndex: ITermToSemanticRefIndex,
](Protocol):
    name_tag: str
    tags: list[str]
    messages: IMessageCollection[TMessage]
    semantic_refs: ISemanticRefCollection
    semantic_ref_index: TTermToSemanticRefIndex
    secondary_indexes: IConversationSecondaryIndexes[TMessage] | None


__all__ = [
    "ConversationMetadata",
    "ICollection",
    "IConversation",
    "IMessageCollection",
    "IReadonlyCollection",
    "ISemanticRefCollection",
    "IStorageProvider",
]
