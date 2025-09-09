# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Memory-based collection implementations."""

from typing import Iterable
from ...knowpro.interfaces import (
    ICollection,
    IMessage,
    ISemanticRefCollection,
    IMessageCollection,
    MessageOrdinal,
    SemanticRef,
    SemanticRefOrdinal,
)


class MemoryCollection[T, TOrdinal: int](ICollection[T, TOrdinal]):
    """A generic in-memory (non-persistent) collection class."""

    def __init__(self, items: list[T] | None = None):
        self.items: list[T] = items or []

    async def size(self) -> int:
        return len(self.items)

    def __aiter__(self):
        """Return an async iterator over the collection."""
        return self._async_iterator()

    async def _async_iterator(self):
        """Async generator that yields items from the collection."""
        for item in self.items:
            yield item

    async def get_item(self, arg: int) -> T:
        """Retrieve an item by its ordinal."""
        return self.items[arg]

    async def get_slice(self, start: int, stop: int) -> list[T]:
        """Retrieve a slice of items."""
        return self.items[start:stop]

    async def get_multiple(self, arg: list[TOrdinal]) -> list[T]:
        """Retrieve multiple items by their ordinals."""
        return [await self.get_item(ordinal) for ordinal in arg]

    @property
    def is_persistent(self) -> bool:
        return False

    async def append(self, item: T) -> None:
        """Append an item to the collection."""
        self.items.append(item)

    async def extend(self, items: Iterable[T]) -> None:
        """Extend the collection with multiple items."""
        self.items.extend(items)


class MemorySemanticRefCollection(MemoryCollection[SemanticRef, SemanticRefOrdinal]):
    """A collection of semantic references."""


class MemoryMessageCollection[TMessage: IMessage](
    MemoryCollection[TMessage, MessageOrdinal]
):
    """A collection of messages."""
