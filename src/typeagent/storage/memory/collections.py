# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Memory-based collection implementations."""

from typing import Iterable

from ...knowpro.interfaces import (
    ICollection,
    IMessage,
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
        return self._async_iterator()

    async def _async_iterator(self):
        for item in self.items:
            yield item

    async def get_item(self, arg: int) -> T:
        return self.items[arg]

    async def get_slice(self, start: int, stop: int) -> list[T]:
        return self.items[start:stop]

    async def get_multiple(self, arg: list[TOrdinal]) -> list[T]:
        size = len(self.items)
        if not all((0 <= i < size) for i in arg):
            raise IndexError("One or more indices are out of bounds")
        return [self.items[ordinal] for ordinal in arg]

    @property
    def is_persistent(self) -> bool:
        return False

    async def append(self, item: T) -> None:
        self.items.append(item)

    async def extend(self, items: Iterable[T]) -> None:
        self.items.extend(items)


class MemorySemanticRefCollection(MemoryCollection[SemanticRef, SemanticRefOrdinal]):
    """A collection of semantic references."""


class MemoryMessageCollection[TMessage: IMessage](
    MemoryCollection[TMessage, MessageOrdinal]
):
    """A collection of messages."""
