# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    Iterator,
    Callable,
)

from .interfaces import (
    ICollection,
    IMessage,
    IStorageProvider,
    JsonSerializer,
    MessageOrdinal,
    SemanticRef,
    SemanticRefOrdinal,
)


class Collection[T, TOrdinal: int](ICollection[T, TOrdinal]):
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
        """Append items to the collection."""
        self.items.append(item)

    async def extend(self, items: Iterable[T]) -> None:
        """Extend the collection with multiple items."""
        self.items.extend(items)


class SemanticRefCollection(Collection[SemanticRef, SemanticRefOrdinal]):
    """A collection of semantic references."""


class MessageCollection[TMessage: IMessage](Collection[TMessage, MessageOrdinal]):
    """A collection of messages."""


class MemoryStorageProvider(IStorageProvider):
    """A storage provider that operates in memory."""

    def create_message_collection[TMessage: IMessage](
        self,
        serializer: JsonSerializer[TMessage] | type[TMessage] | None = None,
    ) -> MessageCollection[TMessage]:
        """Create a new message collection."""
        if isinstance(serializer, JsonSerializer):
            raise ValueError("MemoryStorageProvider does not use a serializer.")
        return MessageCollection[TMessage]()

    def create_semantic_ref_collection(self) -> SemanticRefCollection:
        """Create a new semantic reference collection."""
        return SemanticRefCollection()

    def close(self) -> None:
        """Close the storage provider."""
        pass


# TODO: The rest of this file is not currently used.


@dataclass
class Batch[T]:
    """A batch of items from a collection."""

    start_at: int
    value: list[T]


async def get_batches_from_collection[T](
    collection: Collection[T, int],
    start_at_ordinal: int,
    batch_size: int,
) -> list[Batch[T]]:
    """Generate batches of items from a collection."""
    start_at = start_at_ordinal
    result = []
    while True:
        batch = await collection.get_slice(start_at, start_at + batch_size)
        if not batch:
            break
        result.append(Batch(start_at=start_at, value=batch))
        start_at += batch_size
    return result


async def map_collection[T](
    collection: Collection[T, int],
    callback: Callable[[T, int], T],
) -> list[T]:
    """Map a callback function over a collection."""
    results: list[T] = []
    size = await collection.size()
    for i in range(size):
        item = await collection.get_item(i)
        results.append(callback(item, i))
    return results


async def filter_collection[T](
    collection: Collection[T, int],
    predicate: Callable[[T, int], bool],
) -> list[T]:
    """Filter items in a collection based on a predicate."""
    results: list[T] = []
    size = await collection.size()
    for i in range(size):
        item = await collection.get_item(i)
        if predicate(item, i):
            results.append(item)
    return results
