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
    MessageOrdinal,
    SemanticRef,
    SemanticRefOrdinal,
)


class Collection[T, TOrdinal: int](ICollection[T, TOrdinal]):
    """A generic collection class."""

    def __init__(self, items: list[T] | None = None):
        self.items: list[T] = items or []

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the collection."""
        return iter(self.items)

    def __getitem__(self, arg: Any) -> Any:
        if isinstance(arg, slice):
            assert arg.step in (None, 1)  # type: ignore  # slice weirdness
            return self._get_slice(arg.start, arg.stop)  # type: ignore  # Use of internal; slice weirdness
        if isinstance(arg, int):
            return self._get(arg)
        if isinstance(arg, list):
            return self._get_multiple(arg)
        raise TypeError(f"Invalid argument type for __getitem__: {type(arg)}")

    def _get(self, ordinal: int) -> T:
        """Retrieve an item by its ordinal."""
        return self.items[ordinal]

    def _get_slice(self, start: TOrdinal, end: TOrdinal) -> list[T]:
        """Retrieve a slice of items."""
        return self.items[start:end]

    def _get_multiple(self, ordinals: list[TOrdinal]) -> list[T]:
        """Retrieve multiple items by their ordinals."""
        return [self._get(ordinal) for ordinal in ordinals]

    def get_all(self) -> list[T]:
        """Retrieve all items in the collection."""
        return self.items

    @property
    def is_persistent(self) -> bool:
        return False

    def append(self, item: T) -> None:
        """Append items to the collection."""
        self.items.append(item)

    def extend(self, items: Iterable[T]) -> None:
        """Extend the collection with multiple items."""
        self.items.extend(items)


class SemanticRefCollection(Collection[SemanticRef, SemanticRefOrdinal]):
    """A collection of semantic references."""


class MessageCollection[TMessage: IMessage](Collection[TMessage, MessageOrdinal]):
    """A collection of messages."""


class MemoryStorageProvider[TMessage: IMessage]:
    """A storage provider that operates in memory."""

    def create_message_collection(self) -> MessageCollection[TMessage]:
        """Create a new message collection."""
        return MessageCollection[TMessage]()

    def create_semantic_ref_collection(self) -> SemanticRefCollection:
        """Create a new semantic reference collection."""
        return SemanticRefCollection()

    def close(self) -> None:
        """Close the storage provider."""
        pass


@dataclass
class Batch[T]:
    """A batch of items from a collection."""

    start_at: int
    value: list[T]


def get_batches_from_collection[T](
    collection: Collection[T, int],
    start_at_ordinal: int,
    batch_size: int,
) -> Iterable[Batch[T]]:
    """Generate batches of items from a collection."""
    start_at = start_at_ordinal
    while True:
        batch = collection._get_slice(start_at, start_at + batch_size)
        if not batch:
            break
        yield Batch(start_at=start_at, value=batch)
        start_at += batch_size


def map_collection[T](
    collection: Collection[T, int],
    callback: Callable[[T, int], T],
) -> list[T]:
    """Map a callback function over a collection."""
    results: list[T] = []
    for i, item in enumerate(collection):
        results.append(callback(item, i))
    return results


def filter_collection[T](
    collection: Collection[T, int],
    predicate: Callable[[T, int], bool],
) -> list[T]:
    """Filter items in a collection based on a predicate."""
    results: list[T] = []
    for i, item in enumerate(collection):
        if predicate(item, i):
            results.append(item)
    return results
