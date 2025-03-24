# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Protocol

from .importing import TextEmbeddingIndexSettings
from .interfaces import IConversationThreads, ScoredThreadOrdinal, Thread
from ..aitools.embeddings import NormalizedEmbedding
from ..aitools.vectorbase import VectorBase


class IThreadDataItem(Protocol):
    thread: Thread
    embedding: NormalizedEmbedding


class IConversationThreadData[TThreadDataItem: IThreadDataItem](Protocol):
    """Abstract interface for conversation thread data."""

    threads: list[TThreadDataItem] | None = None


class ConversationThreads(IConversationThreads):
    threads: list[Thread]
    vector_base: VectorBase

    def __init__(self, settings: TextEmbeddingIndexSettings | None = None):
        self.threads = []
        self.vector_base = VectorBase()  # TODO: pass settings

    async def add_thread(self, thread: Thread) -> None:
        assert len(self.threads) == len(self.vector_base)
        await self.vector_base.add_key(thread.description)
        self.threads.append(thread)

    async def lookup_thread(
        self,
        thread_description: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredThreadOrdinal]:
        matches = await self.vector_base.fuzzy_lookup(
            thread_description,
            max_matches,
            threshold_score,
        )
        return [
            ScoredThreadOrdinal(
                match.ordinal,
                match.score,
            )
            for match in matches
        ]

    def remove_thread(self, thread_ordinal: int) -> None:
        raise NotImplementedError  # TODO: Requires support for removal in vectorbase.py.

    def clear(self) -> None:
        self.threads = []
        self.vector_base.clear()

    async def build_index(self) -> None:
        self.vector_base.clear()  # Just in case
        await self.vector_base.add_keys([t.description for t in self.threads])
