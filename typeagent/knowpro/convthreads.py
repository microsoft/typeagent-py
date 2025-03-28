# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .interfaces import (
    IConversationThreadData,
    IConversationThreads,
    IThreadDataItem,
    ScoredThreadOrdinal,
    Thread,
    ThreadData,
)
from ..aitools.vectorbase import TextEmbeddingIndexSettings, VectorBase


class ConversationThreads(IConversationThreads):
    threads: list[Thread]
    vector_base: VectorBase

    def __init__(self, settings: TextEmbeddingIndexSettings | None = None):
        self.threads = []
        self.vector_base = VectorBase(settings)

    async def add_thread(self, thread: Thread) -> None:
        assert len(self.threads) == len(self.vector_base)
        await self.vector_base.add_key(thread.description, cache=False)
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

    def clear(self) -> None:
        self.threads = []
        self.vector_base.clear()

    async def build_index(self) -> None:
        self.vector_base.clear()  # Just in case
        await self.vector_base.add_keys(
            [t.description for t in self.threads], cache=False
        )

    def serialize(self) -> IConversationThreadData[IThreadDataItem]:
        thread_data: list[IThreadDataItem] = []
        embedding_index = self.vector_base

        for i, thread in enumerate(self.threads):
            thread_data.append(
                IThreadDataItem(
                    thread=thread.serialize(),
                    embedding=embedding_index.serialize_embedding_at(i),
                )
            )

        return IConversationThreadData(threads=thread_data)
