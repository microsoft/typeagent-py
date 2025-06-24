# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .interfaces import (
    ConversationThreadData,
    IConversationThreads,
    ThreadDataItem,
    ScoredThreadOrdinal,
    Thread,
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
                match.item,
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

    def serialize(self) -> ConversationThreadData[ThreadDataItem]:
        embedding_index = self.vector_base

        thread_data: list[ThreadDataItem] = []
        for i, thread in enumerate(self.threads):
            emb = embedding_index.serialize_embedding_at(i)
            thread_data.append(
                ThreadDataItem(
                    thread=thread.serialize(),
                    embedding=list(emb) if emb is not None else None,
                )
            )

        return ConversationThreadData(threads=thread_data)

    def deserialize(self, data: ConversationThreadData[ThreadDataItem]) -> None:
        self.clear()
        thread_data = data.get("threads")
        if thread_data is None:
            return
        for item in thread_data:
            thread_data = item["thread"]
            embedding = item["embedding"]
            thread = Thread.deserialize(thread_data)
            self.threads.append(thread)
            if embedding is not None:
                # assert isinstance(embedding, list), "Expected embedding to be a list"
                self.vector_base.add_embedding(thread_data["description"], embedding)
