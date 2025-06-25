# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..aitools import utils
from .embeddings import AsyncEmbeddingModel, NormalizedEmbedding, NormalizedEmbeddings


@dataclass
class ScoredInt:
    item: int
    score: float


@dataclass
class TextEmbeddingIndexSettings:
    embedding_model: AsyncEmbeddingModel
    embedding_size: int  # Always embedding_model.embedding_size
    min_score: float
    max_matches: int | None
    retry_max_attempts: int = 2
    retry_delay: float = 2.0  # Seconds
    batch_size: int = 8

    def __init__(
        self,
        embedding_model: AsyncEmbeddingModel | None = None,
        embedding_size: int | None = None,
        min_score: float | None = None,
        max_matches: int | None = None,
    ):
        self.embedding_model = embedding_model or AsyncEmbeddingModel(embedding_size)
        self.embedding_size = self.embedding_model.embedding_size
        assert (
            embedding_size is None or self.embedding_size == embedding_size
        ), f"Given embedding size {embedding_size} doesn't match model's embedding size {self.embedding_size}"
        self.min_score = min_score if min_score is not None else 0.85
        self.max_matches = max_matches


class VectorBase:
    _vectors: NormalizedEmbeddings

    def __init__(self, settings: TextEmbeddingIndexSettings | None = None):
        model = settings.embedding_model if settings else None
        embedding_size = settings.embedding_size if settings else None
        if model is None:
            model = AsyncEmbeddingModel(embedding_size)
        self._model = model
        self._embedding_size = model.embedding_size
        self.clear()

    async def get_embedding(self, key: str, cache: bool = True) -> NormalizedEmbedding:
        if cache:
            return await self._model.get_embedding(key)
        else:
            return await self._model.get_embedding_nocache(key)

    async def get_embeddings(
        self, keys: list[str], cache: bool = True
    ) -> NormalizedEmbeddings:
        if cache:
            return await self._model.get_embeddings(keys)
        else:
            return await self._model.get_embeddings_nocache(keys)

    def __len__(self) -> int:
        return len(self._vectors)

    # Needed because otherwise an empty index would be falsy.
    def __bool__(self) -> bool:
        return True

    def add_embedding(
        self, key: str | None, embedding: NormalizedEmbedding | list[float]
    ) -> None:
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        embeddings = embedding.reshape(1, -1)  # Make it 2D
        self._vectors = np.append(self._vectors, embeddings, axis=0)
        if key is not None:
            self._model.add_embedding(key, embedding)

    def add_embeddings(self, embeddings: NormalizedEmbeddings) -> None:
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == self._embedding_size
        self._vectors = np.concatenate((self._vectors, embeddings), axis=0)

    async def add_key(self, key: str, cache: bool = True) -> None:
        embeddings = (await self.get_embedding(key, cache=cache)).reshape(
            1, -1
        )  # Make it 2D
        self._vectors = np.append(self._vectors, embeddings, axis=0)

    async def add_keys(self, keys: list[str], cache: bool = True) -> None:
        embeddings = await self.get_embeddings(keys, cache=cache)
        self._vectors = np.concatenate((self._vectors, embeddings), axis=0)

    def fuzzy_lookup_embedding(
        self,
        embedding: NormalizedEmbedding,
        max_hits: int | None = None,
        min_score: float | None = None,
        predicate: Callable[[int], bool] | None = None,
    ) -> list[ScoredInt]:
        if max_hits is None:
            max_hits = 10
        if min_score is None:
            min_score = 0.0
        # This line does most of the work:
        scores: Iterable[float] = np.dot(self._vectors, embedding)
        scored_ordinals = [
            ScoredInt(i, score)
            for i, score in enumerate(scores)
            if score >= min_score and (predicate is None or predicate(i))
        ]
        scored_ordinals.sort(key=lambda x: x.score, reverse=True)
        return scored_ordinals[:max_hits]

    # TODO: Make this and fizzy_lookup_embedding() more similar.
    def fuzzy_lookup_embedding_in_subset(
        self,
        embedding: NormalizedEmbedding,
        ordinals_of_subset: list[int],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[ScoredInt]:
        return self.fuzzy_lookup_embedding(
            embedding, max_hits, min_score, lambda i: i in ordinals_of_subset
        )

    async def fuzzy_lookup(
        self,
        key: str,
        max_hits: int | None = None,
        min_score: float | None = None,
        predicate: Callable[[int], bool] | None = None,
    ) -> list[ScoredInt]:
        embedding = await self.get_embedding(key)
        return self.fuzzy_lookup_embedding(
            embedding, max_hits=max_hits, min_score=min_score, predicate=predicate
        )

    def clear(self) -> None:
        self._vectors = np.array([], dtype=np.float32)
        self._vectors.shape = (0, self._embedding_size)

    def get_embedding_at(self, pos: int) -> NormalizedEmbedding:
        if 0 <= pos < len(self._vectors):
            return self._vectors[pos]
        raise IndexError(
            f"Index {pos} out of bounds for embedding index of size {len(self)}"
        )

    def serialize_embedding_at(self, pos: int) -> NormalizedEmbedding | None:
        return self._vectors[pos] if 0 <= pos < len(self._vectors) else None

    def serialize(self) -> NormalizedEmbeddings:
        assert self._vectors.shape == (len(self._vectors), self._embedding_size)
        return self._vectors  # TODO: Should we make a copy?

    def deserialize(self, data: NormalizedEmbeddings | None) -> None:
        if data is None:
            self.clear()
            return
        assert data.shape == (len(data), self._embedding_size), [
            data.shape,
            self._embedding_size,
        ]
        self._vectors = data  # TODO: Should we make a copy?


async def main():
    import time
    from . import auth

    epoch = time.time()

    def log(*args: object, end: str = "\n"):
        stamp = f"{time.time()-epoch:7.3f}"
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, str) and "\n" in arg:
                new_args[i] = arg.replace("\n", f"\n{stamp}: ")
        print(f"{stamp}:", *new_args, end=end)

    def debugv(heading: str):
        log(f"{heading}: bool={bool(v)}, len={len(v)}")

    utils.load_dotenv()
    v = VectorBase()
    debugv("\nEmpty vector base")

    words: list[str] = (
        "Mostly about multi-agent frameworks, "
        + "but also about answering questions about podcast transcripts."
    ).split()  # type: ignore  # pyright complains about list[LiteralString] -> list[str]
    cut = 2
    for word in words[:cut]:
        log("\nAdding:", word)
        await v.add_key(word)
        scored_ordinals = await v.fuzzy_lookup(word, max_hits=1)
        assert (
            round(scored_ordinals[0].score, 5) == 1.0
        ), f"{word} scores {scored_ordinals[0]}"
        debugv(word)

    log("\nAdding remaining words")
    await v.add_keys(words[cut:])
    debugv("After adding all")

    log("\nChecking presence")
    for word in words:
        scored_ordinals = await v.fuzzy_lookup(word, max_hits=1)
        assert (
            round(scored_ordinals[0].score, 5) == 1.0
        ), f"{word} scores {scored_ordinals[0]}"
    log("All words are present")
    word = "pancakes"
    scored_ordinals = await v.fuzzy_lookup(word, max_hits=1)
    assert scored_ordinals[0].score < 0.7, f"{word} scores {scored_ordinals[0]}"

    log("\nFuzzy lookups:")
    for word in words + ["pancakes", "hello world", "book", "author"]:
        neighbors = await v.fuzzy_lookup(word, max_hits=3)
        log(f"{word}:", [(nb.item, nb.score) for nb in neighbors])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
