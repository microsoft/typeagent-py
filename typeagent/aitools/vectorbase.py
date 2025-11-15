# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_RETRIES = 3  # simple replacement, matching OpenAI's default

from .embeddings import AsyncEmbeddingModel, NormalizedEmbedding, NormalizedEmbeddings


@dataclass
class ScoredInt:
    item: int
    score: float


@dataclass
class TextEmbeddingIndexSettings:
    embedding_model: AsyncEmbeddingModel
    embedding_size: int
    min_score: float
    max_matches: int | None
    batch_size: int
    max_retries: int

    def __init__(
        self,
        embedding_model: AsyncEmbeddingModel | None = None,
        embedding_size: int | None = None,
        min_score: float | None = None,
        max_matches: int | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
    ):
        self.min_score = min_score if min_score is not None else 0.85
        self.max_matches = max_matches if max_matches and max_matches >= 1 else None
        self.batch_size = batch_size if batch_size and batch_size >= 1 else 10
        self.max_retries = max_retries if max_retries is not None else DEFAULT_MAX_RETRIES

        # LiteLLM embedding model
        self.embedding_model = embedding_model or AsyncEmbeddingModel(
            embedding_size=embedding_size,
            max_retries=self.max_retries,
        )

        self.embedding_size = self.embedding_model.embedding_size
        assert (
            embedding_size is None or self.embedding_size == embedding_size
        ), f"Given embedding size {embedding_size} doesn't match model's size {self.embedding_size}"


class VectorBase:
    settings: TextEmbeddingIndexSettings
    _vectors: NormalizedEmbeddings
    _model: AsyncEmbeddingModel
    _embedding_size: int

    def __init__(self, settings: TextEmbeddingIndexSettings):
        self.settings = settings
        self._model = settings.embedding_model
        self._embedding_size = self._model.embedding_size
        self.clear()

    async def get_embedding(self, key: str, cache: bool = True) -> NormalizedEmbedding:
        if cache:
            return await self._model.get_embedding(key)
        return await self._model.get_embedding_nocache(key)

    async def get_embeddings(
        self, keys: list[str], cache: bool = True
    ) -> NormalizedEmbeddings:
        if cache:
            return await self._model.get_embeddings(keys)
        return await self._model.get_embeddings_nocache(keys)

    def __len__(self) -> int:
        return len(self._vectors)

    def __bool__(self) -> bool:
        return True

    def add_embedding(
        self, key: str | None, embedding: NormalizedEmbedding | list[float]
    ) -> None:
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        embedding = embedding.reshape(1, -1)
        self._vectors = np.append(self._vectors, embedding, axis=0)

        if key is not None:
            self._model.add_embedding(key, embedding[0])

    def add_embeddings(self, embeddings: NormalizedEmbeddings) -> None:
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == self._embedding_size
        self._vectors = np.concatenate((self._vectors, embeddings), axis=0)

    async def add_key(self, key: str, cache: bool = True) -> None:
        emb = (await self.get_embedding(key, cache)).reshape(1, -1)
        self._vectors = np.append(self._vectors, emb, axis=0)

    async def add_keys(self, keys: list[str], cache: bool = True) -> None:
        embeddings = await self.get_embeddings(keys, cache)
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

        scores: Iterable[float] = np.dot(self._vectors, embedding)

        scored = [
            ScoredInt(i, score)
            for i, score in enumerate(scores)
            if score >= min_score and (predicate is None or predicate(i))
        ]

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:max_hits]

    def fuzzy_lookup_embedding_in_subset(
        self,
        embedding: NormalizedEmbedding,
        ordinals_of_subset: list[int],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[ScoredInt]:
        return self.fuzzy_lookup_embedding(
            embedding,
            max_hits,
            min_score,
            predicate=lambda i: i in ordinals_of_subset,
        )

    async def fuzzy_lookup(
        self,
        key: str,
        max_hits: int | None = None,
        min_score: float | None = None,
        predicate: Callable[[int], bool] | None = None,
    ) -> list[ScoredInt]:

        if max_hits is None:
            max_hits = self.settings.max_matches
        if min_score is None:
            min_score = self.settings.min_score

        embedding = await self.get_embedding(key)
        return self.fuzzy_lookup_embedding(
            embedding,
            max_hits=max_hits,
            min_score=min_score,
            predicate=predicate,
        )

    def clear(self) -> None:
        self._vectors = np.array([], dtype=np.float32)
        self._vectors.shape = (0, self._embedding_size)

    def get_embedding_at(self, pos: int) -> NormalizedEmbedding:
        if 0 <= pos < len(self._vectors):
            return self._vectors[pos]
        raise IndexError(f"Index {pos} out of bounds")

    def serialize_embedding_at(self, pos: int) -> NormalizedEmbedding | None:
        return self._vectors[pos] if 0 <= pos < len(self._vectors) else None

    def serialize(self) -> NormalizedEmbeddings:
        return self._vectors

    def deserialize(self, data: NormalizedEmbeddings | None) -> None:
        if data is None:
            self.clear()
            return
        assert data.shape == (len(data), self._embedding_size)
        self._vectors = data
