# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .embeddings import (
    IEmbeddingModel,
    NormalizedEmbedding,
    NormalizedEmbeddings,
)
from .model_adapters import create_embedding_model

DEFAULT_MAX_RETRIES = 2


@dataclass
class ScoredInt:
    item: int
    score: float


@dataclass
class TextEmbeddingIndexSettings:
    embedding_model: IEmbeddingModel
    embedding_size: int  # Set to embedding_model.embedding_size
    min_score: float  # Between 0.0 and 1.0
    max_matches: int | None  # >= 1; None means no limit
    batch_size: int  # >= 1
    max_retries: int

    def __init__(
        self,
        embedding_model: IEmbeddingModel | None = None,
        embedding_size: int | None = None,
        min_score: float | None = None,
        max_matches: int | None = None,
        batch_size: int | None = None,
        max_retries: int | None = None,
    ):
        self.min_score = min_score if min_score is not None else 0.85
        self.max_matches = max_matches if max_matches and max_matches >= 1 else None
        self.batch_size = batch_size if batch_size and batch_size >= 1 else 8
        self.max_retries = (
            max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
        )
        self.embedding_model = embedding_model or create_embedding_model(
            embedding_size=embedding_size or 0,
        )
        self.embedding_size = self.embedding_model.embedding_size
        assert (
            embedding_size is None or self.embedding_size == embedding_size
        ), f"Given embedding size {embedding_size} doesn't match model's embedding size {self.embedding_size}"


class VectorBase:
    settings: TextEmbeddingIndexSettings
    _vectors: NormalizedEmbeddings
    _model: IEmbeddingModel
    _embedding_size: int

    def __init__(self, settings: TextEmbeddingIndexSettings):
        self.settings = settings
        self._model = settings.embedding_model
        self._embedding_size = self._model.embedding_size
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
        if self._embedding_size == 0:
            self._set_embedding_size(len(embedding))
            self._vectors.shape = (0, self._embedding_size)
        embeddings = embedding.reshape(1, -1)  # Make it 2D: 1xN
        self._vectors = np.append(self._vectors, embeddings, axis=0)
        if key is not None:
            self._model.add_embedding(key, embedding)

    def add_embeddings(
        self, keys: None | list[str], embeddings: NormalizedEmbeddings
    ) -> None:
        assert embeddings.ndim == 2
        if self._embedding_size == 0:
            self._set_embedding_size(embeddings.shape[1])
            self._vectors.shape = (0, self._embedding_size)
        assert embeddings.shape[1] == self._embedding_size
        self._vectors = np.concatenate((self._vectors, embeddings), axis=0)
        if keys is not None:
            for key, embedding in zip(keys, embeddings):
                self._model.add_embedding(key, embedding)

    async def add_key(self, key: str, cache: bool = True) -> None:
        embeddings = (await self.get_embedding(key, cache=cache)).reshape(1, -1)
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

    # TODO: Make this and fuzzy_lookup_embedding() more similar.
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
        if max_hits is None:
            max_hits = self.settings.max_matches
        if min_score is None:
            min_score = self.settings.min_score
        embedding = await self.get_embedding(key)
        return self.fuzzy_lookup_embedding(
            embedding, max_hits=max_hits, min_score=min_score, predicate=predicate
        )

    def _set_embedding_size(self, size: int) -> None:
        """Adopt *size* when it was not known at construction time."""
        assert size > 0
        self._embedding_size = size
        self.settings.embedding_size = size

    def clear(self) -> None:
        self._vectors = np.array([], dtype=np.float32)
        if self._embedding_size > 0:
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
        if self._embedding_size > 0:
            assert self._vectors.shape == (len(self._vectors), self._embedding_size)
        return self._vectors  # TODO: Should we make a copy?

    def deserialize(self, data: NormalizedEmbeddings | None) -> None:
        if data is None:
            self.clear()
            return
        if self._embedding_size == 0:
            if data.ndim < 2 or data.shape[0] == 0:
                # Empty data — can't determine size; just clear.
                self.clear()
                return
            self._set_embedding_size(data.shape[1])
        assert data.shape == (len(data), self._embedding_size), [
            data.shape,
            self._embedding_size,
        ]
        self._vectors = data  # TODO: Should we make a copy?
