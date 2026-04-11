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

DEFAULT_MIN_SCORE = 0.25

# Empirical defaults for built-in OpenAI embedding models.
# These values come from repeated runs of the Adrian Tchaikovsky Episode 53
# search benchmark in `tools/benchmark_embeddings.py`, with raw outputs stored
# under `benchmark_results/`.
# They are intended as repository defaults for known models, not universal
# truths; callers can always override `min_score` explicitly for their own use
# cases or models.
MODEL_DEFAULT_MIN_SCORES: dict[str, float] = {
    "text-embedding-3-large": 0.25,
    "text-embedding-3-small": 0.25,
    "text-embedding-ada-002": 0.25,
}


def get_default_min_score(model_name: str) -> float:
    return MODEL_DEFAULT_MIN_SCORES.get(model_name, DEFAULT_MIN_SCORE)


@dataclass
class ScoredInt:
    item: int
    score: float


@dataclass
class TextEmbeddingIndexSettings:
    embedding_model: IEmbeddingModel
    min_score: float  # Between 0.0 and 1.0
    max_matches: int | None  # >= 1; None means no limit
    batch_size: int  # >= 1

    def __init__(
        self,
        embedding_model: IEmbeddingModel | None = None,
        min_score: float | None = None,
        max_matches: int | None = None,
        batch_size: int | None = None,
    ):
        self.embedding_model = embedding_model or create_embedding_model()
        model_name = getattr(self.embedding_model, "model_name", "")
        default_min_score = get_default_min_score(model_name)
        self.min_score = min_score if min_score is not None else default_min_score
        self.max_matches = max_matches  # None means no limit
        self.batch_size = batch_size if batch_size and batch_size >= 1 else 8


class VectorBase:
    settings: TextEmbeddingIndexSettings
    _vectors: NormalizedEmbeddings
    _model: IEmbeddingModel
    _embedding_size: int

    def __init__(self, settings: TextEmbeddingIndexSettings):
        self.settings = settings
        self._model = settings.embedding_model
        self._embedding_size = 0
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
        if len(embedding) != self._embedding_size:
            raise ValueError(
                f"Embedding size mismatch: expected {self._embedding_size}, "
                f"got {len(embedding)}"
            )
        embeddings = embedding.reshape(1, -1)  # Make it 2D: 1xN
        self._vectors = np.append(self._vectors, embeddings, axis=0)
        if key is not None:
            self._model.add_embedding(key, embedding)

    def add_embeddings(
        self, keys: None | list[str], embeddings: NormalizedEmbeddings
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D")
        if self._embedding_size == 0:
            self._set_embedding_size(embeddings.shape[1])
            self._vectors.shape = (0, self._embedding_size)
        if embeddings.shape[1] != self._embedding_size:
            raise ValueError(
                f"Embedding size mismatch: expected {self._embedding_size}, "
                f"got {embeddings.shape[1]}"
            )
        self._vectors = np.concatenate((self._vectors, embeddings), axis=0)
        if keys is not None:
            for key, embedding in zip(keys, embeddings):
                self._model.add_embedding(key, embedding)

    async def add_key(self, key: str, cache: bool = True) -> None:
        embedding = await self.get_embedding(key, cache=cache)
        self.add_embedding(key if cache else None, embedding)

    async def add_keys(self, keys: list[str], cache: bool = True) -> None:
        if not keys:
            return
        embeddings = await self.get_embeddings(keys, cache=cache)
        self.add_embeddings(keys if cache else None, embeddings)

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
        if len(self._vectors) == 0:
            return []
        # This line does most of the work:
        scores: Iterable[float] = np.dot(self._vectors, embedding)
        scored_ordinals = [
            ScoredInt(i, score)
            for i, score in enumerate(scores)
            if score >= min_score and (predicate is None or predicate(i))
        ]
        scored_ordinals.sort(key=lambda x: x.score, reverse=True)
        return scored_ordinals[:max_hits]

    def fuzzy_lookup_embedding_in_subset(
        self,
        embedding: NormalizedEmbedding,
        ordinals_of_subset: list[int],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[ScoredInt]:
        if max_hits is None:
            max_hits = 10
        if min_score is None:
            min_score = 0.0
        if len(self._vectors) == 0 or not ordinals_of_subset:
            return []

        subset_ordinals = np.fromiter(set(ordinals_of_subset), dtype=np.intp)
        if len(subset_ordinals) == 0:
            return []

        scores: Iterable[float] = np.dot(self._vectors[subset_ordinals], embedding)
        scored_ordinals = [
            ScoredInt(int(ordinal), score)
            for ordinal, score in zip(subset_ordinals, scores)
            if score >= min_score
        ]
        scored_ordinals.sort(key=lambda x: x.score, reverse=True)
        return scored_ordinals[:max_hits]

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
                # Empty data can't determine size; just clear.
                self.clear()
                return
            self._set_embedding_size(data.shape[1])
        assert data.shape == (len(data), self._embedding_size), [
            data.shape,
            self._embedding_size,
        ]
        self._vectors = data  # TODO: Should we make a copy?
