# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .embeddings import (
    IEmbeddingModel,
    NormalizedEmbedding,
    NormalizedEmbeddings,
)
from .model_adapters import create_embedding_model

DEFAULT_MIN_SCORE = 0.85

# Empirical defaults for built-in OpenAI embedding models.
# These values come from the Adrian Tchaikovsky Episode 53
# search benchmark in `tools/repeat_embedding_benchmarks.py`, using an
# exhaustive 0.00..1.00 min_score sweep. The benchmark recomputes corpus and
# query embeddings for each model and ignores the fixture's serialized
# embedding sidecar. Scores are normalized from cosine similarity to the public
# 0..1 min_score scale.
# These are repository defaults for known models, not universal truths.
# Unknown models keep the long-standing fallback score of 0.85. Callers can
# always override `min_score` explicitly for their own use cases or models. We
# intentionally leave `max_matches` out of this table: the benchmark still
# reports a best `max_hits` row, but the library default remains `None` unless
# a caller opts into a specific limit.
MODEL_DEFAULT_MIN_SCORES: dict[str, float] = {
    "text-embedding-3-large": 0.74,
    "text-embedding-3-small": 0.73,
    "text-embedding-ada-002": 0.93,
}


def get_default_min_score(model_name: str) -> float:
    """Return the repository default score cutoff for a known model name."""

    return MODEL_DEFAULT_MIN_SCORES.get(model_name, DEFAULT_MIN_SCORE)


def cosine_to_score(cosine_similarity: np.ndarray) -> np.ndarray:
    """Map cosine similarity from -1..1 to the public 0..1 score scale."""

    return np.clip((cosine_similarity + 1.0) / 2.0, 0.0, 1.0)


@dataclass
class ScoredInt:
    """Associate an integer ordinal with its similarity score."""

    item: int
    score: float


@dataclass
class TextEmbeddingIndexSettings:
    """Runtime settings for embedding-backed fuzzy lookup."""

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
        self.max_matches = max_matches if max_matches and max_matches >= 1 else None
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
        embedding_array = np.asarray(embedding, dtype=np.float32)
        if self._embedding_size == 0:
            self._set_embedding_size(len(embedding_array))
            self._vectors.shape = (0, self._embedding_size)
        if len(embedding_array) != self._embedding_size:
            raise ValueError(
                f"Embedding size mismatch: expected {self._embedding_size}, "
                f"got {len(embedding_array)}"
            )
        embeddings = embedding_array.reshape(1, -1)  # Make it 2D: 1xN
        self._vectors = np.append(self._vectors, embeddings, axis=0)
        if key is not None:
            self._model.add_embedding(key, embedding_array)

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
        scores = cosine_to_score(np.dot(self._vectors, embedding))
        if predicate is None:
            # Stay in numpy: filter by score, then top-k via argpartition.
            indices = np.flatnonzero(scores >= min_score)
            if len(indices) == 0:
                return []
            filtered_scores = scores[indices]
            if len(indices) <= max_hits:
                order = np.argsort(filtered_scores)[::-1]
            else:
                top_k = np.argpartition(filtered_scores, -max_hits)[-max_hits:]
                order = top_k[np.argsort(filtered_scores[top_k])[::-1]]
            return [
                ScoredInt(int(indices[i]), float(filtered_scores[i])) for i in order
            ]
        else:
            # Predicate path: pre-filter by score in numpy, apply predicate
            # only to candidates above the threshold.
            candidates = np.flatnonzero(scores >= min_score)
            scored_ordinals = [
                ScoredInt(int(i), float(scores[i]))
                for i in candidates
                if predicate(int(i))
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
        if not ordinals_of_subset or len(self._vectors) == 0:
            return []
        # Compute dot products only for the subset instead of all vectors.
        subset = np.asarray(ordinals_of_subset)
        scores = cosine_to_score(np.dot(self._vectors[subset], embedding))
        indices = np.flatnonzero(scores >= min_score)
        if len(indices) == 0:
            return []
        filtered_scores = scores[indices]
        if len(indices) <= max_hits:
            order = np.argsort(filtered_scores)[::-1]
        else:
            top_k = np.argpartition(filtered_scores, -max_hits)[-max_hits:]
            order = top_k[np.argsort(filtered_scores[top_k])[::-1]]
        return [
            ScoredInt(int(subset[indices[i]]), float(filtered_scores[i])) for i in order
        ]

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
