# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

type NormalizedEmbedding = NDArray[np.float32]  # A single embedding
type NormalizedEmbeddings = NDArray[np.float32]  # An array of embeddings


@runtime_checkable
class IEmbedder(Protocol):
    """Minimal provider interface for embedding models.

    Implement this protocol to add support for a new embedding provider
    (e.g. Anthropic, Gemini, local models).  Only raw embedding computation
    is required; caching is handled by :class:`CachingEmbeddingModel`.

    The production implementation is
    :class:`~typeagent.aitools.model_adapters.PydanticAIEmbedder`.
    """

    @property
    def model_name(self) -> str: ...

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        """Compute a single embedding without caching."""
        ...

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        """Compute embeddings for a batch of strings without caching.

        Raises :class:`ValueError` if *input* is empty.
        """
        ...


@runtime_checkable
class IEmbeddingModel(Protocol):
    """Consumer-facing interface for embedding models with caching.

    This extends the provider interface (:class:`IEmbedder`) with caching
    methods.  Use :class:`CachingEmbeddingModel` to wrap an :class:`IEmbedder`
    and get a ready-to-use ``IEmbeddingModel``.
    """

    @property
    def model_name(self) -> str: ...

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        """Cache an already-computed embedding under the given key."""
        ...

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        """Compute a single embedding without caching."""
        ...

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        """Compute embeddings for a batch of strings without caching."""
        ...

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        """Retrieve a single embedding, using cache if available."""
        ...

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        """Retrieve embeddings for multiple keys, using cache if available."""
        ...


class CachingEmbeddingModel:
    """Wraps an :class:`IEmbedder` with an in-memory embedding cache.

    This shared base class implements the caching logic once, so individual
    embedding providers only need to implement the minimal :class:`IEmbedder`
    protocol (``get_embedding_nocache`` / ``get_embeddings_nocache``).
    """

    def __init__(self, embedder: IEmbedder) -> None:
        self._embedder = embedder
        self._cache: dict[str, NormalizedEmbedding] = {}

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        self._cache[key] = embedding

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        return await self._embedder.get_embedding_nocache(input)

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        return await self._embedder.get_embeddings_nocache(input)

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embedding = await self._embedder.get_embedding_nocache(key)
        self._cache[key] = embedding
        return embedding

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        if not keys:
            raise ValueError("Cannot embed an empty list")
        missing_keys = [k for k in keys if k not in self._cache]
        if missing_keys:
            fresh = await self._embedder.get_embeddings_nocache(missing_keys)
            for i, k in enumerate(missing_keys):
                self._cache[k] = fresh[i]
        return np.array([self._cache[k] for k in keys], dtype=np.float32)


TEST_MODEL_NAME = "test"

model_to_envvar: dict[str, str] = {
    "text-embedding-ada-002": "AZURE_OPENAI_ENDPOINT_EMBEDDING",
    "text-embedding-3-small": "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL",
    "text-embedding-3-large": "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE",
}
