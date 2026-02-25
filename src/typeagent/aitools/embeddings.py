# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

type NormalizedEmbedding = NDArray[np.float32]  # A single embedding
type NormalizedEmbeddings = NDArray[np.float32]  # An array of embeddings


@runtime_checkable
class IEmbeddingModel(Protocol):
    """Provider-agnostic interface for embedding models.

    Implement this protocol to add support for a new embedding provider
    (e.g. Anthropic, Gemini, local models).  The production implementation
    is :class:`~typeagent.aitools.model_adapters.PydanticAIEmbeddingModel`.
    """

    model_name: str
    embedding_size: int

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


DEFAULT_MODEL_NAME = "text-embedding-ada-002"
DEFAULT_EMBEDDING_SIZE = 1536  # Default embedding size (required for ada-002)
DEFAULT_ENVVAR = "AZURE_OPENAI_ENDPOINT_EMBEDDING"  # We support OpenAI and Azure OpenAI
TEST_MODEL_NAME = "test"

model_to_embedding_size_and_envvar: dict[str, tuple[int | None, str]] = {
    DEFAULT_MODEL_NAME: (DEFAULT_EMBEDDING_SIZE, DEFAULT_ENVVAR),
    "text-embedding-3-small": (1536, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"),
    "text-embedding-3-large": (3072, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE"),
    # For testing only, not a real model (insert real embeddings above)
    TEST_MODEL_NAME: (3, "SIR_NOT_APPEARING_IN_THIS_FILM"),
}
