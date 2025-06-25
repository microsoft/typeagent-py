# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Optional, TypedDict, List
from dataclasses import dataclass
import numpy as np

from ..aitools.vectorbase import VectorBase, TextEmbeddingIndexSettings, ScoredInt
from ..aitools.embeddings import NormalizedEmbedding, NormalizedEmbeddings


class EmbeddingIndex:
    """Wrapper around VectorBase."""

    # TODO: Don't use self._vector_base._vectors directly; use VectorBase methods.

    def __init__(
        self,
        embeddings: NormalizedEmbeddings | None = None,
        settings: TextEmbeddingIndexSettings | None = None,
    ):
        # Use VectorBase for storage and operations on embeddings.
        settings = settings or TextEmbeddingIndexSettings()
        self._vector_base = VectorBase(settings)

        # Initialize with embeddings if provided.
        if embeddings is not None:
            for embedding in embeddings:
                self._vector_base.add_embedding(None, embedding)

    def __len__(self) -> int:
        return len(self._vector_base)

    async def get_embedding(self, key: str, cache: bool = True) -> NormalizedEmbedding:
        return await self._vector_base.get_embedding(key, cache)

    def get(self, pos: int) -> NormalizedEmbedding:
        return self._vector_base.get_embedding_at(pos)

    def push(self, embeddings: NormalizedEmbeddings) -> None:
        self._vector_base.add_embeddings(embeddings)

    async def add_texts(self, texts: list[str]) -> None:
        await self._vector_base.add_keys(texts)

    # def insert_at(
    #     self, index: int, embeddings: NormalizedEmbedding | list[NormalizedEmbedding]
    # ) -> None:
    #     """Insert one or more embeddings at the specified position.

    #     Args:
    #         index: Position to insert at
    #         embeddings: A single embedding or list of embeddings to insert
    #     """
    #     # Convert input to list
    #     emb_list = embeddings if isinstance(embeddings, list) else [embeddings]

    #     # Create a new array with space for the insertions
    #     old_vectors = self._vector_base._vectors
    #     size = len(old_vectors)

    #     if index < 0 or index > size:
    #         raise IndexError(
    #             f"Index {index} out of bounds for insertion in embedding index of size {size}"
    #         )

    #     # Convert embeddings to 2D array
    #     new_vectors = np.vstack([e.reshape(1, -1) for e in emb_list])

    #     # Split and recombine the vectors
    #     if index == 0:
    #         result = np.vstack([new_vectors, old_vectors])
    #     elif index >= size:
    #         result = np.vstack([old_vectors, new_vectors])
    #     else:
    #         result = np.vstack([old_vectors[:index], new_vectors, old_vectors[index:]])

    #     # Update the vector base
    #     self._vector_base._vectors = result

    def get_indexes_of_nearest(
        self,
        embedding: NormalizedEmbedding,
        max_matches: int | None = None,
        min_score: float | None = None,
        predicate: Callable[[int], bool] | None = None,
    ) -> list[ScoredInt]:
        return self._vector_base.fuzzy_lookup_embedding(
            embedding,
            max_hits=max_matches,
            min_score=min_score,
            predicate=predicate,
        )

    def get_indexes_of_nearest_in_subset(
        self,
        embedding: NormalizedEmbedding,
        ordinals_of_subset: list[int],
        max_matches: int | None = None,
        min_score: float | None = None,
    ) -> list[ScoredInt]:
        return self._vector_base.fuzzy_lookup_embedding_in_subset(
            embedding,
            ordinals_of_subset,
            max_matches,
            min_score,
        )

    # def remove_at(self, pos: int) -> None:
    #     """Remove the embedding at the specified position.

    #     Args:
    #         pos: The position to remove
    #     """
    #     if 0 <= pos < len(self._vector_base):
    #         # Create new array without the element at pos
    #         self._vector_base._vectors = np.delete(
    #             self._vector_base._vectors, pos, axis=0
    #         )
    #     else:
    #         raise IndexError(
    #             f"Index {pos} out of bounds for embedding index of size {len(self._vector_base)}"
    #         )

    def clear(self) -> None:
        self._vector_base.clear()

    def serialize(self) -> NormalizedEmbeddings:
        return self._vector_base.serialize()

    def deserialize(self, embeddings: NormalizedEmbedding) -> None:
        assert isinstance(embeddings, np.ndarray), type(embeddings)
        assert embeddings.dtype == np.float32, embeddings.dtype
        assert embeddings.ndim == 2, embeddings.shape
        assert (
            embeddings.shape[1] == self._vector_base._embedding_size
        ), embeddings.shape
        self.clear()
        self.push(embeddings)
