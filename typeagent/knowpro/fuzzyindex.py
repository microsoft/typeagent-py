# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Optional, TypedDict, List
from dataclasses import dataclass
import numpy as np

from ..aitools.vectorbase import VectorBase, TextEmbeddingIndexSettings, ScoredOrdinal
from ..aitools.embeddings import NormalizedEmbedding, NormalizedEmbeddings


@dataclass
class Scored:
    """An item with a similarity score."""

    item: int
    score: float


class EmbeddingIndex:
    """Index for storing and querying embeddings efficiently."""

    # TODO: Don't use self._vector_base._vectors directly; use VectorBase methods.

    def __init__(self, embeddings: NormalizedEmbeddings | None = None):
        """Initialize the embedding index.

        Args:
            embeddings: Optional 2D array of embeddings to initialize with
        """
        # Use VectorBase for storage and operations on embeddings
        settings = TextEmbeddingIndexSettings()
        self._vector_base = VectorBase(settings)

        # Initialize with embeddings if provided
        if embeddings is not None:
            for embedding in embeddings:
                self._vector_base.add_embedding(None, embedding)

    def __len__(self) -> int:
        """Get the number of embeddings in the index."""
        return len(self._vector_base)

    async def get_embedding(self, key: str, cache: bool = True) -> NormalizedEmbedding:
        return await self._vector_base.get_embedding(key, cache)

    def get(self, pos: int) -> NormalizedEmbedding:
        """Get the embedding at the specified position.

        Args:
            pos: The position to retrieve

        Returns:
            The normalized embedding at that position
        """
        if 0 <= pos < len(self._vector_base):
            return self._vector_base._vectors[pos]
        raise IndexError(
            f"Index {pos} out of bounds for embedding index of size {len(self._vector_base)}"
        )

    def push(self, embeddings: NormalizedEmbeddings) -> None:
        """Add one or more embeddings to the index.

        Args:
            embeddings: A 2D array whose 2nd dimension is self._embedding_size;
                its 1st dimension is the number of embeddings
        """
        assert (
            embeddings.ndim == 2
            and embeddings.shape[1] == self._vector_base._embedding_size
        )
        for embedding in embeddings:
            self._vector_base.add_embedding(None, embedding)

    def insert_at(
        self, index: int, embeddings: NormalizedEmbedding | list[NormalizedEmbedding]
    ) -> None:
        """Insert one or more embeddings at the specified position.

        Args:
            index: Position to insert at
            embeddings: A single embedding or list of embeddings to insert
        """
        # Convert input to list
        emb_list = embeddings if isinstance(embeddings, list) else [embeddings]

        # Create a new array with space for the insertions
        old_vectors = self._vector_base._vectors
        size = len(old_vectors)

        if index < 0 or index > size:
            raise IndexError(
                f"Index {index} out of bounds for insertion in embedding index of size {size}"
            )

        # Convert embeddings to 2D array
        new_vectors = np.vstack([e.reshape(1, -1) for e in emb_list])

        # Split and recombine the vectors
        if index == 0:
            result = np.vstack([new_vectors, old_vectors])
        elif index >= size:
            result = np.vstack([old_vectors, new_vectors])
        else:
            result = np.vstack([old_vectors[:index], new_vectors, old_vectors[index:]])

        # Update the vector base
        self._vector_base._vectors = result

    def get_indexes_of_nearest(
        self,
        embedding: NormalizedEmbedding,
        max_matches: int | None = None,
        min_score: float | None = None,
        predicate: Callable[[int], bool] | None = None,
    ) -> list[Scored]:
        """Find the indexes of embeddings nearest to the given embedding.

        Args:
            embedding: The embedding to compare against
            max_matches: Maximum number of matches to return
            min_score: Minimum similarity score required (0-1)
            predicate: Optional function to filter results by index

        Returns:
            List of matches with scores, sorted by descending score
        """
        scores = np.dot(self._vector_base._vectors, embedding)

        # Filter by predicate if provided
        if predicate:
            scored = [
                Scored(item=i, score=float(score))
                for i, score in enumerate(scores)
                if (min_score is None or score >= min_score) and predicate(i)
            ]
        else:
            scored = [
                Scored(item=i, score=float(score))
                for i, score in enumerate(scores)
                if (min_score is None or score >= min_score)
            ]

        # Sort by score in descending order
        scored.sort(key=lambda x: x.score, reverse=True)

        # Limit to max_matches if specified
        if max_matches is not None and max_matches > 0:
            return scored[:max_matches]

        return scored

    def get_indexes_of_nearest_in_subset(
        self,
        embedding: NormalizedEmbedding,
        ordinals_of_subset: list[int],
        max_matches: int | None = None,
        min_score: float | None = None,
    ) -> list[Scored]:
        """Find the indexes of nearest embeddings within a specified subset.

        Args:
            embedding: The embedding to compare against
            ordinals_of_subset: List of indexes defining the subset to search
            max_matches: Maximum number of matches to return
            min_score: Minimum similarity score required (0-1)

        Returns:
            List of matches with scores, sorted by descending score
        """
        # Create a subset of embeddings to search
        embeddings_to_search = np.vstack(
            [self._vector_base._vectors[i] for i in ordinals_of_subset]
        )

        # Get scores against the subset
        scores = np.dot(embeddings_to_search, embedding)

        # Create scored results
        scored = [
            Scored(item=ordinals_of_subset[i], score=float(score))
            for i, score in enumerate(scores)
            if min_score is None or score >= min_score
        ]

        # Sort by score in descending order
        scored.sort(key=lambda x: x.score, reverse=True)

        # Limit to max_matches if specified
        if max_matches is not None and max_matches > 0:
            return scored[:max_matches]

        return scored

    def remove_at(self, pos: int) -> None:
        """Remove the embedding at the specified position.

        Args:
            pos: The position to remove
        """
        if 0 <= pos < len(self._vector_base):
            # Create new array without the element at pos
            self._vector_base._vectors = np.delete(
                self._vector_base._vectors, pos, axis=0
            )
        else:
            raise IndexError(
                f"Index {pos} out of bounds for embedding index of size {len(self._vector_base)}"
            )

    def clear(self) -> None:
        """Remove all embeddings from the index."""
        self._vector_base.clear()

    def serialize(self) -> NormalizedEmbeddings:
        """Serialize the embeddings for storage.

        Returns:
            List of embeddings
        """
        return self._vector_base.serialize()

    def deserialize(self, embeddings: NormalizedEmbedding) -> None:
        """Deserialize stored embeddings.

        Args:
            embeddings: List of embeddings to load
        """
        assert isinstance(embeddings, np.ndarray), type(embeddings)
        assert embeddings.dtype == np.float32, embeddings.dtype
        assert embeddings.ndim == 2, embeddings.shape
        assert (
            embeddings.shape[1] == self._vector_base._embedding_size
        ), embeddings.shape
        self.clear()
        self.push(embeddings)
