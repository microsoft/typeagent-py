# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmarks for VectorBase fuzzy lookup methods.

Measures fuzzy_lookup_embedding and fuzzy_lookup_embedding_in_subset
with varying vector counts and result sizes.
"""

import numpy as np
import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings, VectorBase

EMBEDDING_DIM = 384  # Typical small embedding model dimension


def make_populated_vector_base(n_vectors: int) -> tuple[VectorBase, np.ndarray]:
    """Create a VectorBase with n_vectors random normalized embeddings."""
    settings = TextEmbeddingIndexSettings(create_test_embedding_model())
    vb = VectorBase(settings)
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((n_vectors, EMBEDDING_DIM)).astype(np.float32)
    # Normalize to unit vectors (as the real pipeline does).
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    vb.add_embeddings(None, embeddings)
    # Query vector: also normalized.
    query = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    query = query / np.linalg.norm(query)
    return vb, query


# --- fuzzy_lookup_embedding ---


@pytest.mark.asyncio
async def test_benchmark_fuzzy_lookup_1k(async_benchmark):
    vb, query = make_populated_vector_base(1_000)

    async def target():
        vb.fuzzy_lookup_embedding(query, max_hits=10, min_score=0.0)

    await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)


@pytest.mark.asyncio
async def test_benchmark_fuzzy_lookup_10k(async_benchmark):
    vb, query = make_populated_vector_base(10_000)

    async def target():
        vb.fuzzy_lookup_embedding(query, max_hits=10, min_score=0.0)

    await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)


@pytest.mark.asyncio
async def test_benchmark_fuzzy_lookup_10k_with_predicate(async_benchmark):
    vb, query = make_populated_vector_base(10_000)
    # Predicate that accepts ~50% of indices.
    even_only = lambda i: i % 2 == 0

    async def target():
        vb.fuzzy_lookup_embedding(
            query, max_hits=10, min_score=0.0, predicate=even_only
        )

    await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)


# --- fuzzy_lookup_embedding_in_subset ---


@pytest.mark.asyncio
async def test_benchmark_fuzzy_lookup_subset_1k_of_10k(async_benchmark):
    vb, query = make_populated_vector_base(10_000)
    subset = list(range(0, 10_000, 10))  # 1000 indices

    async def target():
        vb.fuzzy_lookup_embedding_in_subset(query, subset, max_hits=10, min_score=0.0)

    await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)
