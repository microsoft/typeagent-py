"""Benchmarks for VectorBase.fuzzy_lookup_embedding.

Measures whether the numpy-vectorized path is meaningfully faster than
Python-level iteration at realistic conversation sizes (1K–10K vectors).

Requires ``pytest-benchmark`` (``uv pip install pytest-benchmark``)::

    uv run python -m pytest tests/benchmarks/test_benchmark_vectorbase.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.aitools.vectorbase import (
    ScoredInt,
    TextEmbeddingIndexSettings,
    VectorBase,
)

# -- Helpers ------------------------------------------------------------------


def make_vectorbase(n: int, dim: int = 384) -> tuple[VectorBase, np.ndarray]:
    """Create a VectorBase with *n* random normalized embeddings."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms

    settings = TextEmbeddingIndexSettings(embedding_model=create_test_embedding_model())
    vb = VectorBase(settings)
    vb.add_embeddings(None, vecs)

    query = rng.standard_normal(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    return vb, query


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def vb_1k() -> tuple[VectorBase, np.ndarray]:
    return make_vectorbase(1_000)


@pytest.fixture(scope="module")
def vb_10k() -> tuple[VectorBase, np.ndarray]:
    return make_vectorbase(10_000)


# -- Benchmarks ---------------------------------------------------------------


class TestFuzzyLookupEmbedding:
    """fuzzy_lookup_embedding at realistic conversation sizes."""

    def test_1k_vectors(self, benchmark, vb_1k: tuple[VectorBase, np.ndarray]) -> None:
        vb, query = vb_1k
        result = benchmark(vb.fuzzy_lookup_embedding, query, max_hits=10, min_score=0.0)
        assert len(result) == 10
        assert all(isinstance(r, ScoredInt) for r in result)

    def test_10k_vectors(
        self, benchmark, vb_10k: tuple[VectorBase, np.ndarray]
    ) -> None:
        vb, query = vb_10k
        result = benchmark(vb.fuzzy_lookup_embedding, query, max_hits=10, min_score=0.0)
        assert len(result) == 10


class TestFuzzyLookupEmbeddingInSubset:
    """fuzzy_lookup_embedding_in_subset — subset of 1K from 10K vectors."""

    def test_subset_1k_of_10k(
        self, benchmark, vb_10k: tuple[VectorBase, np.ndarray]
    ) -> None:
        vb, query = vb_10k
        rng = np.random.default_rng(99)
        subset = rng.choice(10_000, size=1_000, replace=False).tolist()
        result = benchmark(
            vb.fuzzy_lookup_embedding_in_subset,
            query,
            subset,
            max_hits=10,
            min_score=0.0,
        )
        assert len(result) == 10
