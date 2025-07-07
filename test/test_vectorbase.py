# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import numpy as np

from typeagent.aitools.vectorbase import (
    VectorBase,
    TextEmbeddingIndexSettings,
)
from typeagent.aitools.embeddings import AsyncEmbeddingModel, NormalizedEmbedding

from fixtures import needs_auth


@pytest.fixture
def vector_base(scope="function") -> VectorBase:
    """Fixture to create a VectorBase instance with default settings."""
    return make_vector_base()


def make_vector_base() -> VectorBase:
    settings = TextEmbeddingIndexSettings(AsyncEmbeddingModel(model_name="test"))
    return VectorBase(settings)


type Samples = dict[str, NormalizedEmbedding]


@pytest.fixture(scope="function")
def sample_embeddings() -> Samples:
    """Fixture to provide sample embeddings for testing."""
    return {
        "word1": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "word2": np.array([0.4, 0.5, 0.6], dtype=np.float32),
        "word3": np.array([0.7, 0.8, 0.9], dtype=np.float32),
    }


def test_add_embedding(vector_base: VectorBase, sample_embeddings: Samples, needs_auth):
    """Test adding embeddings to the VectorBase."""
    for key, embedding in sample_embeddings.items():
        vector_base.add_embedding(key, embedding)

    assert len(vector_base) == len(sample_embeddings)
    for i, (key, embedding) in enumerate(sample_embeddings.items()):
        np.testing.assert_array_equal(vector_base.serialize_embedding_at(i), embedding)


@pytest.mark.asyncio
async def test_add_key(vector_base: VectorBase, sample_embeddings: Samples, needs_auth):
    """Test adding keys to the VectorBase."""
    for key in sample_embeddings:
        await vector_base.add_key(key)

    assert len(vector_base) == len(sample_embeddings)


@pytest.mark.asyncio
async def test_add_key_no_cache(
    vector_base: VectorBase, sample_embeddings: Samples, needs_auth
):
    """Test adding keys to the VectorBase with cache disabled."""
    for key in sample_embeddings:
        await vector_base.add_key(key, cache=False)

    assert len(vector_base) == len(sample_embeddings)
    assert (
        vector_base._model._embedding_cache == {}
    ), "Cache should remain empty when cache=False"


@pytest.mark.asyncio
async def test_add_keys(
    vector_base: VectorBase, sample_embeddings: Samples, needs_auth
):
    """Test adding multiple keys to the VectorBase."""
    keys = list(sample_embeddings.keys())
    await vector_base.add_keys(keys)

    assert len(vector_base) == len(sample_embeddings)


@pytest.mark.asyncio
async def test_add_keys_no_cache(
    vector_base: VectorBase, sample_embeddings: Samples, needs_auth
):
    """Test adding multiple keys to the VectorBase with cache disabled."""
    keys = list(sample_embeddings.keys())
    await vector_base.add_keys(keys, cache=False)

    assert len(vector_base) == len(sample_embeddings)
    assert (
        vector_base._model._embedding_cache == {}
    ), "Cache should remain empty when cache=False"


@pytest.mark.asyncio
async def test_fuzzy_lookup(
    vector_base: VectorBase, sample_embeddings: Samples, needs_auth
):
    """Test fuzzy lookup functionality."""
    for key in sample_embeddings:
        await vector_base.add_key(key)

    results = await vector_base.fuzzy_lookup("word1", max_hits=2, min_score=0.0)
    assert len(results) == 2
    assert results[0].item == 0
    assert results[0].score > 0.9  # High similarity score for the same word


def test_clear(vector_base: VectorBase, sample_embeddings: Samples):
    """Test clearing the VectorBase."""
    for key, embedding in sample_embeddings.items():
        vector_base.add_embedding(key, embedding)

    assert len(vector_base) == len(sample_embeddings)
    vector_base.clear()
    assert len(vector_base) == 0


def test_serialize_deserialize(
    vector_base: VectorBase, sample_embeddings: Samples, needs_auth
):
    """Test serialization and deserialization of the VectorBase."""
    for key, embedding in sample_embeddings.items():
        vector_base.add_embedding(key, embedding)

    serialized = vector_base.serialize()
    new_vector_base = make_vector_base()
    new_vector_base.deserialize(serialized)

    assert len(new_vector_base) == len(vector_base)
    for i in range(len(vector_base)):
        np.testing.assert_array_equal(
            new_vector_base.serialize_embedding_at(i),
            vector_base.serialize_embedding_at(i),
        )


def test_vectorbase_bool(vector_base: VectorBase):
    """__bool__ should always return True."""
    assert bool(vector_base) is True


def test_get_embedding_at(
    vector_base: VectorBase, sample_embeddings: Samples, needs_auth
):
    """Test get_embedding_at returns correct embedding and raises IndexError."""
    for key, embedding in sample_embeddings.items():
        vector_base.add_embedding(key, embedding)
    # Check retrieval
    for i, embedding in enumerate(sample_embeddings.values()):
        result = vector_base.get_embedding_at(i)
        np.testing.assert_array_equal(result, embedding)
    # Out of bounds
    with pytest.raises(IndexError):
        vector_base.get_embedding_at(len(sample_embeddings))


def test_fuzzy_lookup_embedding_in_subset(
    vector_base: VectorBase, sample_embeddings: Samples, needs_auth
):
    """Test fuzzy_lookup_embedding_in_subset returns best match in subset or None."""
    keys = list(sample_embeddings.keys())
    for key, embedding in sample_embeddings.items():
        vector_base.add_embedding(key, embedding)
    # Query close to first embedding
    query = sample_embeddings[keys[0]]
    subset = list(range(len(keys)))
    result = vector_base.fuzzy_lookup_embedding_in_subset(query, subset)
    # Should return a non-empty list of ScoredInt, with the closest index first
    assert isinstance(result, list)
    assert len(result) > 0
    assert hasattr(result[0], "item")
    # The closest embedding should be among the results; check if 0 is present
    items = [scored.item for scored in result]
    assert 0 in items

    # Subset restricts to one index
    result = vector_base.fuzzy_lookup_embedding_in_subset(query, [1])
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].item == 1

    # Empty subset returns empty list
    result = vector_base.fuzzy_lookup_embedding_in_subset(query, [])
    assert result == []
