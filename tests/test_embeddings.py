# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest
from pytest_mock import MockerFixture

from typeagent.aitools.embeddings import CachingEmbeddingModel, IEmbeddingModel

from conftest import (
    embedding_model,  # type: ignore  # Magic, prevents side effects of mocking
)


@pytest.mark.asyncio
async def test_get_embedding_nocache(embedding_model: CachingEmbeddingModel):
    """Test retrieving an embedding without using the cache."""
    input_text = "Hello, world"
    embedding = await embedding_model.get_embedding_nocache(input_text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.embedding_size,)
    assert embedding.dtype == np.float32


@pytest.mark.asyncio
async def test_get_embeddings_nocache(embedding_model: CachingEmbeddingModel):
    """Test retrieving multiple embeddings without using the cache."""
    inputs = ["Hello, world", "Foo bar baz"]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(inputs), embedding_model.embedding_size)
    assert embeddings.dtype == np.float32


@pytest.mark.asyncio
async def test_get_embedding_with_cache(
    embedding_model: CachingEmbeddingModel, mocker: MockerFixture
):
    """Test retrieving an embedding with caching."""
    input_text = "Hello, world"

    # First call should populate the cache
    embedding1 = await embedding_model.get_embedding(input_text)
    assert input_text in embedding_model._cache

    # Mock the nocache method on the underlying embedder to ensure it's not called
    mock_get_embedding_nocache = mocker.patch.object(
        embedding_model._embedder, "get_embedding_nocache", autospec=True
    )

    # Second call should retrieve from the cache
    embedding2 = await embedding_model.get_embedding(input_text)
    assert np.array_equal(embedding1, embedding2)

    # Ensure the nocache method was not called
    mock_get_embedding_nocache.assert_not_called()


@pytest.mark.asyncio
async def test_get_embeddings_with_cache(
    embedding_model: CachingEmbeddingModel, mocker: MockerFixture
):
    """Test retrieving multiple embeddings with caching."""
    inputs = ["Hello, world", "Foo bar baz"]

    # First call should populate the cache
    embeddings1 = await embedding_model.get_embeddings(inputs)
    for input_text in inputs:
        assert input_text in embedding_model._cache

    # Mock the nocache method on the underlying embedder to ensure it's not called
    mock_get_embeddings_nocache = mocker.patch.object(
        embedding_model._embedder, "get_embeddings_nocache", autospec=True
    )

    # Second call should retrieve from the cache
    embeddings2 = await embedding_model.get_embeddings(inputs)
    assert np.array_equal(embeddings1, embeddings2)

    # Ensure the nocache method was not called
    mock_get_embeddings_nocache.assert_not_called()


@pytest.mark.asyncio
async def test_get_embeddings_empty_input(embedding_model: CachingEmbeddingModel):
    """Test retrieving embeddings for an empty input list."""
    inputs: list[str] = []
    embeddings = await embedding_model.get_embeddings(inputs)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, embedding_model.embedding_size)
    assert embeddings.dtype == np.float32


@pytest.mark.asyncio
async def test_add_embedding_to_cache(embedding_model: CachingEmbeddingModel):
    """Test adding an embedding to the cache."""
    key = "test_key"
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    embedding_model.add_embedding(key, embedding)
    assert key in embedding_model._cache
    assert np.array_equal(embedding_model._cache[key], embedding)


@pytest.mark.asyncio
async def test_get_embedding_nocache_empty_input(
    embedding_model: CachingEmbeddingModel,
):
    """Test retrieving an embedding with no cache for an empty input."""
    with pytest.raises(ValueError, match="Empty input text"):
        await embedding_model.get_embedding_nocache("")


@pytest.mark.asyncio
async def test_embeddings_are_normalized(embedding_model: CachingEmbeddingModel):
    """Test that returned embeddings are unit-normalized."""
    inputs = ["Hello, world", "Foo bar baz", "Testing normalization"]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)

    for i in range(len(inputs)):
        norm = float(np.linalg.norm(embeddings[i]))
        assert abs(norm - 1.0) < 1e-6, f"Embedding {i} not normalized: norm={norm}"


@pytest.mark.asyncio
async def test_embeddings_are_deterministic(
    embedding_model: CachingEmbeddingModel,
):
    """Test that the same input always produces the same embedding."""
    input_text = "Deterministic test"
    e1 = await embedding_model.get_embedding_nocache(input_text)
    e2 = await embedding_model.get_embedding_nocache(input_text)
    assert np.array_equal(e1, e2)


@pytest.mark.asyncio
async def test_different_inputs_produce_different_embeddings(
    embedding_model: CachingEmbeddingModel,
):
    """Test that different inputs produce different embeddings."""
    e1 = await embedding_model.get_embedding_nocache("Hello")
    e2 = await embedding_model.get_embedding_nocache("World")
    assert not np.array_equal(e1, e2)


@pytest.mark.asyncio
async def test_implements_iembedding_model(
    embedding_model: CachingEmbeddingModel,
):
    """Test that CachingEmbeddingModel satisfies the IEmbeddingModel protocol."""
    assert isinstance(embedding_model, IEmbeddingModel)
