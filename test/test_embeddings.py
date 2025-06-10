# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import openai
import pytest
from pytest_mock import MockerFixture
import numpy as np

from typeagent.aitools.embeddings import AsyncEmbeddingModel

from fixtures import needs_auth  # type: ignore  # Needed!


@pytest.fixture
def embedding_model() -> AsyncEmbeddingModel:
    """Fixture to create an instance of AsyncEmbeddingModel."""
    return AsyncEmbeddingModel(embedding_size=3)


@pytest.mark.asyncio
async def test_get_embedding_nocache(
    embedding_model: AsyncEmbeddingModel, needs_auth: None
):
    """Test retrieving an embedding without using the cache."""
    input_text = "Hello, world"
    embedding = await embedding_model.get_embedding_nocache(input_text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.embedding_size,)
    assert embedding.dtype == np.float32


@pytest.mark.asyncio
async def test_get_embeddings_nocache(
    embedding_model: AsyncEmbeddingModel, needs_auth: None
):
    """Test retrieving multiple embeddings without using the cache."""
    inputs = ["Hello, world", "Foo bar baz"]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(inputs), embedding_model.embedding_size)
    assert embeddings.dtype == np.float32


@pytest.mark.asyncio
async def test_get_embedding_with_cache(
    embedding_model: AsyncEmbeddingModel, needs_auth: None, mocker: MockerFixture
):
    """Test retrieving an embedding with caching."""
    input_text = "Hello, world"

    # First call should populate the cache
    embedding1 = await embedding_model.get_embedding(input_text)
    assert input_text in embedding_model._embedding_cache

    # Mock the nocache method to ensure it's not called
    mock_get_embedding_nocache = mocker.patch.object(
        embedding_model, "get_embedding_nocache", autospec=True
    )

    # Second call should retrieve from the cache
    embedding2 = await embedding_model.get_embedding(input_text)
    assert np.array_equal(embedding1, embedding2)

    # Ensure the nocache method was not called
    mock_get_embedding_nocache.assert_not_called()


@pytest.mark.asyncio
async def test_get_embeddings_with_cache(
    embedding_model: AsyncEmbeddingModel, needs_auth: None, mocker: MockerFixture
):
    """Test retrieving multiple embeddings with caching."""
    inputs = ["Hello, world", "Foo bar baz"]

    # First call should populate the cache
    embeddings1 = await embedding_model.get_embeddings(inputs)
    for input_text in inputs:
        assert input_text in embedding_model._embedding_cache

    # Mock the nocache method to ensure it's not called
    mock_get_embeddings_nocache = mocker.patch.object(
        embedding_model, "get_embeddings_nocache", autospec=True
    )

    # Second call should retrieve from the cache
    embeddings2 = await embedding_model.get_embeddings(inputs)
    assert np.array_equal(embeddings1, embeddings2)

    # Ensure the nocache method was not called
    mock_get_embeddings_nocache.assert_not_called()


@pytest.mark.asyncio
async def test_get_embeddings_empty_input(
    embedding_model: AsyncEmbeddingModel, needs_auth: None
):
    """Test retrieving embeddings for an empty input list."""
    inputs = []
    embeddings = await embedding_model.get_embeddings(inputs)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, embedding_model.embedding_size)
    assert embeddings.dtype == np.float32


@pytest.mark.asyncio
async def test_add_embedding_to_cache(
    embedding_model: AsyncEmbeddingModel, needs_auth: None
):
    """Test adding an embedding to the cache."""
    key = "test_key"
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    embedding_model.add_embedding(key, embedding)
    assert key in embedding_model._embedding_cache
    assert np.array_equal(embedding_model._embedding_cache[key], embedding)


@pytest.mark.asyncio
async def test_get_embedding_nocache_empty_input(
    embedding_model: AsyncEmbeddingModel, needs_auth: None
):
    """Test retrieving an embedding with no cache for an empty input."""
    with pytest.raises(openai.BadRequestError):
        await embedding_model.get_embedding_nocache("")


@pytest.mark.asyncio
async def test_refresh_auth(
    embedding_model: AsyncEmbeddingModel, needs_auth: None, mocker: MockerFixture
):
    """Test refreshing authentication when using Azure."""
    # Note that pyright doesn't understand mocking, hence the `# type: ignore` below
    mocker.patch.object(embedding_model, "azure_token_provider", autospec=True)
    mocker.patch.object(embedding_model, "_setup_azure", autospec=True)

    embedding_model.azure_token_provider.needs_refresh.return_value = True  # type: ignore
    embedding_model.azure_token_provider.refresh_token.return_value = "new_token"  # type: ignore
    embedding_model.azure_api_version = "2023-05-15"
    embedding_model.azure_endpoint = "https://example.azure.com"

    await embedding_model.refresh_auth()

    embedding_model.azure_token_provider.refresh_token.assert_called_once()  # type: ignore
    assert embedding_model.async_client is not None
