# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import openai
import pytest
from pytest_mock import MockerFixture
import numpy as np

from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from fixtures import embedding_model, fake_embeddings, fake_embeddings_tiktoken, FakeEmbeddings  # type: ignore  # Yes it's used!
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_get_embedding_nocache(embedding_model: AsyncEmbeddingModel):
    """Test retrieving an embedding without using the cache."""
    input_text = "Hello, world"
    embedding = await embedding_model.get_embedding_nocache(input_text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedding_model.embedding_size,)
    assert embedding.dtype == np.float32


@pytest.mark.asyncio
async def test_get_embeddings_nocache(embedding_model: AsyncEmbeddingModel):
    """Test retrieving multiple embeddings without using the cache."""
    inputs = ["Hello, world", "Foo bar baz"]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(inputs), embedding_model.embedding_size)
    assert embeddings.dtype == np.float32


@pytest.mark.asyncio
async def test_get_embedding_with_cache(
    embedding_model: AsyncEmbeddingModel, mocker: MockerFixture
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
    embedding_model: AsyncEmbeddingModel, mocker: MockerFixture
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
async def test_get_embeddings_empty_input(embedding_model: AsyncEmbeddingModel):
    """Test retrieving embeddings for an empty input list."""
    inputs = []
    embeddings = await embedding_model.get_embeddings(inputs)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, embedding_model.embedding_size)
    assert embeddings.dtype == np.float32


@pytest.mark.asyncio
async def test_add_embedding_to_cache(embedding_model: AsyncEmbeddingModel):
    """Test adding an embedding to the cache."""
    key = "test_key"
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    embedding_model.add_embedding(key, embedding)
    assert key in embedding_model._embedding_cache
    assert np.array_equal(embedding_model._embedding_cache[key], embedding)


@pytest.mark.asyncio
async def test_get_embedding_nocache_empty_input(embedding_model: AsyncEmbeddingModel):
    """Test retrieving an embedding with no cache for an empty input."""
    with pytest.raises(openai.OpenAIError):
        await embedding_model.get_embedding_nocache("")


@pytest.mark.asyncio
async def test_refresh_auth(
    embedding_model: AsyncEmbeddingModel, mocker: MockerFixture
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


@pytest.mark.asyncio
async def test_set_endpoint(monkeypatch):
    """Test creating of model with custom endpoint."""

    monkeypatch.setenv("OPENAI_API_KEY", "does-not-matter")
    monkeypatch.setenv("INFINITY_EMBEDDING_URL", "http://localhost:7997")

    embedding_model = AsyncEmbeddingModel(
        1024, "custom_model", "INFINITY_EMBEDDING_URL"
    )

    assert embedding_model.async_client is not None
    assert embedding_model.async_client.base_url == "http://localhost:7997"
    assert embedding_model.async_client.api_key == "does-not-matter"

    with pytest.raises(
        ValueError,
        match="Environment variable for embedding endpoint WRONG_ENDPOINT does not match required environment"
        " variable AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL for embedding model text-embedding-small.",
    ):
        embedding_model = AsyncEmbeddingModel(
            2000, "text-embedding-small", "WRONG_ENDPOINT"
        )


@pytest.mark.asyncio
async def test_embeddings_batching_tiktoken(
    fake_embeddings_tiktoken: FakeEmbeddings, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")

    embedding_model = AsyncEmbeddingModel()
    assert embedding_model.max_chunk_size == 4096

    embedding_model.async_client.embeddings = fake_embeddings_tiktoken  # type: ignore

    # Check max batch size
    inputs = ["a"] * 2049
    embeddings = await embedding_model.get_embeddings(inputs)
    assert len(embeddings) == 2049
    assert fake_embeddings_tiktoken.call_count == 2

    # Check max token size
    inputs = ["Very long input longer than 4096 tokens needs to split in 2" * 500]
    embeddings = await embedding_model.get_embeddings(inputs)
    assert len(embeddings) == 1
    assert np.all(embeddings[0] == 0.5)


@pytest.mark.asyncio
async def test_embeddings_batching(fake_embeddings: FakeEmbeddings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")

    MAX_CHUNK_SIZE = 4096 * 3

    embedding_model = AsyncEmbeddingModel(1024, "custom_model")
    assert embedding_model.max_chunk_size == MAX_CHUNK_SIZE
    embedding_model.async_client.embeddings = fake_embeddings  # type: ignore

    # Check max token size
    inputs = ["a" * MAX_CHUNK_SIZE]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)
    assert len(embeddings) == 1
    assert np.all(embeddings[0] == 0)

    # Check one over max token size
    inputs = ["a" * (MAX_CHUNK_SIZE + 1)]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)
    assert len(embeddings) == 1
    assert fake_embeddings.call_count == 2
    assert np.all(embeddings[0] == 0.5)

    fake_embeddings.reset_counter()

    # Check input larger than max_size_per_batch

    s = "very long fake string"
    mock_string = MagicMock(wraps=s)
    mock_string.__len__.return_value = 100_000

    inputs = [mock_string, mock_string, mock_string]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)  # type: ignore
    assert fake_embeddings.call_count == 1
    assert len(embeddings) == 3

    fake_embeddings.reset_counter()

    mock_string.__len__.return_value = 100_001

    # First batch input1 + input 2 with length 200_002
    # Second batch input3 with length 100_001

    inputs = [mock_string, mock_string, mock_string]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)  # type: ignore
    assert fake_embeddings.call_count == 2
    assert len(embeddings) == 3
