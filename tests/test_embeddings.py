# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest
from pytest import MonkeyPatch
from pytest_mock import MockerFixture

import openai

from typeagent.aitools.embeddings import AsyncEmbeddingModel

from conftest import (
    embedding_model,  # type: ignore  # Magic, prevents side effects of mocking
)
from conftest import (
    FakeEmbeddings,
)


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
async def test_set_endpoint(monkeypatch: MonkeyPatch):
    """Test creating of model with custom endpoint."""

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "does-not-matter")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # Ensure Azure path is used

    # Default
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT_EMBEDDING",
        "http://localhost:7997?api-version=2024-06-01",
    )
    embedding_model = AsyncEmbeddingModel()
    assert embedding_model.embedding_size == 1536
    assert embedding_model.model_name == "text-embedding-ada-002"
    assert embedding_model.endpoint_envvar == "AZURE_OPENAI_ENDPOINT_EMBEDDING"

    # 3-large
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE",
        "http://localhost:7997?api-version=2024-06-01",
    )
    embedding_model = AsyncEmbeddingModel(model_name="text-embedding-3-large")
    assert embedding_model.embedding_size == 3072
    assert embedding_model.model_name == "text-embedding-3-large"
    assert embedding_model.endpoint_envvar == "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE"

    # 3-small
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL",
        "http://localhost:7998?api-version=2024-06-01",
    )
    embedding_model = AsyncEmbeddingModel(model_name="text-embedding-3-small")
    assert embedding_model.embedding_size == 1536
    assert embedding_model.model_name == "text-embedding-3-small"
    assert embedding_model.endpoint_envvar == "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"

    # Fully custom with OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "does-not-matter")
    monkeypatch.setenv("INFINITY_EMBEDDING_URL", "http://localhost:7997")
    embedding_model = AsyncEmbeddingModel(
        1024, "custom_model", endpoint_envvar="INFINITY_EMBEDDING_URL"
    )
    assert embedding_model.embedding_size == 1024
    assert embedding_model.model_name == "custom_model"
    # NOTE: checking openai.AsyncOpenAI internals
    assert embedding_model.async_client is not None
    assert embedding_model.async_client.base_url == "http://localhost:7997"
    assert embedding_model.async_client.api_key == "does-not-matter"
    assert embedding_model.endpoint_envvar == "INFINITY_EMBEDDING_URL"

    # Customized 3-small with Azure (endpoint_envvar must contain "AZURE")
    monkeypatch.delenv("OPENAI_API_KEY")  # Force Azure path
    monkeypatch.setenv(
        "AZURE_ALTERNATE_ENDPOINT",
        "http://localhost:7999?api-version=2024-06-01",
    )
    embedding_model = AsyncEmbeddingModel(
        2000, "text-embedding-3-small", endpoint_envvar="AZURE_ALTERNATE_ENDPOINT"
    )
    assert embedding_model.embedding_size == 2000
    assert embedding_model.model_name == "text-embedding-3-small"
    assert embedding_model.endpoint_envvar == "AZURE_ALTERNATE_ENDPOINT"

    # Allow explicitly setting default embedding size
    AsyncEmbeddingModel(1536)

    # Can't customize embedding_size for default model
    with pytest.raises(ValueError):
        AsyncEmbeddingModel(1024)

    # Not even when default model name specified explicitly
    with pytest.raises(ValueError):
        AsyncEmbeddingModel(1024, "text-embedding-ada-002")


@pytest.mark.asyncio
async def test_embeddings_batching_tiktoken(
    fake_embeddings_tiktoken: FakeEmbeddings, monkeypatch: MonkeyPatch
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
    inputs = ["Very long input longer than 4096 tokens will be truncated" * 500]
    embeddings = await embedding_model.get_embeddings(inputs)
    assert len(embeddings) == 1

    fake_embeddings_tiktoken.reset_counter()

    TEST_MAX_TOKEN_SIZE = 10
    TEST_MAX_TOKENS_PER_BATCH = 20
    embedding_model.max_chunk_size = TEST_MAX_TOKEN_SIZE
    embedding_model.max_size_per_batch = TEST_MAX_TOKENS_PER_BATCH
    fake_embeddings_tiktoken.max_elements_per_batch = TEST_MAX_TOKENS_PER_BATCH

    assert embedding_model.encoding is not None

    token = [500] * 20  # --> 20 tokens
    input = [embedding_model.encoding.decode(token)] * 4
    embeddings = await embedding_model.get_embeddings_nocache(input)  # type: ignore

    # each input gets truncated to 10 tokens, so 4 inputs fit in 2 batches of 20 tokens
    assert fake_embeddings_tiktoken.call_count == 2
    assert len(embeddings) == 4

    fake_embeddings_tiktoken.reset_counter()

    TEST_MAX_TOKEN_SIZE = 7
    embedding_model.max_chunk_size = TEST_MAX_TOKEN_SIZE

    token = [500] * 20  # --> 20 tokens
    input = [embedding_model.encoding.decode(token)] * 5
    embeddings = await embedding_model.get_embeddings_nocache(input)  # type: ignore

    # each input gets truncated to 7 tokens, so each batch can hold 2 inputs (14 tokens)
    # 5 inputs require 3 batches
    assert fake_embeddings_tiktoken.call_count == 3
    assert len(embeddings) == 5


@pytest.mark.asyncio
async def test_embeddings_batching(
    fake_embeddings: FakeEmbeddings, monkeypatch: MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")

    embedding_model = AsyncEmbeddingModel(1024, "custom_model")
    embedding_model.async_client.embeddings = fake_embeddings  # type: ignore

    # Check max batch size
    inputs = ["a"] * 2049
    embeddings = await embedding_model.get_embeddings(inputs)
    assert len(embeddings) == 2049
    assert fake_embeddings.call_count == 2

    TEST_MAX_CHAR_SIZE = 10
    TEST_MAX_CHARS_PER_BATCH = 20
    embedding_model.max_chunk_size = TEST_MAX_CHAR_SIZE
    embedding_model.max_size_per_batch = TEST_MAX_CHARS_PER_BATCH
    fake_embeddings.max_elements_per_batch = TEST_MAX_CHARS_PER_BATCH

    # Check max token size
    inputs = ["a" * TEST_MAX_CHAR_SIZE]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)
    assert len(embeddings) == 1
    assert np.all(embeddings[0] == 0)

    fake_embeddings.reset_counter()

    # Check one over max token size
    inputs = ["a" * (TEST_MAX_CHAR_SIZE + 1)]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)
    assert len(embeddings) == 1
    assert fake_embeddings.call_count == 1

    fake_embeddings.reset_counter()

    # Check input as large as max_size_per_batch
    inputs = ["a" * 10, "a" * 5, "a" * 5]
    embeddings = await embedding_model.get_embeddings_nocache(inputs)  # type: ignore
    assert fake_embeddings.call_count == 1
    assert len(embeddings) == 3

    fake_embeddings.reset_counter()

    # Check input larger than max_size_per_batch
    # max chars per batch is 20, so 10*10 chars requires 5 batches
    inputs = ["a" * 10] * 10
    embeddings = await embedding_model.get_embeddings_nocache(inputs)  # type: ignore
    assert fake_embeddings.call_count == 5
    assert len(embeddings) == 10
