# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest

import typechat

from typeagent.aitools.embeddings import IEmbeddingModel, NormalizedEmbedding
from typeagent.aitools.model_registry import (
    _KNOWN_EMBEDDING_SIZES,
    configure_models,
    create_chat_model,
    create_embedding_model,
    PydanticAIChatModel,
    PydanticAIEmbeddingModel,
)

# ---------------------------------------------------------------------------
# Spec format
# ---------------------------------------------------------------------------


def test_spec_uses_colon_separator() -> None:
    """Specs use ``provider:model`` format matching pydantic_ai conventions."""
    with pytest.raises(Exception):
        # A nonsense provider should fail
        create_chat_model("nonexistent_provider_xyz:fake-model")


# ---------------------------------------------------------------------------
# Known embedding sizes
# ---------------------------------------------------------------------------


def test_known_embedding_sizes() -> None:
    assert _KNOWN_EMBEDDING_SIZES["text-embedding-3-small"] == 1536
    assert _KNOWN_EMBEDDING_SIZES["text-embedding-3-large"] == 3072
    assert _KNOWN_EMBEDDING_SIZES["text-embedding-ada-002"] == 1536


def test_unknown_embedding_size_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding size"):
        create_embedding_model("openai:completely-unknown-model-xyz")


def test_explicit_embedding_size() -> None:
    """Passing embedding_size= bypasses the lookup table."""
    # This should not raise even though the model name is unknown
    model = create_embedding_model(
        "openai:completely-unknown-model-xyz", embedding_size=42
    )
    assert model.embedding_size == 42


# ---------------------------------------------------------------------------
# PydanticAIChatModel adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_adapter_complete() -> None:
    """PydanticAIChatModel wraps a pydantic_ai Model."""
    from unittest.mock import AsyncMock

    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models import Model

    mock_model = AsyncMock(spec=Model)
    mock_model.request.return_value = ModelResponse(parts=[TextPart(content="hello")])

    adapter = PydanticAIChatModel(mock_model)
    result = await adapter.complete("test prompt")
    assert isinstance(result, typechat.Success)
    assert result.value == "hello"


@pytest.mark.asyncio
async def test_chat_adapter_prompt_sections() -> None:
    """PydanticAIChatModel handles list[PromptSection] prompts."""
    from unittest.mock import AsyncMock

    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models import Model

    mock_model = AsyncMock(spec=Model)
    mock_model.request.return_value = ModelResponse(
        parts=[TextPart(content="response")]
    )

    adapter = PydanticAIChatModel(mock_model)
    sections: list[typechat.PromptSection] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    result = await adapter.complete(sections)
    assert isinstance(result, typechat.Success)
    assert result.value == "response"

    # Verify the request was called with proper message structure
    call_args = mock_model.request.call_args
    messages = call_args[0][0]
    assert len(messages) == 1
    request = messages[0]
    from pydantic_ai.messages import SystemPromptPart, UserPromptPart

    assert isinstance(request.parts[0], SystemPromptPart)
    assert isinstance(request.parts[1], UserPromptPart)


# ---------------------------------------------------------------------------
# PydanticAIEmbeddingModel adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_adapter_single() -> None:
    """PydanticAIEmbeddingModel computes a single normalized embedding."""
    from unittest.mock import AsyncMock

    from pydantic_ai import Embedder
    from pydantic_ai.embeddings import EmbeddingResult

    mock_embedder = AsyncMock(spec=Embedder)
    raw_vec = [3.0, 4.0, 0.0]
    mock_embedder.embed.return_value = EmbeddingResult(
        embeddings=[raw_vec],
        inputs=["test"],
        input_type="document",
        model_name="test-model",
        provider_name="test",
    )

    adapter = PydanticAIEmbeddingModel(mock_embedder, "test-model", 3)
    result = await adapter.get_embedding_nocache("test")
    assert result.shape == (3,)
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_embedding_adapter_batch() -> None:
    """PydanticAIEmbeddingModel computes batch embeddings."""
    from unittest.mock import AsyncMock

    from pydantic_ai import Embedder
    from pydantic_ai.embeddings import EmbeddingResult

    mock_embedder = AsyncMock(spec=Embedder)
    mock_embedder.embed.return_value = EmbeddingResult(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        inputs=["a", "b"],
        input_type="document",
        model_name="test-model",
        provider_name="test",
    )

    adapter = PydanticAIEmbeddingModel(mock_embedder, "test-model", 2)
    result = await adapter.get_embeddings_nocache(["a", "b"])
    assert result.shape == (2, 2)


@pytest.mark.asyncio
async def test_embedding_adapter_caching() -> None:
    """Caching avoids re-computing embeddings."""
    from unittest.mock import AsyncMock

    from pydantic_ai import Embedder
    from pydantic_ai.embeddings import EmbeddingResult

    mock_embedder = AsyncMock(spec=Embedder)
    mock_embedder.embed.return_value = EmbeddingResult(
        embeddings=[[1.0, 0.0, 0.0]],
        inputs=["cached"],
        input_type="document",
        model_name="test-model",
        provider_name="test",
    )

    adapter = PydanticAIEmbeddingModel(mock_embedder, "test-model", 3)
    first = await adapter.get_embedding("cached")
    second = await adapter.get_embedding("cached")
    np.testing.assert_array_equal(first, second)
    # embed() should only be called once
    assert mock_embedder.embed.call_count == 1


@pytest.mark.asyncio
async def test_embedding_adapter_add_embedding() -> None:
    """add_embedding() populates the cache."""
    from unittest.mock import AsyncMock

    from pydantic_ai import Embedder

    mock_embedder = AsyncMock(spec=Embedder)
    adapter = PydanticAIEmbeddingModel(mock_embedder, "test-model", 3)
    vec: NormalizedEmbedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    adapter.add_embedding("key", vec)
    result = await adapter.get_embedding("key")
    np.testing.assert_array_equal(result, vec)
    # No embed() call needed
    mock_embedder.embed.assert_not_called()


@pytest.mark.asyncio
async def test_embedding_adapter_empty_batch() -> None:
    """Empty batch returns empty array."""
    from unittest.mock import AsyncMock

    from pydantic_ai import Embedder

    mock_embedder = AsyncMock(spec=Embedder)
    adapter = PydanticAIEmbeddingModel(mock_embedder, "test-model", 4)
    result = await adapter.get_embeddings_nocache([])
    assert result.shape == (0, 4)


# ---------------------------------------------------------------------------
# configure_models
# ---------------------------------------------------------------------------


def test_configure_models_returns_correct_types() -> None:
    """configure_models creates both adapters."""
    chat, embedder = configure_models("openai:gpt-4o", "openai:text-embedding-3-small")
    assert isinstance(chat, PydanticAIChatModel)
    assert isinstance(embedder, PydanticAIEmbeddingModel)
    assert isinstance(embedder, IEmbeddingModel)
    assert embedder.embedding_size == 1536
