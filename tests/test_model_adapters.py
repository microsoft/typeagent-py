# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import AsyncMock

import numpy as np
import pytest

from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingResult
from pydantic_ai.messages import (
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
import typechat

from typeagent.aitools.embeddings import CachingEmbeddingModel, NormalizedEmbedding
from typeagent.aitools.model_adapters import (
    configure_models,
    create_chat_model,
    PydanticAIChatModel,
    PydanticAIEmbedder,
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
# PydanticAIChatModel adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_adapter_complete() -> None:
    """PydanticAIChatModel wraps a pydantic_ai Model."""
    mock_model = AsyncMock(spec=Model)
    mock_model.request.return_value = ModelResponse(parts=[TextPart(content="hello")])

    adapter = PydanticAIChatModel(mock_model)
    result = await adapter.complete("test prompt")
    assert isinstance(result, typechat.Success)
    assert result.value == "hello"


@pytest.mark.asyncio
async def test_chat_adapter_prompt_sections() -> None:
    """PydanticAIChatModel handles list[PromptSection] prompts."""
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
    assert isinstance(request.parts[0], SystemPromptPart)
    assert isinstance(request.parts[1], UserPromptPart)


# ---------------------------------------------------------------------------
# PydanticAIEmbedder adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_adapter_single() -> None:
    """PydanticAIEmbedder computes a single normalized embedding."""
    mock_embedder = AsyncMock(spec=Embedder)
    raw_vec = [3.0, 4.0, 0.0]
    mock_embedder.embed_documents.return_value = EmbeddingResult(
        embeddings=[raw_vec],
        inputs=["test"],
        input_type="document",
        model_name="test-model",
        provider_name="test",
    )

    adapter = PydanticAIEmbedder(mock_embedder, "test-model")
    result = await adapter.get_embedding_nocache("test")
    assert result.shape == (3,)
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_embedding_adapter_empty_batch_raises() -> None:
    """Empty batch raises ValueError."""
    mock_embedder = AsyncMock(spec=Embedder)
    adapter = PydanticAIEmbedder(mock_embedder, "test-model")
    with pytest.raises(ValueError, match="Cannot embed an empty list"):
        await adapter.get_embeddings_nocache([])


@pytest.mark.asyncio
async def test_embedding_adapter_batch() -> None:
    """PydanticAIEmbedder computes batch embeddings."""
    mock_embedder = AsyncMock(spec=Embedder)
    mock_embedder.embed_documents.return_value = EmbeddingResult(
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        inputs=["a", "b"],
        input_type="document",
        model_name="test-model",
        provider_name="test",
    )

    adapter = PydanticAIEmbedder(mock_embedder, "test-model")
    result = await adapter.get_embeddings_nocache(["a", "b"])
    assert result.shape == (2, 2)


@pytest.mark.asyncio
async def test_embedding_adapter_caching() -> None:
    """CachingEmbeddingModel avoids re-computing embeddings."""
    mock_embedder = AsyncMock(spec=Embedder)
    mock_embedder.embed_documents.return_value = EmbeddingResult(
        embeddings=[[1.0, 0.0, 0.0]],
        inputs=["cached"],
        input_type="document",
        model_name="test-model",
        provider_name="test",
    )

    embedder = PydanticAIEmbedder(mock_embedder, "test-model")
    adapter = CachingEmbeddingModel(embedder)
    first = await adapter.get_embedding("cached")
    second = await adapter.get_embedding("cached")
    np.testing.assert_array_equal(first, second)
    # embed_documents() should only be called once
    assert mock_embedder.embed_documents.call_count == 1


@pytest.mark.asyncio
async def test_embedding_adapter_add_embedding() -> None:
    """add_embedding() populates the cache."""
    mock_embedder = AsyncMock(spec=Embedder)
    embedder = PydanticAIEmbedder(mock_embedder, "test-model")
    adapter = CachingEmbeddingModel(embedder)
    vec: NormalizedEmbedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    adapter.add_embedding("key", vec)
    result = await adapter.get_embedding("key")
    np.testing.assert_array_equal(result, vec)
    # No embed_documents() call needed
    mock_embedder.embed_documents.assert_not_called()


@pytest.mark.asyncio
async def test_embedding_adapter_empty_batch_returns_empty() -> None:
    """Empty batch via CachingEmbeddingModel raises ValueError."""
    mock_embedder = AsyncMock(spec=Embedder)
    embedder = PydanticAIEmbedder(mock_embedder, "test-model")
    adapter = CachingEmbeddingModel(embedder)
    with pytest.raises(ValueError, match="Cannot embed an empty list"):
        await adapter.get_embeddings([])


# ---------------------------------------------------------------------------
# configure_models
# ---------------------------------------------------------------------------


def test_configure_models_returns_correct_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """configure_models creates both adapters."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    chat, embedder = configure_models("openai:gpt-4o", "openai:text-embedding-3-small")
    assert isinstance(chat, PydanticAIChatModel)
    assert isinstance(embedder, CachingEmbeddingModel)
