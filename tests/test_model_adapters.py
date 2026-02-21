# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest

import typechat

from typeagent.aitools.embeddings import IEmbeddingModel, NormalizedEmbedding
from typeagent.aitools.model_adapters import (
    configure_models,
    create_chat_model,
    create_embedding_model,
    DEFAULT_CHAT_MODEL_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    infer_provider_prefix,
    PydanticAIChatModel,
    PydanticAIEmbeddingModel,
)


@pytest.fixture
def provider() -> str:
    """Current provider prefix based on available API keys."""
    return infer_provider_prefix()


# ---------------------------------------------------------------------------
# Spec format
# ---------------------------------------------------------------------------


def test_spec_uses_colon_separator() -> None:
    """Specs use ``provider:model`` format matching pydantic_ai conventions."""
    with pytest.raises(Exception):
        # A nonsense provider should fail
        create_chat_model("nonexistent_provider_xyz:fake-model")


# ---------------------------------------------------------------------------
# Embedding size
# ---------------------------------------------------------------------------


def test_explicit_embedding_size(provider: str) -> None:
    """Passing embedding_size= sets it immediately."""
    model = create_embedding_model(
        f"{provider}:text-embedding-3-small", embedding_size=42
    )
    assert model.embedding_size == 42


def test_default_embedding_size_is_zero(provider: str) -> None:
    """Without embedding_size=, it defaults to 0 (probed on first call)."""
    model = create_embedding_model(f"{provider}:text-embedding-3-small")
    assert model.embedding_size == 0


# ---------------------------------------------------------------------------
# PydanticAIChatModel adapter
# ---------------------------------------------------------------------------


def test_chat_model_is_typechat_model() -> None:
    """PydanticAIChatModel inherits from TypeChatLanguageModel."""
    assert typechat.TypeChatLanguageModel in PydanticAIChatModel.__mro__


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


def test_embedding_model_is_iembedding_model() -> None:
    """PydanticAIEmbeddingModel inherits from IEmbeddingModel."""
    assert IEmbeddingModel in PydanticAIEmbeddingModel.__mro__


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
async def test_embedding_adapter_probes_size() -> None:
    """embedding_size is discovered from the first embedding call."""
    from unittest.mock import AsyncMock

    from pydantic_ai import Embedder
    from pydantic_ai.embeddings import EmbeddingResult

    mock_embedder = AsyncMock(spec=Embedder)
    mock_embedder.embed.return_value = EmbeddingResult(
        embeddings=[[1.0, 0.0, 0.0]],
        inputs=["probe"],
        input_type="document",
        model_name="test-model",
        provider_name="test",
    )

    adapter = PydanticAIEmbeddingModel(mock_embedder, "test-model")
    assert adapter.embedding_size == 0
    await adapter.get_embedding_nocache("probe")
    assert adapter.embedding_size == 3


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
    """Empty batch returns empty array with known size."""
    from unittest.mock import AsyncMock

    from pydantic_ai import Embedder

    mock_embedder = AsyncMock(spec=Embedder)
    adapter = PydanticAIEmbeddingModel(mock_embedder, "test-model", 4)
    result = await adapter.get_embeddings_nocache([])
    assert result.shape == (0, 4)


# ---------------------------------------------------------------------------
# configure_models
# ---------------------------------------------------------------------------


def test_configure_models_returns_correct_types(provider: str) -> None:
    """configure_models creates both adapters."""
    chat, embedder = configure_models(
        f"{provider}:gpt-4o", f"{provider}:text-embedding-3-small"
    )
    assert isinstance(chat, PydanticAIChatModel)
    assert isinstance(embedder, PydanticAIEmbeddingModel)
    assert typechat.TypeChatLanguageModel in type(chat).__mro__


# ---------------------------------------------------------------------------
# Environment-variable defaults
# ---------------------------------------------------------------------------


def test_create_chat_model_env_default(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """create_chat_model() reads PYDANTIC_AI_MODEL from the environment."""
    monkeypatch.setenv("PYDANTIC_AI_MODEL", f"{provider}:gpt-4o-mini")
    model = create_chat_model()
    assert isinstance(model, PydanticAIChatModel)


def test_create_chat_model_builtin_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without PYDANTIC_AI_MODEL, falls back to DEFAULT_CHAT_MODEL."""
    monkeypatch.delenv("PYDANTIC_AI_MODEL", raising=False)
    model = create_chat_model()
    assert isinstance(model, PydanticAIChatModel)


def test_create_embedding_model_env_default(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """create_embedding_model() reads PYDANTIC_AI_EMBEDDING_MODEL from the environment."""
    monkeypatch.setenv(
        "PYDANTIC_AI_EMBEDDING_MODEL", f"{provider}:text-embedding-3-small"
    )
    model = create_embedding_model()
    assert isinstance(model, PydanticAIEmbeddingModel)


def test_create_embedding_model_builtin_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without PYDANTIC_AI_EMBEDDING_MODEL, falls back to auto-detected provider."""
    monkeypatch.delenv("PYDANTIC_AI_EMBEDDING_MODEL", raising=False)
    model = create_embedding_model()
    assert isinstance(model, PydanticAIEmbeddingModel)
    assert model.model_name == DEFAULT_EMBEDDING_MODEL_NAME


def test_configure_models_env_defaults(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """configure_models() falls back to env vars when specs are omitted."""
    monkeypatch.setenv("PYDANTIC_AI_MODEL", f"{provider}:gpt-4o-mini")
    monkeypatch.setenv(
        "PYDANTIC_AI_EMBEDDING_MODEL", f"{provider}:text-embedding-3-small"
    )
    chat, embedder = configure_models()
    assert isinstance(chat, PydanticAIChatModel)
    assert isinstance(embedder, PydanticAIEmbeddingModel)


def test_explicit_spec_overrides_env(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """An explicit model_spec takes precedence over the env var."""
    monkeypatch.setenv("PYDANTIC_AI_MODEL", f"{provider}:gpt-4o-mini")
    model = create_chat_model(f"{provider}:gpt-4o")
    assert isinstance(model, PydanticAIChatModel)


def test_default_constants() -> None:
    """Verify the built-in default model name constants."""
    assert DEFAULT_CHAT_MODEL_NAME == "gpt-4o"
    assert DEFAULT_EMBEDDING_MODEL_NAME == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Provider auto-detection
# ---------------------------------------------------------------------------


def test_infer_provider_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefers openai when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    assert infer_provider_prefix() == "openai"


def test_infer_provider_openai_over_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prefers openai when both keys are set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-test")
    assert infer_provider_prefix() == "openai"


def test_infer_provider_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falls back to azure when only AZURE_OPENAI_API_KEY is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-test")
    assert infer_provider_prefix() == "azure"


def test_infer_provider_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises RuntimeError when no API key is available."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    with pytest.raises(
        RuntimeError, match="Neither OPENAI_API_KEY nor AZURE_OPENAI_API_KEY"
    ):
        infer_provider_prefix()
