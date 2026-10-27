# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest

from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingResult
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings
import typechat

from typeagent.aitools import model_adapters
from typeagent.aitools.embeddings import CachingEmbeddingModel, NormalizedEmbedding
from typeagent.aitools.model_adapters import (
    configure_models,
    create_chat_model,
    create_embedding_model,
    PydanticAIChatModel,
    PydanticAIEmbedder,
    should_disable_thinking,
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


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("gpt-6-luna", True),
        ("gpt-6-luna-2026-10-01", True),
        ("gpt-6-lunar", False),
        ("gpt-6-astra", False),
        ("claude-sonnet-4-20250514", False),
    ],
)
def test_should_disable_thinking(model_name: str, expected: bool) -> None:
    assert should_disable_thinking(model_name) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name, settings, expected_thinking",
    [
        ("gpt-6-luna", None, "low"),
        ("gpt-6-astra", None, None),
        ("gpt-6-luna", ModelSettings(thinking=False), False),
        ("gpt-6-luna", ModelSettings(thinking="high"), "high"),
    ],
)
async def test_chat_adapter_preserves_and_overrides_model_settings(
    model_name: str,
    settings: ModelSettings | None,
    expected_thinking: bool | str | None,
) -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.thinking == expected_thinking
        return ModelResponse(parts=[TextPart(content="hello")])

    adapter = PydanticAIChatModel(
        FunctionModel(
            respond,
            model_name=model_name,
            profile={"supports_thinking": True},
            settings=(
                ModelSettings(thinking="low") if model_name == "gpt-6-luna" else None
            ),
        ),
        settings=settings,
    )
    result = await adapter.complete("test prompt")
    assert isinstance(result, typechat.Success)
    assert result.value == "hello"


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
    chat, embedder = configure_models(
        "openai:gpt-6-luna", "openai:text-embedding-3-small"
    )
    assert isinstance(chat, PydanticAIChatModel)
    assert isinstance(embedder, CachingEmbeddingModel)


def test_create_embedding_model_uses_azure_deployment_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure embedding endpoints contribute the deployment name."""
    captured: dict[str, object] = {}
    provider = object()

    class FakeOpenAIEmbeddingModel:
        def __init__(self, model_name: str, provider: object) -> None:
            captured["azure_model_name"] = model_name
            captured["provider"] = provider

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT_EMBEDDING",
        "https://myhost.openai.azure.com/openai/deployments/ada-002/embeddings?api-version=2025-01-01-preview",
    )
    monkeypatch.setattr(
        model_adapters,
        "_make_azure_provider",
        lambda endpoint_envvar, api_key_envvar: provider,
    )
    monkeypatch.setattr(
        "pydantic_ai.embeddings.openai.OpenAIEmbeddingModel", FakeOpenAIEmbeddingModel
    )
    monkeypatch.setattr(
        model_adapters, "_PydanticAIEmbedder", lambda embedding_model: embedding_model
    )

    embedder = create_embedding_model()

    assert isinstance(embedder, CachingEmbeddingModel)
    assert captured["azure_model_name"] == "ada-002"
    assert captured["provider"] is provider
    assert embedder.model_name == "text-embedding-ada-002"


@pytest.mark.parametrize(
    "model_name, expected_thinking",
    [("gpt-6-luna", False), ("gpt-6-astra", None)],
)
def test_create_chat_model_azure_profile_from_requested_model(
    monkeypatch: pytest.MonkeyPatch, model_name: str, expected_thinking: bool | None
) -> None:
    """Azure chat models use the deployment name for requests, but take their
    profile and thinking setting from the requested model name."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "unused-key")
    monkeypatch.setenv(
        "AZURE_OPENAI_ENDPOINT",
        "https://myhost.openai.azure.com/openai/deployments/my-deploy/chat/completions?api-version=2025-01-01-preview",
    )

    chat_model = create_chat_model(f"openai:{model_name}")

    assert chat_model.settings == (
        {"thinking": False} if expected_thinking is False else None
    )
    model = chat_model._model
    assert model.model_name == "my-deploy"
    assert model.profile.get("supports_thinking") is True
    # Without the requested model's profile, pydantic_ai drops thinking=False.
    _, params = model.prepare_request(chat_model.settings, ModelRequestParameters())
    assert params.thinking is expected_thinking


@pytest.mark.parametrize(
    "model_name, expected_settings",
    [("gpt-6-luna", ModelSettings(thinking=False)), ("gpt-6-astra", None)],
)
def test_create_chat_model_openai_thinking_default(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    expected_settings: ModelSettings | None,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unused-key")
    chat_model = create_chat_model(f"openai:{model_name}")
    assert chat_model.settings == expected_settings


@pytest.mark.asyncio
@pytest.mark.parametrize("use_azure", [False, True], ids=["openai", "azure"])
async def test_create_chat_model_sends_reasoning_effort_none(
    monkeypatch: pytest.MonkeyPatch, use_azure: bool
) -> None:
    """The real model and SDK serialize disabled thinking for both providers."""
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    if use_azure:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "unused-key")
        monkeypatch.setenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://myhost.openai.azure.com/openai/deployments/my-deploy/chat/completions?api-version=2025-01-01-preview",
        )
    else:
        monkeypatch.setenv("OPENAI_API_KEY", "unused-key")

    expected_model = "my-deploy" if use_azure else "gpt-6-luna"
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = json.loads(request.content)
        assert body["model"] == expected_model
        assert body["reasoning_effort"] == "none"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": expected_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    # Select Chat Completions explicitly: newer pydantic_ai versions route
    # the generic openai spec to the Responses API instead.
    spec = "openai:gpt-6-luna" if use_azure else "openai-chat:gpt-6-luna"
    chat_model = create_chat_model(spec)
    model = chat_model._model
    assert isinstance(model, OpenAIChatModel)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        # Keep the factory's model, provider, and SDK request serialization;
        # replace only the HTTP transport so no network access is needed.
        monkeypatch.setattr(model.client, "_client", client)
        result = await chat_model.complete("test prompt")

    assert isinstance(result, typechat.Success)
    assert result.value == "hello"
    assert len(requests) == 1
    if use_azure:
        assert requests[0].url.path == "/openai/deployments/my-deploy/chat/completions"
