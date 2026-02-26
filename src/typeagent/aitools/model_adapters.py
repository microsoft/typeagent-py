# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Provider-agnostic model configuration backed by pydantic_ai.

Create chat and embedding models from ``provider:model`` spec strings::

    from typeagent.aitools.model_adapters import configure_models

    chat, embedder = configure_models(
        "openai:gpt-4o",
        "openai:text-embedding-3-small",
    )

The spec format is ``provider:model``, matching pydantic_ai conventions.
Provider wiring (API keys, endpoints, etc.) is handled by pydantic_ai's
model registry, which supports 25+ providers including ``openai``,
``azure``, ``anthropic``, ``google``, ``bedrock``, ``groq``, ``mistral``,
``ollama``, ``cohere``, and many more.

When a spec uses ``openai:`` as the provider and ``OPENAI_API_KEY`` is not
set, but ``AZURE_OPENAI_API_KEY`` is available, the provider is
automatically switched to Azure OpenAI.

See https://ai.pydantic.dev/models/ for all supported providers and their
required environment variables.
"""

from collections.abc import Sequence
import os

import numpy as np
from numpy.typing import NDArray

from pydantic_ai import Embedder as _PydanticAIEmbedder
from pydantic_ai.embeddings.base import EmbeddingModel as _PydanticAIEmbeddingModelBase
from pydantic_ai.embeddings.result import EmbeddingResult, EmbedInputType
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import infer_model, Model, ModelRequestParameters
import typechat

from .embeddings import (
    CachingEmbeddingModel,
    NormalizedEmbedding,
    NormalizedEmbeddings,
)

# ---------------------------------------------------------------------------
# Chat model adapter
# ---------------------------------------------------------------------------


class PydanticAIChatModel(typechat.TypeChatLanguageModel):
    """Adapter from :class:`pydantic_ai.models.Model` to TypeChat's
    :class:`~typechat.TypeChatLanguageModel`.

    This lets any pydantic_ai chat model (OpenAI, Anthropic, Google, …) be
    used wherever TypeChat expects a ``TypeChatLanguageModel``.
    """

    def __init__(self, model: Model) -> None:
        self._model = model

    async def complete(
        self, prompt: str | list[typechat.PromptSection]
    ) -> typechat.Result[str]:
        parts: list[SystemPromptPart | UserPromptPart] = []
        if isinstance(prompt, str):
            parts.append(UserPromptPart(content=prompt))
        else:
            for section in prompt:
                if section["role"] == "system":
                    parts.append(SystemPromptPart(content=section["content"]))
                else:
                    parts.append(UserPromptPart(content=section["content"]))

        messages: list[ModelMessage] = [ModelRequest(parts=parts)]
        params = ModelRequestParameters()

        response = await self._model.request(messages, None, params)
        text_parts = [p.content for p in response.parts if isinstance(p, TextPart)]
        if text_parts:
            return typechat.Success("".join(text_parts))
        return typechat.Failure("No text content in model response")


# ---------------------------------------------------------------------------
# Embedding model adapter
# ---------------------------------------------------------------------------


class PydanticAIEmbedder:
    """Adapter from :class:`pydantic_ai.Embedder` to :class:`IEmbedder`.

    This lets any pydantic_ai embedding provider (OpenAI, Cohere, Google, …)
    be used wherever the codebase expects an ``IEmbedder``.  Wrap in
    :class:`~typeagent.aitools.embeddings.CachingEmbeddingModel` to get a
    ready-to-use ``IEmbeddingModel`` with caching.

    If *embedding_size* is not given, it is probed automatically by making a
    single embedding call.
    """

    model_name: str
    embedding_size: int

    def __init__(
        self,
        embedder: _PydanticAIEmbedder,
        model_name: str,
        embedding_size: int = 0,
    ) -> None:
        self._embedder = embedder
        self.model_name = model_name
        self.embedding_size = embedding_size

    async def _probe_embedding_size(self) -> None:
        """Discover embedding_size by making a single API call."""
        result = await self._embedder.embed_documents(["probe"])
        self.embedding_size = len(result.embeddings[0])

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        result = await self._embedder.embed_documents([input])
        embedding: NDArray[np.float32] = np.array(
            result.embeddings[0], dtype=np.float32
        )
        if self.embedding_size == 0:
            self.embedding_size = len(embedding)
        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = (embedding / norm).astype(np.float32)
        return embedding

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        if not input:
            if self.embedding_size == 0:
                await self._probe_embedding_size()
            return np.empty((0, self.embedding_size), dtype=np.float32)
        result = await self._embedder.embed_documents(input)
        embeddings: NDArray[np.float32] = np.array(result.embeddings, dtype=np.float32)
        if self.embedding_size == 0:
            self.embedding_size = embeddings.shape[1]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms > 0, norms, np.float32(1.0))
        embeddings = (embeddings / norms).astype(np.float32)
        return embeddings


# ---------------------------------------------------------------------------
# Provider auto-detection
# ---------------------------------------------------------------------------


def _needs_azure_fallback(provider: str) -> bool:
    """Return True if *provider* is ``openai`` but only Azure credentials exist."""
    return (
        provider == "openai"
        and not os.getenv("OPENAI_API_KEY")
        and bool(os.getenv("AZURE_OPENAI_API_KEY"))
    )


def _make_azure_provider(
    endpoint_envvar: str = "AZURE_OPENAI_ENDPOINT",
    api_key_envvar: str = "AZURE_OPENAI_API_KEY",
):
    """Create a :class:`pydantic_ai.providers.azure.AzureProvider`.

    Constructs an ``AsyncAzureOpenAI`` client from the given environment
    variables and wraps it in an ``AzureProvider``.  The endpoint env-var
    may contain a full Azure deployment URL (including path and
    ``api-version`` query parameter) — the same format used throughout
    this codebase.

    When ``AZURE_OPENAI_API_KEY`` is set to ``"identity"``, the client
    uses Azure Managed Identity via a token provider callback, which
    refreshes tokens automatically before each request.
    """
    from openai import AsyncAzureOpenAI
    from pydantic_ai.providers.azure import AzureProvider

    from .utils import parse_azure_endpoint

    raw_key = os.environ[api_key_envvar]
    azure_endpoint, api_version = parse_azure_endpoint(endpoint_envvar)

    if raw_key.lower() == "identity":
        from .auth import get_shared_token_provider

        token_provider = get_shared_token_provider()
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_ad_token_provider=token_provider.get_token,
        )
    else:
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=raw_key,
        )
    return AzureProvider(openai_client=client)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


DEFAULT_CHAT_SPEC = "openai:gpt-4o"


def create_chat_model(
    model_spec: str | None = None,
) -> PydanticAIChatModel:
    """Create a chat model from a ``provider:model`` spec.

    Delegates to :func:`pydantic_ai.models.infer_model` for provider wiring.
    If the spec uses ``openai:`` and ``OPENAI_API_KEY`` is not set but
    ``AZURE_OPENAI_API_KEY`` is, Azure OpenAI is used automatically.

    If *model_spec* is ``None``, it is constructed from the ``OPENAI_MODEL``
    environment variable (falling back to :data:`DEFAULT_CHAT_SPEC`).

    Examples::

        model = create_chat_model()  # uses OPENAI_MODEL or gpt-4o
        model = create_chat_model("openai:gpt-4o")
        model = create_chat_model("anthropic:claude-sonnet-4-20250514")
        model = create_chat_model("google:gemini-2.0-flash")
    """
    if model_spec is None:
        openai_model = os.getenv("OPENAI_MODEL")
        if openai_model:
            model_spec = f"openai:{openai_model}"
        else:
            model_spec = DEFAULT_CHAT_SPEC
    provider, _, model_name = model_spec.partition(":")
    if _needs_azure_fallback(provider):
        from pydantic_ai.models.openai import OpenAIChatModel

        model = OpenAIChatModel(model_name, provider=_make_azure_provider())
    else:
        model = infer_model(model_spec)
    return PydanticAIChatModel(model)


DEFAULT_EMBEDDING_SPEC = "openai:text-embedding-3-small"


def create_embedding_model(
    model_spec: str | None = None,
    *,
    embedding_size: int = 0,
) -> CachingEmbeddingModel:
    """Create an embedding model from a ``provider:model`` spec.

    Delegates to :class:`pydantic_ai.Embedder` for provider wiring.
    If the spec uses ``openai:`` and ``OPENAI_API_KEY`` is not set but
    ``AZURE_OPENAI_API_KEY`` is, Azure OpenAI is used automatically.

    If *model_spec* is ``None``, :data:`DEFAULT_EMBEDDING_SPEC` is used.
    If *embedding_size* is not given, it will be probed automatically
    on the first embedding call.

    Returns a :class:`~typeagent.aitools.embeddings.CachingEmbeddingModel`
    wrapping a :class:`PydanticAIEmbedder`.

    Examples::

        model = create_embedding_model("openai:text-embedding-3-small")
        model = create_embedding_model("cohere:embed-english-v3.0")
        model = create_embedding_model("google:text-embedding-004")
    """
    if model_spec is None:
        model_spec = DEFAULT_EMBEDDING_SPEC
    provider, _, model_name = model_spec.partition(":")
    if not model_name:
        model_name = provider  # No colon in spec
    if _needs_azure_fallback(provider):
        from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel

        from .embeddings import model_to_embedding_size_and_envvar

        # Look up model-specific Azure endpoint, falling back to the generic one.
        _, suggested_envvar = model_to_embedding_size_and_envvar.get(
            model_name, (None, None)
        )
        if suggested_envvar and os.getenv(suggested_envvar):
            endpoint_envvar = suggested_envvar
        else:
            endpoint_envvar = "AZURE_OPENAI_ENDPOINT_EMBEDDING"
        # Allow a model-specific API key, falling back to the generic one.
        api_key_envvar = "AZURE_OPENAI_API_KEY_EMBEDDING"
        if not os.getenv(api_key_envvar):
            api_key_envvar = "AZURE_OPENAI_API_KEY"

        azure_provider = _make_azure_provider(endpoint_envvar, api_key_envvar)
        embedding_model = OpenAIEmbeddingModel(model_name, provider=azure_provider)
        embedder = _PydanticAIEmbedder(embedding_model)
    else:
        embedder = _PydanticAIEmbedder(model_spec)
    return CachingEmbeddingModel(
        PydanticAIEmbedder(embedder, model_name, embedding_size)
    )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _hashish(s: str) -> int:
    """Deterministic hash function for fake embeddings (hash() varies per run)."""
    h = 0
    for ch in s:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def _compute_fake_embeddings(
    input_texts: list[str], embedding_size: int
) -> list[list[float]]:
    """Generate deterministic fake embeddings for testing (unnormalized).

    Raises :class:`ValueError` on empty input strings.
    """
    prime = 1961
    result: list[list[float]] = []
    for item in input_texts:
        if not item:
            raise ValueError("Empty input text")
        length = len(item)
        floats: list[float] = []
        for i in range(embedding_size):
            cut = i % length
            scrambled = item[cut:] + item[:cut]
            hashed = _hashish(scrambled)
            reduced = (hashed % prime) / prime
            floats.append(reduced)
        result.append(floats)
    return result


class _FakePydanticAIEmbeddingModel(_PydanticAIEmbeddingModelBase):
    """A pydantic_ai :class:`EmbeddingModel` that returns deterministic fake
    embeddings.  Used only for testing — no network calls are made."""

    def __init__(self, embedding_size: int = 3) -> None:
        super().__init__()
        self._embedding_size = embedding_size

    @property
    def model_name(self) -> str:
        return "test"

    @property
    def system(self) -> str:
        return "test"

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        inputs_list, settings = self.prepare_embed(inputs, settings)
        embeddings = _compute_fake_embeddings(inputs_list, self._embedding_size)
        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs_list,
            input_type=input_type,
            model_name="test",
            provider_name="test",
        )


def create_test_embedding_model(
    embedding_size: int = 3,
) -> CachingEmbeddingModel:
    """Create a :class:`CachingEmbeddingModel` with deterministic fake
    embeddings for testing.  No API keys or network access required."""
    fake_model = _FakePydanticAIEmbeddingModel(embedding_size)
    pydantic_embedder = _PydanticAIEmbedder(fake_model)
    return CachingEmbeddingModel(
        PydanticAIEmbedder(pydantic_embedder, "test", embedding_size)
    )


def configure_models(
    chat_model_spec: str,
    embedding_model_spec: str,
    *,
    embedding_size: int = 0,
) -> tuple[PydanticAIChatModel, CachingEmbeddingModel]:
    """Configure both a chat model and an embedding model at once.

    Delegates to pydantic_ai's model registry for provider wiring.

    Example::

        chat, embedder = configure_models(
            "openai:gpt-4o",
            "openai:text-embedding-3-small",
        )

        settings = ConversationSettings(model=embedder)
        extractor = KnowledgeExtractor(model=chat)
    """
    return (
        create_chat_model(chat_model_spec),
        create_embedding_model(embedding_model_spec, embedding_size=embedding_size),
    )
