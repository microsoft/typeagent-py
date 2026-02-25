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

See https://ai.pydantic.dev/models/ for all supported providers and their
required environment variables.
"""

import numpy as np
from numpy.typing import NDArray

from pydantic_ai import Embedder as _PydanticAIEmbedder
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import infer_model, Model, ModelRequestParameters
import typechat

from .embeddings import IEmbeddingModel, NormalizedEmbedding, NormalizedEmbeddings

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


class PydanticAIEmbeddingModel(IEmbeddingModel):
    """Adapter from :class:`pydantic_ai.Embedder` to :class:`IEmbeddingModel`.

    This lets any pydantic_ai embedding provider (OpenAI, Cohere, Google, …)
    be used wherever the codebase expects an ``IEmbeddingModel``, including
    :class:`~typeagent.aitools.vectorbase.VectorBase` and
    :class:`~typeagent.knowpro.convsettings.ConversationSettings`.

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
        self._cache: dict[str, NormalizedEmbedding] = {}

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        self._cache[key] = embedding

    async def _probe_embedding_size(self) -> None:
        """Discover embedding_size by making a single API call."""
        result = await self._embedder.embed(["probe"], input_type="document")
        self.embedding_size = len(result.embeddings[0])

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        result = await self._embedder.embed([input], input_type="document")
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
        result = await self._embedder.embed(input, input_type="document")
        embeddings: NDArray[np.float32] = np.array(result.embeddings, dtype=np.float32)
        if self.embedding_size == 0:
            self.embedding_size = embeddings.shape[1]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms > 0, norms, np.float32(1.0))
        embeddings = (embeddings / norms).astype(np.float32)
        return embeddings

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embedding = await self.get_embedding_nocache(key)
        self._cache[key] = embedding
        return embedding

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        missing_keys = [k for k in keys if k not in self._cache]
        if missing_keys:
            fresh = await self.get_embeddings_nocache(missing_keys)
            for i, k in enumerate(missing_keys):
                self._cache[k] = fresh[i]
        return np.array([self._cache[k] for k in keys], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_chat_model(
    model_spec: str,
) -> PydanticAIChatModel:
    """Create a chat model from a ``provider:model`` spec.

    Delegates to :func:`pydantic_ai.models.infer_model` for provider wiring.

    Examples::

        model = create_chat_model("openai:gpt-4o")
        model = create_chat_model("anthropic:claude-sonnet-4-20250514")
        model = create_chat_model("google:gemini-2.0-flash")
    """
    model = infer_model(model_spec)
    return PydanticAIChatModel(model)


DEFAULT_EMBEDDING_SPEC = "openai:text-embedding-3-small"


def create_embedding_model(
    model_spec: str | None = None,
    *,
    embedding_size: int = 0,
) -> PydanticAIEmbeddingModel:
    """Create an embedding model from a ``provider:model`` spec.

    Delegates to :class:`pydantic_ai.Embedder` for provider wiring.

    If *model_spec* is ``None``, :data:`DEFAULT_EMBEDDING_SPEC` is used.
    If *embedding_size* is not given, it will be probed automatically
    on the first embedding call.

    Examples::

        model = create_embedding_model("openai:text-embedding-3-small")
        model = create_embedding_model("cohere:embed-english-v3.0")
        model = create_embedding_model("google:text-embedding-004")
    """
    if model_spec is None:
        model_spec = DEFAULT_EMBEDDING_SPEC
    model_name = model_spec.split(":")[-1] if ":" in model_spec else model_spec
    embedder = _PydanticAIEmbedder(model_spec)
    return PydanticAIEmbeddingModel(embedder, model_name, embedding_size)


def configure_models(
    chat_model_spec: str,
    embedding_model_spec: str,
    *,
    embedding_size: int = 0,
) -> tuple[PydanticAIChatModel, PydanticAIEmbeddingModel]:
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
