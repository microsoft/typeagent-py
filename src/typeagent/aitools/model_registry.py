# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Provider-agnostic model configuration.

Create chat and embedding models from ``provider/model`` spec strings::

    from typeagent.aitools.model_registry import configure_models

    chat, embedder = configure_models(
        "openai/gpt-4o",
        "openai/text-embedding-3-small",
    )

Supported built-in providers
-----------------------------

* ``openai/<model>`` — requires ``OPENAI_API_KEY`` env var.
* ``azure/<deployment>`` — requires ``AZURE_OPENAI_API_KEY`` (and
  ``AZURE_OPENAI_ENDPOINT``) env vars.  For Azure, the *model* part of the
  spec is the **deployment name**.

Extending with new providers
----------------------------

Implement ``typechat.TypeChatLanguageModel`` for chat, or
``IEmbeddingModel`` for embeddings, then register a factory::

    from typeagent.aitools.model_registry import (
        register_chat_provider,
        register_embedding_provider,
    )

    register_chat_provider("anthropic", my_anthropic_chat_factory)
    register_embedding_provider("gemini", my_gemini_embedding_factory)

Each factory is a callable ``(model_name: str) -> Model``.
"""

from collections.abc import Callable
import os

import typechat

from .embeddings import AsyncEmbeddingModel, IEmbeddingModel

# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------

type ChatModelFactory = Callable[[str], typechat.TypeChatLanguageModel]
type EmbeddingModelFactory = Callable[[str], IEmbeddingModel]


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse ``'provider/model'`` into ``(provider, model_name)``.

    Raises ``ValueError`` on malformed specs.
    """
    parts = spec.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid model spec {spec!r}. "
            f"Expected 'provider/model', e.g. 'openai/gpt-4o'."
        )
    return parts[0], parts[1]


# ---------------------------------------------------------------------------
# Chat model registry
# ---------------------------------------------------------------------------

_chat_providers: dict[str, ChatModelFactory] = {}


def register_chat_provider(provider: str, factory: ChatModelFactory) -> None:
    """Register a factory that creates chat models for *provider*."""
    _chat_providers[provider] = factory


def _openai_chat(model_name: str) -> typechat.TypeChatLanguageModel:
    env: dict[str, str | None] = dict(os.environ)
    if not env.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY required for openai/ chat models.")
    env["OPENAI_MODEL"] = model_name
    # Force the OpenAI path even when Azure env vars are also present.
    env.pop("AZURE_OPENAI_API_KEY", None)
    return typechat.create_language_model(env)


def _azure_chat(model_name: str) -> typechat.TypeChatLanguageModel:
    from .auth import AzureTokenProvider, get_shared_token_provider
    from .utils import DEFAULT_MAX_RETRY_ATTEMPTS, DEFAULT_TIMEOUT_SECONDS, ModelWrapper

    env: dict[str, str | None] = dict(os.environ)
    key = env.get("AZURE_OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AZURE_OPENAI_API_KEY required for azure/ chat models.")
    env["OPENAI_MODEL"] = model_name
    # Force the Azure path even when OPENAI_API_KEY is also present.
    env.pop("OPENAI_API_KEY", None)

    shared_token_provider: AzureTokenProvider | None = None
    if isinstance(key, str) and key.lower() == "identity":
        shared_token_provider = get_shared_token_provider()
        env["AZURE_OPENAI_API_KEY"] = shared_token_provider.get_token()

    model = typechat.create_language_model(env)
    model.timeout_seconds = DEFAULT_TIMEOUT_SECONDS
    model.max_retry_attempts = DEFAULT_MAX_RETRY_ATTEMPTS
    if shared_token_provider is not None:
        model = ModelWrapper(model, shared_token_provider)
    return model


register_chat_provider("openai", _openai_chat)
register_chat_provider("azure", _azure_chat)


# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------

_embedding_providers: dict[str, EmbeddingModelFactory] = {}


def register_embedding_provider(provider: str, factory: EmbeddingModelFactory) -> None:
    """Register a factory that creates embedding models for *provider*."""
    _embedding_providers[provider] = factory


def _openai_embedding(model_name: str) -> IEmbeddingModel:
    return AsyncEmbeddingModel(model_name=model_name, use_azure=False)


def _azure_embedding(model_name: str) -> IEmbeddingModel:
    return AsyncEmbeddingModel(model_name=model_name, use_azure=True)


register_embedding_provider("openai", _openai_embedding)
register_embedding_provider("azure", _azure_embedding)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_chat_model(
    model_spec: str,
) -> typechat.TypeChatLanguageModel:
    """Create a chat model from a ``provider/model`` spec.

    Examples::

        model = create_chat_model("openai/gpt-4o")
        model = create_chat_model("azure/my-gpt4o-deployment")

    For Azure, *model* is the **deployment name**, not the underlying
    model name.
    """
    provider, model_name = _parse_model_spec(model_spec)
    factory = _chat_providers.get(provider)
    if factory is None:
        avail = ", ".join(sorted(_chat_providers)) or "(none)"
        raise ValueError(
            f"Unknown chat provider {provider!r}. "
            f"Available: {avail}. "
            f"Use register_chat_provider() to add support."
        )
    return factory(model_name)


def create_embedding_model(
    model_spec: str,
) -> IEmbeddingModel:
    """Create an embedding model from a ``provider/model`` spec.

    Examples::

        model = create_embedding_model("openai/text-embedding-3-small")
        model = create_embedding_model("azure/text-embedding-3-small")
    """
    provider, model_name = _parse_model_spec(model_spec)
    factory = _embedding_providers.get(provider)
    if factory is None:
        avail = ", ".join(sorted(_embedding_providers)) or "(none)"
        raise ValueError(
            f"Unknown embedding provider {provider!r}. "
            f"Available: {avail}. "
            f"Use register_embedding_provider() to add support."
        )
    return factory(model_name)


def configure_models(
    chat_model_spec: str,
    embedding_model_spec: str,
) -> tuple[typechat.TypeChatLanguageModel, IEmbeddingModel]:
    """Configure both a chat model and an embedding model at once.

    Example::

        chat, embedder = configure_models(
            "openai/gpt-4o",
            "openai/text-embedding-3-small",
        )

        settings = ConversationSettings(model=embedder)
        extractor = KnowledgeExtractor(model=chat)
    """
    return create_chat_model(chat_model_spec), create_embedding_model(
        embedding_model_spec
    )
