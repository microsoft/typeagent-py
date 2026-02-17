# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

import typechat

from typeagent.aitools.embeddings import (
    AsyncEmbeddingModel,
    IEmbeddingModel,
    TEST_MODEL_NAME,
)
from typeagent.aitools.model_registry import (
    _chat_providers,
    _embedding_providers,
    _parse_model_spec,
    configure_models,
    create_chat_model,
    create_embedding_model,
    register_chat_provider,
    register_embedding_provider,
)

# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


def test_parse_valid_specs() -> None:
    assert _parse_model_spec("openai/gpt-4o") == ("openai", "gpt-4o")
    assert _parse_model_spec("azure/my-deployment") == ("azure", "my-deployment")
    assert _parse_model_spec("anthropic/claude-3.5-sonnet") == (
        "anthropic",
        "claude-3.5-sonnet",
    )


def test_parse_spec_preserves_slashes() -> None:
    """Only the first '/' is a separator; the rest belong to the model name."""
    assert _parse_model_spec("provider/model/variant") == (
        "provider",
        "model/variant",
    )


def test_parse_invalid_specs() -> None:
    with pytest.raises(ValueError, match="Invalid model spec"):
        _parse_model_spec("noslash")
    with pytest.raises(ValueError, match="Invalid model spec"):
        _parse_model_spec("/model")
    with pytest.raises(ValueError, match="Invalid model spec"):
        _parse_model_spec("provider/")
    with pytest.raises(ValueError, match="Invalid model spec"):
        _parse_model_spec("")


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------


def test_builtin_providers_registered() -> None:
    assert "openai" in _chat_providers
    assert "azure" in _chat_providers
    assert "openai" in _embedding_providers
    assert "azure" in _embedding_providers


# ---------------------------------------------------------------------------
# Unknown provider errors
# ---------------------------------------------------------------------------


def test_unknown_chat_provider() -> None:
    with pytest.raises(ValueError, match="Unknown chat provider"):
        create_chat_model("magical/unicorn")


def test_unknown_embedding_provider() -> None:
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedding_model("magical/unicorn")


# ---------------------------------------------------------------------------
# Custom provider registration
# ---------------------------------------------------------------------------


class FakeChatModel(typechat.TypeChatLanguageModel):
    """Minimal chat model for registry tests."""

    async def complete(
        self, prompt: str | list[typechat.PromptSection]
    ) -> typechat.Result[str]:
        return typechat.Success("fake")


def test_register_and_use_custom_chat_provider() -> None:
    instance = FakeChatModel()
    register_chat_provider("_test_chat", lambda name: instance)
    try:
        result = create_chat_model("_test_chat/any-model")
        assert result is instance
    finally:
        _chat_providers.pop("_test_chat", None)


def test_register_and_use_custom_embedding_provider() -> None:
    instance = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    register_embedding_provider("_test_embed", lambda name: instance)
    try:
        result = create_embedding_model("_test_embed/any-model")
        assert result is instance
        assert isinstance(result, IEmbeddingModel)
    finally:
        _embedding_providers.pop("_test_embed", None)


def test_model_name_forwarded_to_factory() -> None:
    """The model portion of the spec is passed to the factory."""
    received: list[str] = []

    def capture_factory(model_name: str) -> IEmbeddingModel:
        received.append(model_name)
        return AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)

    register_embedding_provider("_test_fwd", capture_factory)
    try:
        create_embedding_model("_test_fwd/text-embedding-3-small")
        assert received == ["text-embedding-3-small"]
    finally:
        _embedding_providers.pop("_test_fwd", None)


# ---------------------------------------------------------------------------
# configure_models
# ---------------------------------------------------------------------------


def test_configure_models() -> None:
    chat_instance = FakeChatModel()
    embed_instance = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)

    register_chat_provider("_test_cm", lambda name: chat_instance)
    register_embedding_provider("_test_cm", lambda name: embed_instance)
    try:
        chat, embedder = configure_models("_test_cm/chat", "_test_cm/embed")
        assert chat is chat_instance
        assert embedder is embed_instance
    finally:
        _chat_providers.pop("_test_cm", None)
        _embedding_providers.pop("_test_cm", None)
