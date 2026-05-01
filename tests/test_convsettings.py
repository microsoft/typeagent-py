# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

from typeagent.aitools.embeddings import NormalizedEmbedding, NormalizedEmbeddings
from typeagent.knowpro.convsettings import (
    ConversationSettings,
    DEFAULT_MESSAGE_TEXT_MIN_SCORE,
    DEFAULT_RELATED_TERM_MIN_SCORE,
)


class FakeEmbeddingModel:
    """Minimal embedding model stub for settings tests."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        del key, embedding

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        del input
        return np.array([1.0], dtype=np.float32)

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        del input
        return np.array([[1.0]], dtype=np.float32)

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        del key
        return np.array([1.0], dtype=np.float32)

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        del keys
        return np.array([[1.0]], dtype=np.float32)


def test_conversation_settings_keep_normal_application_thresholds() -> None:
    settings = ConversationSettings(model=FakeEmbeddingModel("text-embedding-3-small"))

    assert (
        settings.related_term_index_settings.embedding_index_settings.min_score
        == DEFAULT_RELATED_TERM_MIN_SCORE
    )
    assert settings.thread_settings.min_score == DEFAULT_RELATED_TERM_MIN_SCORE
    assert (
        settings.message_text_index_settings.embedding_index_settings.min_score
        == DEFAULT_MESSAGE_TEXT_MIN_SCORE
    )
    assert (
        settings.related_term_index_settings.embedding_index_settings.max_matches == 50
    )
    assert (
        settings.message_text_index_settings.embedding_index_settings.max_matches
        is None
    )
