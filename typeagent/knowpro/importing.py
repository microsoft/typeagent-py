# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field

from ..aitools.embeddings import AsyncEmbeddingModel


# TODO: TextEmbeddingIndexSettings belongs in vectorbase.py.
@dataclass
class TextEmbeddingIndexSettings:
    embedding_model: AsyncEmbeddingModel | None = None
    embedding_size: int | None = None
    min_score: float = 0.0
    max_matches: int | None = None
    retry_max_attempts: int = 2
    retry_delay: float = 2.0  # Seconds
    batch_size: int = 8

    # TODO: Add embedding_model, embedding_size arguments (one supported).
    def __init__(
        self,
        embedding_model: AsyncEmbeddingModel | None = None,
        embedding_size: int | None = None,
        min_score: float | None = None,
        max_matches: int | None = None,
    ):
        if embedding_model is None:
            embedding_model = AsyncEmbeddingModel()
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size
        if min_score is None:
            min_score = 0.85
        self.min_score = min_score
        self.max_matches = max_matches


# TODO: RelatedTermIndexSettings belongs in relatedtermsindex.py.
@dataclass
class RelatedTermIndexSettings:
    embedding_index_settings: TextEmbeddingIndexSettings

    def __init__(
        self, embedding_index_settings: TextEmbeddingIndexSettings | None = None
    ):
        if embedding_index_settings is None:
            embedding_index_settings = TextEmbeddingIndexSettings()
        self.embedding_index_settings = embedding_index_settings


# TODO: MessageTextIndexSettings belongs in messageindex.py.
@dataclass
class MessageTextIndexSettings:
    embedding_index_settings: TextEmbeddingIndexSettings

    def __init__(
        self, embedding_index_settings: TextEmbeddingIndexSettings | None = None
    ):
        if embedding_index_settings is None:
            embedding_index_settings = TextEmbeddingIndexSettings()
        self.embedding_index_settings = embedding_index_settings


@dataclass
class ConversationSettings:
    related_term_index_settings: RelatedTermIndexSettings
    thread_settings: TextEmbeddingIndexSettings
    message_text_index_settings: MessageTextIndexSettings

    def __init__(self):
        # All settings share the same model, so they share the embedding cache.
        model = AsyncEmbeddingModel()
        min_score = 0.85
        self.related_term_index_settings = RelatedTermIndexSettings(
            TextEmbeddingIndexSettings(model, min_score=min_score, max_matches=50)
        )
        self.thread_settings = TextEmbeddingIndexSettings(model, min_score=min_score)
        self.message_text_index_settings = MessageTextIndexSettings(
            TextEmbeddingIndexSettings(model, min_score=min_score)
        )
