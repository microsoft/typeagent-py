# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field


# TODO: TextEmbeddingIndexSettings belongs in vectorbase.py.
@dataclass
class TextEmbeddingIndexSettings:
    # TODO: Add these (currently hardcoded in VectorBase):
    # embedding_model
    # embedding_size
    min_score: float
    max_matches: int | None = None
    retry_max_attempts: int = 2
    retry_delay: float = 2.0  # Seconds
    batch_size: int = 8

    # TODO: Add embedding_model, embedding_size arguments (one supported).
    def __init__(self, min_score: float | None = None, max_matches: int | None = None):
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
        min_score = 0.85
        self.related_term_index_settings = RelatedTermIndexSettings(
            TextEmbeddingIndexSettings(min_score, max_matches=50)
        )
        self.thread_settings = TextEmbeddingIndexSettings(min_score)
        self.message_text_index_settings = MessageTextIndexSettings(
            TextEmbeddingIndexSettings(min_score)
        )
