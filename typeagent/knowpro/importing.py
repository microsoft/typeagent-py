# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass

from ..aitools.embeddings import AsyncEmbeddingModel
from ..aitools.vectorbase import TextEmbeddingIndexSettings

from typeagent.knowpro.convknowledge import KnowledgeExtractor


# TODO: RelatedTermIndexSettings belongs in reltermsindex.py.
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
class SemanticRefIndexSettings:
    batch_size: int
    auto_extract_knowledge: bool
    knowledge_extractor: KnowledgeExtractor | None = None


@dataclass
class ConversationSettings:
    related_term_index_settings: RelatedTermIndexSettings
    thread_settings: TextEmbeddingIndexSettings
    message_text_index_settings: MessageTextIndexSettings
    semantic_ref_index_settings: SemanticRefIndexSettings

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
        self.semantic_ref_index_settings = SemanticRefIndexSettings(
            batch_size=10,
            auto_extract_knowledge=False,
        )
