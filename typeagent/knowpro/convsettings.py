# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from dataclasses import dataclass

from ..aitools.embeddings import AsyncEmbeddingModel
from ..aitools.vectorbase import TextEmbeddingIndexSettings
from .interfaces import IKnowledgeExtractor, IStorageProvider


@dataclass
class MessageTextIndexSettings:
    embedding_index_settings: TextEmbeddingIndexSettings

    def __init__(self, embedding_index_settings: TextEmbeddingIndexSettings):
        self.embedding_index_settings = embedding_index_settings


@dataclass
class RelatedTermIndexSettings:
    embedding_index_settings: TextEmbeddingIndexSettings

    def __init__(self, embedding_index_settings: TextEmbeddingIndexSettings):
        self.embedding_index_settings = embedding_index_settings


@dataclass
class SemanticRefIndexSettings:
    batch_size: int
    auto_extract_knowledge: bool
    knowledge_extractor: IKnowledgeExtractor | None = None


class ConversationSettings:
    """Settings for conversation processing and indexing."""

    def __init__(
        self,
        model: AsyncEmbeddingModel | None = None,
        storage_provider: IStorageProvider | None = None,
    ):
        # All settings share the same model, so they share the embedding cache.
        model = model or AsyncEmbeddingModel()
        self.embedding_model = model
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

        # Storage provider will be created lazily if not provided
        self._storage_provider: IStorageProvider | None = storage_provider
        self._storage_provider_created = storage_provider is not None

    @property
    def storage_provider(self) -> IStorageProvider:
        if not self._storage_provider_created:
            raise RuntimeError(
                "Storage provider not initialized. Use await ConversationSettings.get_storage_provider() "
                "or provide storage_provider in constructor."
            )
        assert (
            self._storage_provider is not None
        ), "Storage provider should be set when _storage_provider_created is True"
        return self._storage_provider

    @storage_provider.setter
    def storage_provider(self, value: IStorageProvider) -> None:
        self._storage_provider = value
        self._storage_provider_created = True

    async def get_storage_provider(self) -> IStorageProvider:
        """Get or create the storage provider asynchronously."""
        if not self._storage_provider_created:
            from ..storage.memory import MemoryStorageProvider

            self._storage_provider = MemoryStorageProvider(
                message_text_settings=self.message_text_index_settings,
                related_terms_settings=self.related_term_index_settings,
            )
            self._storage_provider_created = True
        assert (
            self._storage_provider is not None
        ), "Storage provider should be set after creation"
        return self._storage_provider
