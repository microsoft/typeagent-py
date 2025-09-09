# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Test that ConversationSecondaryIndexes now uses storage provider properly
import pytest

from fixtures import needs_auth, memory_storage, embedding_model  # type: ignore  # It's used!
from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.convsettings import RelatedTermIndexSettings
from typeagent.knowpro.secindex import ConversationSecondaryIndexes
from typeagent.storage.memory import MemoryStorageProvider


@pytest.mark.asyncio
async def test_secondary_indexes_use_storage_provider(
    memory_storage: MemoryStorageProvider, needs_auth: None
):
    """Test that ConversationSecondaryIndexes gets indexes from storage provider."""
    storage_provider = memory_storage

    # Create test settings
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(test_model)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    indexes = await ConversationSecondaryIndexes.create(
        storage_provider, related_terms_settings
    )

    assert indexes.property_to_semantic_ref_index is not None
    assert indexes.timestamp_index is not None
    assert indexes.term_to_related_terms_index is not None
    assert indexes.threads is not None
    assert indexes.message_index is not None

    # Verify they are the same instances as those from storage provider
    storage_prop_index = await storage_provider.get_property_index()
    storage_timestamp_index = await storage_provider.get_timestamp_index()
    storage_related_terms = await storage_provider.get_related_terms_index()
    storage_threads = await storage_provider.get_conversation_threads()
    storage_message_index = await storage_provider.get_message_text_index()

    assert indexes.property_to_semantic_ref_index is storage_prop_index
    assert indexes.timestamp_index is storage_timestamp_index
    assert indexes.term_to_related_terms_index is storage_related_terms
    assert indexes.threads is storage_threads
    assert indexes.message_index is storage_message_index
