# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from fixtures import needs_auth, memory_storage, embedding_model  # type: ignore  # It's used!
from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.convsettings import MessageTextIndexSettings
from typeagent.knowpro.convsettings import RelatedTermIndexSettings
from typeagent.storage.memory import MemoryStorageProvider


@pytest.mark.asyncio
async def test_all_index_creation(
    memory_storage: MemoryStorageProvider, needs_auth: None
):
    """Test that all 6 index types are created and accessible."""
    # storage fixture already initializes indexes

    # Test all index types are created and return objects
    conv_index = await memory_storage.get_semantic_ref_index()
    assert conv_index is not None

    prop_index = await memory_storage.get_property_index()
    assert prop_index is not None

    time_index = await memory_storage.get_timestamp_index()
    assert time_index is not None

    msg_index = await memory_storage.get_message_text_index()
    assert msg_index is not None

    rel_index = await memory_storage.get_related_terms_index()
    assert rel_index is not None

    threads = await memory_storage.get_conversation_threads()
    assert threads is not None


@pytest.mark.asyncio
async def test_index_persistence(
    memory_storage: MemoryStorageProvider, needs_auth: None
):
    """Test that same index instance is returned across calls."""
    # storage fixture already initializes indexes

    # All index types should return same instance across calls
    conv1 = await memory_storage.get_semantic_ref_index()
    conv2 = await memory_storage.get_semantic_ref_index()
    assert conv1 is conv2

    prop1 = await memory_storage.get_property_index()
    prop2 = await memory_storage.get_property_index()
    assert prop1 is prop2


@pytest.mark.asyncio
async def test_indexes_work_independently(needs_auth):
    """Test that different storage providers have independent indexes."""
    # Create two separate storage providers with test settings
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(test_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    storage1 = MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )
    storage2 = MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )

    # Get indexes from both storage providers
    index1 = await storage1.get_semantic_ref_index()
    index2 = await storage2.get_semantic_ref_index()

    # Should be different instances
    assert index1 is not index2


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_indexes_available_after_create(needs_auth):
    """Test that indexes are available after using create() factory method."""
    # Create storage provider with test settings
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(test_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    # Use the async factory method
    storage = MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )

    # Should work immediately after create
    conv_index = await storage.get_semantic_ref_index()
    assert conv_index is not None


@pytest.mark.asyncio
async def test_storage_provider_collections_still_work(needs_auth):
    """Test that existing collection functionality still works."""
    # Create storage provider with test settings
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(test_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    storage = MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )

    # Test message collection creation
    msg_collection = await storage.get_message_collection()
    assert msg_collection is not None
    assert await msg_collection.size() == 0

    # Test semantic ref collection creation
    ref_collection = await storage.get_semantic_ref_collection()
    assert ref_collection is not None
    assert await ref_collection.size() == 0
