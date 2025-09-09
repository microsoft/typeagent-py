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
