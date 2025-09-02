# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from fixtures import (
    memory_storage,
    needs_auth,
    embedding_model,
    FakeConversation,
    FakeMessage,
)  # Import the storage fixture
from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.convsettings import MessageTextIndexSettings
from typeagent.knowpro.convsettings import RelatedTermIndexSettings
from typeagent.storage.memory.timestampindex import TimestampToTextRangeIndex
from typeagent.storage.memory import MemoryStorageProvider
from typeagent.knowpro.secindex import (
    ConversationSecondaryIndexes,
    build_secondary_indexes,
    build_transient_secondary_indexes,
)
from typeagent.storage.memory import (
    MemoryMessageCollection as MemoryMessageCollection,
    MemorySemanticRefCollection,
)

from fixtures import needs_auth  # type: ignore  # Yes it is used!


@pytest.fixture
def simple_conversation() -> FakeConversation:
    return FakeConversation()


@pytest.fixture
def conversation_settings(needs_auth: None) -> ConversationSettings:
    from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME

    model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    return ConversationSettings(model)


def test_conversation_secondary_indexes_initialization(
    memory_storage: MemoryStorageProvider, needs_auth: None
):
    """Test initialization of ConversationSecondaryIndexes."""
    storage_provider = memory_storage
    # Create proper settings for testing
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(test_model)
    settings = RelatedTermIndexSettings(embedding_settings)
    indexes = ConversationSecondaryIndexes(storage_provider, settings)
    # Note: indexes are None until initialize() is called
    assert indexes.property_to_semantic_ref_index is None
    assert indexes.timestamp_index is None
    assert indexes.term_to_related_terms_index is None

    # Test with custom settings
    settings2 = RelatedTermIndexSettings(embedding_settings)
    indexes_with_settings = ConversationSecondaryIndexes(storage_provider, settings2)
    assert indexes_with_settings.property_to_semantic_ref_index is None


@pytest.mark.asyncio
async def test_build_secondary_indexes(
    simple_conversation: FakeConversation, conversation_settings: ConversationSettings
):
    """Test building secondary indexes asynchronously."""
    # Ensure the conversation is properly initialized
    await simple_conversation.ensure_initialized()
    assert simple_conversation.secondary_indexes is not None
    simple_conversation.secondary_indexes.timestamp_index = TimestampToTextRangeIndex()

    # Add some dummy data to the conversation
    await simple_conversation.messages.append(FakeMessage("Message 1"))
    await simple_conversation.messages.append(FakeMessage("Message 2"))

    await build_secondary_indexes(simple_conversation, conversation_settings)

    # Verify that the indexes were built by checking they exist
    assert simple_conversation.secondary_indexes is not None


@pytest.mark.asyncio
async def test_build_transient_secondary_indexes(
    simple_conversation: FakeConversation, needs_auth: None
):
    """Test building transient secondary indexes."""
    # Ensure the conversation is properly initialized
    await simple_conversation.ensure_initialized()
    assert simple_conversation.secondary_indexes is not None
    simple_conversation.secondary_indexes.timestamp_index = TimestampToTextRangeIndex()

    # Add some dummy data to the conversation
    await simple_conversation.messages.append(FakeMessage("Message 1"))
    await simple_conversation.messages.append(FakeMessage("Message 2"))

    await build_transient_secondary_indexes(
        simple_conversation, simple_conversation.settings
    )

    # Verify that the indexes were built by checking they exist
    assert simple_conversation.secondary_indexes is not None
