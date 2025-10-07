# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test create_conversation factory function."""

import pytest

from typeagent import create_conversation
from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.transcripts.transcript import TranscriptMessage, TranscriptMessageMeta


@pytest.mark.asyncio
async def test_create_conversation_minimal():
    """Test creating a conversation with minimal parameters."""
    # Create empty conversation with test model
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    settings = ConversationSettings(model=test_model)
    conversation = await create_conversation(
        None,
        TranscriptMessage,
        name="My Conversation",
        settings=settings,
    )

    # Verify conversation is properly initialized
    assert conversation.name_tag == "My Conversation"
    assert conversation.tags == []
    assert await conversation.messages.size() == 0
    assert await conversation.semantic_refs.size() == 0
    assert conversation.semantic_ref_index is not None
    assert conversation.secondary_indexes is not None


@pytest.mark.asyncio
async def test_create_conversation_with_tags():
    """Test creating a conversation with tags."""
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    settings = ConversationSettings(model=test_model)
    conversation = await create_conversation(
        None,
        TranscriptMessage,
        name="Tagged Conversation",
        tags=["test", "example"],
        settings=settings,
    )

    assert conversation.name_tag == "Tagged Conversation"
    assert conversation.tags == ["test", "example"]


@pytest.mark.asyncio
async def test_create_conversation_and_add_messages():
    """Test the complete workflow: create conversation and add messages."""
    # 1. Create empty conversation
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    settings = ConversationSettings(model=test_model)
    conversation = await create_conversation(
        None,
        TranscriptMessage,
        name="Test Conversation",
        settings=settings,
    )

    # 2. Prepare messages
    messages = [
        TranscriptMessage(
            text_chunks=["Hello, how are you?"],
            metadata=TranscriptMessageMeta(speaker="Alice"),
        ),
        TranscriptMessage(
            text_chunks=["I'm doing great, thanks!"],
            metadata=TranscriptMessageMeta(speaker="Bob"),
        ),
    ]

    # 3. Ingest with automatic indexing
    result = await conversation.add_messages_with_indexing(messages)

    # 4. Verify results
    assert result.messages_added == 2
    assert result.semrefs_added >= 0  # May extract metadata knowledge
    assert await conversation.messages.size() == 2

    # Verify messages were stored correctly
    msg0 = await conversation.messages.get_item(0)
    assert msg0.text_chunks == ["Hello, how are you?"]
    assert msg0.metadata.speaker == "Alice"

    msg1 = await conversation.messages.get_item(1)
    assert msg1.text_chunks == ["I'm doing great, thanks!"]
    assert msg1.metadata.speaker == "Bob"
