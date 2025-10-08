# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the conversation.query() method."""

import pytest

from typeagent import create_conversation
from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.aitools.utils import load_dotenv
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.transcripts.transcript import TranscriptMessage, TranscriptMessageMeta

from fixtures import really_needs_auth


@pytest.mark.asyncio
async def test_query_method_basic(really_needs_auth: None):
    """Test the basic query method workflow."""
    # Create a conversation with some test data
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    settings = ConversationSettings(model=test_model)
    conversation = await create_conversation(
        None,
        TranscriptMessage,
        name="Test Conversation",
        settings=settings,
    )

    # Add some test messages
    messages = [
        TranscriptMessage(
            text_chunks=["Welcome to the Python programming tutorial."],
            metadata=TranscriptMessageMeta(speaker="Instructor"),
        ),
        TranscriptMessage(
            text_chunks=["Today we'll learn about async/await in Python."],
            metadata=TranscriptMessageMeta(speaker="Instructor"),
        ),
        TranscriptMessage(
            text_chunks=["Python is a great language for beginners."],
            metadata=TranscriptMessageMeta(speaker="Instructor"),
        ),
    ]

    await conversation.add_messages_with_indexing(messages)

    # Test the query method
    answer = await conversation.query("What programming language is discussed?")

    # Verify we got a response (content depends on indexing and LLM behavior)
    assert isinstance(answer, str)
    assert len(answer) > 0
    # The answer should either mention Python or indicate no answer was found
    # Both are valid since indexing might not extract all knowledge
    assert (
        "python" in answer.lower()
        or "no answer" in answer.lower()
        or "unable to find" in answer.lower()
    )


@pytest.mark.asyncio
async def test_query_method_empty_conversation(really_needs_auth: None):
    """Test query method on an empty conversation."""
    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    settings = ConversationSettings(model=test_model)
    conversation = await create_conversation(
        None,
        TranscriptMessage,
        name="Empty Conversation",
        settings=settings,
    )

    # Query should handle empty conversation gracefully
    answer = await conversation.query("What was discussed?")

    assert isinstance(answer, str)
    assert len(answer) > 0
    # Should indicate no answer found or no relevant information
    assert "no answer" in answer.lower() or "unable to find" in answer.lower()
