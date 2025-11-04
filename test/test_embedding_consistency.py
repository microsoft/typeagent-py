# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test embedding consistency checks between database and settings."""

import pytest
import tempfile
import os
from typeagent import create_conversation
from typeagent.transcripts.transcript import TranscriptMessage, TranscriptMessageMeta
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.storage.sqlite import SqliteStorageProvider


@pytest.mark.asyncio
async def test_embedding_size_mismatch_in_message_index():
    """Test that opening a DB with mismatched embedding size raises an error."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create a conversation with test model (embedding size 3)
        settings1 = ConversationSettings(
            model=AsyncEmbeddingModel(embedding_size=3, model_name="test")
        )
        # Disable LLM knowledge extraction to avoid API key requirement
        settings1.semantic_ref_index_settings.auto_extract_knowledge = False
        conv1 = await create_conversation(
            db_path, TranscriptMessage, settings=settings1
        )

        # Add some messages to populate the index
        messages = [
            TranscriptMessage(
                text_chunks=["Hello world"],
                metadata=TranscriptMessageMeta(speaker="Alice"),
            )
        ]
        await conv1.add_messages_with_indexing(messages)
        await conv1.storage_provider.close()

        # Now try to open the same database with a different embedding size
        # This should raise an error
        settings2 = ConversationSettings(
            model=AsyncEmbeddingModel(embedding_size=5, model_name="test")
        )

        with pytest.raises(ValueError, match="embedding size mismatch"):
            provider = SqliteStorageProvider(
                db_path=db_path,
                message_type=TranscriptMessage,
                message_text_index_settings=settings2.message_text_index_settings,
                related_term_index_settings=settings2.related_term_index_settings,
            )
            await provider.close()

    finally:
        # Clean up the temporary database
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_embedding_size_mismatch_in_related_terms():
    """Test that opening a DB with mismatched embedding size in related terms raises an error."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create a conversation with default embedding size
        settings1 = ConversationSettings(
            model=AsyncEmbeddingModel(embedding_size=3, model_name="test")
        )
        # Disable LLM knowledge extraction to avoid API key requirement
        settings1.semantic_ref_index_settings.auto_extract_knowledge = False
        conv1 = await create_conversation(
            db_path, TranscriptMessage, settings=settings1
        )

        # Add some messages to populate the related terms index
        messages = [
            TranscriptMessage(
                text_chunks=["Apple is a fruit"],
                metadata=TranscriptMessageMeta(speaker="Alice"),
            )
        ]
        await conv1.add_messages_with_indexing(messages)
        await conv1.storage_provider.close()

        # Now try to open the same database with a different embedding size
        # This should raise an error
        settings2 = ConversationSettings(
            model=AsyncEmbeddingModel(embedding_size=5, model_name="test")
        )

        with pytest.raises(ValueError, match="embedding size mismatch"):
            provider = SqliteStorageProvider(
                db_path=db_path,
                message_type=TranscriptMessage,
                message_text_index_settings=settings2.message_text_index_settings,
                related_term_index_settings=settings2.related_term_index_settings,
            )
            await provider.close()

    finally:
        # Clean up the temporary database
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_empty_db_no_error():
    """Test that opening an empty DB doesn't raise an error regardless of embedding size."""
    # Create a temporary database file
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create an empty database
        settings1 = ConversationSettings(
            model=AsyncEmbeddingModel(embedding_size=3, model_name="test")
        )
        # Disable LLM knowledge extraction to avoid API key requirement
        settings1.semantic_ref_index_settings.auto_extract_knowledge = False
        conv1 = await create_conversation(
            db_path, TranscriptMessage, settings=settings1
        )
        await conv1.storage_provider.close()

        # Open with different embedding size should work since DB is empty
        settings2 = ConversationSettings(
            model=AsyncEmbeddingModel(embedding_size=5, model_name="test")
        )
        provider = SqliteStorageProvider(
            db_path=db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings2.message_text_index_settings,
            related_term_index_settings=settings2.related_term_index_settings,
        )
        await provider.close()

    finally:
        # Clean up the temporary database
        if os.path.exists(db_path):
            os.unlink(db_path)
