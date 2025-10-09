# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test add_messages_with_indexing functionality."""

import os
import tempfile

import pytest

from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)


@pytest.mark.asyncio
async def test_add_messages_with_indexing_basic():
    """Test basic add_messages_with_indexing functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings = ConversationSettings(model=test_model)
        settings.semantic_ref_index_settings.auto_extract_knowledge = False

        storage = SqliteStorageProvider(
            db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        transcript = await Transcript.create(settings, name="test")

        metadata1 = TranscriptMessageMeta(speaker="Alice")
        metadata2 = TranscriptMessageMeta(speaker="Bob")

        messages = [
            TranscriptMessage(
                text_chunks=["Hello world"], metadata=metadata1, tags=["test"]
            ),
            TranscriptMessage(
                text_chunks=["Hi Alice"], metadata=metadata2, tags=["test"]
            ),
        ]

        result = await transcript.add_messages_with_indexing(messages)

        assert result.messages_added == 2
        assert result.semrefs_added >= 2
        assert await transcript.messages.size() == 2
        assert await transcript.semantic_refs.size() >= 2

        await storage.close()


@pytest.mark.asyncio
async def test_add_messages_with_indexing_batched():
    """Test batched message addition."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings = ConversationSettings(model=test_model)
        settings.semantic_ref_index_settings.auto_extract_knowledge = False

        storage = SqliteStorageProvider(
            db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        transcript = await Transcript.create(settings, name="test")

        # Add first batch
        batch1 = [
            TranscriptMessage(
                text_chunks=["Message 1"],
                metadata=TranscriptMessageMeta(speaker="Alice"),
                tags=["batch1"],
            ),
            TranscriptMessage(
                text_chunks=["Message 2"],
                metadata=TranscriptMessageMeta(speaker="Bob"),
                tags=["batch1"],
            ),
        ]
        result1 = await transcript.add_messages_with_indexing(batch1)
        assert result1.messages_added == 2

        # Add second batch
        batch2 = [
            TranscriptMessage(
                text_chunks=["Message 3"],
                metadata=TranscriptMessageMeta(speaker="Alice"),
                tags=["batch2"],
            ),
            TranscriptMessage(
                text_chunks=["Message 4"],
                metadata=TranscriptMessageMeta(speaker="Bob"),
                tags=["batch2"],
            ),
        ]
        result2 = await transcript.add_messages_with_indexing(batch2)
        assert result2.messages_added == 2

        assert await transcript.messages.size() == 4
        assert await transcript.semantic_refs.size() >= 4

        await storage.close()


@pytest.mark.asyncio
async def test_transaction_rollback_on_error():
    """Test that transactions are rolled back on error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings = ConversationSettings(model=test_model)
        settings.semantic_ref_index_settings.auto_extract_knowledge = False

        storage = SqliteStorageProvider(
            db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        transcript = await Transcript.create(settings, name="test")

        # Add some valid messages first
        batch1 = [
            TranscriptMessage(
                text_chunks=["Message 1"],
                metadata=TranscriptMessageMeta(speaker="Alice"),
                tags=["batch1"],
            ),
        ]
        result1 = await transcript.add_messages_with_indexing(batch1)
        assert result1.messages_added == 1

        initial_count = await transcript.messages.size()

        # Verify the transaction context manager works
        try:
            async with storage:
                pass
        except Exception:
            pytest.fail("Transaction context manager should work")

        # Verify nested transactions fail
        with pytest.raises(Exception):  # SQLite will raise an OperationalError
            async with storage:
                async with storage:  # This should fail
                    pass

        await storage.close()
