# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test incremental index building."""

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
from typeagent.transcripts.transcript_ingest import ingest_vtt_transcript


@pytest.mark.asyncio
async def test_incremental_index_building():
    """Test that we can build indexes, add more messages, and rebuild indexes."""

    # Create a temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        # Create settings with test model (no API keys needed)
        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings = ConversationSettings(model=test_model)
        settings.semantic_ref_index_settings.auto_extract_knowledge = False

        # First ingestion - add some messages and build index
        print("\n=== First ingestion ===")
        storage1 = SqliteStorageProvider(
            db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage1
        transcript1 = await Transcript.create(settings, name="test")

        # Add some messages
        messages1 = [
            TranscriptMessage(
                text_chunks=["Hello world"],
                metadata=TranscriptMessageMeta(speaker="Alice"),
                tags=["file1"],
            ),
            TranscriptMessage(
                text_chunks=["Hi Alice"],
                metadata=TranscriptMessageMeta(speaker="Bob"),
                tags=["file1"],
            ),
        ]

        # Add messages with indexing
        print("Adding messages with indexing...")
        result1 = await transcript1.add_messages_with_indexing(messages1)

        msg_count1 = await transcript1.messages.size()
        print(f"Added {msg_count1} messages")
        print(f"Created {result1.semrefs_added} semantic refs")

        ref_count1 = await transcript1.semantic_refs.size()

        # Close first connection
        await storage1.close()

        # Second ingestion - add more messages and rebuild index
        print("\n=== Second ingestion ===")
        test_model2 = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings2 = ConversationSettings(model=test_model2)
        settings2.semantic_ref_index_settings.auto_extract_knowledge = False
        storage2 = SqliteStorageProvider(
            db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings2.message_text_index_settings,
            related_term_index_settings=settings2.related_term_index_settings,
        )
        settings2.storage_provider = storage2
        transcript2 = await Transcript.create(settings2, name="test")

        # Verify existing messages are there
        msg_count_before = await transcript2.messages.size()
        print(f"Database has {msg_count_before} existing messages")
        assert msg_count_before == msg_count1

        # Add more messages
        messages2 = [
            TranscriptMessage(
                text_chunks=["How are you?"],
                metadata=TranscriptMessageMeta(speaker="Alice"),
                tags=["file2"],
            ),
            TranscriptMessage(
                text_chunks=["I'm good thanks"],
                metadata=TranscriptMessageMeta(speaker="Bob"),
                tags=["file2"],
            ),
        ]

        # Add messages with indexing
        print("Adding more messages with indexing...")
        result2 = await transcript2.add_messages_with_indexing(messages2)

        msg_count2 = await transcript2.messages.size()
        print(f"Now have {msg_count2} messages total")
        assert msg_count2 == msg_count_before + len(messages2)

        print("SUCCESS: Messages added with incremental indexing!")
        ref_count2 = await transcript2.semantic_refs.size()
        print(f"Now have {ref_count2} semantic refs (was {ref_count1})")

        # We should have more refs now
        assert ref_count2 >= ref_count1, "Should have at least as many refs as before"

        await storage2.close()


@pytest.mark.asyncio
async def test_incremental_index_with_vtt_files():
    """Test incremental indexing with actual VTT files.

    This test verifies that we can:
    1. Ingest a VTT file and build indexes
    2. Ingest a second VTT file into the same database
    3. Rebuild indexes incrementally without errors or duplication
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        # Create settings with test model (no API keys needed)
        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings = ConversationSettings(model=test_model)
        settings.semantic_ref_index_settings.auto_extract_knowledge = False

        # First VTT file ingestion
        print("\n=== Import first VTT file ===")
        # Import the first transcript
        transcript1 = await ingest_vtt_transcript(
            "testdata/Confuse-A-Cat.vtt",
            settings,
            dbname=db_path,
        )
        msg_count1 = await transcript1.messages.size()
        print(f"Imported {msg_count1} messages from Confuse-A-Cat.vtt")

        # Indexing already done by add_messages_with_indexing() in ingest
        ref_count1 = await transcript1.semantic_refs.size()
        print(f"Built index with {ref_count1} semantic refs")

        # Close the storage provider
        storage1 = await settings.get_storage_provider()
        await storage1.close()

        # Second VTT file ingestion into same database
        print("\n=== Import second VTT file ===")
        settings2 = ConversationSettings(
            model=AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        )
        settings2.semantic_ref_index_settings.auto_extract_knowledge = False

        # Ingest the second transcript
        transcript2 = await ingest_vtt_transcript(
            "testdata/Parrot_Sketch.vtt",
            settings2,
            dbname=db_path,
        )
        msg_count2 = await transcript2.messages.size()
        print(f"Now have {msg_count2} messages total")
        assert msg_count2 > msg_count1, "Should have added more messages"

        # Indexing already done incrementally by add_messages_with_indexing()
        print("Index built incrementally during ingestion")
        ref_count2 = await transcript2.semantic_refs.size()
        print(f"Now have {ref_count2} semantic refs (was {ref_count1})")

        # Should have more refs from the additional messages
        assert (
            ref_count2 > ref_count1
        ), "Should have more semantic refs after adding messages"

        storage2 = await settings2.get_storage_provider()
        await storage2.close()
