# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test embedding consistency checks between database indexes."""

import os
import sqlite3
import tempfile

import numpy as np
import pytest

from typeagent import create_conversation
from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.storage.sqlite import SqliteStorageProvider
from typeagent.storage.sqlite.schema import serialize_embedding
from typeagent.transcripts.transcript import TranscriptMessage, TranscriptMessageMeta


@pytest.mark.asyncio
async def test_same_embedding_size_no_error():
    """Test that opening a DB with the same model works fine."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        settings1 = ConversationSettings(
            model=create_test_embedding_model(embedding_size=3)
        )
        settings1.semantic_ref_index_settings.auto_extract_knowledge = False
        conv1 = await create_conversation(
            db_path, TranscriptMessage, settings=settings1
        )

        messages = [
            TranscriptMessage(
                text_chunks=["Hello world"],
                metadata=TranscriptMessageMeta(speaker="Alice"),
            )
        ]
        await conv1.add_messages_with_indexing(messages)
        await conv1.storage_provider.close()

        # Reopen with same settings — should work
        settings2 = ConversationSettings(
            model=create_test_embedding_model(embedding_size=3)
        )
        provider = SqliteStorageProvider(
            db_path=db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings2.message_text_index_settings,
            related_term_index_settings=settings2.related_term_index_settings,
        )
        await provider.close()

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_empty_db_no_error():
    """Test that opening an empty DB doesn't raise an error regardless of embedding size."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        settings1 = ConversationSettings(
            model=create_test_embedding_model(embedding_size=3)
        )
        settings1.semantic_ref_index_settings.auto_extract_knowledge = False
        conv1 = await create_conversation(
            db_path, TranscriptMessage, settings=settings1
        )
        await conv1.storage_provider.close()

        # Open with different embedding size should work since DB is empty
        settings2 = ConversationSettings(
            model=create_test_embedding_model(embedding_size=5)
        )
        provider = SqliteStorageProvider(
            db_path=db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings2.message_text_index_settings,
            related_term_index_settings=settings2.related_term_index_settings,
        )
        await provider.close()

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_embedding_size_mismatch_raises():
    """Test that mismatched embedding sizes between indexes raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create a conversation so that the schema is set up
        settings = ConversationSettings(
            model=create_test_embedding_model(embedding_size=3)
        )
        settings.semantic_ref_index_settings.auto_extract_knowledge = False
        conv = await create_conversation(db_path, TranscriptMessage, settings=settings)
        await conv.storage_provider.close()

        # Manually insert embeddings of different sizes into the two tables
        conn = sqlite3.connect(db_path)
        msg_emb = serialize_embedding(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        term_emb = serialize_embedding(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        )
        conn.execute(
            "INSERT INTO MessageTextIndex "
            "(msg_id, chunk_ordinal, embedding, index_position) "
            "VALUES (0, 0, ?, 0)",
            (msg_emb,),
        )
        conn.execute(
            "INSERT INTO RelatedTermsFuzzy (term, term_embedding) VALUES (?, ?)",
            ("hello", term_emb),
        )
        conn.commit()
        conn.close()

        # Reopening should detect the mismatch
        settings2 = ConversationSettings(
            model=create_test_embedding_model(embedding_size=3)
        )
        with pytest.raises(ValueError, match="Embedding size mismatch"):
            SqliteStorageProvider(
                db_path=db_path,
                message_type=TranscriptMessage,
                message_text_index_settings=settings2.message_text_index_settings,
                related_term_index_settings=settings2.related_term_index_settings,
            )

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


@pytest.mark.asyncio
async def test_adding_mismatched_embeddings_raises():
    """Test that adding messages with a different embedding size raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create and populate with size-3 embeddings
        settings1 = ConversationSettings(
            model=create_test_embedding_model(embedding_size=3)
        )
        settings1.semantic_ref_index_settings.auto_extract_knowledge = False
        conv1 = await create_conversation(
            db_path, TranscriptMessage, settings=settings1
        )
        await conv1.add_messages_with_indexing(
            [
                TranscriptMessage(
                    text_chunks=["Hello world"],
                    metadata=TranscriptMessageMeta(speaker="Alice"),
                )
            ]
        )
        await conv1.storage_provider.close()

        # Reopen with size-5 embeddings and try to add more messages
        settings2 = ConversationSettings(
            model=create_test_embedding_model(embedding_size=5)
        )
        settings2.semantic_ref_index_settings.auto_extract_knowledge = False
        conv2 = await create_conversation(
            db_path, TranscriptMessage, settings=settings2
        )
        with pytest.raises(ValueError, match="Embedding size mismatch"):
            await conv2.add_messages_with_indexing(
                [
                    TranscriptMessage(
                        text_chunks=["Goodbye world"],
                        metadata=TranscriptMessageMeta(speaker="Bob"),
                    )
                ]
            )
        await conv2.storage_provider.close()

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
