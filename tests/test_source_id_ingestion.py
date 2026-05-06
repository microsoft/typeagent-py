# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for source_id-based ingestion tracking in add_messages_with_indexing."""

import os
import tempfile

import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)


def _make_message(
    text: str, speaker: str = "Alice", source_id: str | None = None
) -> TranscriptMessage:
    return TranscriptMessage(
        text_chunks=[text],
        metadata=TranscriptMessageMeta(speaker=speaker),
        tags=["test"],
        source_id=source_id,
    )


async def _create_transcript(
    db_path: str,
) -> tuple[Transcript, SqliteStorageProvider]:
    model = create_test_embedding_model()
    settings = ConversationSettings(model=model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = False
    storage = SqliteStorageProvider(
        db_path,
        message_type=TranscriptMessage,
        message_text_index_settings=settings.message_text_index_settings,
        related_term_index_settings=settings.related_term_index_settings,
    )
    settings.storage_provider = storage
    transcript = await Transcript.create(settings, name="test")
    return transcript, storage


def _ingested_count(storage: SqliteStorageProvider) -> int:
    """Count rows in IngestedSources table."""
    cursor = storage.db.cursor()
    cursor.execute("SELECT COUNT(*) FROM IngestedSources")
    return cursor.fetchone()[0]


@pytest.mark.asyncio
async def test_explicit_source_ids_marks_ingested() -> None:
    """Passing source_ids= explicitly marks those IDs as ingested."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message("Hello"), _make_message("World")]
        await transcript.add_messages_with_indexing(msgs, source_ids=["src-1", "src-2"])

        assert await storage.is_source_ingested("src-1")
        assert await storage.is_source_ingested("src-2")
        assert not await storage.is_source_ingested("src-3")

        await storage.close()


@pytest.mark.asyncio
async def test_message_source_id_marks_ingested() -> None:
    """When source_ids is omitted, message.source_id is used."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [
            _make_message("Hello", source_id="msg-src-1"),
            _make_message("World", source_id="msg-src-2"),
        ]
        await transcript.add_messages_with_indexing(msgs)

        assert await storage.is_source_ingested("msg-src-1")
        assert await storage.is_source_ingested("msg-src-2")

        await storage.close()


@pytest.mark.asyncio
async def test_message_source_id_none_skipped() -> None:
    """Messages with source_id=None are silently skipped (no ingestion mark)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [
            _make_message("Hello", source_id="only-one"),
            _make_message("World"),  # source_id=None
        ]
        await transcript.add_messages_with_indexing(msgs)

        assert await storage.is_source_ingested("only-one")
        # The second message had no source_id, so nothing extra was marked
        assert await storage.get_source_status("only-one") == "ingested"
        assert _ingested_count(storage) == 1

        await storage.close()


@pytest.mark.asyncio
async def test_explicit_source_ids_overrides_message_source_id() -> None:
    """Passing source_ids= takes precedence; message.source_id is ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [
            _make_message("Hello", source_id="msg-level"),
        ]
        await transcript.add_messages_with_indexing(msgs, source_ids=["explicit-id"])

        assert await storage.is_source_ingested("explicit-id")
        assert not await storage.is_source_ingested("msg-level")

        await storage.close()


@pytest.mark.asyncio
async def test_source_ids_length_mismatch_raises() -> None:
    """Passing source_ids with wrong length raises ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message("Hello"), _make_message("World")]
        with pytest.raises(ValueError, match="Length of source_ids"):
            await transcript.add_messages_with_indexing(msgs, source_ids=["only-one"])

        await storage.close()


@pytest.mark.asyncio
async def test_no_source_ids_no_message_source_id() -> None:
    """When neither source_ids nor message.source_id is set, nothing is marked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message("Hello"), _make_message("World")]
        result = await transcript.add_messages_with_indexing(msgs)

        assert result.messages_added == 2
        # No source tracking happened
        assert _ingested_count(storage) == 0

        await storage.close()
