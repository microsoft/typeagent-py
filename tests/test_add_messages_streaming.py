# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for add_messages_streaming."""

from collections.abc import AsyncIterator
import os
import tempfile

import pytest

import typechat

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro import knowledge_schema as kplib
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces_core import IKnowledgeExtractor
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    text: str,
    speaker: str = "Alice",
    source_id: str | None = None,
) -> TranscriptMessage:
    return TranscriptMessage(
        text_chunks=[text],
        metadata=TranscriptMessageMeta(speaker=speaker),
        tags=["test"],
        source_id=source_id,
    )


async def _create_transcript(
    db_path: str,
    *,
    auto_extract: bool = False,
    knowledge_extractor: IKnowledgeExtractor | None = None,
) -> tuple[Transcript, SqliteStorageProvider]:
    model = create_test_embedding_model()
    settings = ConversationSettings(model=model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = auto_extract
    if knowledge_extractor is not None:
        settings.semantic_ref_index_settings.knowledge_extractor = knowledge_extractor
    storage = SqliteStorageProvider(
        db_path,
        message_type=TranscriptMessage,
        message_text_index_settings=settings.message_text_index_settings,
        related_term_index_settings=settings.related_term_index_settings,
    )
    settings.storage_provider = storage
    transcript = await Transcript.create(settings, name="test")
    return transcript, storage


async def _async_iter(
    items: list[TranscriptMessage],
) -> AsyncIterator[TranscriptMessage]:
    for item in items:
        yield item


def _ingested_count(storage: SqliteStorageProvider) -> int:
    cursor = storage.db.cursor()
    cursor.execute("SELECT COUNT(*) FROM IngestedSources")
    return cursor.fetchone()[0]


def _failure_count(storage: SqliteStorageProvider) -> int:
    cursor = storage.db.cursor()
    cursor.execute("SELECT COUNT(*) FROM ChunkFailures")
    return cursor.fetchone()[0]


# ---------------------------------------------------------------------------
# A test IKnowledgeExtractor that lets us control per-call results
# ---------------------------------------------------------------------------

_EMPTY_RESPONSE = kplib.KnowledgeResponse(
    entities=[], actions=[], inverse_actions=[], topics=[]
)


class ControlledExtractor:
    """An IKnowledgeExtractor that returns Success or Failure per call.

    ``fail_on`` is a set of 0-based call indices for which the extractor
    returns a Failure instead of a Success.
    ``raise_on`` is a set of call indices that raise an exception.
    """

    def __init__(
        self,
        *,
        fail_on: set[int] | None = None,
        raise_on: set[int] | None = None,
    ) -> None:
        self.fail_on = fail_on or set()
        self.raise_on = raise_on or set()
        self.call_count = 0

    async def extract(self, message: str) -> typechat.Result[kplib.KnowledgeResponse]:
        idx = self.call_count
        self.call_count += 1
        if idx in self.raise_on:
            raise RuntimeError(f"Systemic failure at call {idx}")
        if idx in self.fail_on:
            return typechat.Failure(f"Extraction failed for call {idx}")
        return typechat.Success(_EMPTY_RESPONSE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_basic() -> None:
    """Streaming ingest of a few messages with no extraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message(f"msg-{i}") for i in range(5)]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        assert result.messages_added == 5
        assert await transcript.messages.size() == 5

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_batching() -> None:
    """Messages are committed in batches of the requested size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(7)]
        result = await transcript.add_messages_streaming(
            _async_iter(msgs), batch_size=3
        )

        # 3 batches: [0,1,2], [3,4,5], [6]
        assert result.messages_added == 7
        assert await transcript.messages.size() == 7
        # All 7 sources marked
        assert _ingested_count(storage) == 7

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_skips_already_ingested() -> None:
    """Messages whose source_id is already ingested are skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        # Pre-mark some sources as ingested
        async with storage:
            await storage.mark_source_ingested("s-1")
            await storage.mark_source_ingested("s-3")

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(5)]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        # s-1 and s-3 skipped -> only 3 added
        assert result.messages_added == 3
        assert await transcript.messages.size() == 3
        assert _ingested_count(storage) == 5  # 2 pre-existing + 3 new

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_no_source_id_always_ingested() -> None:
    """Messages without source_id are always ingested (never skipped)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message(f"msg-{i}") for i in range(3)]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        assert result.messages_added == 3
        assert _ingested_count(storage) == 0  # no source IDs to track

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_records_chunk_failures() -> None:
    """Extraction Failure results are recorded, not raised."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        extractor = ControlledExtractor(fail_on={1})  # second chunk fails
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [
            _make_message("good chunk 0"),
            _make_message("bad chunk 1"),
            _make_message("good chunk 2"),
        ]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        assert result.messages_added == 3
        assert _failure_count(storage) == 1

        failures = await storage.get_chunk_failures()
        assert len(failures) == 1
        assert failures[0].message_ordinal == 1
        assert failures[0].chunk_ordinal == 0
        assert "Extraction failed" in failures[0].error_message

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_exception_stops_run() -> None:
    """A raised exception stops processing; committed batches survive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        # Raise on the 4th extract call (first chunk of second batch)
        extractor = ControlledExtractor(raise_on={3})
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(6)]

        with pytest.raises(ExceptionGroup) as exc_info:
            await transcript.add_messages_streaming(_async_iter(msgs), batch_size=3)

        # Verify the wrapped exception is our RuntimeError
        assert any(
            isinstance(e, RuntimeError) and "Systemic failure" in str(e)
            for e in exc_info.value.exceptions
        )

        # First batch (3 messages, 3 extract calls 0-2) committed
        assert await transcript.messages.size() == 3
        assert _ingested_count(storage) == 3

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_empty_iterable() -> None:
    """Streaming with no messages returns zeros."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        result = await transcript.add_messages_streaming(_async_iter([]))

        assert result.messages_added == 0
        assert result.semrefs_added == 0

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_all_skipped_batch() -> None:
    """A batch where all messages are already ingested produces no commit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        # Pre-mark all sources
        async with storage:
            for i in range(3):
                await storage.mark_source_ingested(f"s-{i}")

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(3)]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        assert result.messages_added == 0
        assert await transcript.messages.size() == 0

        await storage.close()
