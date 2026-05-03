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
from typeagent.knowpro.interfaces_core import AddMessagesResult, IKnowledgeExtractor
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


# ---------------------------------------------------------------------------
# Pipeline overlap and DB batching tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_on_batch_committed_fires_per_batch() -> None:
    """on_batch_committed fires once per non-empty batch with the pipelined approach."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(7)]
        batch_results: list[int] = []
        result = await transcript.add_messages_streaming(
            _async_iter(msgs),
            batch_size=3,
            on_batch_committed=lambda r: batch_results.append(r.messages_added),
        )

        assert result.messages_added == 7
        # 3 batches: [0,1,2], [3,4,5], [6]
        assert batch_results == [3, 3, 1]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_extraction_with_multiple_batches() -> None:
    """Extraction results are correctly applied across batches with ordinal remapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        extractor = ControlledExtractor()
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(6)]
        result = await transcript.add_messages_streaming(
            _async_iter(msgs), batch_size=3
        )

        assert result.messages_added == 6
        assert await transcript.messages.size() == 6
        # All 6 chunks extracted (no failures)
        assert extractor.call_count == 6
        assert _failure_count(storage) == 0

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_extraction_failure_across_batches() -> None:
    """Extraction failures are recorded with correct global ordinals across batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        # Fail on call index 1 (batch 0, msg 1) and 4 (batch 1, msg 1)
        extractor = ControlledExtractor(fail_on={1, 4})
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(6)]
        result = await transcript.add_messages_streaming(
            _async_iter(msgs), batch_size=3
        )

        assert result.messages_added == 6
        assert _failure_count(storage) == 2

        failures = await storage.get_chunk_failures()
        failure_ordinals = sorted(f.message_ordinal for f in failures)
        # msg 1 in batch 0 → global ordinal 1, msg 1 in batch 1 → global ordinal 4
        assert failure_ordinals == [1, 4]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_exception_in_later_batch_preserves_earlier() -> None:
    """A raised exception in batch 1 stops processing; batch 0 is committed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        # Raise on call 4 (first call of batch 1, since batch 0 has 3 msgs)
        extractor = ControlledExtractor(raise_on={3})
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(6)]
        with pytest.raises(ExceptionGroup) as exc_info:
            await transcript.add_messages_streaming(_async_iter(msgs), batch_size=3)

        assert any(
            isinstance(e, RuntimeError) and "Systemic failure" in str(e)
            for e in exc_info.value.exceptions
        )

        # Batch 0 committed (3 messages), batch 1 rolled back
        assert await transcript.messages.size() == 3
        assert _ingested_count(storage) == 3

        await storage.close()


@pytest.mark.asyncio
async def test_mark_sources_ingested_batch_sqlite() -> None:
    """mark_sources_ingested_batch marks multiple sources in one call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        _, storage = await _create_transcript(db_path)

        async with storage:
            await storage.mark_sources_ingested_batch(["a", "b", "c"])

        assert await storage.is_source_ingested("a")
        assert await storage.is_source_ingested("b")
        assert await storage.is_source_ingested("c")
        assert not await storage.is_source_ingested("d")
        assert _ingested_count(storage) == 3

        await storage.close()


@pytest.mark.asyncio
async def test_mark_sources_ingested_batch_empty() -> None:
    """mark_sources_ingested_batch with empty list is a no-op."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        _, storage = await _create_transcript(db_path)

        async with storage:
            await storage.mark_sources_ingested_batch([])

        assert _ingested_count(storage) == 0

        await storage.close()


@pytest.mark.asyncio
async def test_mark_sources_ingested_batch_idempotent() -> None:
    """mark_sources_ingested_batch is idempotent via INSERT OR REPLACE."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        _, storage = await _create_transcript(db_path)

        async with storage:
            await storage.mark_sources_ingested_batch(["a", "b"])
        async with storage:
            await storage.mark_sources_ingested_batch(["b", "c"])

        assert _ingested_count(storage) == 3
        assert await storage.is_source_ingested("a")
        assert await storage.is_source_ingested("b")
        assert await storage.is_source_ingested("c")

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_extraction_with_empty_text_chunks() -> None:
    """Messages with empty text_chunks skip extraction gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        extractor = ControlledExtractor()
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [
            TranscriptMessage(
                text_chunks=[],
                metadata=TranscriptMessageMeta(speaker="Alice"),
                tags=["test"],
                source_id="empty-chunks",
            ),
            _make_message("has content", source_id="has-content"),
        ]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        assert result.messages_added == 2
        # Only the message with content triggers extraction
        assert extractor.call_count == 1

        await storage.close()


# ---------------------------------------------------------------------------
# Multi-chunk messages and chunk-based batching
# ---------------------------------------------------------------------------


def _make_multi_chunk_message(
    chunks: list[str],
    speaker: str = "Alice",
    source_id: str | None = None,
) -> TranscriptMessage:
    return TranscriptMessage(
        text_chunks=chunks,
        metadata=TranscriptMessageMeta(speaker=speaker),
        tags=["test"],
        source_id=source_id,
    )


@pytest.mark.asyncio
async def test_streaming_multi_chunk_extraction() -> None:
    """Each chunk in a multi-chunk message triggers a separate extraction call."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        extractor = ControlledExtractor()
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [
            _make_multi_chunk_message(["c0", "c1", "c2"], source_id="s-0"),
            _make_message("single chunk", source_id="s-1"),
        ]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        assert result.messages_added == 2
        assert result.chunks_added == 4  # 3 + 1
        # 4 extraction calls: one per chunk
        assert extractor.call_count == 4

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_batch_size_counts_chunks() -> None:
    """batch_size counts chunks, not messages — a 3-chunk message fills batch_size=3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [
            _make_multi_chunk_message(["a", "b", "c"], source_id="s-0"),  # 3 chunks
            _make_message("d", source_id="s-1"),  # 1 chunk
        ]
        batch_results: list[int] = []
        result = await transcript.add_messages_streaming(
            _async_iter(msgs),
            batch_size=3,
            on_batch_committed=lambda r: batch_results.append(r.messages_added),
        )

        assert result.messages_added == 2
        # First message (3 chunks) fills batch_size=3, second message goes to batch 2
        assert batch_results == [1, 1]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_large_message_exceeds_batch_size() -> None:
    """A single message with more chunks than batch_size becomes its own batch."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [
            _make_multi_chunk_message(
                [f"chunk-{i}" for i in range(5)], source_id="s-big"
            ),
            _make_message("small", source_id="s-small"),
        ]
        batch_results: list[int] = []
        result = await transcript.add_messages_streaming(
            _async_iter(msgs),
            batch_size=3,
            on_batch_committed=lambda r: batch_results.append(r.messages_added),
        )

        assert result.messages_added == 2
        # 5-chunk msg in batch 1, then 1-chunk msg in batch 2
        assert batch_results == [1, 1]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_mixed_chunk_sizes_batching() -> None:
    """Messages of varying chunk counts are batched by cumulative chunk count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [
            _make_message("a", source_id="s-0"),  # 1 chunk, total=1
            _make_multi_chunk_message(
                ["b1", "b2"], source_id="s-1"
            ),  # 2 chunks, total=3 → flush
            _make_message("c", source_id="s-2"),  # 1 chunk, total=1
            _make_message("d", source_id="s-3"),  # 1 chunk, total=2
            _make_message("e", source_id="s-4"),  # 1 chunk, total=3 → flush
        ]
        batch_results: list[int] = []
        result = await transcript.add_messages_streaming(
            _async_iter(msgs),
            batch_size=3,
            on_batch_committed=lambda r: batch_results.append(r.messages_added),
        )

        assert result.messages_added == 5
        assert result.chunks_added == 6
        # Batch 1: msgs 0+1 (3 chunks), Batch 2: msgs 2+3+4 (3 chunks)
        assert batch_results == [2, 3]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_multi_chunk_failure_ordinals() -> None:
    """Extraction failures in multi-chunk messages record correct ordinals."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        # Fail on call index 1 (chunk 1 of first message) and 3 (chunk 0 of second message)
        extractor = ControlledExtractor(fail_on={1, 3})
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [
            _make_multi_chunk_message(
                ["c0", "c1", "c2"], source_id="s-0"
            ),  # calls 0,1,2
            _make_multi_chunk_message(["d0", "d1"], source_id="s-1"),  # calls 3,4
        ]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        assert result.messages_added == 2
        assert extractor.call_count == 5
        assert _failure_count(storage) == 2

        failures = await storage.get_chunk_failures()
        failure_locs = sorted((f.message_ordinal, f.chunk_ordinal) for f in failures)
        # call 1 → msg 0, chunk 1; call 3 → msg 1, chunk 0
        assert failure_locs == [(0, 1), (1, 0)]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_multi_chunk_exception_preserves_earlier_batch() -> None:
    """Exception during extraction of multi-chunk batch preserves committed batches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        # Batch 1: 3-chunk msg (calls 0,1,2).  Batch 2: 2-chunk msg (calls 3,4) — raise on 3
        extractor = ControlledExtractor(raise_on={3})
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        msgs = [
            _make_multi_chunk_message(["a", "b", "c"], source_id="s-0"),  # batch 1
            _make_multi_chunk_message(["d", "e"], source_id="s-1"),  # batch 2
        ]

        with pytest.raises(ExceptionGroup):
            await transcript.add_messages_streaming(_async_iter(msgs), batch_size=3)

        # Batch 1 committed (1 message, 3 chunks), batch 2 rolled back
        assert await transcript.messages.size() == 1
        assert _ingested_count(storage) == 1

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_multi_chunk_skip_and_ingest_mixed() -> None:
    """Multi-chunk messages are skipped or ingested as a whole based on source_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        extractor = ControlledExtractor()
        transcript, storage = await _create_transcript(
            db_path, auto_extract=True, knowledge_extractor=extractor
        )

        # Pre-mark s-1 as ingested
        async with storage:
            await storage.mark_source_ingested("s-1")

        msgs = [
            _make_multi_chunk_message(["a", "b"], source_id="s-0"),  # ingested
            _make_multi_chunk_message(["c", "d", "e"], source_id="s-1"),  # skipped
            _make_message("f", source_id="s-2"),  # ingested
        ]
        result = await transcript.add_messages_streaming(_async_iter(msgs))

        assert result.messages_added == 2
        assert result.messages_skipped == 1
        assert result.chunks_added == 3  # 2 + 1 (not 5)
        # Only 3 extraction calls (2 chunks from s-0 + 1 chunk from s-2)
        assert extractor.call_count == 3

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_batch_size_1_separates_all() -> None:
    """batch_size=1 commits every single-chunk message individually."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(4)]
        batch_results: list[int] = []
        result = await transcript.add_messages_streaming(
            _async_iter(msgs),
            batch_size=1,
            on_batch_committed=lambda r: batch_results.append(r.messages_added),
        )

        assert result.messages_added == 4
        assert batch_results == [1, 1, 1, 1]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_callback_reports_skipped_multi_chunk() -> None:
    """on_batch_committed reports skipped count for batches with multi-chunk messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        # Pre-mark s-1 as ingested
        async with storage:
            await storage.mark_source_ingested("s-1")

        msgs = [
            _make_multi_chunk_message(["a", "b"], source_id="s-0"),  # 2 chunks
            _make_multi_chunk_message(["c", "d"], source_id="s-1"),  # 2 chunks, skipped
            _make_message("e", source_id="s-2"),  # 1 chunk → total = 5 chunks in batch
        ]
        callback_results: list[tuple[int, int]] = []
        result = await transcript.add_messages_streaming(
            _async_iter(msgs),
            batch_size=10,  # all fit in one batch
            on_batch_committed=lambda r: callback_results.append(
                (r.messages_added, r.messages_skipped)
            ),
        )

        assert result.messages_added == 2
        assert result.messages_skipped == 1
        assert callback_results == [(2, 1)]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_preflush_avoids_oversized_batch() -> None:
    """Adding a message that would exceed batch_size flushes first.

    With batch_size=10 and four 3-chunk messages, batches should be
    [msg0,msg1,msg2] (9 chunks) and [msg3] (3 chunks) — never a single
    batch of 12 chunks.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [
            _make_multi_chunk_message(
                [f"m{i}c{j}" for j in range(3)], source_id=f"s-{i}"
            )
            for i in range(4)
        ]
        batch_chunks: list[int] = []
        result = await transcript.add_messages_streaming(
            _async_iter(msgs),
            batch_size=10,
            on_batch_committed=lambda r: batch_chunks.append(r.chunks_added),
        )

        assert result.messages_added == 4
        assert result.chunks_added == 12
        # Batch 1: 3 msgs × 3 chunks = 9, Batch 2: 1 msg × 3 chunks = 3
        assert batch_chunks == [9, 3]

        await storage.close()


@pytest.mark.asyncio
async def test_streaming_all_skipped_batch_after_real_batch() -> None:
    """A batch of all-duplicates reports skipped correctly.

    First call ingests messages s-0..s-2. Second call re-submits the same
    source_ids — they should all be filtered by _filter_ingested, exercising
    the all-skipped + pending_commit path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        transcript, storage = await _create_transcript(db_path)

        msgs = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(3)]

        # First call — ingest originals
        result1 = await transcript.add_messages_streaming(
            _async_iter(msgs),
            batch_size=3,
        )
        assert result1.messages_added == 3
        assert result1.messages_skipped == 0

        # Second call — all duplicates
        dupes = [_make_message(f"msg-{i}", source_id=f"s-{i}") for i in range(3)]
        batch_results: list[AddMessagesResult] = []
        result2 = await transcript.add_messages_streaming(
            _async_iter(dupes),
            batch_size=3,
            on_batch_committed=lambda r: batch_results.append(r),
        )

        assert result2.messages_added == 0
        assert result2.messages_skipped == 3

        # One callback for the all-skipped batch
        assert len(batch_results) == 1
        assert batch_results[0].messages_added == 0
        assert batch_results[0].messages_skipped == 3

        await storage.close()
