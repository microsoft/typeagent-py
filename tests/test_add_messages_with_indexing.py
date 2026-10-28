# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test add_messages_with_indexing functionality."""

from collections.abc import AsyncIterator
import os
import tempfile

import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.add_messages import add_messages_streaming
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.knowledge_schema import ConcreteEntity, KnowledgeResponse
from typeagent.storage.memory.related_terms_index import TermEmbeddingIndex
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.storage.sqlite.related_terms_index import SqliteRelatedTermsFuzzy
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)

from conftest import RichKnowledgeExtractor


async def _async_iter(
    items: list[TranscriptMessage],
) -> AsyncIterator[TranscriptMessage]:
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_add_messages_with_indexing_basic():
    """Test basic add_messages_with_indexing functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = create_test_embedding_model()
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

        test_model = create_test_embedding_model()
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

        test_model = create_test_embedding_model()
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

        _initial_count = await transcript.messages.size()

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


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_extract", [False, True])
async def test_add_messages_with_indexing_fuzzy_terms_match_term_index(
    auto_extract: bool,
):
    """The fuzzy related-terms index gets every semantic ref index term (#346)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = create_test_embedding_model()
        settings = ConversationSettings(model=test_model)
        settings.semantic_ref_index_settings.auto_extract_knowledge = auto_extract
        settings.semantic_ref_index_settings.knowledge_extractor = (
            RichKnowledgeExtractor(subject="Alice")
        )

        storage = SqliteStorageProvider(
            db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        transcript = await Transcript.create(settings, name="test")

        messages = [
            TranscriptMessage(
                text_chunks=["Hello world"],
                metadata=TranscriptMessageMeta(speaker="Alice", recipients=["Bob"]),
            ),
        ]
        await transcript.add_messages_with_indexing(messages)

        fuzzy_index = storage.related_terms_index.fuzzy_index
        assert isinstance(fuzzy_index, SqliteRelatedTermsFuzzy)
        term_index_terms = await storage.semantic_ref_index.get_terms()
        assert {"alice", "bob", "person"} <= set(term_index_terms)
        if auto_extract:
            assert {"looks up", "height", "seattle"} <= set(term_index_terms)
        assert sorted(await fuzzy_index.get_terms()) == sorted(term_index_terms)

        await storage.close()


@pytest.mark.asyncio
async def test_add_messages_with_indexing_fuzzy_terms_include_nameless_metadata_entity(
    monkeypatch: pytest.MonkeyPatch,
):
    """Metadata entities are indexed without validation; so are their terms."""
    original_get_knowledge = TranscriptMessage.get_knowledge

    def get_knowledge(self: TranscriptMessage) -> KnowledgeResponse:
        knowledge = original_get_knowledge(self)
        knowledge.entities.append(ConcreteEntity(name="", type=["Ghost"]))
        return knowledge

    monkeypatch.setattr(TranscriptMessage, "get_knowledge", get_knowledge)
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = create_test_embedding_model()
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

        messages = [
            TranscriptMessage(
                text_chunks=["Hello world"],
                metadata=TranscriptMessageMeta(speaker="Alice", recipients=["Bob"]),
            ),
        ]
        await transcript.add_messages_with_indexing(messages)

        fuzzy_index = storage.related_terms_index.fuzzy_index
        assert isinstance(fuzzy_index, SqliteRelatedTermsFuzzy)
        term_index_terms = await storage.semantic_ref_index.get_terms()
        assert "ghost" in term_index_terms
        assert sorted(await fuzzy_index.get_terms()) == sorted(term_index_terms)

        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_memory_fuzzy_terms_match_term_index_across_batches(
    streaming: bool,
) -> None:
    """In memory, repeated batches leave one fuzzy entry per term-index term."""
    settings = ConversationSettings(model=create_test_embedding_model())
    settings.semantic_ref_index_settings.auto_extract_knowledge = True
    settings.semantic_ref_index_settings.knowledge_extractor = RichKnowledgeExtractor(
        subject="Alice"
    )
    transcript = await Transcript.create(settings, name="test")

    batches = [
        [
            TranscriptMessage(
                text_chunks=["Hello world"],
                metadata=TranscriptMessageMeta(speaker="Alice", recipients=["Bob"]),
            )
        ],
        [
            TranscriptMessage(
                text_chunks=["Hello again"],
                metadata=TranscriptMessageMeta(speaker="Bob", recipients=["Alice"]),
            )
        ],
    ]
    for batch in batches:
        if streaming:
            await add_messages_streaming(transcript, _async_iter(batch))
        else:
            await transcript.add_messages_with_indexing(batch)

    assert await transcript.messages.size() == 2
    secondary_indexes = transcript.secondary_indexes
    assert secondary_indexes is not None
    related_terms_index = secondary_indexes.term_to_related_terms_index
    assert related_terms_index is not None
    fuzzy_index = related_terms_index.fuzzy_index
    assert isinstance(fuzzy_index, TermEmbeddingIndex)
    fuzzy_terms = fuzzy_index.serialize()["textItems"]
    term_index_terms = await transcript.semantic_ref_index.get_terms()
    assert {"alice", "bob", "person", "looks up", "seattle"} <= set(term_index_terms)
    assert sorted(fuzzy_terms) == sorted(term_index_terms)


@pytest.mark.asyncio
async def test_add_messages_with_indexing_retry_after_rollback(
    monkeypatch: pytest.MonkeyPatch,
):
    """A retry after a rolled-back transaction persists the dropped terms (#362)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = create_test_embedding_model()
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

        messages = [
            TranscriptMessage(
                text_chunks=["Hello world"],
                metadata=TranscriptMessageMeta(speaker="Alice", recipients=["Bob"]),
            ),
        ]

        # Fail once at the end of the indexing transaction, after the fuzzy
        # index and the message text index have been updated.
        original_update = storage.update_conversation_timestamps
        calls = 0

        async def update_timestamps_failing_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("simulated failure")
            await original_update(*args, **kwargs)

        monkeypatch.setattr(
            storage, "update_conversation_timestamps", update_timestamps_failing_once
        )

        with pytest.raises(RuntimeError, match="simulated failure"):
            await transcript.add_messages_with_indexing(messages)

        fuzzy_index = storage.related_terms_index.fuzzy_index
        assert await storage.messages.size() == 0
        assert await fuzzy_index.get_terms() == []
        assert fuzzy_index.serialize()["textItems"] == []
        assert await storage.message_text_index.size() == 0

        result = await transcript.add_messages_with_indexing(messages)
        assert result.messages_added == 1

        term_index_terms = await storage.semantic_ref_index.get_terms()
        assert {"alice", "bob"} <= set(term_index_terms)
        assert sorted(await fuzzy_index.get_terms()) == sorted(term_index_terms)
        assert sorted(fuzzy_index.serialize()["textItems"]) == sorted(term_index_terms)
        hits = await fuzzy_index.lookup_term("alice", max_hits=1, min_score=0.0)
        assert [t.text for t in hits] == ["alice"]
        assert await storage.message_text_index.size() == 1
        matches = await storage.message_text_index.lookup_messages(
            "Hello world", max_matches=1
        )
        assert [m.message_ordinal for m in matches] == [0]

        await storage.close()
