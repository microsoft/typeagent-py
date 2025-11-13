# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for conversation metadata operations in SQLite storage provider."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import field
from datetime import datetime, timezone
import os
import sqlite3
import tempfile
import time

import pytest
import pytest_asyncio
from pydantic.dataclasses import dataclass

from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.convsettings import (
    ConversationSettings,
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.knowpro.interfaces import ConversationMetadata, IMessage
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)

from fixtures import embedding_model, temp_db_path


def parse_iso_datetime(iso_string: str) -> datetime:
    """Helper to parse ISO datetime strings to datetime objects."""
    # Handle Z timezone marker
    if iso_string.endswith("Z"):
        iso_string = iso_string[:-1] + "+00:00"
    return datetime.fromisoformat(iso_string)


# Dummy IMessage for testing
@dataclass
class DummyMessage(IMessage):
    text_chunks: list[str]
    tags: list[str] = field(default_factory=list)
    timestamp: str | None = None

    def get_knowledge(self) -> KnowledgeResponse:
        raise NotImplementedError("Should not be called")


@pytest_asyncio.fixture
async def storage_provider(
    temp_db_path: str, embedding_model: AsyncEmbeddingModel
) -> AsyncGenerator[SqliteStorageProvider[DummyMessage], None]:
    """Create a SqliteStorageProvider for testing conversation metadata."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = SqliteStorageProvider(
        db_path=temp_db_path,
        message_type=DummyMessage,
        message_text_index_settings=message_text_settings,
        related_term_index_settings=related_terms_settings,
    )
    yield provider
    await provider.close()


@pytest_asyncio.fixture
async def storage_provider_memory() -> (
    AsyncGenerator[SqliteStorageProvider[DummyMessage], None]
):
    """Create an in-memory SqliteStorageProvider for testing conversation metadata."""
    embedding_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = SqliteStorageProvider(
        db_path=":memory:",
        message_type=DummyMessage,
        message_text_index_settings=message_text_settings,
        related_term_index_settings=related_terms_settings,
    )
    yield provider
    await provider.close()


class TestConversationMetadata:
    """Test conversation metadata operations."""

    def test_get_conversation_metadata_nonexistent(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test getting metadata before any writes returns empty metadata."""
        metadata = storage_provider.get_conversation_metadata()
        # Metadata is not initialized until first write, so all fields are None
        assert metadata.name_tag is None
        assert metadata.schema_version is None
        assert metadata.created_at is None
        assert metadata.updated_at is None
        assert metadata.embedding_size is None
        assert metadata.embedding_model is None
        assert metadata.tags is None
        assert metadata.extra is None

    def test_update_conversation_timestamps_new(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating conversation metadata timestamps."""
        created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        updated_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        storage_provider.update_conversation_timestamps(
            created_at=created_at,
            updated_at=updated_at,
        )

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.name_tag == "conversation"
        assert metadata.schema_version == 1
        assert metadata.created_at == created_at
        assert metadata.updated_at == updated_at
        settings = storage_provider.message_text_index_settings.embedding_index_settings
        expected_size = settings.embedding_size
        expected_model = settings.embedding_model.model_name
        assert metadata.embedding_size == expected_size
        assert metadata.embedding_model == expected_model
        assert metadata.tags is None
        assert metadata.extra is None

    def test_update_conversation_timestamps_existing(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating existing conversation metadata."""
        # Create initial metadata
        initial_created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        initial_updated = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        storage_provider.update_conversation_timestamps(
            created_at=initial_created,
            updated_at=initial_updated,
        )

        # Update only the updated_at timestamp
        new_updated = datetime(2024, 1, 2, 15, 30, 0, tzinfo=timezone.utc)
        storage_provider.update_conversation_timestamps(updated_at=new_updated)

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == initial_created  # Unchanged
        assert metadata.updated_at == new_updated  # Changed

    def test_update_conversation_timestamps_partial_created_at(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating only created_at of existing conversation metadata."""
        # Create initial metadata
        initial_created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        initial_updated = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        storage_provider.update_conversation_timestamps(
            created_at=initial_created,
            updated_at=initial_updated,
        )

        # Update only the created_at timestamp
        new_created = datetime(2023, 12, 1, 10, 0, 0, tzinfo=timezone.utc)
        storage_provider.update_conversation_timestamps(created_at=new_created)

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == new_created  # Changed
        assert metadata.updated_at == initial_updated  # Unchanged

    def test_update_conversation_timestamps_both_timestamps(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating both timestamps of existing conversation metadata."""
        # Create initial metadata
        initial_created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        initial_updated = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        storage_provider.update_conversation_timestamps(
            created_at=initial_created,
            updated_at=initial_updated,
        )

        # Update both timestamps
        new_created = datetime(2023, 12, 1, 10, 0, 0, tzinfo=timezone.utc)
        new_updated = datetime(2024, 1, 2, 15, 30, 0, tzinfo=timezone.utc)
        storage_provider.update_conversation_timestamps(
            created_at=new_created,
            updated_at=new_updated,
        )

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == new_created
        assert metadata.updated_at == new_updated

    def test_update_conversation_timestamps_no_params(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test calling update with no parameters on existing conversation."""
        # Create initial metadata
        initial_created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        initial_updated = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        storage_provider.update_conversation_timestamps(
            created_at=initial_created,
            updated_at=initial_updated,
        )

        # Call update with no parameters - should not change anything
        storage_provider.update_conversation_timestamps()

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == initial_created
        assert metadata.updated_at == initial_updated

    def test_update_conversation_timestamps_none_values(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating with explicit None values."""
        # Create initial metadata
        initial_created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        initial_updated = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        storage_provider.update_conversation_timestamps(
            created_at=initial_created,
            updated_at=initial_updated,
        )

        # Update with None values - should not change anything
        storage_provider.update_conversation_timestamps(
            created_at=None, updated_at=None
        )

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == initial_created
        assert metadata.updated_at == initial_updated

    def test_get_db_version(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test getting database schema version."""
        version = storage_provider.get_db_version()
        assert isinstance(version, int)
        assert version >= 1  # Schema version is now 1

    def test_get_db_version_with_metadata(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test getting database version after creating metadata."""
        # Metadata is automatically initialized, so version should be available
        version = storage_provider.get_db_version()
        assert isinstance(version, int)
        assert version >= 1  # Schema version is now 1

    @pytest.mark.asyncio
    async def test_multiple_conversations_different_dbs(
        self, embedding_model: AsyncEmbeddingModel
    ):
        """Test multiple conversations in different database files."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        # Create temporary database files
        temp_file1 = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        db_path1 = temp_file1.name
        temp_file1.close()
        temp_file2 = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
        db_path2 = temp_file2.name
        temp_file2.close()

        try:
            # Create first provider with conversation "conv1"
            metadata1 = ConversationMetadata(name_tag="conversation_conv1")
            provider1 = SqliteStorageProvider(
                db_path=db_path1,
                message_type=DummyMessage,
                message_text_index_settings=message_text_settings,
                related_term_index_settings=related_terms_settings,
                metadata=metadata1,
            )

            # Create second provider with conversation "conv2" on different DB
            metadata2 = ConversationMetadata(name_tag="conversation_conv2")
            provider2 = SqliteStorageProvider(
                db_path=db_path2,
                message_type=DummyMessage,
                message_text_index_settings=message_text_settings,
                related_term_index_settings=related_terms_settings,
                metadata=metadata2,
            )

            try:
                # Update timestamps for both conversations
                provider1.update_conversation_timestamps(
                    created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    updated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                )

                provider2.update_conversation_timestamps(
                    created_at=datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc),
                    updated_at=datetime(2024, 1, 2, 14, 0, 0, tzinfo=timezone.utc),
                )

                # Verify each conversation sees its own metadata
                read_metadata1 = provider1.get_conversation_metadata()
                read_metadata2 = provider2.get_conversation_metadata()

                assert read_metadata1.name_tag == "conversation_conv1"
                assert read_metadata2.name_tag == "conversation_conv2"

                assert read_metadata1.created_at == parse_iso_datetime(
                    "2024-01-01T12:00:00+00:00"
                )
                assert read_metadata2.created_at == parse_iso_datetime(
                    "2024-01-02T14:00:00+00:00"
                )
            finally:
                await provider1.close()
                await provider2.close()

        finally:
            if os.path.exists(db_path1):
                os.remove(db_path1)
            if os.path.exists(db_path2):
                os.remove(db_path2)

    @pytest.mark.asyncio
    async def test_conversation_metadata_single_per_db(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test that only one conversation metadata can exist per database."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        # Create first provider with specific metadata
        metadata_alpha = ConversationMetadata(name_tag="conversation_alpha")
        provider_alpha = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
            metadata=metadata_alpha,
        )

        # Create second provider on same DB (different metadata preference)
        metadata_beta = ConversationMetadata(name_tag="conversation_beta")
        provider_beta = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
            metadata=metadata_beta,
        )

        try:
            # Write metadata with alpha provider (first write wins)
            provider_alpha.update_conversation_timestamps(
                created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
            provider_alpha.db.commit()

            # Both providers should see the same metadata since it's the same DB
            alpha_metadata = provider_alpha.get_conversation_metadata()
            beta_metadata = provider_beta.get_conversation_metadata()

            # They should be the same since there's only one metadata row per DB
            assert alpha_metadata.name_tag == "conversation_alpha"  # First write wins
            assert beta_metadata.name_tag == alpha_metadata.name_tag  # Same metadata
            assert alpha_metadata.created_at == beta_metadata.created_at
            assert alpha_metadata.updated_at == beta_metadata.updated_at
        finally:
            await provider_alpha.close()
            await provider_beta.close()

    def test_conversation_metadata_with_special_characters(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test conversation metadata with special characters in timestamps."""
        # Test with various ISO 8601 timestamp formats
        test_timestamps = [
            "2024-01-01T12:00:00Z",  # UTC with Z
            "2024-01-01T12:00:00+00:00",  # UTC with offset
            "2024-01-01T12:00:00.123456+05:30",  # With microseconds and timezone
            "2024-12-31T23:59:59-08:00",  # Different timezone
        ]

        # Convert test timestamps to datetime objects for the API
        for timestamp_str in test_timestamps:
            timestamp_dt = parse_iso_datetime(timestamp_str)
            storage_provider.update_conversation_timestamps(
                created_at=timestamp_dt, updated_at=timestamp_dt
            )

            metadata = storage_provider.get_conversation_metadata()
            assert metadata is not None
            assert metadata.created_at == timestamp_dt
            assert metadata.updated_at == timestamp_dt

    @pytest.mark.asyncio
    async def test_conversation_metadata_persistence(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test that conversation metadata persists across provider instances."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        created_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        updated_at = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Create first provider and add metadata
        metadata_input = ConversationMetadata(name_tag="conversation_persistent_test")
        provider1 = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
            metadata=metadata_input,
        )

        provider1.update_conversation_timestamps(
            created_at=created_at,
            updated_at=updated_at,
        )
        await provider1.close()

        # Create second provider on same DB (doesn't need metadata, will read from DB)
        provider2 = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        try:
            metadata = provider2.get_conversation_metadata()
            assert metadata.name_tag == "conversation_persistent_test"
            assert metadata.created_at == created_at
            assert metadata.updated_at == updated_at
            expected_size = embedding_settings.embedding_size
            expected_model = embedding_settings.embedding_model.model_name
            assert metadata.embedding_size == expected_size
            assert metadata.embedding_model == expected_model
        finally:
            await provider2.close()


class TestConversationMetadataEdgeCases:
    """Test edge cases for conversation metadata operations."""

    def test_empty_string_timestamps(
        self, storage_provider_memory: SqliteStorageProvider[DummyMessage]
    ):
        """Test behavior with None timestamps (should remain None)."""
        storage_provider_memory.update_conversation_timestamps(
            created_at=None, updated_at=None
        )

        metadata = storage_provider_memory.get_conversation_metadata()
        # Calling with None values creates a row but leaves timestamps None
        assert metadata.name_tag == "conversation"
        assert metadata.schema_version == 1
        assert metadata.created_at is None
        assert metadata.updated_at is None

    @pytest.mark.asyncio
    async def test_very_long_name_tag(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test conversation metadata with very long name_tag."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        long_name = "conversation_" + ("a" * 1000)  # Very long name_tag

        metadata_input = ConversationMetadata(name_tag=long_name)
        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
            metadata=metadata_input,
        )

        try:
            provider.update_conversation_timestamps(
                created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )

            metadata = provider.get_conversation_metadata()
            assert metadata.name_tag == long_name
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_unicode_name_tag(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test conversation metadata with Unicode name_tag."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        unicode_name = "conversation_конверсация_🚀_测试"  # Mixed Unicode

        metadata_input = ConversationMetadata(name_tag=unicode_name)
        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
            metadata=metadata_input,
        )

        try:
            provider.update_conversation_timestamps(
                created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )

            metadata = provider.get_conversation_metadata()
            assert metadata.name_tag == unicode_name
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_conversation_metadata_shared_access(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test shared access to metadata using the same database file."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        # Create two providers pointing to same database
        provider1 = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        provider2 = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        try:
            # Update from provider1
            provider1.update_conversation_timestamps(
                created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                updated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            )
            provider1.db.commit()

            # Update from provider2 - should update the same metadata row
            provider2.update_conversation_timestamps(
                updated_at=datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
            )
            provider2.db.commit()

            # Both should see the latest state
            metadata1 = provider1.get_conversation_metadata()
            metadata2 = provider2.get_conversation_metadata()

            assert metadata1 is not None
            assert metadata2 is not None
            assert metadata1.created_at == metadata2.created_at
            expected_updated = parse_iso_datetime("2024-01-01T13:00:00+00:00")
            assert metadata1.updated_at == metadata2.updated_at == expected_updated
        finally:
            await provider1.close()
            await provider2.close()

    def test_embedding_metadata_mismatch_raises(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Ensure a mismatch between stored metadata and provided settings raises."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        provider.update_conversation_timestamps(
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        provider.db.commit()
        asyncio.run(provider.close())

        mismatched_model = AsyncEmbeddingModel(
            embedding_size=embedding_settings.embedding_size + 1,
            model_name=embedding_model.model_name,
        )
        mismatched_settings = TextEmbeddingIndexSettings(
            embedding_model=mismatched_model,
            embedding_size=mismatched_model.embedding_size,
        )

        with pytest.raises(ValueError, match="embedding_size"):
            SqliteStorageProvider(
                db_path=temp_db_path,
                message_type=DummyMessage,
                message_text_index_settings=MessageTextIndexSettings(
                    mismatched_settings
                ),
                related_term_index_settings=RelatedTermIndexSettings(
                    mismatched_settings
                ),
            )

    def test_embedding_model_mismatch_raises(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Ensure providing a different embedding model name raises."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        provider.update_conversation_timestamps(
            created_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            updated_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        provider.db.commit()
        asyncio.run(provider.close())

        with sqlite3.connect(temp_db_path) as conn:
            conn.execute(
                "UPDATE ConversationMetadata SET value = ? WHERE key = 'embedding_name'",
                ("mismatched-model",),
            )
            conn.commit()

        mismatch_settings = TextEmbeddingIndexSettings(embedding_model)

        with pytest.raises(ValueError, match="embedding_model"):
            SqliteStorageProvider(
                db_path=temp_db_path,
                message_type=DummyMessage,
                message_text_index_settings=MessageTextIndexSettings(mismatch_settings),
                related_term_index_settings=RelatedTermIndexSettings(mismatch_settings),
            )

    @pytest.mark.asyncio
    async def test_updated_at_changes_on_add_messages(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test that updated_at timestamp is updated when messages are added."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=TranscriptMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        try:
            # Get initial metadata (should be empty due to lazy initialization)
            initial_metadata = provider.get_conversation_metadata()
            initial_updated_at = initial_metadata.updated_at
            assert initial_updated_at is None  # No writes yet

            # Wait a tiny bit to ensure timestamp difference
            time.sleep(0.01)

            # Create a conversation and add messages
            settings = ConversationSettings(model=embedding_model)
            settings.storage_provider = provider
            settings.semantic_ref_index_settings.auto_extract_knowledge = False
            transcript = await Transcript.create(settings, name="test")

            messages = [
                TranscriptMessage(
                    text_chunks=["Test message"],
                    metadata=TranscriptMessageMeta(speaker="Alice"),
                    tags=["test"],
                ),
            ]

            await transcript.add_messages_with_indexing(messages)

            # Get updated metadata (should now have timestamps)
            updated_metadata = provider.get_conversation_metadata()
            updated_updated_at = updated_metadata.updated_at

            # The updated_at timestamp should now be set
            assert updated_updated_at is not None

        finally:
            await provider.close()
