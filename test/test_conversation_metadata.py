# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for conversation metadata operations in SQLite storage provider."""

import os
import tempfile
from collections.abc import AsyncGenerator, Generator
from dataclasses import field
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from pydantic.dataclasses import dataclass

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.convsettings import (
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.knowpro.interfaces import IMessage
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.storage.sqlite.schema import ConversationMetadata

from fixtures import embedding_model, temp_db_path


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
        conversation_id="test_conversation",
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
    from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
    from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings

    embedding_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = SqliteStorageProvider(
        db_path=":memory:",
        conversation_id="test_conversation",
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
        """Test getting metadata for a non-existent conversation returns None."""
        metadata = storage_provider.get_conversation_metadata()
        assert metadata is None

    def test_update_conversation_metadata_new(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test creating new conversation metadata."""
        created_at = "2024-01-01T12:00:00+00:00"
        updated_at = "2024-01-01T12:00:00+00:00"

        storage_provider.update_conversation_metadata(
            created_at=created_at, updated_at=updated_at
        )

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.name_tag == "conversation_test_conversation"
        assert metadata.schema_version == "1.0"
        assert metadata.created_at == created_at
        assert metadata.updated_at == updated_at
        assert metadata.tags == []
        assert metadata.extra == {}

    def test_update_conversation_metadata_existing(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating existing conversation metadata."""
        # Create initial metadata
        initial_created = "2024-01-01T12:00:00+00:00"
        initial_updated = "2024-01-01T12:00:00+00:00"
        storage_provider.update_conversation_metadata(
            created_at=initial_created, updated_at=initial_updated
        )

        # Update only the updated_at timestamp
        new_updated = "2024-01-02T15:30:00+00:00"
        storage_provider.update_conversation_metadata(updated_at=new_updated)

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == initial_created  # Unchanged
        assert metadata.updated_at == new_updated  # Changed

    def test_update_conversation_metadata_partial_created_at(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating only created_at of existing conversation metadata."""
        # Create initial metadata
        initial_created = "2024-01-01T12:00:00+00:00"
        initial_updated = "2024-01-01T12:00:00+00:00"
        storage_provider.update_conversation_metadata(
            created_at=initial_created, updated_at=initial_updated
        )

        # Update only the created_at timestamp
        new_created = "2023-12-01T10:00:00+00:00"
        storage_provider.update_conversation_metadata(created_at=new_created)

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == new_created  # Changed
        assert metadata.updated_at == initial_updated  # Unchanged

    def test_update_conversation_metadata_both_timestamps(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating both timestamps of existing conversation metadata."""
        # Create initial metadata
        initial_created = "2024-01-01T12:00:00+00:00"
        initial_updated = "2024-01-01T12:00:00+00:00"
        storage_provider.update_conversation_metadata(
            created_at=initial_created, updated_at=initial_updated
        )

        # Update both timestamps
        new_created = "2023-12-01T10:00:00+00:00"
        new_updated = "2024-01-02T15:30:00+00:00"
        storage_provider.update_conversation_metadata(
            created_at=new_created, updated_at=new_updated
        )

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == new_created
        assert metadata.updated_at == new_updated

    def test_update_conversation_metadata_no_params(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test calling update with no parameters on existing conversation."""
        # Create initial metadata
        initial_created = "2024-01-01T12:00:00+00:00"
        initial_updated = "2024-01-01T12:00:00+00:00"
        storage_provider.update_conversation_metadata(
            created_at=initial_created, updated_at=initial_updated
        )

        # Call update with no parameters - should not change anything
        storage_provider.update_conversation_metadata()

        metadata = storage_provider.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == initial_created
        assert metadata.updated_at == initial_updated

    def test_update_conversation_metadata_none_values(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test updating with explicit None values."""
        # Create initial metadata
        initial_created = "2024-01-01T12:00:00+00:00"
        initial_updated = "2024-01-01T12:00:00+00:00"
        storage_provider.update_conversation_metadata(
            created_at=initial_created, updated_at=initial_updated
        )

        # Update with None values - should not change anything
        storage_provider.update_conversation_metadata(created_at=None, updated_at=None)

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
        assert (
            version >= 0
        )  # Should be at least version 0 (since schema version is "0.1")

    def test_get_db_version_with_metadata(
        self, storage_provider: SqliteStorageProvider[DummyMessage]
    ):
        """Test getting database version after creating metadata."""
        # Create metadata first
        storage_provider.update_conversation_metadata(
            created_at="2024-01-01T12:00:00+00:00",
            updated_at="2024-01-01T12:00:00+00:00",
        )

        version = storage_provider.get_db_version()
        assert isinstance(version, int)
        assert version >= 1

    @pytest.mark.asyncio
    async def test_multiple_conversations_different_dbs(
        self, embedding_model: AsyncEmbeddingModel
    ):
        """Test multiple conversations in different database files."""
        import tempfile
        import os

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
            provider1 = SqliteStorageProvider(
                db_path=db_path1,
                conversation_id="conv1",
                message_type=DummyMessage,
                message_text_index_settings=message_text_settings,
                related_term_index_settings=related_terms_settings,
            )

            # Create second provider with conversation "conv2" on different DB
            provider2 = SqliteStorageProvider(
                db_path=db_path2,
                conversation_id="conv2",
                message_type=DummyMessage,
                message_text_index_settings=message_text_settings,
                related_term_index_settings=related_terms_settings,
            )

            try:
                # Add metadata for both conversations
                provider1.update_conversation_metadata(
                    created_at="2024-01-01T12:00:00+00:00",
                    updated_at="2024-01-01T12:00:00+00:00",
                )

                provider2.update_conversation_metadata(
                    created_at="2024-01-02T14:00:00+00:00",
                    updated_at="2024-01-02T14:00:00+00:00",
                )

                # Verify each conversation sees its own metadata
                metadata1 = provider1.get_conversation_metadata()
                metadata2 = provider2.get_conversation_metadata()

                assert metadata1 is not None
                assert metadata2 is not None

                assert metadata1.name_tag == "conversation_conv1"
                assert metadata2.name_tag == "conversation_conv2"

                assert metadata1.created_at == "2024-01-01T12:00:00+00:00"
                assert metadata2.created_at == "2024-01-02T14:00:00+00:00"

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

        # Create providers for different conversation IDs but same DB
        provider_alpha = SqliteStorageProvider(
            db_path=temp_db_path,
            conversation_id="alpha",
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        provider_beta = SqliteStorageProvider(
            db_path=temp_db_path,
            conversation_id="beta",
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        try:
            # Create metadata with alpha provider
            provider_alpha.update_conversation_metadata(
                created_at="2024-01-01T12:00:00+00:00",
                updated_at="2024-01-01T12:00:00+00:00",
            )
            provider_alpha.db.commit()

            # Both providers should see the same metadata since it's the same DB
            alpha_metadata = provider_alpha.get_conversation_metadata()
            beta_metadata = provider_beta.get_conversation_metadata()

            assert alpha_metadata is not None
            assert beta_metadata is not None

            # They should be the same since there's only one row in the table
            assert (
                alpha_metadata.name_tag == "conversation_alpha"
            )  # Uses alpha's conversation_id
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

        for i, timestamp in enumerate(test_timestamps):
            storage_provider.update_conversation_metadata(
                created_at=timestamp, updated_at=timestamp
            )

            metadata = storage_provider.get_conversation_metadata()
            assert metadata is not None
            assert metadata.created_at == timestamp
            assert metadata.updated_at == timestamp

    @pytest.mark.asyncio
    async def test_conversation_metadata_persistence(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test that conversation metadata persists across provider instances."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        created_at = "2024-01-01T12:00:00+00:00"
        updated_at = "2024-01-01T12:00:00+00:00"

        # Create first provider and add metadata
        provider1 = SqliteStorageProvider(
            db_path=temp_db_path,
            conversation_id="persistent_test",
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        provider1.update_conversation_metadata(
            created_at=created_at, updated_at=updated_at
        )
        await provider1.close()

        # Create second provider with same conversation_id and check metadata persists
        provider2 = SqliteStorageProvider(
            db_path=temp_db_path,
            conversation_id="persistent_test",
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        try:
            metadata = provider2.get_conversation_metadata()
            assert metadata is not None
            assert metadata.name_tag == "conversation_persistent_test"
            assert metadata.created_at == created_at
            assert metadata.updated_at == updated_at

        finally:
            await provider2.close()


class TestConversationMetadataEdgeCases:
    """Test edge cases for conversation metadata operations."""

    def test_empty_string_timestamps(
        self, storage_provider_memory: SqliteStorageProvider[DummyMessage]
    ):
        """Test behavior with empty string timestamps."""
        storage_provider_memory.update_conversation_metadata(
            created_at="", updated_at=""
        )

        metadata = storage_provider_memory.get_conversation_metadata()
        assert metadata is not None
        assert metadata.created_at == ""
        assert metadata.updated_at == ""

    @pytest.mark.asyncio
    async def test_very_long_conversation_id(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test conversation metadata with very long conversation ID."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        long_id = "a" * 1000  # Very long conversation ID

        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            conversation_id=long_id,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        try:
            provider.update_conversation_metadata(
                created_at="2024-01-01T12:00:00+00:00",
                updated_at="2024-01-01T12:00:00+00:00",
            )

            metadata = provider.get_conversation_metadata()
            assert metadata is not None
            assert metadata.name_tag == f"conversation_{long_id}"

        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_unicode_conversation_id(
        self, temp_db_path: str, embedding_model: AsyncEmbeddingModel
    ):
        """Test conversation metadata with Unicode conversation ID."""
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        unicode_id = "конверсация_🚀_测试"  # Mixed Unicode

        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            conversation_id=unicode_id,
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        try:
            provider.update_conversation_metadata(
                created_at="2024-01-01T12:00:00+00:00",
                updated_at="2024-01-01T12:00:00+00:00",
            )

            metadata = provider.get_conversation_metadata()
            assert metadata is not None
            assert metadata.name_tag == f"conversation_{unicode_id}"

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
            conversation_id="shared_test1",
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        provider2 = SqliteStorageProvider(
            db_path=temp_db_path,
            conversation_id="shared_test2",
            message_type=DummyMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        try:
            # Update from provider1
            provider1.update_conversation_metadata(
                created_at="2024-01-01T12:00:00+00:00",
                updated_at="2024-01-01T12:00:00+00:00",
            )
            provider1.db.commit()

            # Update from provider2 - should update the same metadata row
            provider2.update_conversation_metadata(
                updated_at="2024-01-01T13:00:00+00:00"
            )
            provider2.db.commit()

            # Both should see the latest state
            metadata1 = provider1.get_conversation_metadata()
            metadata2 = provider2.get_conversation_metadata()

            assert metadata1 is not None
            assert metadata2 is not None
            assert metadata1.created_at == metadata2.created_at
            assert (
                metadata1.updated_at
                == metadata2.updated_at
                == "2024-01-01T13:00:00+00:00"
            )

        finally:
            await provider1.close()
            await provider2.close()
