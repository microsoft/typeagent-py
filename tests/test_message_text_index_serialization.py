# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test for MessageTextIndex serialization to ensure it's no longer a no-op."""

import sqlite3

import numpy as np
import pytest

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.knowpro.convsettings import (
    MessageTextIndexSettings,
    TextEmbeddingIndexSettings,
)
from typeagent.storage.sqlite.messageindex import SqliteMessageTextIndex
from typeagent.storage.sqlite.schema import init_db_schema


class TestMessageTextIndexSerialization:
    """Test MessageTextIndex serialization/deserialization functionality."""

    @pytest.fixture
    def sqlite_db(self) -> sqlite3.Connection:
        """Create a temporary SQLite database."""
        db = sqlite3.connect(":memory:")
        init_db_schema(db)

        # Add test messages to the database
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO Messages (msg_id, chunks, extra, tags, metadata)
            VALUES (1, '["First test message", "Second chunk"]', '{}', '[]', '{}')
        """)
        cursor.execute("""
            INSERT INTO Messages (msg_id, chunks, extra, tags, metadata)
            VALUES (2, '["Another message"]', '{}', '[]', '{}')
        """)
        db.commit()

        return db

    @pytest.mark.asyncio
    async def test_message_text_index_serialize_not_empty(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test that MessageTextIndex serialization produces non-empty data when populated."""
        # Create settings
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        settings = MessageTextIndexSettings(embedding_settings)

        index = SqliteMessageTextIndex(sqlite_db, settings)

        # Add some data to the index by directly inserting into the table
        # (avoiding the async embedding generation for this test)
        cursor = sqlite_db.cursor()
        cursor.execute(
            """
            INSERT INTO MessageTextIndex (msg_id, chunk_ordinal, embedding)
            VALUES (1, 0, ?)
            """,
            [np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()],
        )
        cursor.execute(
            """
            INSERT INTO MessageTextIndex (msg_id, chunk_ordinal, embedding)
            VALUES (1, 1, ?)
            """,
            [np.array([0.4, 0.5, 0.6], dtype=np.float32).tobytes()],
        )
        cursor.execute(
            """
            INSERT INTO MessageTextIndex (msg_id, chunk_ordinal, embedding)
            VALUES (2, 0, ?)
            """,
            [np.array([0.7, 0.8, 0.9], dtype=np.float32).tobytes()],
        )
        sqlite_db.commit()

        # Verify the index has data
        size = await index.size()
        assert size == 3

        # Serialize the index
        serialized_data = await index.serialize()

        # Verify it's not just an empty dict (which was the previous bug)
        assert serialized_data != {}
        assert "indexData" in serialized_data

        index_data = serialized_data["indexData"]
        assert index_data is not None
        assert "textLocations" in index_data

        text_locations = index_data["textLocations"]
        assert len(text_locations) == 3

        # Verify the structure of text locations
        expected_locations = [
            {"messageOrdinal": 1, "chunkOrdinal": 0},
            {"messageOrdinal": 1, "chunkOrdinal": 1},
            {"messageOrdinal": 2, "chunkOrdinal": 0},
        ]

        assert text_locations == expected_locations

    @pytest.mark.asyncio
    async def test_message_text_index_deserialize_restores_data(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test that MessageTextIndex deserialization actually restores data."""
        # Create settings
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        settings = MessageTextIndexSettings(embedding_settings)

        index = SqliteMessageTextIndex(sqlite_db, settings)

        # Create test data to deserialize
        from typeagent.knowpro.interfaces import (
            MessageTextIndexData,
            TextToTextLocationIndexData,
        )

        test_data: MessageTextIndexData = {
            "indexData": TextToTextLocationIndexData(
                textLocations=[
                    {"messageOrdinal": 1, "chunkOrdinal": 0},
                    {"messageOrdinal": 1, "chunkOrdinal": 1},
                    {"messageOrdinal": 2, "chunkOrdinal": 0},
                ],
                embeddings=np.array(
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                    dtype=np.float32,
                ),
            )
        }

        # Verify index is initially empty
        assert await index.size() == 0

        # Deserialize the data
        await index.deserialize(test_data)

        # Verify the data was restored
        size_after = await index.size()
        assert size_after == 3

        # Verify the actual data in the database
        cursor = sqlite_db.cursor()
        cursor.execute(
            "SELECT msg_id, chunk_ordinal FROM MessageTextIndex ORDER BY msg_id, chunk_ordinal"
        )

        rows = cursor.fetchall()
        expected_rows = [
            (1, 0),
            (1, 1),
            (2, 0),
        ]

        assert rows == expected_rows
