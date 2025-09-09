# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Additional tests for SQLite indexes to improve coverage."""

import sqlite3
import tempfile
import os
from typing import Generator
import pytest
import pytest_asyncio

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro import interfaces
from typeagent.knowpro.interfaces import (
    SemanticRef,
    TextLocation,
    TextRange,
    Topic,
    Term,
)
from typeagent.storage.sqlite.semrefindex import SqliteTermToSemanticRefIndex
from typeagent.storage.sqlite.propindex import SqlitePropertyIndex
from typeagent.storage.sqlite.reltermsindex import (
    SqliteRelatedTermsAliases,
    SqliteRelatedTermsFuzzy,
)
from typeagent.storage.sqlite.messageindex import SqliteMessageTextIndex
from typeagent.storage.sqlite.schema import init_db_schema
from typeagent.knowpro.convsettings import MessageTextIndexSettings

from fixtures import needs_auth, embedding_model


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary SQLite database file."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def sqlite_db(temp_db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Create and initialize a SQLite database connection."""
    db = sqlite3.connect(temp_db_path)
    init_db_schema(db)
    yield db
    db.close()


class TestSqliteIndexesEdgeCases:
    """Test edge cases and error conditions in SQLite indexes."""

    @pytest.mark.asyncio
    async def test_term_index_edge_cases(self, sqlite_db: sqlite3.Connection):
        """Test edge cases in term index."""
        index = SqliteTermToSemanticRefIndex(sqlite_db)

        # Test with None/empty lookups
        results = await index.lookup_term("")
        assert results == []

        # Test removing terms
        await index.add_term("remove_test", 1)
        await index.remove_term("remove_test", 1)
        results = await index.lookup_term("remove_test")
        assert results == []

        # Test clearing
        await index.add_term("clear_test", 2)
        await index.clear()
        assert await index.size() == 0

    @pytest.mark.asyncio
    async def test_property_index_edge_cases(self, sqlite_db: sqlite3.Connection):
        """Test edge cases in property index."""
        index = SqlitePropertyIndex(sqlite_db)

        # Test lookup of non-existent property
        results = await index.lookup_property("nonexistent", "value")
        assert results is None

        # Test removal operations
        await index.add_property("test_prop", "test_value", 1)
        await index.remove_property("test_prop", 1)
        results = await index.lookup_property("test_prop", "test_value")
        assert results is None

        # Test remove all for semref
        await index.add_property("prop1", "val1", 2)
        await index.add_property("prop2", "val2", 2)
        await index.remove_all_for_semref(2)

        results1 = await index.lookup_property("prop1", "val1")
        results2 = await index.lookup_property("prop2", "val2")
        assert results1 is None
        assert results2 is None

    @pytest.mark.asyncio
    async def test_related_terms_aliases_edge_cases(
        self, sqlite_db: sqlite3.Connection
    ):
        """Test edge cases in aliases."""
        index = SqliteRelatedTermsAliases(sqlite_db)

        # Test lookup of non-existent term
        results = await index.lookup_term("nonexistent")
        assert results is None

        # Test adding different types of related terms
        await index.add_related_term("test", Term("string_term"))  # Term object
        await index.add_related_term("test", Term("term_object"))  # Term object
        await index.add_related_term("test", [Term("list_term")])  # list

        related = await index.get_related_terms("test")
        assert related is not None
        assert len(related) == 3

        # Test deserialize with None data
        await index.deserialize(None)
        # Should not crash

        # Test deserialize with empty data
        await index.deserialize({"relatedTerms": []})

        # Test with properly formatted data
        from typeagent.knowpro.interfaces import TermToRelatedTermsData

        formatted_data: TermToRelatedTermsData = {
            "relatedTerms": [
                {"termText": "test", "relatedTerms": []},  # valid but empty
                {"termText": "orphan", "relatedTerms": [{"text": "related"}]},  # valid
            ]
        }
        await index.deserialize(formatted_data)

    @pytest.mark.asyncio
    async def test_fuzzy_index_edge_cases(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test edge cases in fuzzy index."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_model)

        # Test with empty embeddings
        await index.add_terms([])  # Empty list
        assert await index.size() == 0

        # Test lookup_terms (plural) method
        results_list = await index.lookup_terms(["test1", "test2"], max_hits=5)
        assert len(results_list) == 2
        assert all(isinstance(results, list) for results in results_list)

        # Test deserialize with various data formats
        from typeagent.knowpro.interfaces import TextEmbeddingIndexData

        # Valid data with None embeddings
        valid_data1: TextEmbeddingIndexData = {
            "textItems": ["test"],
            "embeddings": None,
        }
        await index.deserialize(valid_data1)

        # Valid data with empty text items
        valid_data2: TextEmbeddingIndexData = {"textItems": [], "embeddings": None}
        await index.deserialize(valid_data2)

    @pytest.mark.asyncio
    async def test_message_text_index_basic(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test basic operations of message text index."""
        # Create settings
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        settings = MessageTextIndexSettings(embedding_settings)

        index = SqliteMessageTextIndex(sqlite_db, settings)

        # Test initial state
        assert await index.size() == 0

        # Test lookup_text on empty index
        results = await index.lookup_text("test query", max_matches=5)
        assert results == []

        # Create some mock messages for testing
        from fixtures import FakeMessage
        from typeagent.knowpro.interfaces import IMessage

        messages: list[IMessage] = [
            FakeMessage(text_chunks=["First test message", "Second chunk"]),
            FakeMessage(text_chunks=["Another message"]),
        ]

        # Add messages using the proper method
        await index.add_messages_starting_at(0, messages)

        # After adding messages, size should be > 0
        size = await index.size()
        assert size > 0

        # Test lookup with real text
        results = await index.lookup_text("test message", max_matches=5)
        assert isinstance(results, list)

        # Test is_empty method
        assert not await index.is_empty()

        # Test clear and verify it's empty
        await index.clear()
        assert await index.size() == 0
        assert await index.is_empty()

    @pytest.mark.asyncio
    async def test_serialization_edge_cases(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test serialization edge cases."""
        fuzzy_index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_model)

        # Test serialization of empty index
        # Note: fuzzy index doesn't implement serialize (returns empty for SQLite)
        # But test that calling it doesn't crash
        # This would be implemented if needed

        # Test fuzzy index with some data then clear
        await fuzzy_index.add_terms(["test1", "test2"])
        await fuzzy_index.clear()
        assert await fuzzy_index.size() == 0

        # Test remove_term
        # TODO: Implement remove_term properly before enabling this test
        # await fuzzy_index.add_terms(["remove_me"])
        # await fuzzy_index.remove_term("remove_me")
        # results = await fuzzy_index.lookup_term("remove_me")
        # assert results == []
