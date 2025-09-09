# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for SQLite index implementations with real embeddings."""

import sqlite3
import tempfile
import os
from typing import Generator
import pytest

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
from typeagent.storage.sqlite.timestampindex import SqliteTimestampToTextRangeIndex
from typeagent.storage.sqlite.reltermsindex import (
    SqliteRelatedTermsAliases,
    SqliteRelatedTermsFuzzy,
    SqliteRelatedTermsIndex,
)
from typeagent.storage.sqlite.schema import init_db_schema

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


def make_semantic_ref(ordinal: int = 0, text: str = "test_topic") -> SemanticRef:
    """Helper to create a SemanticRef for testing."""
    topic = Topic(text=text)
    location = TextLocation(message_ordinal=0)
    text_range = TextRange(start=location)
    return SemanticRef(
        semantic_ref_ordinal=ordinal,
        range=text_range,
        knowledge=topic,
    )


class TestSqliteTermToSemanticRefIndex:
    """Test SqliteTermToSemanticRefIndex functionality."""

    @pytest.mark.asyncio
    async def test_empty_index(self, sqlite_db: sqlite3.Connection):
        """Test empty index behavior."""
        index = SqliteTermToSemanticRefIndex(sqlite_db)

        assert await index.size() == 0
        assert await index.get_terms() == []

        # Search non-existent term
        results = await index.lookup_term("nonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_add_and_lookup_terms(self, sqlite_db: sqlite3.Connection):
        """Test adding and looking up terms."""
        index = SqliteTermToSemanticRefIndex(sqlite_db)

        # Add terms
        term = await index.add_term("test", 1)
        assert term == "test"

        term = await index.add_term(
            "another", interfaces.ScoredSemanticRefOrdinal(2, 0.8)
        )
        assert term == "another"

        # Check size and terms
        assert await index.size() == 2
        terms = await index.get_terms()
        assert sorted(terms) == ["another", "test"]

        # Lookup terms
        results = await index.lookup_term("test")
        assert results is not None
        assert len(results) == 1
        assert results[0].semantic_ref_ordinal == 1

        results = await index.lookup_term("another")
        assert results is not None
        assert len(results) == 1
        assert results[0].semantic_ref_ordinal == 2

    @pytest.mark.asyncio
    async def test_empty_term_handling(self, sqlite_db: sqlite3.Connection):
        """Test that empty terms are handled correctly."""
        index = SqliteTermToSemanticRefIndex(sqlite_db)

        # Empty term should return as-is
        term = await index.add_term("", 1)
        assert term == ""

        # Should not affect size
        assert await index.size() == 0


class TestSqlitePropertyIndex:
    """Test SqlitePropertyIndex functionality."""

    @pytest.mark.asyncio
    async def test_property_operations(self, sqlite_db: sqlite3.Connection):
        """Test property index operations."""
        index = SqlitePropertyIndex(sqlite_db)

        # Add property mappings
        await index.add_property("topic", "ai", 1)
        await index.add_property("topic", "machine learning", 2)
        await index.add_property("author", "john doe", 3)

        # Lookup by property
        results = await index.lookup_property("topic", "ai")
        assert results is not None
        assert len(results) == 1
        assert results[0].semantic_ref_ordinal == 1

        # Get all values
        values = await index.get_values()
        assert len(values) >= 3  # Should contain our values

        # Check size
        assert await index.size() >= 3


class TestSqliteTimestampToTextRangeIndex:
    """Test SqliteTimestampToTextRangeIndex functionality."""

    @pytest.mark.asyncio
    async def test_timestamp_operations(self, sqlite_db: sqlite3.Connection):
        """Test timestamp index operations."""
        index = SqliteTimestampToTextRangeIndex(sqlite_db)

        # First, we need to create some messages in the database for the timestamps to reference
        cursor = sqlite_db.cursor()
        cursor.execute(
            """
            INSERT INTO Messages (msg_id, chunks, start_timestamp)
            VALUES (1, '["test message 1"]', NULL)
        """
        )
        cursor.execute(
            """
            INSERT INTO Messages (msg_id, chunks, start_timestamp)
            VALUES (2, '["test message 2"]', NULL)
        """
        )
        sqlite_db.commit()

        # Add timestamps to existing messages
        success = await index.add_timestamp(1, "2023-01-01T10:00:00Z")
        assert success

        success = await index.add_timestamp(2, "2023-01-01T11:00:00Z")
        assert success

        # Check size
        assert await index.size() == 2

        # Get timestamp ranges
        results = await index.get_timestamp_ranges("2023-01-01T10:00:00Z")
        assert len(results) == 1
        assert results[0].timestamp == "2023-01-01T10:00:00Z"

        # Range query
        results = await index.get_timestamp_ranges(
            "2023-01-01T10:00:00Z", "2023-01-01T11:00:00Z"
        )
        assert len(results) == 2


class TestSqliteRelatedTermsAliases:
    """Test SqliteRelatedTermsAliases functionality."""

    @pytest.mark.asyncio
    async def test_aliases_operations(self, sqlite_db: sqlite3.Connection):
        """Test aliases operations."""
        index = SqliteRelatedTermsAliases(sqlite_db)

        # Initially empty
        assert await index.is_empty()
        assert await index.size() == 0

        # Add related terms
        await index.add_related_term("ai", Term("artificial intelligence"))
        await index.add_related_term("ai", [Term("machine learning"), Term("ML")])

        # Check size and emptiness
        assert not await index.is_empty()
        assert await index.size() == 1

        # Lookup related terms
        results = await index.lookup_term("ai")
        assert results is not None
        assert len(results) == 3
        term_texts = [term.text for term in results]
        assert "artificial intelligence" in term_texts
        assert "machine learning" in term_texts
        assert "ML" in term_texts

        # Get related terms
        related = await index.get_related_terms("ai")
        assert related is not None
        assert len(related) == 3

        # Set related terms (replace existing)
        await index.set_related_terms("ai", ["neural networks", "deep learning"])
        related = await index.get_related_terms("ai")
        assert related is not None
        assert len(related) == 2
        assert "neural networks" in related
        assert "deep learning" in related

    @pytest.mark.asyncio
    async def test_serialize_deserialize(self, sqlite_db: sqlite3.Connection):
        """Test serialization and deserialization of aliases."""
        index = SqliteRelatedTermsAliases(sqlite_db)

        # Add some data
        await index.add_related_term(
            "ai", [Term("artificial intelligence"), Term("ML")]
        )
        await index.add_related_term("python", Term("programming language"))

        # Serialize
        data = await index.serialize()
        assert data is not None
        assert "relatedTerms" in data
        related_terms = data["relatedTerms"]
        assert related_terms is not None
        assert len(related_terms) == 2

        # Clear and deserialize
        await index.clear()
        assert await index.is_empty()

        await index.deserialize(data)

        # Verify data was restored
        assert not await index.is_empty()
        assert await index.size() == 2

        ai_related = await index.get_related_terms("ai")
        assert ai_related is not None
        assert len(ai_related) == 2

        python_related = await index.get_related_terms("python")
        assert python_related is not None
        assert len(python_related) == 1


class TestSqliteRelatedTermsFuzzy:
    """Test SqliteRelatedTermsFuzzy with real embeddings."""

    @pytest.mark.asyncio
    async def test_fuzzy_operations(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test fuzzy operations with real embeddings."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_model)

        # Initially empty
        assert await index.size() == 0
        assert await index.get_terms() == []

        # Add terms with embeddings
        await index.add_terms(
            ["artificial intelligence", "machine learning", "neural networks"]
        )

        # Check size
        assert await index.size() == 3
        terms = await index.get_terms()
        assert len(terms) == 3
        assert "artificial intelligence" in terms

        # Test fuzzy lookup
        results = await index.lookup_term("AI", max_hits=5, min_score=0.1)
        assert len(results) > 0
        # Should find semantically similar terms
        result_texts = [term.text for term in results]
        assert any(
            "artificial intelligence" in text.lower()
            or "machine learning" in text.lower()
            for text in result_texts
        )

    @pytest.mark.asyncio
    async def test_fuzzy_deserialize(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test deserialization of fuzzy index data - the critical fix we made."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_model)

        # Create test data similar to what would be in JSON
        text_items = ["chess", "artificial intelligence", "machine learning"]

        # Create embeddings data (simulate what VectorBase would serialize)
        from typeagent.aitools.vectorbase import VectorBase

        settings = TextEmbeddingIndexSettings(embedding_model)
        temp_vectorbase = VectorBase(settings)

        # Add embeddings to the vector base using add_key
        for text in text_items:
            await temp_vectorbase.add_key(text)

        # Serialize the embeddings
        embeddings_data = temp_vectorbase.serialize()

        # Create the data structure that would come from JSON
        index_data = {"textItems": text_items, "embeddings": embeddings_data}

        # Test deserialization - this is the critical fix
        await index.deserialize(index_data)  # type: ignore

        # Verify data was loaded
        assert await index.size() == 3
        terms = await index.get_terms()
        assert len(terms) == 3
        assert "chess" in terms
        assert "artificial intelligence" in terms
        assert "machine learning" in terms

        # Test that fuzzy lookup works after deserialization
        results = await index.lookup_term("AI", max_hits=5, min_score=0.1)
        assert len(results) > 0
        # Should find the semantically similar "artificial intelligence"
        result_texts = [term.text for term in results]
        assert any("artificial intelligence" in text for text in result_texts)

    @pytest.mark.asyncio
    async def test_fuzzy_lookup_edge_cases(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test edge cases in fuzzy lookup."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_model)

        # Empty index
        results = await index.lookup_term("anything")
        assert results == []

        # Add a term
        await index.add_terms(["test term"])

        # Exact match should be filtered out (self-match)
        results = await index.lookup_term("test term", min_score=0.0)
        # Should not return the exact same term
        assert not any(term.text == "test term" for term in results)

        # Test with multiple terms and verify behavior
        results = await index.lookup_term("xyzabc123")
        # Should return some results (may have unexpected similarity)
        assert isinstance(results, list)
        # All results should have weights (even if surprisingly high)
        assert all(result.weight is not None for result in results)


class TestSqliteRelatedTermsIndex:
    """Test SqliteRelatedTermsIndex combined functionality."""

    @pytest.mark.asyncio
    async def test_combined_index_basic(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """Test the combined related terms index basic functionality."""
        index = SqliteRelatedTermsIndex(sqlite_db, embedding_model)

        # Test that both sub-indexes are accessible
        assert index.aliases is not None
        assert index.fuzzy_index is not None

        # Add some terms to both sub-indexes
        await index.fuzzy_index.add_terms(
            ["artificial intelligence", "machine learning"]
        )
        await index.aliases.add_related_term("AI", Term("artificial intelligence"))

        # Test that both indexes work
        fuzzy_results = await index.fuzzy_index.lookup_term("AI", max_hits=5)
        assert len(fuzzy_results) > 0

        alias_results = await index.aliases.lookup_term("AI")
        assert alias_results is not None
        assert len(alias_results) == 1
        assert alias_results[0].text == "artificial intelligence"


# Integration test to verify the fix for the regression we encountered
class TestRegressionPrevention:
    """Tests to prevent the regressions we've fixed."""

    @pytest.mark.asyncio
    async def test_fuzzy_index_first_run_scenario(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_model: AsyncEmbeddingModel,
        needs_auth: None,
    ):
        """
        Test the specific scenario that was failing:
        Fresh database -> load from JSON -> query should work immediately.

        This test prevents the regression where SQLite deserialize was a no-op.
        """
        # Create a fresh fuzzy index
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_model)

        # Simulate JSON data that would be loaded on first run
        # This represents the scenario where we have podcast data with pre-computed embeddings
        text_items = [
            "chess",
            "Magnus Carlsen",
            "grandmaster",
            "artificial intelligence",
        ]

        # Create embeddings as they would exist in the JSON
        from typeagent.aitools.vectorbase import VectorBase

        settings = TextEmbeddingIndexSettings(embedding_model)
        temp_vectorbase = VectorBase(settings)

        for text in text_items:
            await temp_vectorbase.add_key(text)

        # Create the data structure that comes from JSON
        json_data = {"textItems": text_items, "embeddings": temp_vectorbase.serialize()}

        # This is the critical test: deserialize should populate the fuzzy index
        await index.deserialize(json_data)  # type: ignore

        # Verify the index is populated (size check)
        size = await index.size()
        assert size == len(text_items), f"Expected {len(text_items)} terms, got {size}"

        # Verify we can immediately query and get results (the actual regression test)
        # This should find "Magnus Carlsen" when searching for chess-related terms
        chess_results = await index.lookup_term(
            "chess grandmaster", max_hits=10, min_score=0.1
        )

        # Should find related terms
        assert len(chess_results) > 0, "Should find related terms after deserialize"

        # Should find Magnus Carlsen among the results for chess-related queries
        result_texts = [term.text.lower() for term in chess_results]
        assert any(
            "magnus carlsen" in text for text in result_texts
        ), f"Should find Magnus Carlsen in results: {result_texts}"

        # Test the specific query that was failing
        magnus_results = await index.lookup_term(
            "grandmaster", max_hits=10, min_score=0.1
        )
        assert len(magnus_results) > 0, "Should find results for grandmaster query"
