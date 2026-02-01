# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for SQLite index implementations with real embeddings."""

import os
import sqlite3
import tempfile
from typing import Generator

import pytest

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro import interfaces
from typeagent.knowpro.convsettings import MessageTextIndexSettings
from typeagent.knowpro.interfaces import (
    SemanticRef,
    Term,
    TextLocation,
    TextRange,
    Topic,
)
from typeagent.storage.sqlite.messageindex import SqliteMessageTextIndex
from typeagent.storage.sqlite.propindex import SqlitePropertyIndex
from typeagent.storage.sqlite.reltermsindex import (
    SqliteRelatedTermsAliases,
    SqliteRelatedTermsFuzzy,
    SqliteRelatedTermsIndex,
)
from typeagent.storage.sqlite.schema import init_db_schema
from typeagent.storage.sqlite.semrefindex import SqliteTermToSemanticRefIndex
from typeagent.storage.sqlite.timestampindex import SqliteTimestampToTextRangeIndex


@pytest.fixture
def embedding_settings(
    embedding_model: AsyncEmbeddingModel,
) -> TextEmbeddingIndexSettings:
    """Create TextEmbeddingIndexSettings for testing."""
    return TextEmbeddingIndexSettings(embedding_model)


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
        cursor.execute("""
            INSERT INTO Messages (msg_id, chunks, start_timestamp)
            VALUES (1, '["test message 1"]', NULL)
        """)
        cursor.execute("""
            INSERT INTO Messages (msg_id, chunks, start_timestamp)
            VALUES (2, '["test message 2"]', NULL)
        """)
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


class TestSqliteRelatedTermsFuzzy:
    """Test SqliteRelatedTermsFuzzy with real embeddings."""

    @pytest.mark.asyncio
    async def test_fuzzy_operations(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test fuzzy operations with real embeddings."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_settings)

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
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test deserialization of fuzzy index data - the critical fix we made."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_settings)

        # Create test data similar to what would be in JSON
        text_items = ["chess", "artificial intelligence", "machine learning"]

        # Create embeddings data (simulate what VectorBase would serialize)
        from typeagent.aitools.vectorbase import VectorBase

        settings = TextEmbeddingIndexSettings(embedding_settings.embedding_model)
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
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test edge cases in fuzzy lookup."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_settings)

        # Empty index
        results = await index.lookup_term("anything")
        assert results == []

        # Add a term
        await index.add_terms(["test term"])

        # Exact match should return score 1.0
        results = await index.lookup_term("test term", min_score=0.0)
        assert any(term.text == "test term" for term in results)

        # Test with multiple terms and verify behavior
        results = await index.lookup_term("xyzabc123")
        # Should return some results (may have unexpected similarity)
        assert isinstance(results, list)
        # All results should have weights (even if surprisingly high)
        assert all(result.weight is not None for result in results)

    @pytest.mark.asyncio
    async def test_fuzzy_serialize(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test serialization of fuzzy index data."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_settings)

        # Add some terms
        test_terms = ["chess", "grandmaster", "artificial intelligence"]
        await index.add_terms(test_terms)

        # Serialize the fuzzy index
        data = index.serialize()

        # Verify serialized data structure
        assert data is not None
        assert "textItems" in data
        assert "embeddings" in data

        # Verify text items
        text_items = data["textItems"]
        assert len(text_items) == 3
        assert "chess" in text_items
        assert "grandmaster" in text_items
        assert "artificial intelligence" in text_items

        # Verify embeddings exist
        embeddings = data["embeddings"]
        assert embeddings is not None


class TestSqliteRelatedTermsIndex:
    """Test SqliteRelatedTermsIndex combined functionality."""

    @pytest.mark.asyncio
    async def test_combined_index_basic(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test the combined related terms index basic functionality."""
        index = SqliteRelatedTermsIndex(sqlite_db, embedding_settings)

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

    @pytest.mark.asyncio
    async def test_serialize_and_deserialize(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test serialization and deserialization of the combined related terms index."""
        index = SqliteRelatedTermsIndex(sqlite_db, embedding_settings)

        # Ensure fuzzy_index is available
        assert index.fuzzy_index is not None

        # Add data to both sub-indexes
        await index.fuzzy_index.add_terms(["chess", "grandmaster", "tournament"])
        await index.aliases.add_related_term("AI", Term("artificial intelligence"))
        await index.aliases.add_related_term("ML", Term("machine learning"))

        # Serialize the combined index
        data = await index.serialize()

        # Verify serialized data structure
        assert data is not None
        assert "aliasData" in data
        assert "textEmbeddingData" in data

        # Verify alias data
        alias_data = data["aliasData"]
        assert alias_data is not None
        assert "relatedTerms" in alias_data
        related_terms = alias_data["relatedTerms"]
        assert related_terms is not None
        assert len(related_terms) == 2

        # Verify text embedding data
        text_embedding_data = data["textEmbeddingData"]
        assert text_embedding_data is not None
        assert "textItems" in text_embedding_data
        assert len(text_embedding_data["textItems"]) == 3
        assert "chess" in text_embedding_data["textItems"]
        assert "grandmaster" in text_embedding_data["textItems"]
        assert "embeddings" in text_embedding_data

        # Create a fresh database and index to test deserialization
        with tempfile.TemporaryDirectory() as tmpdir:
            fresh_db_path = os.path.join(tmpdir, "fresh_test.db")
            fresh_db = sqlite3.connect(fresh_db_path)
            init_db_schema(fresh_db)

            fresh_index = SqliteRelatedTermsIndex(fresh_db, embedding_settings)

            # Deserialize into fresh index
            await fresh_index.deserialize(data)

            # Verify alias data was restored
            ai_results = await fresh_index.aliases.lookup_term("AI")
            assert ai_results is not None
            assert len(ai_results) == 1
            assert ai_results[0].text == "artificial intelligence"

            ml_results = await fresh_index.aliases.lookup_term("ML")
            assert ml_results is not None
            assert len(ml_results) == 1
            assert ml_results[0].text == "machine learning"

            # Verify fuzzy index data was restored
            fresh_fuzzy = fresh_index.fuzzy_index
            assert fresh_fuzzy is not None
            assert await fresh_fuzzy.size() == 3

            # Use concrete type to access get_terms (not in interface)
            assert isinstance(fresh_fuzzy, SqliteRelatedTermsFuzzy)
            fuzzy_terms = await fresh_fuzzy.get_terms()
            assert "chess" in fuzzy_terms
            assert "grandmaster" in fuzzy_terms
            assert "tournament" in fuzzy_terms

            # Verify fuzzy lookup works after deserialization
            fuzzy_results = await fresh_fuzzy.lookup_term(
                "chess", max_hits=5, min_score=0.1
            )
            assert len(fuzzy_results) > 0

            fresh_db.close()


# Integration test to verify the fix for the regression we encountered
class TestRegressionPrevention:
    """Tests to prevent the regressions we've fixed."""

    @pytest.mark.asyncio
    async def test_fuzzy_index_first_run_scenario(
        self,
        sqlite_db: sqlite3.Connection,
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """
        Test the specific scenario that was failing:
        Fresh database -> load from JSON -> query should work immediately.

        This test prevents the regression where SQLite deserialize was a no-op.
        """
        # Create a fresh fuzzy index
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_settings)

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

        settings = TextEmbeddingIndexSettings(embedding_settings.embedding_model)
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


class TestSqliteIndexesEdgeCases:
    """Test edge cases and error conditions in SQLite indexes."""

    @pytest.mark.asyncio
    async def test_term_index_edge_cases(self, sqlite_db: sqlite3.Connection):
        """Test edge cases in term index."""
        index = SqliteTermToSemanticRefIndex(sqlite_db)
        assert await index.size() == 0

        # Test with None/empty lookups
        results = await index.lookup_term("")
        assert results == []
        assert await index.size() == 0

        # Test removing terms
        await index.add_term("remove_test", 1)
        assert await index.size() == 1
        await index.remove_term("remove_test", 1)
        results = await index.lookup_term("remove_test")
        assert results == []
        assert await index.size() == 0

        # Test clearing
        await index.add_term("clear_test", 2)
        assert await index.size() == 1
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
        results = await index.lookup_property("test_prop", "test_value")
        assert results is not None
        assert len(results) == 1
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
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test edge cases in fuzzy index."""
        index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_settings)

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
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test basic operations of message text index."""
        # Create settings
        embedding_settings_local = TextEmbeddingIndexSettings(
            embedding_settings.embedding_model
        )
        settings = MessageTextIndexSettings(embedding_settings_local)

        index = SqliteMessageTextIndex(sqlite_db, settings)

        # Test initial state
        assert await index.size() == 0

        # Test lookup_text on empty index
        results = await index.lookup_text("test query", max_matches=5)
        assert results == []

        # Create some mock messages for testing
        from typeagent.knowpro.interfaces import IMessage

        from conftest import FakeMessage

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
        embedding_settings: TextEmbeddingIndexSettings,
        needs_auth: None,
    ):
        """Test serialization edge cases."""
        fuzzy_index = SqliteRelatedTermsFuzzy(sqlite_db, embedding_settings)

        # Test serialization of empty index
        empty_data = fuzzy_index.serialize()
        assert empty_data is not None
        assert empty_data["textItems"] == []

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

    @pytest.mark.asyncio
    async def test_term_normalization_whitespace(self, sqlite_db: sqlite3.Connection):
        """Test whitespace normalization in _prepare_term()."""
        index = SqliteTermToSemanticRefIndex(sqlite_db)

        # Test that whitespace variations normalize to the same term
        whitespace_variants = [
            "hello world",  # baseline
            "  hello world  ",  # leading/trailing spaces
            "hello\tworld",  # tab instead of space
            "hello\nworld",  # newline instead of space
            "hello   world",  # multiple spaces
            "hello \t world",  # mixed whitespace
        ]

        # Add all variants - they should normalize to the same internal form
        for i, variant in enumerate(whitespace_variants):
            await index.add_term(variant, i + 1)

        # All variants should find the same normalized term
        for i, variant in enumerate(whitespace_variants):
            results = await index.lookup_term(variant)
            assert results is not None, f"Should find results for '{variant}'"
            assert len(results) == len(
                whitespace_variants
            ), f"Expected {len(whitespace_variants)} results for '{variant}', got {len(results)}"
            # All should map to the same normalized form, so should find all semantic refs
            expected_semrefs = set(range(1, len(whitespace_variants) + 1))
            actual_semrefs = {r.semantic_ref_ordinal for r in results}
            assert actual_semrefs == expected_semrefs

    @pytest.mark.asyncio
    async def test_term_normalization_unicode(self, sqlite_db: sqlite3.Connection):
        """Test Unicode normalization and roundtripping."""
        index = SqliteTermToSemanticRefIndex(sqlite_db)

        # Test Unicode normalization - these should be equivalent after NFC normalization
        unicode_variants = [
            "café",  # NFC form (single é character)
            "cafe\u0301",  # NFD form (e + combining acute accent)
        ]

        # Test higher Unicode planes
        high_plane_terms = [
            "test🏠house",  # Emoji (U+1F3E0)
            "math𝑨𝑩𝑪",  # Mathematical symbols (U+1D400 range)
            "ancient𓀀𓀁",  # Egyptian hieroglyphs (U+13000 range)
        ]

        # Add Unicode variants
        for i, variant in enumerate(unicode_variants):
            await index.add_term(variant, 100 + i)

        # Both variants should resolve to the same normalized form
        results1 = await index.lookup_term(unicode_variants[0])
        results2 = await index.lookup_term(unicode_variants[1])
        assert (
            results1 is not None and results2 is not None
        ), "Both Unicode forms should return results"
        assert len(results1) == len(
            results2
        ), "NFC and NFD forms should normalize to same term"
        assert len(results1) == 2, f"Expected 2 results, got {len(results1)}"

        # Test higher plane Unicode roundtripping
        for i, term in enumerate(high_plane_terms):
            await index.add_term(term, 200 + i)
            results = await index.lookup_term(term)
            assert (
                results is not None
            ), f"Should find results for higher plane Unicode: '{term}'"
            assert len(results) == 1, f"Should roundtrip higher plane Unicode: '{term}'"
            assert results[0].semantic_ref_ordinal == 200 + i

    @pytest.mark.asyncio
    async def test_term_case_sensitivity(self, sqlite_db: sqlite3.Connection):
        """Test case normalization in _prepare_term()."""
        index = SqliteTermToSemanticRefIndex(sqlite_db)

        # Test case variations
        case_variants = [
            "Hello",
            "HELLO",
            "hello",
            "HeLLo",
        ]

        # Add all case variants
        for i, variant in enumerate(case_variants):
            await index.add_term(variant, 300 + i)

        # All should normalize to same lowercase form
        for variant in case_variants:
            results = await index.lookup_term(variant)
            assert (
                results is not None
            ), f"Should find results for case variant '{variant}'"
            assert len(results) == len(
                case_variants
            ), f"Case variant '{variant}' should find all normalized forms"
            expected_semrefs = set(range(300, 300 + len(case_variants)))
            actual_semrefs = {r.semantic_ref_ordinal for r in results}
            assert actual_semrefs == expected_semrefs

        # Test Unicode case sensitivity
        unicode_cases = ["Café", "CAFÉ", "café"]
        for i, variant in enumerate(unicode_cases):
            await index.add_term(variant, 400 + i)

        for variant in unicode_cases:
            results = await index.lookup_term(variant)
            assert (
                results is not None
            ), f"Should find results for Unicode case variant '{variant}'"
            assert len(results) == len(
                unicode_cases
            ), f"Unicode case variant '{variant}' should find all forms"
