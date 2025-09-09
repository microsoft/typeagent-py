# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based related terms index implementations."""

import sqlite3
import typing

from ...aitools.embeddings import AsyncEmbeddingModel
from ...aitools.vectorbase import TextEmbeddingIndexSettings, VectorBase
from ...knowpro import interfaces


class SqliteRelatedTermsAliases(interfaces.ITermToRelatedTerms):
    """SQLite-backed implementation of term to related terms aliases."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    async def lookup_term(self, text: str) -> list[interfaces.Term] | None:
        cursor = self.db.cursor()
        cursor.execute("SELECT alias FROM RelatedTermsAliases WHERE term = ?", (text,))
        results = [interfaces.Term(row[0]) for row in cursor.fetchall()]
        return results if results else None

    async def add_related_term(
        self, text: str, related_terms: interfaces.Term | list[interfaces.Term]
    ) -> None:
        # Convert single Term to list
        if isinstance(related_terms, interfaces.Term):
            related_terms = [related_terms]

        cursor = self.db.cursor()
        # Add new aliases (use INSERT OR IGNORE to avoid duplicates)
        for related_term in related_terms:
            cursor.execute(
                "INSERT OR IGNORE INTO RelatedTermsAliases (term, alias) VALUES (?, ?)",
                (text, related_term.text),
            )

    async def remove_term(self, text: str) -> None:
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM RelatedTermsAliases WHERE term = ?", (text,))

    async def clear(self) -> None:
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM RelatedTermsAliases")

    async def get_related_terms(self, term: str) -> list[str] | None:
        cursor = self.db.cursor()
        cursor.execute("SELECT alias FROM RelatedTermsAliases WHERE term = ?", (term,))
        results = [row[0] for row in cursor.fetchall()]
        return results if results else None

    async def set_related_terms(self, term: str, related_terms: list[str]) -> None:
        cursor = self.db.cursor()
        # Clear existing aliases for this term
        cursor.execute("DELETE FROM RelatedTermsAliases WHERE term = ?", (term,))
        # Add new aliases
        for alias in related_terms:
            cursor.execute(
                "INSERT INTO RelatedTermsAliases (term, alias) VALUES (?, ?)",
                (term, alias),
            )

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(DISTINCT term) FROM RelatedTermsAliases")
        return cursor.fetchone()[0]

    async def get_terms(self) -> list[str]:
        cursor = self.db.cursor()
        cursor.execute("SELECT DISTINCT term FROM RelatedTermsAliases ORDER BY term")
        return [row[0] for row in cursor.fetchall()]

    async def is_empty(self) -> bool:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM RelatedTermsAliases")
        return cursor.fetchone()[0] == 0

    async def serialize(self) -> interfaces.TermToRelatedTermsData:
        """Serialize the aliases data."""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT term, alias FROM RelatedTermsAliases ORDER BY term, alias"
        )

        # Group by term
        term_to_aliases: dict[str, list[str]] = {}
        for term, alias in cursor.fetchall():
            if term not in term_to_aliases:
                term_to_aliases[term] = []
            term_to_aliases[term].append(alias)

        # Convert to the expected format
        items = []
        for term, aliases in term_to_aliases.items():
            term_data_list = [interfaces.TermData(text=alias) for alias in aliases]
            items.append(
                interfaces.TermsToRelatedTermsDataItem(
                    termText=term, relatedTerms=term_data_list
                )
            )

        return interfaces.TermToRelatedTermsData(relatedTerms=items)

    async def deserialize(self, data: interfaces.TermToRelatedTermsData | None) -> None:
        """Deserialize alias data."""
        cursor = self.db.cursor()

        # Clear existing data
        cursor.execute("DELETE FROM RelatedTermsAliases")

        if data is None:
            return

        related_terms = data.get("relatedTerms", [])

        if related_terms:
            # Prepare all insertion data for bulk operation
            insertion_data = []
            for item in related_terms:
                if item and item.get("termText") and item.get("relatedTerms"):
                    term = item["termText"]
                    for term_data in item["relatedTerms"]:
                        alias = term_data["text"]
                        insertion_data.append((term, alias))

            # Bulk insert all the data
            if insertion_data:
                cursor.executemany(
                    "INSERT INTO RelatedTermsAliases (term, alias) VALUES (?, ?)",
                    insertion_data,
                )


class SqliteRelatedTermsFuzzy(interfaces.ITermToRelatedTermsFuzzy):
    """SQLite-backed implementation of fuzzy term relationships with persistent embeddings."""

    # TODO: Require settings to be passed in so embedding_model doesn't need to be.
    def __init__(self, db: sqlite3.Connection, embedding_model: AsyncEmbeddingModel):
        self.db = db
        # Create a VectorBase for caching and fuzzy matching
        self._embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        self._vector_base = VectorBase(self._embedding_settings)
        # Keep reference to embedding model for direct access if needed
        self._embedding_model = embedding_model
        # Maintain our own list of terms to map ordinals back to keys
        self._terms_list: list[str] = []
        self._terms_to_ordinal: dict[str, int] = {}

    async def lookup_term(
        self,
        text: str,
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[interfaces.Term]:
        """Look up similar terms using fuzzy matching."""

        # Use VectorBase for fuzzy embedding search instead of manual similarity calculation
        try:
            # Search for similar terms using VectorBase
            similar_results = await self._vector_base.fuzzy_lookup(
                text, max_hits=max_hits, min_score=min_score or 0.7
            )

            # Convert VectorBase results to Term objects
            results = []
            for scored_int in similar_results:
                # Get the term text from our ordinal mapping
                if scored_int.item < len(self._terms_list):
                    term_text = self._terms_list[scored_int.item]

                    # Skip exact self-match
                    if term_text == text and abs(scored_int.score - 1.0) < 0.001:
                        continue

                    results.append(interfaces.Term(term_text, scored_int.score))

            return results

        except Exception:
            # Fallback to direct database query if VectorBase fails
            return await self._lookup_term_fallback(text, max_hits, min_score)

    async def _lookup_term_fallback(
        self,
        text: str,
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[interfaces.Term]:
        """Fallback method using direct embedding comparison."""
        # Generate embedding for query text
        query_embedding = await self._embedding_model.get_embedding(text)
        if query_embedding is None:
            return []

        # Get all stored terms and their embeddings
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT DISTINCT term, term_embedding FROM RelatedTermsFuzzy WHERE term_embedding IS NOT NULL"
        )

        results = []
        from .schema import deserialize_embedding
        import numpy as np

        for term, embedding_blob in cursor.fetchall():
            if embedding_blob is None:
                continue

            # Deserialize the stored embedding
            stored_embedding = deserialize_embedding(embedding_blob)
            if stored_embedding is None:
                continue

            # Compute cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )

            # Skip if below minimum score threshold
            if min_score is not None and similarity < min_score:
                continue

            # Skip exact self-match (similarity 1.0 with identical text)
            if term == text and abs(similarity - 1.0) < 0.001:
                continue

            results.append(interfaces.Term(term, float(similarity)))

        # Sort by similarity score descending
        results.sort(key=lambda x: x.weight, reverse=True)

        # Apply max_hits limit
        if max_hits is not None:
            results = results[:max_hits]

        return results

    async def remove_term(self, term: str) -> None:
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM RelatedTermsFuzzy WHERE term = ?", (term,))
        # Also remove any entries where this term appears as a related_term
        cursor.execute("DELETE FROM RelatedTermsFuzzy WHERE related_term = ?", (term,))

        # Clear VectorBase and local mappings - they will be rebuilt on next lookup
        self._vector_base.clear()
        self._terms_list.clear()
        self._terms_to_ordinal.clear()

    async def clear(self) -> None:
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM RelatedTermsFuzzy")

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(DISTINCT term) FROM RelatedTermsFuzzy")
        return cursor.fetchone()[0]

    async def get_terms(self) -> list[str]:
        cursor = self.db.cursor()
        cursor.execute("SELECT DISTINCT term FROM RelatedTermsFuzzy ORDER BY term")
        return [row[0] for row in cursor.fetchall()]

    async def add_terms(self, texts: list[str]) -> None:
        """Add terms with self-related embeddings."""
        from .schema import serialize_embedding

        cursor = self.db.cursor()
        for text in texts:
            # Add to VectorBase for fuzzy lookup if not already present
            if text not in self._terms_to_ordinal:
                await self._vector_base.add_key(text)
                ordinal = len(self._terms_list)
                self._terms_list.append(text)
                self._terms_to_ordinal[text] = ordinal

            # Generate embedding for term and store in database
            embed = await self._embedding_model.get_embedding(text)
            serialized = serialize_embedding(embed)
            # Insert term as related to itself, only storing term_embedding once
            cursor.execute(
                """
                    INSERT OR REPLACE INTO RelatedTermsFuzzy
                    (term, related_term, score, term_embedding)
                    VALUES (?, ?, 1.0, ?)
                    """,
                (text, text, serialized),
            )

    async def get_related_terms(
        self, term: str, max_matches: int | None = None, min_score: float | None = None
    ) -> list[interfaces.Term] | None:
        cursor = self.db.cursor()

        query = "SELECT related_term, score FROM RelatedTermsFuzzy WHERE term = ?"
        params: list[typing.Any] = [term]

        if min_score is not None:
            query += " AND score >= ?"
            params.append(min_score)

        query += " ORDER BY score DESC"

        if max_matches is not None:
            query += " LIMIT ?"
            params.append(max_matches)

        cursor.execute(query, params)

        results = [
            interfaces.Term(related_term, score)
            for related_term, score in cursor.fetchall()
        ]
        return results if results else None

    async def lookup_terms(
        self,
        texts: list[str],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[list[interfaces.Term]]:
        """Look up multiple terms at once."""
        results = []
        for text in texts:
            term_results = await self.lookup_term(text, max_hits, min_score)
            results.append(term_results)
        return results

    async def deserialize(self, data: interfaces.TextEmbeddingIndexData) -> None:
        """Deserialize fuzzy index data from JSON into SQLite database."""
        # Clear existing data
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM RelatedTermsFuzzy")

        # Clear local mappings
        self._terms_list.clear()
        self._terms_to_ordinal.clear()

        # Get text items and embeddings from the data
        text_items = data.get("textItems")
        embeddings_data = data.get("embeddings")

        if not text_items or embeddings_data is None:
            return

        # Use persistent VectorBase to deserialize embeddings (preserves caching)
        self._vector_base.deserialize(embeddings_data)

        # Prepare all insertion data for bulk operation
        from .schema import serialize_embedding

        insertion_data = []
        for i, text in enumerate(text_items):
            if i < len(self._vector_base):
                # Get embedding from persistent VectorBase
                embedding = self._vector_base.get_embedding_at(i)
                if embedding is not None:
                    serialized_embedding = serialize_embedding(embedding)
                    # Insert as self-referential entry with only term_embedding
                    insertion_data.append((text, text, 1.0, serialized_embedding))
                    # Update local mappings
                    self._terms_list.append(text)
                    self._terms_to_ordinal[text] = len(self._terms_to_ordinal)

        # Bulk insert all the data
        if insertion_data:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO RelatedTermsFuzzy
                (term, related_term, score, term_embedding)
                VALUES (?, ?, ?, ?)
                """,
                insertion_data,
            )


class SqliteRelatedTermsIndex(interfaces.ITermToRelatedTermsIndex):
    """SQLite-backed implementation of ITermToRelatedTermsIndex combining aliases and fuzzy index."""

    def __init__(self, db: sqlite3.Connection, embedding_model: AsyncEmbeddingModel):
        self.db = db
        # Initialize alias and fuzzy related terms indexes
        self._aliases = SqliteRelatedTermsAliases(db)
        # Pass embedding_model to fuzzy index for persistent embeddings
        self._fuzzy_index = SqliteRelatedTermsFuzzy(db, embedding_model)

    @property
    def aliases(self) -> interfaces.ITermToRelatedTerms:
        return self._aliases

    @property
    def fuzzy_index(self) -> interfaces.ITermToRelatedTermsFuzzy | None:
        return self._fuzzy_index

    async def serialize(self) -> interfaces.TermsToRelatedTermsIndexData:
        raise NotImplementedError("TODO")

    async def deserialize(self, data: interfaces.TermsToRelatedTermsIndexData) -> None:
        """Deserialize related terms index data."""
        # Deserialize alias data
        alias_data = data.get("aliasData")
        if alias_data is not None:
            await self._aliases.deserialize(alias_data)

        # Deserialize fuzzy index data
        text_embedding_data = data.get("textEmbeddingData")
        if text_embedding_data is not None:
            await self._fuzzy_index.deserialize(text_embedding_data)
