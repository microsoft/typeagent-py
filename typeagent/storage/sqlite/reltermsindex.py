# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based related terms index implementations."""

import sqlite3

from ...aitools.embeddings import AsyncEmbeddingModel, NormalizedEmbeddings
from ...aitools.vectorbase import TextEmbeddingIndexSettings, VectorBase
from ...knowpro import interfaces

from .schema import serialize_embedding, deserialize_embedding


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

    def __init__(self, db: sqlite3.Connection, settings: TextEmbeddingIndexSettings):
        self.db = db
        self._embedding_settings = settings
        self._vector_base = VectorBase(self._embedding_settings)
        # Maintain our own list of terms to map ordinals back to keys
        self._terms_list: list[str] = []  # TODO: Use the database instead?
        self._added_terms: set[str] = set()  # TODO: Ditto?
        # If items exist in the db, copy them into the VectorBase, terms list, and added terms
        if self._size() > 0:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT term, term_embedding FROM RelatedTermsFuzzy ORDER BY term"
            )
            rows = cursor.fetchall()
            for term, blob in rows:
                assert blob is not None, term
                embedding: NormalizedEmbeddings = deserialize_embedding(blob)
                # Add to VectorBase at the correct ordinal
                self._vector_base.add_embedding(term, embedding)
                self._terms_list.append(term)
                self._added_terms.add(term)

    async def lookup_term(
        self,
        text: str,
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[interfaces.Term]:
        """Look up similar terms using fuzzy matching."""

        # Search for similar terms using VectorBase
        similar_results = await self._vector_base.fuzzy_lookup(
            text, max_hits=max_hits, min_score=min_score
        )

        # Convert VectorBase results to Term objects
        results = []
        for scored_int in similar_results:
            # Get the term text from the list of terms  # TODO: Use the database instead?
            if scored_int.item < len(self._terms_list):
                term_text = self._terms_list[scored_int.item]
                results.append(interfaces.Term(term_text, scored_int.score))

        return results

    async def remove_term(self, term: str) -> None:
        raise NotImplementedError(
            "TODO: Removal from VectorBase, _terms_list, _terms_to_ordinal"
        )
        # cursor = self.db.cursor()
        # cursor.execute("DELETE FROM RelatedTermsFuzzy WHERE term = ?", (term,))

        # Clear VectorBase and local mappings - they will be rebuilt on next lookup
        # NO THEY WON'T
        # self._vector_base.clear()
        # self._terms_list.clear()
        # self._added_terms.clear()

    async def clear(self) -> None:
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM RelatedTermsFuzzy")

    async def size(self) -> int:
        return self._size()

    def _size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(term) FROM RelatedTermsFuzzy")
        return cursor.fetchone()[0]

    async def get_terms(self) -> list[str]:
        cursor = self.db.cursor()
        cursor.execute("SELECT term FROM RelatedTermsFuzzy ORDER BY term")
        return [row[0] for row in cursor.fetchall()]

    async def add_terms(self, texts: list[str]) -> None:
        """Add terms."""
        cursor = self.db.cursor()
        # TODO: Batch additions to database
        for text in texts:
            if text in self._added_terms:
                continue

            # Add to VectorBase for fuzzy lookup
            await self._vector_base.add_key(text)
            self._terms_list.append(text)
            self._added_terms.add(text)

            # Generate embedding for term and store in database
            embedding = await self._vector_base.get_embedding(text)  # Cached
            serialized_embedding = serialize_embedding(embedding)
            # Insert term and embedding
            cursor.execute(
                """
                INSERT OR REPLACE INTO RelatedTermsFuzzy
                (term, term_embedding)
                VALUES (?, ?)
                """,
                (text, serialized_embedding),
            )

    async def lookup_terms(
        self,
        texts: list[str],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[list[interfaces.Term]]:
        """Look up multiple terms at once."""
        # TODO: Some kind of batching?
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
        self._added_terms.clear()

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
                    insertion_data.append((text, serialized_embedding))
                    # Update local mappings
                    self._terms_list.append(text)
                    self._added_terms.add(text)

        # Bulk insert all the data
        if insertion_data:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO RelatedTermsFuzzy
                (term, term_embedding)
                VALUES (?, ?)
                """,
                insertion_data,
            )


class SqliteRelatedTermsIndex(interfaces.ITermToRelatedTermsIndex):
    """SQLite-backed implementation of ITermToRelatedTermsIndex combining aliases and fuzzy index."""

    def __init__(self, db: sqlite3.Connection, settings: TextEmbeddingIndexSettings):
        self.db = db
        # Initialize alias and fuzzy related terms indexes
        self._aliases = SqliteRelatedTermsAliases(db)
        self._fuzzy_index = SqliteRelatedTermsFuzzy(db, settings)

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
