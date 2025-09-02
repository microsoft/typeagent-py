# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based index implementations."""

import sqlite3
import typing

from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings
from ...knowpro.textlocindex import ScoredTextLocation


class SqliteTermToSemanticRefIndex(interfaces.ITermToSemanticRefIndex):
    """SQLite-backed implementation of term to semantic ref index."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(DISTINCT term) FROM SemanticRefIndex")
        return cursor.fetchone()[0]

    async def get_terms(self) -> list[str]:
        cursor = self.db.cursor()
        cursor.execute("SELECT DISTINCT term FROM SemanticRefIndex ORDER BY term")
        return [row[0] for row in cursor.fetchall()]

    async def add_term(
        self,
        term: str,
        semantic_ref_ordinal: (
            interfaces.SemanticRefOrdinal | interfaces.ScoredSemanticRefOrdinal
        ),
    ) -> str:
        if not term:
            return term

        term = self._prepare_term(term)

        # Extract semref_id from the ordinal
        if isinstance(semantic_ref_ordinal, interfaces.ScoredSemanticRefOrdinal):
            semref_id = semantic_ref_ordinal.semantic_ref_ordinal
        else:
            semref_id = semantic_ref_ordinal

        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO SemanticRefIndex (term, semref_id)
                VALUES (?, ?)
                """,
                (term, semref_id),
            )

        return term

    async def remove_term(
        self, term: str, semantic_ref_ordinal: interfaces.SemanticRefOrdinal
    ) -> None:
        term = self._prepare_term(term)
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "DELETE FROM SemanticRefIndex WHERE term = ? AND semref_id = ?",
                (term, semantic_ref_ordinal),
            )

    async def lookup_term(
        self, term: str
    ) -> list[interfaces.ScoredSemanticRefOrdinal] | None:
        term = self._prepare_term(term)
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT semref_id FROM SemanticRefIndex WHERE term = ?",
            (term,),
        )

        # Return as ScoredSemanticRefOrdinal with default score of 1.0
        results = []
        for row in cursor.fetchall():
            semref_id = row[0]
            results.append(interfaces.ScoredSemanticRefOrdinal(semref_id, 1.0))
        return results

    async def clear(self) -> None:
        """Clear all terms from the semantic ref index."""
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM SemanticRefIndex")

    def serialize(self) -> interfaces.TermToSemanticRefIndexData:
        """Serialize the index data for compatibility with in-memory version."""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT term, semref_id FROM SemanticRefIndex ORDER BY term, semref_id"
        )

        # Group by term
        term_to_semrefs: dict[str, list[interfaces.ScoredSemanticRefOrdinalData]] = {}
        for term, semref_id in cursor.fetchall():
            if term not in term_to_semrefs:
                term_to_semrefs[term] = []
            scored_ref = interfaces.ScoredSemanticRefOrdinal(semref_id, 1.0)
            term_to_semrefs[term].append(scored_ref.serialize())

        # Convert to the expected format
        items = []
        for term, semref_ordinals in term_to_semrefs.items():
            items.append(
                interfaces.TermToSemanticRefIndexItemData(
                    term=term, semanticRefOrdinals=semref_ordinals
                )
            )

        return interfaces.TermToSemanticRefIndexData(items=items)

    def deserialize(self, data: interfaces.TermToSemanticRefIndexData) -> None:
        """Deserialize index data by populating the SQLite table."""
        # Clear existing data
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM SemanticRefIndex")

            # Add all the terms
            for item in data["items"]:
                if item and item["term"]:
                    term = item["term"]
                    for semref_ordinal_data in item["semanticRefOrdinals"]:
                        if isinstance(semref_ordinal_data, dict):
                            semref_id = semref_ordinal_data["semanticRefOrdinal"]
                        else:
                            # Fallback for direct integer
                            semref_id = semref_ordinal_data
                        cursor.execute(
                            "INSERT OR IGNORE INTO SemanticRefIndex (term, semref_id) VALUES (?, ?)",
                            (self._prepare_term(term), semref_id),
                        )

    def _prepare_term(self, term: str) -> str:
        """Normalize term by converting to lowercase."""
        return term.lower()


class SqlitePropertyIndex(interfaces.IPropertyToSemanticRefIndex):
    """SQLite-backed implementation of property to semantic ref index."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT prop_name, value_str FROM PropertyIndex)"
        )
        return cursor.fetchone()[0]

    async def get_values(self) -> list[str]:
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT DISTINCT value_str FROM PropertyIndex ORDER BY value_str"
        )
        return [row[0] for row in cursor.fetchall()]

    async def add_property(
        self,
        property_name: str,
        value: str,
        semantic_ref_ordinal: (
            interfaces.SemanticRefOrdinal | interfaces.ScoredSemanticRefOrdinal
        ),
    ) -> None:
        # Extract semref_id and score from the ordinal
        if isinstance(semantic_ref_ordinal, interfaces.ScoredSemanticRefOrdinal):
            semref_id = semantic_ref_ordinal.semantic_ref_ordinal
            score = semantic_ref_ordinal.score
        else:
            semref_id = semantic_ref_ordinal
            score = 1.0

        # Normalize property name and value (to match in-memory implementation)
        from ...storage.memory.propindex import (
            make_property_term_text,
            split_property_term_text,
        )

        term_text = make_property_term_text(property_name, value)
        term_text = term_text.lower()  # Matches PropertyIndex._prepare_term_text
        property_name, value = split_property_term_text(term_text)
        # Remove "prop." prefix that was added by make_property_term_text
        if property_name.startswith("prop."):
            property_name = property_name[5:]

        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                """
                INSERT INTO PropertyIndex (prop_name, value_str, score, semref_id)
                VALUES (?, ?, ?, ?)
                """,
                (property_name, value, score, semref_id),
            )

    async def clear(self) -> None:
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM PropertyIndex")

    async def lookup_property(
        self,
        property_name: str,
        value: str,
    ) -> list[interfaces.ScoredSemanticRefOrdinal] | None:
        # Normalize property name and value (to match in-memory implementation)
        from ...storage.memory.propindex import (
            make_property_term_text,
            split_property_term_text,
        )

        term_text = make_property_term_text(property_name, value)
        term_text = term_text.lower()  # Matches PropertyIndex._prepare_term_text
        property_name, value = split_property_term_text(term_text)
        # Remove "prop." prefix that was added by make_property_term_text
        if property_name.startswith("prop."):
            property_name = property_name[5:]

        cursor = self.db.cursor()
        cursor.execute(
            "SELECT semref_id, score FROM PropertyIndex WHERE prop_name = ? AND value_str = ?",
            (property_name, value),
        )

        results = [
            interfaces.ScoredSemanticRefOrdinal(semref_id, score)
            for semref_id, score in cursor.fetchall()
        ]

        return results if results else None

    async def remove_property(self, prop_name: str, semref_id: int) -> None:
        """Remove all properties for a specific property name and semantic ref."""
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "DELETE FROM PropertyIndex WHERE prop_name = ? AND semref_id = ?",
                (prop_name, semref_id),
            )

    async def remove_all_for_semref(self, semref_id: int) -> None:
        """Remove all properties for a specific semantic ref."""
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "DELETE FROM PropertyIndex WHERE semref_id = ?",
                (semref_id,),
            )


class SqliteTimestampToTextRangeIndex(interfaces.ITimestampToTextRangeIndex):
    """SQL-based timestamp index that queries Messages table directly."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    async def size(self) -> int:
        return self._size()

    def _size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM Messages WHERE start_timestamp IS NOT NULL"
        )
        return cursor.fetchone()[0]

    async def add_timestamp(
        self, message_ordinal: interfaces.MessageOrdinal, timestamp: str
    ) -> bool:
        return self._add_timestamp(message_ordinal, timestamp)

    def _add_timestamp(
        self, message_ordinal: interfaces.MessageOrdinal, timestamp: str
    ) -> bool:
        """Add timestamp to Messages table start_timestamp column."""
        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "UPDATE Messages SET start_timestamp = ? WHERE msg_id = ?",
                (timestamp, message_ordinal),
            )
            return cursor.rowcount > 0

    async def get_timestamp_ranges(
        self, start_timestamp: str, end_timestamp: str | None = None
    ) -> list[interfaces.TimestampedTextRange]:
        """Get timestamp ranges from Messages table."""
        cursor = self.db.cursor()

        if end_timestamp is None:
            # Single timestamp query
            cursor.execute(
                """
                SELECT msg_id, start_timestamp
                FROM Messages
                WHERE start_timestamp = ?
                ORDER BY msg_id
                """,
                (start_timestamp,),
            )
        else:
            # Range query
            cursor.execute(
                """
                SELECT msg_id, start_timestamp
                FROM Messages
                WHERE start_timestamp >= ? AND start_timestamp <= ?
                ORDER BY msg_id
                """,
                (start_timestamp, end_timestamp),
            )

        results = []
        for msg_id, timestamp in cursor.fetchall():
            # Create text range for message
            from ...knowpro.interfaces import TextLocation, TextRange

            text_range = TextRange(
                start=TextLocation(message_ordinal=msg_id, chunk_ordinal=0)
            )
            results.append(
                interfaces.TimestampedTextRange(range=text_range, timestamp=timestamp)
            )

        return results

    async def add_timestamps(
        self, message_timestamps: list[tuple[interfaces.MessageOrdinal, str]]
    ) -> None:
        """Add multiple timestamps."""
        with self.db:
            cursor = self.db.cursor()
            for message_ordinal, timestamp in message_timestamps:
                cursor.execute(
                    "UPDATE Messages SET start_timestamp = ? WHERE msg_id = ?",
                    (timestamp, message_ordinal),
                )

    async def lookup_range(
        self, date_range: interfaces.DateRange
    ) -> list[interfaces.TimestampedTextRange]:
        """Lookup messages in a date range."""
        cursor = self.db.cursor()

        # Convert datetime objects to ISO format strings for comparison
        start_timestamp = date_range.start.isoformat().replace("+00:00", "Z")
        end_timestamp = (
            date_range.end.isoformat().replace("+00:00", "Z")
            if date_range.end
            else None
        )

        if date_range.end is None:
            # Point query
            cursor.execute(
                """
                SELECT msg_id, start_timestamp, chunks
                FROM Messages 
                WHERE start_timestamp = ? 
                ORDER BY msg_id
                """,
                (start_timestamp,),
            )
        else:
            # Range query
            cursor.execute(
                """
                SELECT msg_id, start_timestamp, chunks
                FROM Messages 
                WHERE start_timestamp >= ? AND start_timestamp < ?
                ORDER BY msg_id
                """,
                (start_timestamp, end_timestamp),
            )

        results = []
        for msg_id, timestamp, chunks in cursor.fetchall():
            text_location = interfaces.TextLocation(
                message_ordinal=msg_id, chunk_ordinal=0
            )
            text_range = interfaces.TextRange(
                start=text_location, end=None  # Point range
            )
            results.append(
                interfaces.TimestampedTextRange(timestamp=timestamp, range=text_range)
            )

        return results


class SqliteMessageTextIndex(interfaces.IMessageTextIndex):
    """SQLite-backed message text index with embedding support."""

    def __init__(self, db: sqlite3.Connection, settings: MessageTextIndexSettings):
        self.db = db
        self.settings = settings
        # For simplicity, keep the embedding index in memory for now
        # In a full implementation, embeddings would also be stored in SQLite
        from ...knowpro.textlocindex import TextToTextLocationIndex

        self._text_to_location_index = TextToTextLocationIndex(
            settings.embedding_index_settings
        )

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM MessageTextIndex")
        return cursor.fetchone()[0]

    async def add_messages(
        self,
        messages: typing.Iterable[interfaces.IMessage],
    ) -> None:
        """Add messages to the text index."""
        for i, message in enumerate(messages):
            message_ordinal = i
            for chunk_ordinal, chunk in enumerate(message.text_chunks):
                # Add to SQLite index
                with self.db:
                    cursor = self.db.cursor()
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO MessageTextIndex (msg_id, chunk_ordinal)
                        VALUES (?, ?)
                        """,
                        (message_ordinal, chunk_ordinal),
                    )

                # Add to embedding index
                text_location = interfaces.TextLocation(
                    message_ordinal=message_ordinal, chunk_ordinal=chunk_ordinal
                )
                await self._text_to_location_index.add_text_location(
                    chunk, text_location
                )
            i += 1

    async def lookup_text(
        self, text: str, max_matches: int | None = None, min_score: float | None = None
    ) -> list[ScoredTextLocation]:
        """Look up text using embeddings."""
        return await self._text_to_location_index.lookup_text(
            text, max_matches, min_score
        )

    async def lookup_messages(
        self,
        message_text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages by text content."""
        # Use the embedding index to find text locations
        scored_locations = await self._text_to_location_index.lookup_text(
            message_text, max_matches, threshold_score
        )

        # Convert to scored message ordinals (group by message)
        message_scores: dict[int, float] = {}
        for scored_loc in scored_locations:
            msg_ord = scored_loc.text_location.message_ordinal
            if msg_ord not in message_scores:
                message_scores[msg_ord] = scored_loc.score
            else:
                # Take the max score across chunks
                message_scores[msg_ord] = max(message_scores[msg_ord], scored_loc.score)

        # Convert to list and sort by score
        results = [
            interfaces.ScoredMessageOrdinal(message_ordinal=ordinal, score=score)
            for ordinal, score in message_scores.items()
        ]
        results.sort(key=lambda x: x.score, reverse=True)

        if max_matches is not None:
            results = results[:max_matches]

        return results

    async def lookup_messages_in_subset(
        self,
        message_text: str,
        ordinals_to_search: list[interfaces.MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages in a subset of ordinals."""
        # Get all matches first
        all_matches = await self.lookup_messages(message_text, None, threshold_score)

        # Filter to only include the specified ordinals
        ordinals_set = set(ordinals_to_search)
        filtered_matches = [
            match for match in all_matches if match.message_ordinal in ordinals_set
        ]

        # Apply max_matches limit
        if max_matches is not None:
            filtered_matches = filtered_matches[:max_matches]

        return filtered_matches

    async def is_empty(self) -> bool:
        """Check if the index is empty."""
        size = await self.size()
        return size == 0

    def serialize(self) -> interfaces.MessageTextIndexData:
        """Serialize the message text index."""
        # Return empty data for now - in a full implementation this would
        # serialize the embedding vectors and other index data
        return {}

    def deserialize(self, data: interfaces.MessageTextIndexData) -> None:
        """Deserialize message text index data."""
        # For now, this is a placeholder
        # In a full implementation, this would restore the index state
        pass

    async def clear(self) -> None:
        """Clear the message text index."""
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM MessageTextIndex")
        self._text_to_location_index.clear()


# Related terms index implementations


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
        if isinstance(related_terms, str):
            related_terms = [interfaces.Term(related_terms)]
        elif isinstance(related_terms, interfaces.Term):
            related_terms = [related_terms]

        with self.db:
            cursor = self.db.cursor()
            # Add new aliases (use INSERT OR IGNORE to avoid duplicates)
            for related_term in related_terms:
                cursor.execute(
                    "INSERT OR IGNORE INTO RelatedTermsAliases (term, alias) VALUES (?, ?)",
                    (text, related_term.text),
                )

    async def remove_term(self, text: str) -> None:
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM RelatedTermsAliases WHERE term = ?", (text,))

    async def clear(self) -> None:
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM RelatedTermsAliases")

    async def get_related_terms(self, term: str) -> list[str] | None:
        cursor = self.db.cursor()
        cursor.execute("SELECT alias FROM RelatedTermsAliases WHERE term = ?", (term,))
        results = [row[0] for row in cursor.fetchall()]
        return results if results else None

    async def set_related_terms(self, term: str, related_terms: list[str]) -> None:
        with self.db:
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
        if data is None:
            return
        related_terms = data.get("relatedTerms", [])
        if related_terms:
            with self.db:
                cursor = self.db.cursor()
                cursor.execute("DELETE FROM RelatedTermsAliases")
                for item in related_terms:
                    if item and item.get("termText") and item.get("relatedTerms"):
                        term = item["termText"]
                        for term_data in item["relatedTerms"]:
                            alias = term_data["text"]
                            cursor.execute(
                                "INSERT INTO RelatedTermsAliases (term, alias) VALUES (?, ?)",
                                (term, alias),
                            )


class SqliteRelatedTermsFuzzy(interfaces.ITermToRelatedTermsFuzzy):
    """SQLite-backed implementation of fuzzy term relationships."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    async def lookup_term(
        self,
        text: str,
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[interfaces.Term]:
        cursor = self.db.cursor()
        query = "SELECT related_term, score FROM RelatedTermsFuzzy WHERE term = ?"
        params: list[str | int | float] = [text]

        if min_score is not None:
            query += " AND score >= ?"
            params.append(min_score)

        query += " ORDER BY score DESC"

        if max_hits is not None:
            query += " LIMIT ?"
            params.append(max_hits)

        cursor.execute(query, params)
        results = [
            interfaces.Term(related_term, score)
            for related_term, score in cursor.fetchall()
        ]
        return results

    async def add_related_term(
        self, term: str, related_terms: list[interfaces.Term]
    ) -> None:
        with self.db:
            cursor = self.db.cursor()
            for related_term in related_terms:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO RelatedTermsFuzzy (term, related_term, score)
                    VALUES (?, ?, ?)
                    """,
                    (term, related_term.text, 1.0),
                )

    async def remove_term(self, term: str) -> None:
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM RelatedTermsFuzzy WHERE term = ?", (term,))

    async def clear(self) -> None:
        with self.db:
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
        # For this implementation, we just add the terms to the database
        # In a real implementation, you might want to extract related terms
        with self.db:
            cursor = self.db.cursor()
            for text in texts:
                # For now, just insert the term with itself as a related term
                cursor.execute(
                    "INSERT OR IGNORE INTO RelatedTermsFuzzy (term, related_term, score) VALUES (?, ?, 1.0)",
                    (text, text),
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
        """Deserialize fuzzy index data."""
        # For now, this is a placeholder since embedding data is complex
        # In a full implementation, we would deserialize embedding vectors
        pass


class SqliteRelatedTermsIndex(interfaces.ITermToRelatedTermsIndex):
    """SQLite-backed implementation of ITermToRelatedTermsIndex combining aliases and fuzzy index."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._aliases = SqliteRelatedTermsAliases(db)
        self._fuzzy_index = SqliteRelatedTermsFuzzy(db)

    @property
    def aliases(self) -> interfaces.ITermToRelatedTerms:
        return self._aliases

    @property
    def fuzzy_index(self) -> interfaces.ITermToRelatedTermsFuzzy | None:
        return self._fuzzy_index

    async def serialize(self) -> interfaces.TermsToRelatedTermsIndexData:
        """Serialize is not needed for SQLite-backed implementation."""
        # Return empty data since persistence is handled by SQLite
        return interfaces.TermsToRelatedTermsIndexData()

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
