# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based semantic reference index implementation."""

import re
import sqlite3
import unicodedata

from ...knowpro import interfaces
from ...knowpro.interfaces import ScoredSemanticRefOrdinal


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
            results.append(ScoredSemanticRefOrdinal(semref_id, 1.0))
        return results

    async def clear(self) -> None:
        """Clear all terms from the semantic ref index."""
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM SemanticRefIndex")

    async def serialize(self) -> interfaces.TermToSemanticRefIndexData:
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
            scored_ref = ScoredSemanticRefOrdinal(semref_id, 1.0)
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

    async def deserialize(self, data: interfaces.TermToSemanticRefIndexData) -> None:
        """Deserialize index data by populating the SQLite table."""
        cursor = self.db.cursor()

        # Clear existing data
        cursor.execute("DELETE FROM SemanticRefIndex")

        # Prepare all insertion data for bulk operation
        insertion_data = []
        for item in data["items"]:
            if item and item["term"]:
                term = self._prepare_term(item["term"])
                for semref_ordinal_data in item["semanticRefOrdinals"]:
                    if isinstance(semref_ordinal_data, dict):
                        semref_id = semref_ordinal_data["semanticRefOrdinal"]
                    else:
                        # Fallback for direct integer
                        semref_id = semref_ordinal_data
                    insertion_data.append((term, semref_id))

        # Bulk insert all the data
        if insertion_data:
            cursor.executemany(
                "INSERT OR IGNORE INTO SemanticRefIndex (term, semref_id) VALUES (?, ?)",
                insertion_data,
            )

    def _prepare_term(self, term: str) -> str:
        """Normalize term by converting to lowercase, stripping whitespace, and normalizing Unicode."""
        # Strip leading/trailing whitespace
        term = term.strip()

        # Normalize Unicode to NFC form (canonical composition)
        term = unicodedata.normalize("NFC", term)

        # Collapse multiple whitespace characters to single space
        term = re.sub(r"\s+", " ", term)

        # Convert to lowercase
        return term.lower()
