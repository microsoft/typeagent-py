# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based semantic reference index implementation."""

from collections.abc import Sequence
import sqlite3

from ...knowpro.common import normalize_term
from ...knowpro.interfaces import (
    ITermToSemanticRefIndex,
    ScoredSemanticRefOrdinal,
    ScoredSemanticRefOrdinalData,
    SemanticRefOrdinal,
    TermToSemanticRefIndexData,
    TermToSemanticRefIndexItemData,
)


def _split_ordinal(
    ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
) -> tuple[SemanticRefOrdinal, float]:
    if isinstance(ordinal, ScoredSemanticRefOrdinal):
        return ordinal.semantic_ref_ordinal, ordinal.score
    return ordinal, 1.0


class SqliteTermToSemanticRefIndex(ITermToSemanticRefIndex):
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
        semantic_ref_ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
    ) -> str:
        if not term:
            return term

        term = self._prepare_term(term)

        semref_id, score = _split_ordinal(semantic_ref_ordinal)

        cursor = self.db.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO SemanticRefIndex (term, semref_id, score)
            VALUES (?, ?, ?)
            """,
            (term, semref_id, score),
        )

        return term

    async def add_terms_batch(
        self,
        terms: Sequence[tuple[str, SemanticRefOrdinal | ScoredSemanticRefOrdinal]],
    ) -> None:
        if not terms:
            return
        rows = []
        for term, ordinal in terms:
            if not term:
                continue
            term = self._prepare_term(term)
            semref_id, score = _split_ordinal(ordinal)
            rows.append((term, semref_id, score))
        if rows:
            cursor = self.db.cursor()
            cursor.executemany(
                "INSERT OR IGNORE INTO SemanticRefIndex (term, semref_id, score) VALUES (?, ?, ?)",
                rows,
            )

    async def remove_term(
        self, term: str, semantic_ref_ordinal: SemanticRefOrdinal
    ) -> None:
        term = self._prepare_term(term)
        cursor = self.db.cursor()
        cursor.execute(
            "DELETE FROM SemanticRefIndex WHERE term = ? AND semref_id = ?",
            (term, semantic_ref_ordinal),
        )

    async def lookup_term(self, term: str) -> list[ScoredSemanticRefOrdinal] | None:
        term = self._prepare_term(term)
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT semref_id, score FROM SemanticRefIndex WHERE term = ? ORDER BY rowid",
            (term,),
        )
        return [
            ScoredSemanticRefOrdinal(semref_id, score)
            for semref_id, score in cursor.fetchall()
        ]

    async def clear(self) -> None:
        """Clear all terms from the semantic ref index."""
        cursor = self.db.cursor()
        cursor.execute("DELETE FROM SemanticRefIndex")

    async def serialize(self) -> TermToSemanticRefIndexData:
        """Serialize the index data for compatibility with in-memory version."""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT term, semref_id, score FROM SemanticRefIndex "
            "ORDER BY term, semref_id, rowid"
        )

        # Group by term
        term_to_semrefs: dict[str, list[ScoredSemanticRefOrdinalData]] = {}
        for term, semref_id, score in cursor.fetchall():
            if term not in term_to_semrefs:
                term_to_semrefs[term] = []
            scored_ref = ScoredSemanticRefOrdinal(semref_id, score)
            term_to_semrefs[term].append(scored_ref.serialize())

        # Convert to the expected format
        items = []
        for term, semref_ordinals in term_to_semrefs.items():
            items.append(
                TermToSemanticRefIndexItemData(
                    term=term, semanticRefOrdinals=semref_ordinals
                )
            )

        return TermToSemanticRefIndexData(items=items)

    async def deserialize(self, data: TermToSemanticRefIndexData) -> None:
        """Deserialize index data by populating the SQLite table."""
        cursor = self.db.cursor()

        # Clear existing data
        cursor.execute("DELETE FROM SemanticRefIndex")

        # Prepare all insertion data for bulk operation
        insertion_data = []
        for item in data["items"]:
            if item and item.get("term") is not None:
                term = self._prepare_term(item["term"])
                for semref_ordinal_data in item["semanticRefOrdinals"]:
                    if isinstance(semref_ordinal_data, dict):
                        semref_id = semref_ordinal_data["semanticRefOrdinal"]
                        score = semref_ordinal_data.get("score", 1.0)
                    else:
                        # Fallback for direct integer
                        semref_id = semref_ordinal_data
                        score = 1.0
                    insertion_data.append((term, semref_id, score))

        # Bulk insert all the data
        if insertion_data:
            cursor.executemany(
                "INSERT OR IGNORE INTO SemanticRefIndex (term, semref_id, score) VALUES (?, ?, ?)",
                insertion_data,
            )

    def _prepare_term(self, term: str) -> str:
        return normalize_term(term)
