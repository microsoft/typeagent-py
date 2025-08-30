# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sqlite3
import typing
from dataclasses import dataclass
from datetime import datetime, timezone

from ..knowpro.convthreads import ConversationThreads
from ..knowpro import interfaces
from ..knowpro.messageindex import MessageTextIndexSettings
from ..knowpro.propindex import PropertyIndex
from ..knowpro.reltermsindex import RelatedTermsIndex, RelatedTermIndexSettings
from ..knowpro.semrefindex import text_range_from_message_chunk
from ..knowpro.serialization import deserialize_object, serialize_object
from ..knowpro.textlocindex import ScoredTextLocation


# Constants
CONVERSATION_SCHEMA_VERSION = "0.1"

MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS Messages (
    msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Messages can store chunks directly in JSON or reference external storage via URI
    chunks JSON NULL,             -- JSON array of text chunks, or NULL if using chunk_uri
    chunk_uri TEXT NULL,          -- URI for external chunk storage, or NULL if using chunks
    start_timestamp TEXT NULL,    -- ISO format with Z timezone
    tags JSON NULL,               -- JSON array of tags
    metadata JSON NULL,           -- Message metadata (source, dest, etc.)
    extra JSON NULL,              -- Extra message fields that were serialized

    CONSTRAINT chunks_xor_chunkuri CHECK (
        (chunks IS NOT NULL AND chunk_uri IS NULL) OR
        (chunks IS NULL AND chunk_uri IS NOT NULL)
    )
);
"""

type ShreddedMessage = tuple[
    str | None, str | None, str | None, str | None, str | None, str | None
]

TIMESTAMP_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_messages_start_timestamp ON Messages(start_timestamp);
"""

# Conversation metadata table (single row)
CONVERSATION_METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS ConversationMetadata (
    name_tag TEXT NOT NULL,           -- User-defined name for this conversation
    schema_version TEXT NOT NULL,     -- Version of the metadata schema
    created_at TEXT NOT NULL,         -- UTC timestamp when conversation was created
    updated_at TEXT NOT NULL,         -- UTC timestamp when metadata was last updated
    tags JSON NOT NULL,               -- JSON array of string tags
    extra JSON NOT NULL               -- JSON object for additional metadata
);
"""

SEMANTIC_REFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS SemanticRefs (
    semref_id INTEGER PRIMARY KEY,
    range_json JSON NOT NULL,          -- JSON of the TextRange object
    knowledge_type TEXT NOT NULL,      -- Required to distinguish JSON types (entity, topic, etc.)
    knowledge_json JSON NOT NULL       -- JSON of the Knowledge object
);
"""

SEMANTIC_REF_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS SemanticRefIndex (
    term TEXT NOT NULL,             -- lowercased, not-unique/normalized
    semref_id INTEGER NOT NULL,

    FOREIGN KEY (semref_id) REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
);
"""

SEMANTIC_REF_INDEX_TERM_INDEX = """
CREATE INDEX IF NOT EXISTS idx_semantic_ref_index_term ON SemanticRefIndex(term);
"""

MESSAGE_TEXT_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS MessageTextIndex (
    msg_id INTEGER NOT NULL,
    chunk_ordinal INTEGER NOT NULL,

    PRIMARY KEY (msg_id, chunk_ordinal),
    FOREIGN KEY (msg_id) REFERENCES Messages(msg_id) ON DELETE CASCADE
);
"""

MESSAGE_TEXT_INDEX_MESSAGE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_message_text_index_message ON MessageTextIndex(msg_id, chunk_ordinal);
"""

PROPERTY_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS PropertyIndex (
    prop_name TEXT NOT NULL,
    value_str TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 1.0,
    semref_id INTEGER NOT NULL,

    FOREIGN KEY (semref_id) REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
);
"""

PROPERTY_INDEX_PROP_NAME_INDEX = """
CREATE INDEX IF NOT EXISTS idx_property_index_prop_name ON PropertyIndex(prop_name);
"""

PROPERTY_INDEX_VALUE_STR_INDEX = """
CREATE INDEX IF NOT EXISTS idx_property_index_value_str ON PropertyIndex(value_str);
"""

PROPERTY_INDEX_COMBINED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_property_index_combined ON PropertyIndex(prop_name, value_str);
"""

RELATED_TERMS_ALIASES_SCHEMA = """
CREATE TABLE IF NOT EXISTS RelatedTermsAliases (
    term TEXT NOT NULL,
    alias TEXT NOT NULL,

    PRIMARY KEY (term, alias)
);
"""

RELATED_TERMS_ALIASES_TERM_INDEX = """
CREATE INDEX IF NOT EXISTS idx_related_aliases_term ON RelatedTermsAliases(term);
"""

RELATED_TERMS_ALIASES_ALIAS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_related_aliases_alias ON RelatedTermsAliases(alias);
"""

RELATED_TERMS_FUZZY_SCHEMA = """
CREATE TABLE IF NOT EXISTS RelatedTermsFuzzy (
    term TEXT NOT NULL,
    related_term TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 1.0,

    PRIMARY KEY (term, related_term)
);
"""

RELATED_TERMS_FUZZY_TERM_INDEX = """
CREATE INDEX IF NOT EXISTS idx_related_fuzzy_term ON RelatedTermsFuzzy(term);
"""

RELATED_TERMS_FUZZY_RELATED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_related_fuzzy_related ON RelatedTermsFuzzy(related_term);
"""

RELATED_TERMS_FUZZY_SCORE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_related_fuzzy_score ON RelatedTermsFuzzy(score);
"""

type ShreddedSemanticRef = tuple[int, str, str, str]
type ShreddedMessageText = tuple[int, int, str, bytes | None]
type ShreddedPropertyIndex = tuple[str, str, float, int]
type ShreddedRelatedTermsAlias = tuple[str, str]
type ShreddedRelatedTermsFuzzy = tuple[str, str, float]


@dataclass
class ConversationMetadata:
    """Metadata for the current conversation stored in SQLite."""

    name_tag: str
    schema_version: str
    created_at: datetime
    updated_at: datetime
    tags: list[str]
    extra: dict[str, typing.Any]


def _datetime_to_utc_string(dt: datetime) -> str:
    """Convert datetime to UTC ISO string. Assumes local timezone if naive."""
    if dt.tzinfo is None:
        # Assume local timezone
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt.astimezone(timezone.utc).isoformat()


def _string_to_utc_datetime(s: str) -> datetime:
    """Convert ISO string to UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _create_default_metadata() -> ConversationMetadata:
    """Create a ConversationMetadata with all defaults."""
    now = datetime.now(timezone.utc)
    return ConversationMetadata(
        name_tag="",
        schema_version=CONVERSATION_SCHEMA_VERSION,
        created_at=now,
        updated_at=now,
        tags=[],
        extra={},
    )


class SqliteMessageCollection[TMessage: interfaces.IMessage](
    interfaces.IMessageCollection
):
    def __init__(
        self, db: sqlite3.Connection, message_type: type[TMessage] | None = None
    ):
        self.db = db
        self.message_type = message_type

    @property
    def is_persistent(self) -> bool:
        return True

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Messages")
        return cursor.fetchone()[0]

    def __aiter__(self) -> typing.AsyncIterator[TMessage]:
        return self._async_iterator()

    async def _async_iterator(self) -> typing.AsyncIterator[TMessage]:
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
            FROM Messages ORDER BY msg_id
            """
        )
        for row in cursor:
            message = self._deserialize_message_from_row(row)
            yield message
            # Potentially add await asyncio.sleep(0) here to yield control

    def _deserialize_message_from_row(self, row: ShreddedMessage) -> TMessage:
        """Rehydrate a message from database row columns."""
        (
            chunks_json,
            chunk_uri,
            start_timestamp,
            tags_json,
            metadata_json,
            extra_json,
        ) = row

        # Parse JSON fields and build a JSON object using camelCase.
        message_data = json.loads(extra_json) if extra_json else {}
        message_data["textChunks"] = json.loads(chunks_json) if chunks_json else []
        message_data["timestamp"] = start_timestamp
        message_data["tags"] = json.loads(tags_json) if tags_json else []
        message_data["metadata"] = json.loads(metadata_json) if metadata_json else {}

        # The serialization.deserialize_object will convert to snake_case Python attributes.
        if self.message_type is None:
            raise ValueError(
                "Deserialization requires message_type passed to either get_message_collection or SqliteMessageCollection"
            )
        return deserialize_object(self.message_type, message_data)

    def _serialize_message_to_row(self, message: TMessage) -> ShreddedMessage:
        """Shred a message object into database columns."""
        # Serialize the message to JSON first (this uses camelCase)
        message_data = serialize_object(message)

        # Extract shredded fields (JSON uses camelCase)
        chunks_json = json.dumps(message_data.pop("textChunks", []))
        chunk_uri = None  # For now, we're not using chunk URIs
        start_timestamp = message_data.pop("timestamp", None)
        tags_json = json.dumps(message_data.pop("tags", []))
        metadata_json = json.dumps(message_data.pop("metadata", {}))

        # What's left in message_data becomes 'extra'.
        extra_json = json.dumps(message_data) if message_data else None

        return (
            chunks_json,
            chunk_uri,
            start_timestamp,
            tags_json,
            metadata_json,
            extra_json,
        )

    async def get_item(self, arg: int) -> TMessage:
        if not isinstance(arg, int):
            raise TypeError(f"Index must be an int, not {type(arg).__name__}")
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
            FROM Messages WHERE msg_id = ?
            """,
            (arg,),
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_message_from_row(row)
        raise IndexError("Message not found")

    async def get_slice(self, start: int, stop: int) -> list[TMessage]:
        if stop <= start:
            return []
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
            FROM Messages WHERE msg_id >= ? AND msg_id < ? ORDER BY msg_id
            """,
            (start, stop),
        )
        rows = cursor.fetchall()
        return [self._deserialize_message_from_row(row) for row in rows]

    async def get_multiple(self, arg: list[int]) -> list[TMessage]:
        results = []
        for i in arg:
            results.append(await self.get_item(i))
        return results

    async def append(self, item: TMessage) -> None:
        with self.db:
            cursor = self.db.cursor()
            (
                chunks_json,
                chunk_uri,
                start_timestamp,
                tags_json,
                metadata_json,
                extra_json,
            ) = self._serialize_message_to_row(item)
            # Use the current size as the ID to maintain 0-based indexing like the old implementation
            msg_id = await self.size()
            cursor.execute(
                """
                INSERT INTO Messages (msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    chunks_json,
                    chunk_uri,
                    start_timestamp,
                    tags_json,
                    metadata_json,
                    extra_json,
                ),
            )

    async def extend(self, items: typing.Iterable[TMessage]) -> None:
        with self.db:
            cursor = self.db.cursor()
            current_size = await self.size()
            for msg_id, item in enumerate(items, current_size):
                (
                    chunks_json,
                    chunk_uri,
                    start_timestamp,
                    tags_json,
                    metadata_json,
                    extra_json,
                ) = self._serialize_message_to_row(item)
                cursor.execute(
                    """
                    INSERT INTO Messages (msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        msg_id,
                        chunks_json,
                        chunk_uri,
                        start_timestamp,
                        tags_json,
                        metadata_json,
                        extra_json,
                    ),
                )


class SqliteSemanticRefCollection(interfaces.ISemanticRefCollection):
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def _deserialize_semantic_ref_from_row(
        self, row: ShreddedSemanticRef
    ) -> interfaces.SemanticRef:
        """Deserialize a semantic ref from database row columns."""
        semref_id, range_json, knowledge_type, knowledge_json = row

        # Build semantic ref data using camelCase (JSON format)
        semantic_ref_data = interfaces.SemanticRefData(
            semanticRefOrdinal=semref_id,
            range=json.loads(range_json),
            knowledgeType=knowledge_type,  # type: ignore
            knowledge=json.loads(knowledge_json),
        )

        return interfaces.SemanticRef.deserialize(semantic_ref_data)

    def _serialize_semantic_ref_to_row(
        self, semantic_ref: interfaces.SemanticRef
    ) -> ShreddedSemanticRef:
        """Serialize a semantic ref object into database columns."""
        # Serialize the semantic ref to JSON first (this uses camelCase)
        semantic_ref_data = semantic_ref.serialize()

        # Extract shredded fields (JSON uses camelCase)
        semref_id = semantic_ref_data["semanticRefOrdinal"]
        range_json = json.dumps(semantic_ref_data["range"])
        knowledge_type = semantic_ref_data["knowledgeType"]
        knowledge_json = json.dumps(semantic_ref_data["knowledge"])

        return (semref_id, range_json, knowledge_type, knowledge_json)

    @property
    def is_persistent(self) -> bool:
        return True

    async def size(self) -> int:
        return self._size()

    def _size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM SemanticRefs")
        return cursor.fetchone()[0]

    async def __aiter__(self) -> typing.AsyncIterator[interfaces.SemanticRef]:
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT semref_id, range_json, knowledge_type, knowledge_json
            FROM SemanticRefs ORDER BY semref_id
            """
        )
        for row in cursor:
            yield self._deserialize_semantic_ref_from_row(row)

    async def get_item(self, arg: int) -> interfaces.SemanticRef:
        if not isinstance(arg, int):
            raise TypeError(f"Index must be an int, not {type(arg).__name__}")
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT semref_id, range_json, knowledge_type, knowledge_json
            FROM SemanticRefs WHERE semref_id = ?
            """,
            (arg,),
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_semantic_ref_from_row(row)
        raise IndexError("SemanticRef not found")

    async def get_slice(self, start: int, stop: int) -> list[interfaces.SemanticRef]:
        if stop <= start:
            return []
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT semref_id, range_json, knowledge_type, knowledge_json
            FROM SemanticRefs WHERE semref_id >= ? AND semref_id < ?
            ORDER BY semref_id
            """,
            (start, stop),
        )
        rows = cursor.fetchall()
        return [self._deserialize_semantic_ref_from_row(row) for row in rows]

    async def get_multiple(self, arg: list[int]) -> list[interfaces.SemanticRef]:
        # TODO: Do we really want to support this?
        # If so, we should probably try to optimize it.
        results = []
        for i in arg:
            results.append(await self.get_item(i))
        return results

    async def append(self, item: interfaces.SemanticRef) -> None:
        with self.db:
            cursor = self.db.cursor()
            semref_id, range_json, knowledge_type, knowledge_json = (
                self._serialize_semantic_ref_to_row(item)
            )
            cursor.execute(
                """
                INSERT INTO SemanticRefs (semref_id, range_json, knowledge_type, knowledge_json)
                VALUES (?, ?, ?, ?)
                """,
                (semref_id, range_json, knowledge_type, knowledge_json),
            )

    async def extend(self, items: typing.Iterable[interfaces.SemanticRef]) -> None:
        with self.db:
            cursor = self.db.cursor()
            for item in items:
                semref_id, range_json, knowledge_type, knowledge_json = (
                    self._serialize_semantic_ref_to_row(item)
                )
                cursor.execute(
                    """
                    INSERT INTO SemanticRefs (semref_id, range_json, knowledge_type, knowledge_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (semref_id, range_json, knowledge_type, knowledge_json),
                )


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
        from ..knowpro.propindex import (
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
        from ..knowpro.propindex import (
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
        if not timestamp:
            return False

        # Normalize timestamp format for consistency
        from datetime import datetime

        try:
            timestamp_datetime = datetime.fromisoformat(timestamp)
            normalized_timestamp = timestamp_datetime.isoformat()
        except ValueError:
            return False

        with self.db:
            cursor = self.db.cursor()
            cursor.execute(
                "UPDATE Messages SET start_timestamp = ? WHERE msg_id = ?",
                (normalized_timestamp, message_ordinal),
            )
        return cursor.rowcount > 0

    async def add_timestamps(
        self, message_timestamps: list[tuple[interfaces.MessageOrdinal, str]]
    ) -> None:
        self._add_timestamps(message_timestamps)

    def _add_timestamps(
        self, message_timestamps: list[tuple[interfaces.MessageOrdinal, str]]
    ) -> None:
        """Add multiple timestamps to Messages table."""
        from datetime import datetime

        with self.db:
            cursor = self.db.cursor()
            # Normalize timestamps and filter out empty/invalid ones
            updates = []
            for message_ordinal, timestamp in message_timestamps:
                if timestamp:
                    try:
                        timestamp_datetime = datetime.fromisoformat(timestamp)
                        normalized_timestamp = timestamp_datetime.isoformat()
                        updates.append((normalized_timestamp, message_ordinal))
                    except ValueError:
                        continue  # Skip invalid timestamps

            cursor.executemany(
                "UPDATE Messages SET start_timestamp = ? WHERE msg_id = ?", updates
            )

    async def lookup_range(
        self, date_range: interfaces.DateRange
    ) -> list[interfaces.TimestampedTextRange]:
        return self._lookup_range(date_range)

    def _lookup_range(
        self, date_range: interfaces.DateRange
    ) -> list[interfaces.TimestampedTextRange]:
        """Look up timestamped text ranges in the given date range."""
        start_timestamp = date_range.start.isoformat()

        cursor = self.db.cursor()

        if date_range.end is None:
            # Point query - find messages exactly at start time
            cursor.execute(
                "SELECT msg_id, start_timestamp FROM Messages WHERE start_timestamp = ? ORDER BY start_timestamp",
                (start_timestamp,),
            )
        else:
            # Range query - start <= timestamp < end (end exclusive)
            end_timestamp = date_range.end.isoformat()
            cursor.execute(
                "SELECT msg_id, start_timestamp FROM Messages "
                "WHERE start_timestamp >= ? AND start_timestamp < ? "
                "ORDER BY start_timestamp",
                (start_timestamp, end_timestamp),
            )

        results = []
        for msg_id, timestamp in cursor.fetchall():
            message_ordinal = msg_id  # msg_id is 0-based like message_ordinal
            text_range = text_range_from_message_chunk(message_ordinal)
            results.append(
                interfaces.TimestampedTextRange(timestamp=timestamp, range=text_range)
            )

        return results


class SqliteMessageTextIndex[TMessage: interfaces.IMessage](
    interfaces.IMessageTextIndex[TMessage]
):
    """SQLite-backed implementation of message text index."""

    def __init__(self, db: sqlite3.Connection, settings: MessageTextIndexSettings):
        self.db = db
        self.settings = settings
        # Import here to avoid circular dependency
        from ..knowpro.fuzzyindex import EmbeddingIndex

        self._embedding_index = EmbeddingIndex(
            settings=settings.embedding_index_settings
        )

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM MessageTextIndex")
        return cursor.fetchone()[0]

    async def is_empty(self) -> bool:
        return await self.size() == 0

    async def add_messages(
        self,
        messages: typing.Iterable[TMessage],
    ) -> None:
        """Add messages to the SQLite index."""
        # Get starting message ordinal
        start_message_ordinal = await self._get_next_message_ordinal()

        # Collect all text chunks with their locations using list comprehension
        all_chunks: list[tuple[str, interfaces.TextLocation]] = [
            (
                chunk,
                interfaces.TextLocation(message_ordinal, chunk_ordinal),
            )
            for message_ordinal, message in enumerate(messages, start_message_ordinal)
            for chunk_ordinal, chunk in enumerate(message.text_chunks)
        ]

        if not all_chunks:
            return

        # Add texts to embedding index
        texts = [chunk for chunk, _ in all_chunks]
        await self._embedding_index.add_texts(texts)

        # Store only the locations in SQLite
        location_tuples = [
            (text_location.message_ordinal, text_location.chunk_ordinal)
            for text, text_location in all_chunks
        ]
        with self.db:
            cursor = self.db.cursor()
            cursor.executemany(
                "INSERT INTO MessageTextIndex (msg_id, chunk_ordinal) VALUES (?, ?)",
                location_tuples,
            )

    async def _get_next_message_ordinal(self) -> int:
        """Get the next message ordinal from the Messages table."""
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Messages")
        return cursor.fetchone()[0]

    async def lookup_messages(
        self,
        message_text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages by text using embedding similarity."""
        max_matches = max_matches or self.settings.embedding_index_settings.max_matches
        threshold_score = (
            threshold_score or self.settings.embedding_index_settings.min_score
        )

        # Generate embedding for search text
        embedding = await self._embedding_index.get_embedding(message_text)

        # Get similar embeddings from the in-memory index
        matches = self._embedding_index.get_indexes_of_nearest(
            embedding,
            max_matches=max_matches,
            min_score=threshold_score,
        )

        if not matches:
            return []

        # Get all text locations from SQLite in the same order as embedding index
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT msg_id, chunk_ordinal FROM MessageTextIndex
            ORDER BY msg_id, chunk_ordinal
            """
        )

        # TODO: Don't look up all text locations, only those relevant to the matches
        all_text_locations = [
            interfaces.TextLocation(msg_id, chunk_ordinal)
            for msg_id, chunk_ordinal in cursor.fetchall()
        ]

        # Convert embedding matches to scored text locations
        scored_text_locations = [
            ScoredTextLocation(
                text_location=all_text_locations[match.item], score=match.score
            )
            for match in matches
            if match.item < len(all_text_locations)
        ]

        return self._to_scored_message_ordinals(scored_text_locations)

    async def lookup_messages_in_subset(
        self,
        message_text: str,
        ordinals_to_search: list[interfaces.MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Look up messages in a subset by text using embedding similarity."""
        # First get all possible matches
        all_matches = await self.lookup_messages(message_text, None, threshold_score)

        # Filter to only include the requested ordinals
        ordinals_set = set(ordinals_to_search)
        filtered_matches = [
            match for match in all_matches if match.message_ordinal in ordinals_set
        ]

        # Apply max_matches limit
        if max_matches is not None:
            filtered_matches = filtered_matches[:max_matches]

        return filtered_matches

    def _to_scored_message_ordinals(
        self, scored_locations: list[ScoredTextLocation]
    ) -> list[interfaces.ScoredMessageOrdinal]:
        """Convert scored text locations to scored message ordinals."""
        matches: dict[interfaces.MessageOrdinal, interfaces.ScoredMessageOrdinal] = {}

        for sl in scored_locations:
            value = sl.text_location.message_ordinal
            score = sl.score
            match = matches.get(value)
            if match is None:
                matches[value] = interfaces.ScoredMessageOrdinal(value, score)
            else:
                match.score = max(score, match.score)

        return [
            interfaces.ScoredMessageOrdinal(
                match.message_ordinal,
                match.score,
            )
            for match in sorted(
                matches.values(), key=lambda match: match.score, reverse=True
            )
        ]

    def serialize(self) -> interfaces.MessageTextIndexData:
        """Serialize the index data."""
        # Get all text locations from SQLite in the same order as embedding index
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT msg_id, chunk_ordinal FROM MessageTextIndex ORDER BY msg_id, chunk_ordinal"
        )

        text_locations = []
        for msg_id, chunk_ordinal in cursor.fetchall():
            text_location = interfaces.TextLocation(msg_id, chunk_ordinal)
            text_locations.append(text_location.serialize())

        # Create TextToTextLocationIndexData
        text_location_index_data = interfaces.TextToTextLocationIndexData(
            textLocations=text_locations,
            embeddings=self._embedding_index.serialize(),
        )

        return interfaces.MessageTextIndexData(
            indexData=text_location_index_data,
        )

    def deserialize(self, data: interfaces.MessageTextIndexData) -> None:
        """Deserialize message text index data into SQLite table and embedding index."""
        index_data = data.get("indexData")
        if index_data is None:
            return

        text_locations = index_data.get("textLocations", [])
        embeddings = index_data.get("embeddings")

        if not text_locations:
            return

        # Clear existing data
        with self.db:
            cursor = self.db.cursor()
            cursor.execute("DELETE FROM MessageTextIndex")

        # Insert text locations into SQLite (without duplicating text content)
        # The text content is already stored in the Messages table
        if text_locations:
            location_tuples = [
                (loc_data["messageOrdinal"], loc_data["chunkOrdinal"])
                for loc_data in text_locations
            ]
            with self.db:
                cursor = self.db.cursor()
                cursor.executemany(
                    "INSERT INTO MessageTextIndex (msg_id, chunk_ordinal) VALUES (?, ?)",
                    location_tuples,
                )

        # Restore the in-memory embedding index from serialized data
        if embeddings is not None:
            self._embedding_index.deserialize(embeddings)


class SqliteRelatedTermsAliases(interfaces.ITermToRelatedTerms):
    async def remove_term(self, text: str) -> None:
        """Remove all aliases for a term."""
        with self.db:
            self.db.execute(
                "DELETE FROM RelatedTermsAliases WHERE term = ?",
                (text,),
            )

    """SQLite-backed implementation of ITermToRelatedTerms for exact aliases."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    async def lookup_term(self, text: str) -> list[interfaces.Term] | None:
        """Look up aliases for a term."""
        cursor = self.db.cursor()
        cursor.execute("SELECT alias FROM RelatedTermsAliases WHERE term = ?", (text,))
        rows = cursor.fetchall()
        if not rows:
            return None
        return [interfaces.Term(alias) for (alias,) in rows]

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM RelatedTermsAliases")
        return cursor.fetchone()[0]

    async def is_empty(self) -> bool:
        return await self.size() == 0

    async def clear(self) -> None:
        """Clear all aliases."""
        with self.db:
            self.db.execute("DELETE FROM RelatedTermsAliases")

    async def add_alias(self, term: str, alias: str) -> None:
        """Add an alias for a term."""
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO RelatedTermsAliases (term, alias) VALUES (?, ?)",
                (term, alias),
            )

    async def add_related_term(
        self, text: str, related_terms: interfaces.Term | list[interfaces.Term]
    ) -> None:
        """Add related terms (aliases) for a term."""
        if not isinstance(related_terms, list):
            related_terms = [related_terms]
        with self.db:
            for related in related_terms:
                self.db.execute(
                    "INSERT OR IGNORE INTO RelatedTermsAliases (term, alias) VALUES (?, ?)",
                    (text, related.text),
                )

    async def remove_alias(self, term: str, alias: str) -> None:
        """Remove an alias for a term."""
        with self.db:
            self.db.execute(
                "DELETE FROM RelatedTermsAliases WHERE term = ? AND alias = ?",
                (term, alias),
            )

    async def serialize(self) -> interfaces.TermToRelatedTermsData:
        """Serialize all aliases to a data structure."""
        cursor = self.db.cursor()
        cursor.execute("SELECT term, alias FROM RelatedTermsAliases ORDER BY term")
        rows = cursor.fetchall()

        # Group aliases by term
        term_to_aliases = {}
        for term, alias in rows:
            if term not in term_to_aliases:
                term_to_aliases[term] = []
            term_to_aliases[term].append(interfaces.Term(alias))

        # Convert to data structure
        related_terms = []
        for term, aliases in term_to_aliases.items():
            related_terms.append(
                interfaces.TermsToRelatedTermsDataItem(
                    termText=term,
                    relatedTerms=[alias.serialize() for alias in aliases],
                )
            )

        return interfaces.TermToRelatedTermsData(relatedTerms=related_terms)

    async def deserialize(self, data: interfaces.TermToRelatedTermsData | None) -> None:
        """Deserialize and replace all aliases from a data structure."""
        await self.clear()
        if data:
            related_terms = data.get("relatedTerms")
            if related_terms:
                with self.db:
                    for item in related_terms:
                        term = item["termText"]
                        for related_data in item["relatedTerms"]:
                            # Create Term from text since Term doesn't have deserialize
                            alias = interfaces.Term(related_data["text"])
                            self.db.execute(
                                "INSERT OR IGNORE INTO RelatedTermsAliases (term, alias) VALUES (?, ?)",
                                (term, alias.text),
                            )


class SqliteRelatedTermsFuzzy(interfaces.ITermToRelatedTermsFuzzy):
    """SQLite-backed implementation of ITermToRelatedTermsFuzzy for conversation-derived fuzzy matches."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM RelatedTermsFuzzy")
        return cursor.fetchone()[0]

    async def add_terms(self, texts: list[str]) -> None:
        """Add terms to the fuzzy index. This would typically compute relationships."""
        # For now, we just store the terms as self-related
        # A full implementation would compute fuzzy relationships between terms
        with self.db:
            for text in texts:
                self.db.execute(
                    "INSERT OR IGNORE INTO RelatedTermsFuzzy (term, related_term, score) VALUES (?, ?, ?)",
                    (text, text, 1.0),
                )

    async def lookup_term(
        self,
        text: str,
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[interfaces.Term]:
        """Look up fuzzy related terms."""
        cursor = self.db.cursor()

        query = "SELECT related_term, score FROM RelatedTermsFuzzy WHERE term = ?"
        params: list[object] = [text]

        if min_score is not None:
            query += " AND score >= ?"
            params.append(min_score)

        query += " ORDER BY score DESC"

        if max_hits is not None:
            query += " LIMIT ?"
            params.append(max_hits)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [
            interfaces.Term(related_term, weight=score) for related_term, score in rows
        ]

    async def lookup_terms(
        self,
        texts: list[str],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[list[interfaces.Term]]:
        """Look up fuzzy related terms for multiple texts."""
        results = []
        for text in texts:
            terms = await self.lookup_term(text, max_hits, min_score)
            results.append(terms)
        return results

    async def add_related_term(
        self, term: str, related_term: str, score: float = 1.0
    ) -> None:
        """Add a fuzzy relationship between terms."""
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO RelatedTermsFuzzy (term, related_term, score) VALUES (?, ?, ?)",
                (term, related_term, score),
            )

    async def clear(self) -> None:
        """Clear all fuzzy relationships."""
        with self.db:
            self.db.execute("DELETE FROM RelatedTermsFuzzy")

    async def deserialize(self, data: interfaces.TextEmbeddingIndexData | None) -> None:
        """Deserialize fuzzy relationships from text embedding data."""
        if data is None:
            return

        # Clear existing data first
        await self.clear()

        text_items = data.get("textItems", [])
        embeddings = data.get("embeddings")

        if not text_items or embeddings is None:
            return

        # For SQLite implementation, we need to compute term relationships
        # from the embeddings and store them in the database
        # For now, we'll add the terms as self-related (placeholder implementation)
        with self.db:
            for text in text_items:
                self.db.execute(
                    "INSERT OR IGNORE INTO RelatedTermsFuzzy (term, related_term, score) VALUES (?, ?, ?)",
                    (text, text, 1.0),
                )


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


class SqliteStorageProvider[TMessage: interfaces.IMessage](
    interfaces.IStorageProvider[TMessage]
):
    """A storage provider backed by SQLite.

    NOTE: You can create only one message collection
    and one semantic ref collection per provider.
    The semantic ref index is persisted to SQLite.
    Other indexes are still stored in memory.
    """

    db_path: str
    message_type: type[TMessage]
    db: sqlite3.Connection | None

    _message_collection: SqliteMessageCollection[TMessage]
    _semantic_ref_collection: SqliteSemanticRefCollection
    _conversation_index: SqliteTermToSemanticRefIndex
    _property_index: SqlitePropertyIndex
    _timestamp_index: SqliteTimestampToTextRangeIndex
    _message_text_index: interfaces.IMessageTextIndex[TMessage]
    _related_terms_index: SqliteRelatedTermsIndex
    _conversation_threads: ConversationThreads

    def __init__(self, db_path: str, message_type: type[TMessage]):
        self.db_path = db_path
        self.message_type = message_type

    @classmethod
    async def create(
        cls,
        message_text_settings: MessageTextIndexSettings,
        related_terms_settings: RelatedTermIndexSettings,
        db_path: str,
        message_type: type[TMessage],
    ) -> "SqliteStorageProvider[TMessage]":
        """Create and initialize a SqliteStorageProvider with all indexes."""
        self = cls(db_path, message_type)

        # Initialize database connection first
        self.db = self._create_db(self.db_path)

        # Check schema version compatibility if metadata exists
        await self._validate_schema_version()

        # Initialize collections
        self._message_collection = SqliteMessageCollection(self.db, message_type)
        self._semantic_ref_collection = SqliteSemanticRefCollection(self.db)

        # Initialize all indexes
        self._conversation_index = SqliteTermToSemanticRefIndex(self.db)
        self._property_index = SqlitePropertyIndex(self.db)
        self._timestamp_index = SqliteTimestampToTextRangeIndex(self.db)
        self._message_text_index = SqliteMessageTextIndex(
            self.db, message_text_settings
        )
        self._related_terms_index = SqliteRelatedTermsIndex(self.db)
        self._conversation_threads = ConversationThreads(
            related_terms_settings.embedding_index_settings
        )

        # Populate indexes from existing data
        msg_size = await self._message_collection.size()
        semref_size = await self._semantic_ref_collection.size()
        if msg_size or semref_size:
            await self._populate_indexes_from_data()

        return self

    async def _populate_indexes_from_data(self) -> None:
        """Populate transient indexes from persisted data when reopening database."""
        # Build timestamp index from messages (timestamp index is transient)
        msg_count = await self._message_collection.size()
        for i in range(msg_count):
            message = await self._message_collection.get_item(i)
            if message.timestamp:
                await self._timestamp_index.add_timestamp(i, message.timestamp)

        # Property index, related terms index, and message text index are SQLite-backed
        # and persistent. They should be populated by the caller when needed, not here.

    async def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None

    def _create_db(self, db_path: str) -> sqlite3.Connection:
        db = sqlite3.connect(db_path)
        with db:
            db.execute(MESSAGES_SCHEMA)
            db.execute(TIMESTAMP_INDEX_SCHEMA)
            db.execute(CONVERSATION_METADATA_SCHEMA)
            db.execute(SEMANTIC_REFS_SCHEMA)
            db.execute(SEMANTIC_REF_INDEX_SCHEMA)
            db.execute(SEMANTIC_REF_INDEX_TERM_INDEX)
            db.execute(MESSAGE_TEXT_INDEX_SCHEMA)
            db.execute(MESSAGE_TEXT_INDEX_MESSAGE_INDEX)
            db.execute(PROPERTY_INDEX_SCHEMA)
            db.execute(PROPERTY_INDEX_PROP_NAME_INDEX)
            db.execute(PROPERTY_INDEX_VALUE_STR_INDEX)
            db.execute(PROPERTY_INDEX_COMBINED_INDEX)
            db.execute(RELATED_TERMS_ALIASES_SCHEMA)
            db.execute(RELATED_TERMS_ALIASES_TERM_INDEX)
            db.execute(RELATED_TERMS_ALIASES_ALIAS_INDEX)
            db.execute(RELATED_TERMS_FUZZY_SCHEMA)
            db.execute(RELATED_TERMS_FUZZY_TERM_INDEX)
            db.execute(RELATED_TERMS_FUZZY_RELATED_INDEX)
            db.execute(RELATED_TERMS_FUZZY_SCORE_INDEX)
        return db

    # Collection getter methods

    async def get_message_collection(self) -> SqliteMessageCollection[TMessage]:
        return self._message_collection

    async def get_semantic_ref_collection(self) -> SqliteSemanticRefCollection:
        return self._semantic_ref_collection

    # Index getter methods

    async def get_semantic_ref_index(self) -> SqliteTermToSemanticRefIndex:
        return self._conversation_index

    async def get_property_index(self) -> interfaces.IPropertyToSemanticRefIndex:
        return self._property_index

    async def get_timestamp_index(self) -> SqliteTimestampToTextRangeIndex:
        return self._timestamp_index

    async def get_message_text_index(self) -> interfaces.IMessageTextIndex[TMessage]:
        return self._message_text_index

    async def get_related_terms_index(self) -> interfaces.ITermToRelatedTermsIndex:
        return self._related_terms_index

    async def get_conversation_threads(self) -> interfaces.IConversationThreads:
        return self._conversation_threads

    # Conversation metadata methods

    async def get_conversation_metadata(self) -> ConversationMetadata:
        """Get conversation metadata from the single row in the table, creating defaults if needed."""
        assert self.db is not None, "Database connection is closed"
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT name_tag, schema_version, created_at, updated_at, tags, extra
            FROM ConversationMetadata
            """,
        )
        row = cursor.fetchone()
        if row is None:
            # No metadata exists, create default row directly
            default_metadata = _create_default_metadata()
            self._insert_metadata_row(default_metadata)
            return default_metadata

        name_tag, schema_version, created_at, updated_at, tags_json, extra_json = row

        # Parse JSON fields with defaults
        tags = json.loads(tags_json) if tags_json else []
        extra = json.loads(extra_json) if extra_json else {}

        # Convert string timestamps to datetime objects with defaults
        now = datetime.now(timezone.utc)
        created_at_dt = _string_to_utc_datetime(created_at) if created_at else now
        updated_at_dt = _string_to_utc_datetime(updated_at) if updated_at else now

        return ConversationMetadata(
            name_tag=name_tag or "",
            schema_version=schema_version,
            created_at=created_at_dt,
            updated_at=updated_at_dt,
            tags=tags,
            extra=extra,
        )

    def _insert_metadata_row(self, metadata: ConversationMetadata) -> None:
        """Insert a metadata row directly without calling get_conversation_metadata."""
        assert self.db is not None, "Database connection is closed"

        # Convert datetime objects to UTC strings for storage
        created_at_str = _datetime_to_utc_string(metadata.created_at)
        updated_at_str = _datetime_to_utc_string(metadata.updated_at)

        # Serialize JSON fields
        tags_json = json.dumps(metadata.tags)
        extra_json = json.dumps(metadata.extra)

        with self.db:
            cursor = self.db.cursor()
            # Clear existing row and insert new one
            cursor.execute("DELETE FROM ConversationMetadata")
            cursor.execute(
                """
                INSERT INTO ConversationMetadata
                (name_tag, schema_version, created_at, updated_at, tags, extra)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.name_tag,
                    metadata.schema_version,
                    created_at_str,
                    updated_at_str,
                    tags_json,
                    extra_json,
                ),
            )

    async def set_conversation_metadata(
        self,
        *,
        name_tag: str | None = None,
        schema_version: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        tags: list[str] | None = None,
        extra: dict[str, typing.Any] | None = None,
    ) -> None:
        """Set conversation metadata with smart defaults and validation."""
        assert self.db is not None, "Database connection is closed"

        # Schema version validation
        if schema_version is not None and schema_version != CONVERSATION_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: expected {CONVERSATION_SCHEMA_VERSION}, got {schema_version}"
            )

        # Get existing metadata as baseline (this will create defaults if none exist)
        baseline = await self.get_conversation_metadata()

        # Use baseline to fill in defaults for arguments given as None
        now = datetime.now(timezone.utc)
        final_name_tag = name_tag if name_tag is not None else baseline.name_tag
        final_schema_version = schema_version or CONVERSATION_SCHEMA_VERSION
        final_created_at = created_at if created_at is not None else baseline.created_at
        final_updated_at = updated_at if updated_at is not None else now
        final_tags = tags if tags is not None else baseline.tags
        final_extra = extra if extra is not None else baseline.extra

        # Create new metadata object
        new_metadata = ConversationMetadata(
            name_tag=final_name_tag,
            schema_version=final_schema_version,
            created_at=final_created_at,
            updated_at=final_updated_at,
            tags=final_tags,
            extra=final_extra,
        )

        # Insert the updated metadata
        self._insert_metadata_row(new_metadata)

    async def _validate_schema_version(self) -> None:
        """Validate that existing schema version matches expected version."""
        assert self.db is not None, "Database connection is closed"
        cursor = self.db.cursor()
        cursor.execute("SELECT schema_version FROM ConversationMetadata LIMIT 1")
        row = cursor.fetchone()
        if row is not None:
            existing_version = row[0]
            if existing_version != CONVERSATION_SCHEMA_VERSION:
                raise ValueError(
                    f"Database schema version mismatch: expected {CONVERSATION_SCHEMA_VERSION}, "
                    f"found {existing_version}. Please migrate the database or use a compatible version."
                )
