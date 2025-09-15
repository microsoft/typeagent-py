# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite database schema definitions."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
import typing
import numpy as np

from ...aitools.embeddings import NormalizedEmbedding

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
    embedding BLOB NOT NULL,        -- Serialized embedding (numpy array as bytes)
    index_position INTEGER,         -- Position in VectorBase index for fast lookup

    PRIMARY KEY (msg_id, chunk_ordinal),
    FOREIGN KEY (msg_id) REFERENCES Messages(msg_id) ON DELETE CASCADE
);
"""

MESSAGE_TEXT_INDEX_MESSAGE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_message_text_index_message ON MessageTextIndex(msg_id, chunk_ordinal);
"""

MESSAGE_TEXT_INDEX_POSITION_INDEX = """
CREATE INDEX IF NOT EXISTS idx_message_text_index_position ON MessageTextIndex(index_position);
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
    term TEXT NOT NULL PRIMARY KEY,
    term_embedding BLOB NOT NULL    -- Serialized embedding for the term
);
"""

RELATED_TERMS_FUZZY_TERM_INDEX = """
CREATE INDEX IF NOT EXISTS idx_related_fuzzy_term ON RelatedTermsFuzzy(term);
"""

# Type aliases for database row tuples
type ShreddedMessage = tuple[
    str | None, str | None, str | None, str | None, str | None, str | None
]
type ShreddedSemanticRef = tuple[int, str, str, str]

type ShreddedMessageText = tuple[int, int, str, bytes | None]
type ShreddedPropertyIndex = tuple[str, str, float, int]
type ShreddedRelatedTermsAlias = tuple[str, str]
type ShreddedRelatedTermsFuzzy = tuple[str, float, bytes]


@dataclass
class ConversationMetadata:
    """Metadata for the current conversation stored in SQLite."""

    name_tag: str
    schema_version: str
    created_at: datetime
    updated_at: datetime
    tags: list[str]
    extra: dict[str, typing.Any]


@typing.overload
def serialize_embedding(embedding: NormalizedEmbedding) -> bytes: ...


@typing.overload
def serialize_embedding(embedding: None) -> None: ...


def serialize_embedding(embedding: NormalizedEmbedding | None) -> bytes | None:
    """Serialize a numpy embedding array to bytes for SQLite storage."""
    if embedding is None:
        return None
    return embedding.tobytes()


@typing.overload
def deserialize_embedding(blob: bytes) -> NormalizedEmbedding: ...


@typing.overload
def deserialize_embedding(blob: None) -> None: ...


def deserialize_embedding(blob: bytes | None) -> NormalizedEmbedding | None:
    """Deserialize bytes back to numpy embedding array."""
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _create_default_metadata() -> ConversationMetadata:
    """Create default conversation metadata."""
    current_time = datetime.now(timezone.utc)
    return ConversationMetadata(
        name_tag="",
        schema_version=CONVERSATION_SCHEMA_VERSION,
        tags=[],
        extra={},
        created_at=current_time,
        updated_at=current_time,
    )


def init_db_schema(db: sqlite3.Connection) -> None:
    """Initialize the database schema with all required tables."""
    cursor = db.cursor()

    # Create all tables
    cursor.execute(CONVERSATION_METADATA_SCHEMA)
    cursor.execute(MESSAGES_SCHEMA)
    cursor.execute(SEMANTIC_REFS_SCHEMA)
    cursor.execute(SEMANTIC_REF_INDEX_SCHEMA)
    cursor.execute(MESSAGE_TEXT_INDEX_SCHEMA)
    cursor.execute(PROPERTY_INDEX_SCHEMA)
    cursor.execute(RELATED_TERMS_ALIASES_SCHEMA)
    cursor.execute(RELATED_TERMS_FUZZY_SCHEMA)
    cursor.execute(TIMESTAMP_INDEX_SCHEMA)

    # Create additional indexes
    cursor.execute(SEMANTIC_REF_INDEX_TERM_INDEX)
    cursor.execute(MESSAGE_TEXT_INDEX_MESSAGE_INDEX)
    cursor.execute(MESSAGE_TEXT_INDEX_POSITION_INDEX)
    cursor.execute(RELATED_TERMS_ALIASES_TERM_INDEX)
    cursor.execute(RELATED_TERMS_ALIASES_ALIAS_INDEX)
    cursor.execute(RELATED_TERMS_FUZZY_TERM_INDEX)


def get_db_schema_version(db: sqlite3.Connection) -> str:
    """Get the database schema version."""
    try:
        cursor = db.cursor()
        cursor.execute("SELECT schema_version FROM ConversationMetadata LIMIT 1")
        row = cursor.fetchone()
        return row[0] if row else CONVERSATION_SCHEMA_VERSION
    except sqlite3.OperationalError:
        # Table doesn't exist, return current version
        return CONVERSATION_SCHEMA_VERSION


# Schema aliases for backward compatibility
CONVERSATIONS_SCHEMA = CONVERSATION_METADATA_SCHEMA
