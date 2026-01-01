# SQLite Storage Provider Specification

## Overview

This specification defines the updated storage provider interface and SQLite schema to support storing most indexes and conversations directly in SQLite, moving away from the current in-memory approach. This design follows the schema outlined in the TODO.md file and aims to provide better persistence, scalability, and query performance.

## Key Design Principles

1. **Schema Alignment**: Database tables correspond closely to existing in-memory data structures, with composite objects decomposed into separate columns and optional fields normalized to explicit values with sensible defaults
2. **In-Memory Compatibility**: Continue to support the existing pure in-memory storage option alongside SQLite storage
3. **Async-First**: All operations are async to support both SQLite and future distributed storage

## Database Schema

### ConversationMetadata Table
```sql
CREATE TABLE ConversationMetadata (
    name_tag TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    created_at TEXT NULL,         -- ISO format with Z timezone
    updated_at TEXT NULL,         -- ISO format with Z timezone
    tags JSON NULL,               -- JSON array of conversation tags
    extra JSON NULL               -- Other per-conversation metadata
);
```

#### Conversation Metadata Behavior
- **Single Row Storage**: Contains exactly one row with all conversation metadata
- **Schema Version**: Enforced to match `CONVERSATION_SCHEMA_VERSION` constant (default: "0.1")
  - Validated at provider creation time - raises error if existing DB has incompatible version
- **Non-null Constraints**: All fields are never None/null:
  - `name_tag`: Can be empty string but never None
  - `tags`: Always a list (empty if no tags), never None
  - `extra`: Always a dict (empty if no extra data), never None
  - `created_at`, `updated_at`: Always datetime objects, default to current time UTC
- **Smart Timestamps**:
  - `updated_at`: Always set to current time when metadata is modified (unless explicitly overridden)
  - `created_at`: Preserved from existing metadata, or set to current time for new conversations
- **Timezone Handling**: All datetime inputs converted to UTC for storage, returned as UTC datetime objects
- **Auto-initialization**: If no metadata row exists, `get_conversation_metadata()` creates one with defaults
- **Selective Updates**: Only specified fields are updated, others retain existing values or baseline defaults### Messages Table
```sql
CREATE TABLE Messages (
    msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Messages can store chunks directly in JSON or reference external storage via URI
    chunks JSON NULL,             -- JSON array of text chunks, or NULL if using chunk_uri
    chunk_uri TEXT NULL,          -- URI for external chunk storage, or NULL if using chunks
    start_timestamp TEXT NULL,    -- ISO format with Z timezone
    end_timestamp TEXT NULL,      -- ISO format with Z timezone
    tags JSON NULL,               -- JSON array of tags
    metadata JSON NULL,           -- Message metadata (source, dest, etc.)
    extra JSON NULL,              -- Extra message fields that were serialized

    CONSTRAINT chunks_xor_chunkuri CHECK (
        (chunks IS NOT NULL AND chunk_uri IS NULL) OR
        (chunks IS NULL AND chunk_uri IS NOT NULL)
    )
);

CREATE INDEX idx_messages_start_timestamp ON Messages(start_timestamp);
-- CREATE INDEX idx_messages_end_timestamp ON Messages(end_timestamp);
```

### SemanticRefs Table
```sql
CREATE TABLE SemanticRefs (
    semref_id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- TextRange decomposed into separate columns for efficient querying
    -- Forms a half-open interval [start, end)
    -- If in-memory TextRange has no end, defaults to: end_msg_id = start_msg_id, end_chunk_ord = start_chunk_ord + 1
    start_msg_id INTEGER NOT NULL,
    start_chunk_ord INTEGER NOT NULL,
    end_msg_id INTEGER NOT NULL,
    end_chunk_ord INTEGER NOT NULL,  -- Points past the last included chunk
    ktype TEXT NOT NULL CHECK (ktype IN ('entity', 'action', 'topic', 'tag')),
    knowledge JSON NOT NULL,

    FOREIGN KEY (start_msg_id) REFERENCES Messages(msg_id) ON DELETE RESTRICT,
    FOREIGN KEY (end_msg_id) REFERENCES Messages(msg_id) ON DELETE RESTRICT
);

CREATE INDEX idx_semantic_refs_start_msg ON SemanticRefs(start_msg_id);
CREATE INDEX idx_semantic_refs_end_msg ON SemanticRefs(end_msg_id);
CREATE INDEX idx_semantic_refs_ktype ON SemanticRefs(ktype);
```

### SemanticRefIndex Table
```sql
CREATE TABLE SemanticRefIndex (
    term TEXT NOT NULL,             -- lowercased, not-unique/normalized
    semref_id INTEGER NOT NULL,

    PRIMARY KEY (term, semref_id),
    FOREIGN KEY (semref_id) REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
);

CREATE INDEX idx_semantic_ref_index_term ON SemanticRefIndex(term);
```

### PropertyIndex Table
```sql
CREATE TABLE PropertyIndex (
    prop_name TEXT NOT NULL,
    value_str TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 1.0,
    semref_id INTEGER NOT NULL,

    FOREIGN KEY (semref_id) REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
);

CREATE INDEX idx_property_index_prop_name ON PropertyIndex(prop_name);
CREATE INDEX idx_property_index_value_str ON PropertyIndex(value_str);
CREATE INDEX idx_property_index_combined ON PropertyIndex(prop_name, value_str);
```

### RelatedTermsAliases Table
```sql
CREATE TABLE RelatedTermsAliases (
    term TEXT NOT NULL,
    alias TEXT NOT NULL,

    PRIMARY KEY (term, alias)
);

CREATE INDEX idx_related_aliases_term ON RelatedTermsAliases(term);
CREATE INDEX idx_related_aliases_alias ON RelatedTermsAliases(alias);
```

### RelatedTermsFuzzy Table
```sql
CREATE TABLE RelatedTermsFuzzy (
    term TEXT NOT NULL,
    related_term TEXT NOT NULL,
    score REAL NOT NULL DEFAULT 1.0,

    PRIMARY KEY (term, related_term)
);

CREATE INDEX idx_related_fuzzy_term ON RelatedTermsFuzzy(term);
CREATE INDEX idx_related_fuzzy_related ON RelatedTermsFuzzy(related_term);
CREATE INDEX idx_related_fuzzy_score ON RelatedTermsFuzzy(score);
```

### ConversationThreads Table
```sql
CREATE TABLE ConversationThreads (
    thread_id INTEGER PRIMARY KEY AUTOINCREMENT,
    description TEXT NOT NULL,
    ranges JSON NOT NULL,            -- JSON array of TextRange objects
    embedding BLOB NULL              -- Optional embedding vector for fuzzy search
);

CREATE INDEX idx_threads_description ON ConversationThreads(description);
-- Note: Embedding searches would use vector similarity, not standard SQL indexes
```

### MessageTextIndex Table
```sql
CREATE TABLE MessageTextIndex (
    msg_id INTEGER NOT NULL,
    chunk_ordinal INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding BLOB NULL,             -- Optional embedding vector for fuzzy search

    FOREIGN KEY (msg_id) REFERENCES Messages(msg_id) ON DELETE CASCADE,
    PRIMARY KEY (msg_id, chunk_ordinal)
);

CREATE INDEX idx_message_text_chunk_text ON MessageTextIndex(chunk_text);
-- Note: Embedding searches would use vector similarity, not standard SQL indexes
```

## Storage Provider Interface Changes

### Key Design Changes

1. **Hybrid Chunk Storage**: Messages can store chunks inline (JSON) or reference external storage (URI)
2. **Persistent Indexes**: All secondary indexes stored in database tables
3. **Async-First Design**: All operations async to support database I/O
4. **Lazy Loading**: Message chunks loaded on demand when using external storage
5. **Index Separation**: Each index type has its own table and interface
6. **Timestamp Queries**: No separate timestamp index table - queries done directly on Messages table using `start_timestamp` and `end_timestamp` columns with existing indexes
7. **Related Terms Split**: Related terms functionality split into two tables - `RelatedTermsAliases` for exact aliases and `RelatedTermsFuzzy` for conversation-derived fuzzy matches

### Message Behavior Changes

#### Chunk Loading Strategy
- **Inline Storage**: `chunks` field contains JSON array, `chunk_uri` is NULL
- **External Storage**: `chunk_uri` contains URI, `chunks` is NULL
- **Lazy Loading**: `get_chunks()` method loads from URI if needed and caches

#### In-Memory Message Implementation
```python
class Message:
    def __init__(self, chunks: list[str] | None = None, chunk_uri: str | None = None, ...):
        self._chunks = chunks
        self._chunk_uri = chunk_uri
        # Exactly one of chunks, chunk_uri must be not-NULL

    async def get_chunks(self) -> list[str]:
        if self._chunks is not None:
            return self._chunks
        assert self._chunk_uri is not None
        # Load chunks using chunk_uri (extractor-specific implementation)
        self._chunks = await self._load_chunks_from_uri(self._chunk_uri)
        return self._chunks
```

### Index Interface Changes

#### Term to SemanticRef Index
```python
class ITermToSemanticRefIndex(Protocol):
    async def add_term_mapping(self, term: str, semref_id: int) -> None: ...
    async def get_semantic_refs_for_term(self, term: str) -> list[int]: ...
    async def remove_term_mapping(self, term: str, semref_id: int) -> None: ...
    async def remove_all_for_semref(self, semref_id: int) -> None: ...
```

#### Property to SemanticRef Index
```python
class IPropertyToSemanticRefIndex(Protocol):
    async def add_property(self, prop_name: str, value_str: str,
                          score: float, semref_id: int) -> None: ...
    async def get_semantic_refs_for_property(self, prop_name: str,
                                           value_str: str | None = None) -> list[tuple[int, float]]: ...
    async def remove_property(self, prop_name: str, semref_id: int) -> None: ...
    async def remove_all_for_semref(self, semref_id: int) -> None: ...
```

### Storage Provider Interface Updates

#### Core Interface
```python
class IStorageProvider(Protocol):
    # Collection creation (existing)
    def create_message_collection[TMessage: IMessage](
        self, message_type: type[TMessage]
    ) -> IMessageCollection[TMessage]: ...

    def create_semantic_ref_collection(self) -> ISemanticRefCollection: ...

    # Index creation (new) - 6 index types (timestamp queries done directly on Messages table)
    def create_term_to_semantic_ref_index(self) -> ITermToSemanticRefIndex: ...
    def create_property_to_semantic_ref_index(self) -> IPropertyToSemanticRefIndex: ...
    def create_message_text_index(self) -> IMessageTextIndex: ...
    def create_related_terms_index(self) -> ITermToRelatedTermsIndex: ...
    def create_conversation_threads(self) -> IConversationThreads: ...
    # Note: EmbeddingIndex is used internally by RelatedTermsIndex and MessageTextIndex
    # Note: Timestamp queries handled directly on Messages table using start_timestamp/end_timestamp columns

    # Conversation metadata (new) - single row storage with smart defaults
    async def get_conversation_metadata(self) -> ConversationMetadata: ...
    async def set_conversation_metadata(
        self,
        *,
        name_tag: str | None = None,
        schema_version: str | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None: ...

    # Resource management
    async def close(self) -> None: ...
```

#### Conversation Metadata Usage Examples
```python
# Get metadata (always returns a valid object, creates defaults if needed)
metadata = await provider.get_conversation_metadata()
print(f"Name: {metadata.name_tag}")  # Never None, could be empty string
print(f"Tags: {metadata.tags}")      # Never None, could be empty list []
print(f"Extra: {metadata.extra}")    # Never None, could be empty dict {}

# Create new conversation with defaults (timestamps auto-set to now UTC)
await provider.set_conversation_metadata(name_tag="my_conversation")

# Update just the tags, timestamp gets updated automatically
await provider.set_conversation_metadata(tags=["important", "work"])

# Update timestamp only (equivalent to refresh/touch)
await provider.set_conversation_metadata()

# Override everything explicitly with timezone handling
from datetime import datetime, timezone
await provider.set_conversation_metadata(
    name_tag="conversation",
    created_at=datetime(2025, 1, 1, 12, 0, 0),      # Assumes local TZ, converted to UTC
    updated_at=datetime.now(timezone.utc),          # Explicit UTC
    tags=["tag1", "tag2"],                          # Never None
    extra={"custom_field": "value"}                 # Never None
)

# Schema version validation (raises ValueError if mismatch)
try:
    await provider.set_conversation_metadata(schema_version="1.0")  # Would raise error
except ValueError as e:
    print(f"Schema mismatch: {e}")

# Baseline behavior - use None to keep existing values
await provider.set_conversation_metadata(
    name_tag="new_name",     # Update name
    tags=None,               # Keep existing tags from baseline
    extra=None,              # Keep existing extra from baseline
    # created_at not specified -> keeps existing, updated_at -> current time
)
```

## Related Documents

- [Immediate Implementation Steps](storage_immediate_implementation.md) - Focused steps for index centralization before SQLite implementation
- [Implementation Plan](storage_implementation_plan.md) - Detailed implementation strategy and timeline
- [Future Extensions](storage_future_extensions.md) - Planned enhancements and research directions
