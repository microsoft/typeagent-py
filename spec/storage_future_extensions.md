# SQLite Storage Future Extensions

## Overview

This document outlines potential future enhancements to the SQLite storage system. These extensions are not part of the initial implementation but represent planned improvements and research directions.

## Enhanced Property Types

### Motivation
Currently, all property values are stored as strings. Future versions could support typed values for better query performance and type safety.

### Schema Extensions
```sql
-- Future columns for PropertyIndex table
ALTER TABLE PropertyIndex ADD COLUMN value_number REAL;        -- For numeric values
ALTER TABLE PropertyIndex ADD COLUMN value_bool INTEGER;       -- For boolean values (0/1)  
ALTER TABLE PropertyIndex ADD COLUMN value_quantity_amount REAL; -- For quantity amounts
ALTER TABLE PropertyIndex ADD COLUMN value_quantity_unit TEXT;   -- For quantity units
ALTER TABLE PropertyIndex ADD COLUMN value_datetime TEXT;        -- For ISO datetime values
ALTER TABLE PropertyIndex ADD COLUMN value_type TEXT;            -- Type indicator

-- Indexes for typed values
CREATE INDEX idx_property_index_value_number ON PropertyIndex(value_number);
CREATE INDEX idx_property_index_value_bool ON PropertyIndex(value_bool);
CREATE INDEX idx_property_index_value_datetime ON PropertyIndex(value_datetime);
CREATE INDEX idx_property_index_type_number ON PropertyIndex(prop_name, value_number);
```

### Interface Updates
```python
class IPropertyToSemanticRefIndex(Protocol):
    # Enhanced methods for typed values
    async def add_property_typed(
        self, 
        prop_name: str, 
        value: str | int | float | bool | datetime | None,
        score: float, 
        semref_id: int
    ) -> None: ...
    
    async def get_semantic_refs_for_numeric_range(
        self, 
        prop_name: str, 
        min_value: float | None = None,
        max_value: float | None = None
    ) -> list[tuple[int, float]]: ...
    
    async def get_semantic_refs_for_datetime_range(
        self, 
        prop_name: str, 
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> list[tuple[int, float]]: ...
```

## Term Normalization and Optimization

### Motivation
Currently, terms are stored directly in the SemanticRefIndex table. For better storage efficiency and advanced search features, terms could be normalized and stored in a separate table.

### Schema Design
```sql
-- Separate Terms table for normalization
CREATE TABLE Terms (
    term_id INTEGER PRIMARY KEY AUTOINCREMENT,
    term_text TEXT UNIQUE NOT NULL,        -- Original term
    normalized_text TEXT NOT NULL,         -- Lowercased, normalized version
    stem_text TEXT,                       -- Stemmed version for fuzzy matching
    language TEXT DEFAULT 'en',           -- Language for proper stemming
    frequency INTEGER DEFAULT 1           -- Usage frequency for optimization
);

CREATE INDEX idx_terms_normalized ON Terms(normalized_text);
CREATE INDEX idx_terms_stem ON Terms(stem_text);
CREATE INDEX idx_terms_frequency ON Terms(frequency DESC);

-- Updated SemanticRefIndex references term_id
CREATE TABLE SemanticRefIndex_v2 (
    term_id INTEGER NOT NULL,
    semref_id INTEGER NOT NULL,
    relevance_score REAL DEFAULT 1.0,     -- Term relevance to semantic ref
    
    PRIMARY KEY (term_id, semref_id),
    FOREIGN KEY (term_id) REFERENCES Terms(term_id) ON DELETE CASCADE,
    FOREIGN KEY (semref_id) REFERENCES SemanticRefs(semref_id) ON DELETE CASCADE
);

CREATE INDEX idx_semantic_ref_index_v2_term ON SemanticRefIndex_v2(term_id);
CREATE INDEX idx_semantic_ref_index_v2_relevance ON SemanticRefIndex_v2(relevance_score DESC);
```

### Advanced Search Features
```python
class IAdvancedTermIndex(Protocol):
    async def add_term_with_variants(
        self, 
        term: str, 
        semref_id: int,
        relevance_score: float = 1.0
    ) -> None: ...
    
    async def search_fuzzy(
        self, 
        query: str, 
        max_distance: int = 2
    ) -> list[tuple[int, float]]: ...  # (semref_id, relevance_score)
    
    async def search_semantic_similar(
        self, 
        term: str, 
        threshold: float = 0.8
    ) -> list[tuple[int, float]]: ...
    
    async def get_term_suggestions(
        self, 
        partial_term: str, 
        limit: int = 10
    ) -> list[str]: ...
```

## Message Versioning and History

### Motivation
Support for message editing, versioning, and history tracking without losing semantic references.

### Schema Design
```sql
-- Message versions table
CREATE TABLE MessageVersions (
    version_id INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_id INTEGER NOT NULL,              -- References Messages.msg_id
    version_number INTEGER NOT NULL,      -- 1, 2, 3, etc.
    chunks JSON NULL,
    chunk_uri TEXT NULL,
    created_at TEXT NOT NULL,
    editor_info JSON,                     -- Who/what made the edit
    change_reason TEXT,                   -- Why was it changed
    
    FOREIGN KEY (msg_id) REFERENCES Messages(msg_id) ON DELETE CASCADE,
    CONSTRAINT chunks_xor_chunkuri_v CHECK (
        (chunks IS NOT NULL AND chunk_uri IS NULL) OR 
        (chunks IS NULL AND chunk_uri IS NOT NULL)
    )
);

CREATE INDEX idx_message_versions_msg_id ON MessageVersions(msg_id);
CREATE INDEX idx_message_versions_created ON MessageVersions(created_at);

-- Update Messages table to reference current version
ALTER TABLE Messages ADD COLUMN current_version_id INTEGER;
ALTER TABLE Messages ADD FOREIGN KEY (current_version_id) REFERENCES MessageVersions(version_id);
```

## Distributed Storage Support

### Motivation
Support for distributed storage backends and replication for scalability.

### Interface Design
```python
class IDistributedStorageProvider(IStorageProvider):
    async def set_replication_factor(self, factor: int) -> None: ...
    async def add_storage_node(self, node_id: str, connection_info: dict) -> None: ...
    async def remove_storage_node(self, node_id: str) -> None: ...
    async def rebalance_data(self) -> None: ...
    async def get_node_health(self) -> dict[str, dict]: ...
```

### Sharding Strategy
```sql
-- Shard mapping table
CREATE TABLE ShardMapping (
    shard_id INTEGER PRIMARY KEY,
    start_hash TEXT NOT NULL,
    end_hash TEXT NOT NULL,
    node_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Conversations could be sharded by name_tag hash
-- Messages could be sharded by msg_id range or timestamp
```

## Full-Text Search Integration

### Motivation
Integration with SQLite FTS5 for advanced text search capabilities.

### Schema Design
```sql
-- FTS5 virtual table for message content
CREATE VIRTUAL TABLE MessageContentFTS USING fts5(
    msg_id UNINDEXED,
    content,
    tags,
    content=Messages,
    content_rowid=msg_id
);

-- Triggers to keep FTS table in sync
CREATE TRIGGER messages_fts_insert AFTER INSERT ON Messages BEGIN
    INSERT INTO MessageContentFTS(msg_id, content, tags) 
    VALUES (new.msg_id, json_extract(new.chunks, '$'), json_extract(new.tags, '$'));
END;

CREATE TRIGGER messages_fts_update AFTER UPDATE ON Messages BEGIN
    UPDATE MessageContentFTS 
    SET content = json_extract(new.chunks, '$'), tags = json_extract(new.tags, '$')
    WHERE msg_id = new.msg_id;
END;

CREATE TRIGGER messages_fts_delete AFTER DELETE ON Messages BEGIN
    DELETE FROM MessageContentFTS WHERE msg_id = old.msg_id;
END;
```

### Search Interface
```python
class IFullTextSearch(Protocol):
    async def search_messages(
        self, 
        query: str,
        limit: int = 100,
        highlight: bool = True
    ) -> list[tuple[int, str, float]]: ...  # (msg_id, snippet, rank)
    
    async def search_with_filters(
        self, 
        query: str,
        timestamp_range: tuple[str, str] | None = None,
        tags: list[str] | None = None,
        limit: int = 100
    ) -> list[tuple[int, str, float]]: ...
```

## Analytics and Metrics

### Motivation
Built-in analytics for understanding conversation patterns and system performance.

### Schema Design
```sql
-- Usage metrics table
CREATE TABLE UsageMetrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    timestamp TEXT NOT NULL,
    metadata JSON
);

CREATE INDEX idx_usage_metrics_name_time ON UsageMetrics(metric_name, timestamp);

-- Query performance tracking
CREATE TABLE QueryPerformance (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_type TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    result_count INTEGER,
    timestamp TEXT NOT NULL,
    query_params JSON
);
```

### Analytics Interface
```python
class IStorageAnalytics(Protocol):
    async def record_query_performance(
        self, 
        query_type: str, 
        duration_ms: int, 
        result_count: int
    ) -> None: ...
    
    async def get_usage_stats(
        self, 
        start_time: str, 
        end_time: str
    ) -> dict[str, Any]: ...
    
    async def get_performance_metrics(
        self, 
        query_type: str | None = None
    ) -> dict[str, Any]: ...
```

## Implementation Priority

### Phase 1 (Short Term)
1. Enhanced Property Types - Most immediately useful
2. Basic Term Normalization - Improves search quality

### Phase 2 (Medium Term)  
3. Message Versioning - Important for editing workflows
4. Full-Text Search - Major search improvement

### Phase 3 (Long Term)
5. Analytics and Metrics - Useful for optimization
6. Distributed Storage - Only needed at scale

### Research Phase
7. Advanced semantic search with embeddings
8. Machine learning-based query optimization
9. Integration with external search engines (Elasticsearch, etc.)

## Migration Strategy

Each extension should be designed as an optional enhancement that doesn't break existing functionality:

1. **Additive Schema Changes**: Use `ALTER TABLE` and new tables
2. **Interface Extensions**: Add new methods without changing existing ones  
3. **Feature Flags**: Allow enabling/disabling extensions
4. **Backward Compatibility**: Maintain existing behavior by default
5. **Gradual Migration**: Provide tools to migrate data incrementally
