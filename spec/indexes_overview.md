# TypeAgent Knowledge Processing Indexes

This document explains all the indexes in the `typeagent/knowpro` folder and how they work.

## How the Indexes Work

The knowpro system uses multiple indexes to find information quickly in conversation data. The main idea is **semantic references** -- structured knowledge pulled from messages that can be searched easily.

### Main Parts

1. **Primary Index**: The main index that stores knowledge from messages
2. **Secondary Indexes**: Specialized indexes for different kinds of searches
3. **Embedding Indexes**: Vector indexes that find words with similar meanings
4. **Helper Indexes**: Extra indexes for timestamps, properties, and related words

## Index Types

### 1. Semantic Reference Index (`semrefindex.py`)

**What it does**: The main index that connects words to semantic references (structured knowledge objects).

**How it works**:
- Maps text words to lists of semantic reference numbers
- Semantic references contain entities, topics, actions, and other structured knowledge
- Uses fuzzy matching with embedding-based similarity

**How it's built**:
```python
async def add_batch_to_semantic_ref_index(
    conversation: IConversation,
    knowledge_responses: list[kplib.KnowledgeResponse],
    storage_provider: IStorageProvider
) -> IndexingResults
```

**How it's used**:
- Main way to search with text
- Supports exact word matching and fuzzy/semantic matching
- Returns scored semantic reference numbers for ranking

### 2. Fuzzy Index (`fuzzyindex.py`)

**What it does**: Finds semantically similar content using embeddings.

**How it works**:
- Wraps a `VectorBase` for storing and searching normalized embeddings
- Supports nearest neighbor search with adjustable thresholds
- Can search within subsets for filtered results

**How it's built**:
```python
class EmbeddingIndex:
    async def add_texts(self, texts: list[str]) -> None:
        await self._vector_base.add_keys(texts)
```

**How it's used**:
```python
def get_indexes_of_nearest(
    self,
    embedding: NormalizedEmbedding,
    max_matches: int | None = None,
    min_score: float | None = None,
    predicate: Callable[[int], bool] | None = None,
) -> list[ScoredInt]
```

### 3. Message Index (`messageindex.py`)

**What it does**: Indexes message content for direct message-level search.

**How it works**:
- Uses `TextToTextLocationIndex` internally
- Maps text chunks to `TextLocation` objects (message number + chunk number)
- Supports embedding-based similarity search on message content

**How it's built**:
```python
async def build_message_index(
    conversation: IConversation,
    storage_provider: IStorageProvider,
) -> None:
    # Called by build_secondary_indexes; lazily obtains
    # the message index from the storage provider and
    # indexes message chunks
```

**How it's used**:
```python
async def lookup_messages(
    self,
    message_text: str,
    max_matches: int | None = None,
    threshold_score: float | None = None,
) -> list[ScoredMessageOrdinal]
```

### 4. Property Index (`propindex.py`)

**What it does**: Indexes structured properties of semantic references for property-based searches.

**How it works**:
- Maps property names and values to semantic reference numbers
- Supports entities, facets, verbs, subjects, objects, tags, and topics
- Enables searches like "find all entities of type 'Person'" or "find all topics about 'AI'"

**How it's built**:
```python
async def build_property_index(
    conversation: IConversation
) -> ListIndexingResult:
    # Goes through all semantic references
    # Gets and indexes properties like entity names, types, facet values
```

**How it's used**:
```python
async def lookup_property_in_property_index(
    property_index: IPropertyToSemanticRefIndex,
    property_name: str,
    property_value: str,
    max_matches: int | None = None,
) -> list[ScoredSemanticRefOrdinal]
```

### 5. Related Terms Index (`reltermsindex.py`)

**What it does**: Stores and manages relationships between terms for query expansion and semantic search.

**How it works**:
- Maps terms to related terms with weights/scores
- Supports fuzzy lookup of related terms using embeddings
- Helps expand searches for better results

**How it's built**:
```python
async def build_related_terms_index(
    conversation: IConversation,
    settings: RelatedTermIndexSettings,
) -> TextIndexingResult | None:
    # Looks at semantic references to find term relationships
    # Uses embedding similarity to find related terms
```

**How it's used**:
```python
async def resolve_related_terms(
    term: Term,
    related_terms_index: ITermToRelatedTermsIndex,
    max_expansion: int = 5,
) -> list[Term]
```

### 6. Text Location Index (`textlocindex.py`)

**What it does**: Maps text content to specific locations within the conversation.

**How it works**:
- Uses an `EmbeddingIndex` for fuzzy text matching
- Stores mappings from text strings to `TextLocation` objects
- Supports batch operations for fast indexing

**How it's built**:
```python
async def add_text_locations(
    self,
    text_and_locations: list[tuple[str, TextLocation]],
) -> ListIndexingResult:
    # Adds embeddings for texts and stores location mappings
```

**How it's used**:
```python
async def lookup_text(
    self,
    text: str,
    max_matches: int | None = None,
    threshold_score: float | None = None,
) -> list[ScoredTextLocation]
```

### 7. Timestamp Index (`timestampindex.py`)

**What it does**: Enables time-based searches by indexing message timestamps.

**How it works**:
- Keeps a sorted list of `TimestampedTextRange` objects
- Uses binary search for fast range searches
- Supports date range filtering

**How it's built**:
```python
async def build_timestamp_index(
    conversation: IConversation
) -> ListIndexingResult:
    # Gets timestamps from messages
    # Creates timestamped text ranges for time indexing
```

**How it's used**:
```python
def lookup_range(self, date_range: DateRange):
    # Uses binary search to find messages in time range
    return get_in_range(
        self._ranges,
        start_at,
        stop_at,
        key=lambda x: x.timestamp,
    )
```

### 8. Secondary Index Coordinator (`secindex.py`)

**What it does**: Manages and coordinates all secondary indexes as one system.

**How it works**:
- Container class that holds references to all index types
- Provides factory methods for creating complete index sets
- Manages index lifecycle and storage

**How it's built**:
```python
async def build_secondary_indexes(
    conversation: IConversation,
    conversation_settings: ConversationSettings,
) -> SecondaryIndexingResults:
    # Controls building of all secondary indexes
    # Handles dependencies between indexes
```

**How it's used**:
- Provides access to all indexes through one interface
- Used by the query system to access the right indexes for different search types

## How Queries Work

### Query Processing Steps

1. **Query Setup**: Raw search terms are turned into structured query expressions
2. **Index Selection**: The query processor picks the right indexes based on query type
3. **Parallel Lookup**: Multiple indexes may be searched at the same time for speed
4. **Result Merging**: Results from different indexes are combined and scored
5. **Post-processing**: Results are filtered, ranked, and formatted for return

### Which Index Gets Used

**Text-based Searches**:
- Primary: Semantic Reference Index
- Expansion: Related Terms Index
- Fallback: Message Index for direct text search

**Property-based Searches**:
- Primary: Property Index
- Examples: "entities named John", "topics about AI", "actions involving travel"

**Time-based Searches**:
- Primary: Timestamp Index
- Examples: "messages from last week", "conversations in January"

**Semantic Similarity Searches**:
- Primary: Fuzzy Index with embeddings
- Examples: "find similar concepts", "semantically related content"

**Location-based Searches**:
- Primary: Text Location Index
- Examples: "find specific text passages", "locate message chunks"

## Performance

### Build Time
- **Semantic Reference Index**: O(n*m) where n = messages, m = avg knowledge items per message
- **Embedding Indexes**: O(n*d) where n = items, d = embedding dimension
- **Property Index**: O(k) where k = total number of properties across all semantic references
- **Timestamp Index**: O(n log n) due to sorting

### Search Time
- **Exact Term Lookup**: O(1) average case with hash-based lookup
- **Fuzzy/Semantic Search**: O(d*n) for vector similarity, O(log n) with approximate methods
- **Range Searches**: O(log n + k) where k = results in range
- **Property Searches**: O(1) average case for property lookup

### Memory Usage
- **Embeddings**: Takes the most memory (n * d * 4 bytes for float32)
- **Term Mappings**: Based on vocabulary size and semantic reference count
- **Metadata**: Relatively small overhead for location and timestamp data

## Storage

All indexes support saving and loading through the `IStorageProvider` interface, which enables:
- Persistent storage across sessions
- Incremental updates without full rebuilds
- Distributed storage scenarios
- Backup and recovery operations

The storage system is designed to be pluggable, allowing different backends (file system, databases, cloud storage) while keeping a consistent API.

## Storage provider integration and access pattern

- Recommended access: use `conversation.secondary_indexes` everywhere. This is the stable API that orchestrates all secondary indexes.
- Under the hood, `ConversationSecondaryIndexes` lazily gets real index instances from the configured `IStorageProvider`.
- Indexes are created on first use. No explicit lifecycle calls (create/drop) are required for in-memory.
- Index-building helpers (e.g., `build_property_index`, `build_timestamp_index`, `build_message_index`) are typically invoked by `build_secondary_indexes` and operate through `conversation.secondary_indexes`.

Example usage:

```python
# Access via the coordinator (recommended)
idx = conversation.secondary_indexes
prop_idx = idx.property_to_semantic_ref_index
# … use prop_idx, timestamp_index, message_index, etc.
```

Notes:
- You usually don’t call storage provider methods directly; the coordinator handles that.
- Multi-conversation support in the in-memory provider is deferred until needed; today we use one provider per conversation.

## SQLite readiness notes

- Timestamp index: in SQLite, time-range queries will go directly against message timestamps in the database rather than an in-memory list.
- Related terms: will be split into exact alias and fuzzy/learned tables. Call sites remain the same via `conversation.secondary_indexes`.
- No changes are required in calling code when switching backends; the storage provider abstraction owns persistence details.

## Summary

The knowpro indexing system provides a complete, multi-layered approach to knowledge retrieval that supports:
- Fast exact and fuzzy text search
- Structured property-based searches
- Time filtering and search
- Semantic similarity matching
- Smart query expansion through related terms

This design enables sophisticated conversational AI applications that can quickly find relevant information across large conversation histories while keeping good performance.
