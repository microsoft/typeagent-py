# Search Indexes

Search across TypeAgent's six specialized indexes for comprehensive knowledge retrieval.

## Usage

This skill allows you to:
- Search semantic references (entities, topics, actions)
- Search by properties (name, type, category)
- Search message text using embeddings
- Search by timestamp and time ranges
- Search related terms and aliases
- Filter by conversation threads

## How to use

When the user wants to search the indexes:

### High-Level Query API

```python
from typeagent.knowpro import create_conversation

conversation = create_conversation(name="my-conv", backend="sqlite")

# Natural language query (uses all indexes)
results = conversation.query("What was discussed about machine learning?")

print(f"Answer: {results.answer}")
for citation in results.citations:
    print(f"Source: {citation.text} (score: {citation.score})")
```

### Direct Index Access

```python
# Get the conversation's indexes
indexes = conversation.get_indexes()

# 1. Semantic Reference Index - Search by terms/keywords
refs = indexes.semantic_ref_index.search("machine learning", limit=10)

# 2. Property Index - Search by properties
refs = indexes.property_index.search_by_property(
    "type", "entity", limit=10
)

# 3. Message Text Index - Semantic similarity search
messages = indexes.message_text_index.search(
    "artificial intelligence developments",
    limit=5
)

# 4. Timestamp Index - Search by time range
from datetime import datetime
messages = indexes.timestamp_index.search_range(
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2024, 12, 31)
)

# 5. Related Terms Index - Find related terms and aliases
related = indexes.related_terms_index.get_related("AI")
# Returns: ["artificial intelligence", "machine learning", "deep learning"]

# 6. Conversation Threads - Filter by thread
messages = indexes.conversation_threads.get_messages(thread_name="podcast-1")
```

## The Six Index Types

### 1. Semantic Reference Index
Maps terms/keywords to semantic references (extracted knowledge).

**Use for:**
- Finding where specific terms are mentioned
- Locating entities, topics, actions by name
- Keyword-based search

**Example:**
```python
# Find all references to "Python"
refs = semantic_ref_index.search("Python")

for ref in refs:
    print(f"{ref.type}: {ref.value}")
    print(f"Location: message {ref.text_location.message_id}")
```

### 2. Property Index
Maps properties (name, type, category, etc.) to semantic references.

**Use for:**
- Filtering by entity type
- Finding items with specific properties
- Structured queries

**Example:**
```python
# Find all entities (people, orgs, places)
entities = property_index.search_by_property("type", "entity")

# Find all topics about technology
tech_topics = property_index.search_by_property("category", "technology")
```

### 3. Message Text Index
Embedding-based semantic similarity search on message text.

**Use for:**
- Finding similar messages
- Semantic search (meaning-based)
- Fallback when structured search fails

**Example:**
```python
# Find messages similar to query
messages = message_text_index.search(
    "How do neural networks learn?",
    limit=5
)

for msg in messages:
    print(f"Similarity: {msg.score:.3f}")
    print(f"Text: {msg.text[:100]}...")
```

### 4. Timestamp Index
Temporal indexing for time-based queries.

**Use for:**
- Finding messages in date ranges
- Temporal filtering
- Timeline queries

**Example:**
```python
from datetime import datetime, timedelta

# Last 7 days
week_ago = datetime.now() - timedelta(days=7)
recent = timestamp_index.search_range(start_time=week_ago)

# Specific date range
start = datetime(2024, 1, 1)
end = datetime(2024, 3, 31)
q1_messages = timestamp_index.search_range(start_time=start, end_time=end)
```

### 5. Related Terms Index
Alias resolution and fuzzy term matching.

**Use for:**
- Finding alternative names/spellings
- Expanding queries with synonyms
- Fuzzy matching

**Example:**
```python
# Find related terms
related = related_terms_index.get_related("ML")
# Returns: ["machine learning", "ML", "ml"]

# Add custom alias
related_terms_index.add_alias("AI", "artificial intelligence")
```

### 6. Conversation Threads
Thread-based message organization and filtering.

**Use for:**
- Grouping related messages
- Multi-conversation management
- Thread-specific queries

**Example:**
```python
# Get all threads
threads = conversation_threads.list_threads()

# Get messages from specific thread
messages = conversation_threads.get_messages("podcast-episode-1")

# Count messages per thread
for thread_name in threads:
    count = conversation_threads.count_messages(thread_name)
    print(f"{thread_name}: {count} messages")
```

## Multi-Index Search

The query system searches across all indexes in parallel:

```python
results = conversation.query("What did Alice say about the project timeline?")
```

This query:
1. Extracts terms: "Alice", "project", "timeline"
2. Searches SemanticRefIndex for these terms
3. Searches PropertyIndex for entities named "Alice"
4. Searches MessageTextIndex for semantic similarity
5. Merges and ranks results
6. Generates answer with citations

## Search Options

Customize search behavior:

```python
from typeagent.knowpro.settings import ConversationSettings

settings = ConversationSettings(
    max_results=20,              # More results per index
    similarity_threshold=0.6,     # Lower threshold (more permissive)
    enable_fallback=True,         # Use text similarity fallback
    use_fuzzy_matching=True       # Enable fuzzy term matching
)

conversation = create_conversation(settings=settings)
```

## Implementation Details

- **Parallel Search** - All indexes searched concurrently
- **Result Fusion** - Results merged using reciprocal rank fusion
- **Score Normalization** - Scores normalized across indexes
- **Deduplication** - Duplicate results removed
- **Citation Tracking** - Source text locations preserved
- **Fallback Strategy** - Text similarity used if structured search fails
