# Manage Conversations

Create and manage conversations in the TypeAgent knowledge base.

## Usage

This skill allows you to:
- Create new conversations
- Add messages to conversations
- Query conversation content
- Manage conversation threads
- Switch between storage backends (memory/SQLite)

## How to use

When the user wants to manage conversations:

### Create a Conversation

```python
from typeagent.knowpro import create_conversation
from typeagent.knowpro.settings import ConversationSettings

# Create with default settings
conversation = create_conversation(
    name="my-conversation",
    backend="sqlite",  # or "memory"
    data_dir="./data"
)

# Create with custom settings
settings = ConversationSettings(
    enable_knowledge_extraction=True,
    batch_size=10,
    llm_model="gpt-4o-mini"
)

conversation = create_conversation(
    name="my-conversation",
    settings=settings,
    backend="sqlite"
)
```

### Add Messages

```python
# Add a simple message
conversation.add_message({
    "text": "Hello, this is a test message.",
    "sender": "Alice",
    "timestamp": "2024-01-01T10:00:00Z"
})

# Add message with metadata
conversation.add_message({
    "text": "This is an important message about the project.",
    "sender": "Bob",
    "recipients": ["Alice", "Charlie"],
    "timestamp": "2024-01-01T10:05:00Z",
    "thread": "project-discussion",
    "metadata": {
        "importance": "high",
        "tags": ["project", "deadline"]
    }
})
```

### Query Conversation

```python
# Query the conversation
results = conversation.query("What was discussed about the project?")

print(f"Answer: {results.answer}")
for citation in results.citations:
    print(f"Source: {citation.text}")
```

### List Messages

```python
# Get all messages
messages = conversation.get_messages()

for msg in messages:
    print(f"{msg.sender}: {msg.text}")
```

### Get Extracted Knowledge

```python
# Get all semantic references (extracted knowledge)
refs = conversation.get_semantic_refs()

for ref in refs:
    print(f"{ref.type}: {ref.value}")
    if ref.description:
        print(f"  Description: {ref.description}")
```

## Storage Backends

### Memory Backend
- Fast in-memory storage
- No persistence (lost on exit)
- Good for testing and demos
- Lower memory overhead

```python
conversation = create_conversation(
    name="temp-conv",
    backend="memory"
)
```

### SQLite Backend
- Persistent storage
- Survives restarts
- Good for production use
- Larger datasets

```python
conversation = create_conversation(
    name="prod-conv",
    backend="sqlite",
    data_dir="./data"
)
```

## Conversation Settings

Available settings in `ConversationSettings`:

```python
from typeagent.knowpro.settings import ConversationSettings

settings = ConversationSettings(
    # Knowledge extraction
    enable_knowledge_extraction=True,
    batch_size=10,
    llm_model="gpt-4o-mini",
    enable_fallback=True,

    # Indexing
    enable_semantic_index=True,
    enable_property_index=True,
    enable_timestamp_index=True,
    enable_related_terms=True,

    # Search
    max_results=10,
    similarity_threshold=0.7,

    # Embeddings
    embedding_model="text-embedding-3-small",
    embedding_dimensions=1536
)
```

## Thread Management

Organize messages by conversation threads:

```python
# Add messages to different threads
conversation.add_message({
    "text": "Let's discuss the budget.",
    "sender": "Manager",
    "thread": "budget-discussion"
})

conversation.add_message({
    "text": "The timeline looks good.",
    "sender": "PM",
    "thread": "timeline-discussion"
})

# Query specific thread
results = conversation.query(
    "What was said about the budget?",
    thread="budget-discussion"
)
```

## Implementation

The conversation system provides:
- Universal message format
- Automatic knowledge extraction
- Multi-index search (6 index types)
- Thread-based organization
- Dual backend support (memory/SQLite)
- Citation tracking
- Temporal indexing
