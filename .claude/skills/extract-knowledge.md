# Extract Knowledge

Extract structured knowledge from text using LLM-powered extraction.

## Usage

This skill allows you to:
- Extract entities (people, places, organizations) from text
- Identify topics and themes
- Extract actions and events
- Find relationships between entities
- Tag and categorize content

## How to use

When the user wants to extract knowledge from text:

1. Use the Python API:
```python
from typeagent.knowpro import create_conversation
from typeagent.knowpro.settings import ConversationSettings

# Create conversation with knowledge extraction enabled
settings = ConversationSettings(
    enable_knowledge_extraction=True,
    batch_size=10
)

conversation = create_conversation(
    name="my-conversation",
    settings=settings
)

# Add messages - knowledge is extracted automatically
conversation.add_message({
    "text": "Your text content here...",
    "sender": "Speaker",
    "timestamp": "2024-01-01T10:00:00Z"
})

# Access extracted knowledge
refs = conversation.get_semantic_refs()
for ref in refs:
    print(f"Type: {ref.type}, Value: {ref.value}")
```

## Configuration Options

In `ConversationSettings`:
- `enable_knowledge_extraction` - Enable/disable extraction (default: True)
- `batch_size` - Concurrent extraction batch size (default: 10)
- `llm_model` - LLM model to use (default: "gpt-4o-mini")
- `enable_fallback` - Use rule-based fallback if LLM fails (default: True)

## Extraction Types

The system extracts:

1. **Entities** (ConcreteEntity)
   - People, organizations, locations
   - Products, technologies, companies
   - Named entities with types and descriptions

2. **Topics** (Topic)
   - Main themes and subjects
   - Categories and classifications
   - Topic hierarchies

3. **Actions** (Action)
   - Events and activities
   - State changes and transitions
   - Who did what, when

4. **Tags**
   - Keywords and labels
   - Metadata markers
   - Classification tags

## Knowledge Schema

Extracted knowledge includes:
- `type` - Entity type (entity, topic, action, tag)
- `value` - The extracted term/phrase
- `description` - Optional description
- `text_location` - Source text location (message ID, offset, length)
- `properties` - Additional metadata (name, category, etc.)

## Implementation Details

The extraction pipeline uses:
- **LLM-based extraction** - GPT-4 or GPT-3.5 via TypeChat
- **Batch processing** - Concurrent extraction for performance
- **Fallback extraction** - Rule-based extraction if LLM fails
- **Deduplication** - Merge similar entities
- **Multi-index storage** - Store in specialized indexes for retrieval

## Advanced Usage

Configure LLM settings:
```python
from typeagent.knowpro.settings import ConversationSettings

settings = ConversationSettings(
    enable_knowledge_extraction=True,
    llm_model="gpt-4o",  # Use GPT-4 for better quality
    batch_size=20,        # Larger batches for speed
    enable_fallback=True  # Use rule-based fallback
)
```

Disable extraction for speed:
```python
settings = ConversationSettings(
    enable_knowledge_extraction=False  # Faster, no knowledge extraction
)
```
