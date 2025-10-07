# Conversation Query Method

The `query()` method provides a simple, end-to-end API for querying conversations using natural language.

## Usage

```python
from typeagent import create_conversation
from typeagent.transcripts.transcript import TranscriptMessage

# Create a conversation
conv = await create_conversation(
    "my_conversation.db",
    TranscriptMessage,
    name="My Conversation",
)

# Add messages
messages: list[TranscriptMessage] = [...]
await conv.add_messages_with_indexing(messages)

# Query the conversation
question: str = input("typeagent> ")
answer: str = await conv.query(question)
print(answer)
```

## How It Works

The `query()` method encapsulates the full TypeAgent query pipeline:

1. **Natural Language Understanding**: Uses TypeChat to translate the natural language question into a structured search query
2. **Search**: Executes the search across the conversation's messages and knowledge base
3. **Answer Generation**: Uses an LLM to generate a natural language answer based on the search results

## Method Signature

```python
async def query(self, question: str) -> str:
    """
    Run an end-to-end query on the conversation.

    Args:
        question: The natural language question to answer

    Returns:
        A natural language answer string. If the answer cannot be determined,
        returns an explanation of why no answer was found.
    """
```

## Behavior

- **Success**: Returns a natural language answer synthesized from the conversation content
- **No Answer Found**: Returns a message explaining why the answer couldn't be determined
- **Search Failure**: Returns an error message describing the failure

## Performance Considerations

The `query()` method caches the TypeChat translators per conversation instance, so repeated queries on the same conversation are more efficient.

## Example: Interactive Loop

```python
while True:
    question: str = input("typeagent> ")
    if not question.strip():
        continue
    if question.lower() in ("quit", "exit"):
        break
    
    answer: str = await conv.query(question)
    print(answer)
```

## Example: Batch Processing

```python
questions = [
    "What was discussed?",
    "Who were the speakers?",
    "What topics came up?",
]

for question in questions:
    answer = await conv.query(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print()
```

## Related APIs

For more control over the query pipeline, you can use the lower-level APIs:

- `searchlang.search_conversation_with_language()` - Search only
- `answers.generate_answers()` - Answer generation from search results

See `tools/utool.py` for examples of using these lower-level APIs with debugging options.
