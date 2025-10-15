# High-level API

NOTE: When an argument's default is given as `[]`, this is a shorthand
for a dynamically assigned default value on each call. We don't mean
the literal meaning of this notation in Python, which would imply
that all calls would share a single empty list object as their default.

## Classes

### Message classes

#### `ConversationMessage`

`typeagent.knowpro.universal_message.ConversationMessage`

Constructor and fields:

```py
class ConversationMessage(
    text_chunks: list[str],  # Text of the message, 1 or more chunks
    tags: list[str] = [],  # Optional tags
    timestamp: str | None = None,  # ISO timestamp in UTC with 'z' suffix
    metadata: ConversationMessageMeta,  # See below 
)
```

- Only `text_chunks` is required.
- Tags are arbitrary pieces of information attached to a message
  that will be indexed; e.g. `["sketch", "pet shop"]`
- If present, the timestamp must be of the form `2025-10-14T09:03:21z`.

#### `ConversationMessageMeta`

`typeagent.knowpro.universal_message.ConversationMessageMeta`

Constructor and fields:

```py
class ConversationMessageMeta(
    speaker: str | None = None,  # Optional entity who sent the message
    recipients: list[str] = [],  # Optional entities to whom the message was sent
)
```

This class represents the metadata for a given `ConversationMessage`.

#### `TranscriptMessage` and `TranscriptMessageMeta`

`typeagent.transcripts.transcript.TranscriptMessage`
`typeagent.transcripts.transcript.TranscriptMessageMeta`

These are simple aliases for `ConversationMessage` and
`ConversationMessageMeta`, respectively.

### Conversation classes

#### `ConversationBase`

`typeagent.knowpro.factory.ConversationBase`

Represents a conversation, which holds ingested messages and the
extracted and indexed knowledge thereof.

It is constructed by calling the factory function
`typeagent.create_conversation` described below.

It has one public method:

- `query`
  ```py
  async def query(
      question: str,
      # Other parameters are not public
  ) -> str
  ```

  Tries to answer the question using (only) the indexed messages.
  If no answer is found, the returned string starts with
  `"No answer found:"`.

## Functions

There is currently only one public function.

#### Factory function

- `create_conversation`
  ```py
  async def create_conversation(
      dbname: str | None,
      message_type: type,
      name: str = "",
      tags: list[str] | None = None,
      settings: ConversationSettings | None = None,
  ) -> ConversationBase
  ```

  - Constructs a conversation object.
  - The required `dbname` argument specifies the SQLite3 database
    name (e.g. `test.db`). If explicitly set to `None` the data is
    stored in RAM and will not persist when the process exits.
  - The required `message_type` is normally `TranscriptMessage`
    or `ConversationMessage` (there are other possibilities too,
    as yet left undocumented).
  - The optional `name` specifies the conversation name, which
    may be used in diagnostics.
  - `tags` gives tags (like `ConversationMessage.tags`) for the whole
    conversation.
  - `settings` provides overrides for various aspects of the knowledge
    extraction and indexing process. Its exact usage is currently left
    as an exercise for the reader.
