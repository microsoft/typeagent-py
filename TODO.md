# TODO for the Python knowpro port

Meta-todo: Gradually move work items from here to repo Issues.

# Leftover TODOs from TADA.md

## Software

Minor:

- Improve load_dotenv() (don't look for `<repo>/ts/.env`, use one loop)

### Specifically for VTT import (minor)

- Reduce duplication between ingest_vtt.py and typeagent/transcripts/
- `get_transcript_speakers` and `get_transcript_duration` should not
  re-parse the transcript -- they should just take the parsed vtt object.

### Embeddings

- Handle input truncation for embeddings by counting tokens (e.g. tiktoken).

## Documentation

- Document how to reproduce the demos from the talk (and podcast)
- Document test/build/release process
- Document how to use gmail_dump.py (set up a project etc.)

Maybe later:

- Document how to run evaluations (but don't reveal all the data)


# TODOs for fully implementing persistence through SQLite

## Now

- Scrutinize sqlite/reltermsindex.py
- Unify tests for storage APIs
- Review the new storage code more carefully, adding notes here
- Conversation id in conversation metadata table feels wrong
- Conversation metadata isn't written -- needs a separate call
- Improve test coverage for search, searchlang, query, sqlite
- Reduce code size
- Make coding style more uniform (e.g. docstrings)

## Also

- Make (de)serialize methods async in interfaces.py if they might execute SQL statements

## Maybe

- Flatten secondary indexes into Conversation (they are no longer optional)
- Split related terms index in two (aliases and fuzzy_index)
- Make the collection/index accessors in StorageProvider synchronous
  (the async work is all done in create())
- Replace the storage accessors with readonly @property functions
- Refactor memory and sqlite indexes to share more code
  (e.g. population and query logic)

## Lower priority

- Try to avoid so many inline imports.
  Break cycles by moving things to their own file if necessary

# From Meeting 8/12/2025 morning (edited)

- "Ordinals" ("ids") should be sequential (ordered) but not contiguous
  - So we can use auto-increment
  - Fix all bugs related to that
- Flatten and reduce IConversation structure:
  - Message collection
  - SemanticRef collection
  - SemanticRef index
  - Property to SemanticRef index
  - Timestamp to TextRange
  - Terms to related terms
- Keep in-memory version (with some compromises) for comparison

# From Meeting 8/12/2025 afternoon (edited)

- Indexing (knowledge extraction) operates chunk by chunk
- TimeRange always points to a TextRange
- Always import VTT, helper to convert podcast to VTT format
  (Probably not, podcast format has listeners but VTT doesn't)
- Rename "Ordinal" to "Id"

# Other stuff

### Left to do here

- Look more into why the search query schema is so instable
- Implement at least some @-commands in query.py
- More debug options (turn on/off various debug prints dynamically)
- Use pydantic.ai for model drivers

## General: Look for things marked as incomplete in source

- `TODO` comments (too numerous)
- `raise NotImplementedError("TODO")` (five found)

## Cleanup:

- Sort out why `IConversation` needs two generic parameters;
  especially `TTermToSemanticRefIndex` is annoying. Can we do better?
- Unify or align or refactor `VectorBase` and `EmbeddingIndex`.

## Serialization

- Remove a bunch of `XxxData` TypedDicts that can be dealt with using
  `deserialize_object` and `serialize_object`
- Catch and report `DeserializationError` better
- Look into whether Pydantic can do our (de)serialization --
  if it can, presumably it's faster?

## Development

- Move `typeagent` into `src`.
- Move test to tests?
- Configure testpaths in pyproject.toml

## Testing

- Review Copilot-generated tests for sanity and minimal mocking
- Add new tests for newly added classes/methods/functions
- Coverage testing (needs to use a mix of indexing and querying)
- Automated end-to-end tests using Umesh's test data files

## Tighter types

- Several places allow `None` and in that case construct a default instance.
  It's probably better to either disallow `None` or skip that functionality.

## Queries and searches

Let me first describe the architecture.
We have several stages (someof which loop):

1. Parse natural language into a `SearchQuery`. (_searchlang.py_)
2. Transform this to a `SearchQueryExpr`. (_search.py_)
3. In `search_conversation` (in _search.py_):
   a. compile to `GetScoredMessageExpr` and run that query.
   b. compile to `GroupSearchResultsExpr` and run that query.
   c. Combine the results into a `ConversationSearchResult`.
4. Turn the results into human language, using an prompt that
   asks the model to generate an answer from the context
   (messages and knowledge from 3c) and he original raw query.
   a. There may be multiple search results; if so, another prompt
      is used to combine the answers.
   b. Similarly, the context from a single search result may be
      too large for a model's token buffer. In that case we split
      the contexts into multiple requests and combine the answers
      in the same way.

All of these stages are at least partially implemented,
so we have some end-to-end functionality.

The TODO items include (in no particular order):

- Implement token budgets -- may leave out messages, favoring only knowledge,
  if it answers the question.
- Change the context to be text, including message texts and timestamps,
  rather than JSON (and especially not just semantic ref ordinals).
- Split large contexts to avoid overflowing the answer generator's
  context buffer (4b).
- Redesign the whole pipeline now that I understand the archtecture better;
  notably make each stage its own function with simpler API.

# Older TODO action items

## Retries for embeddings

For robustness -- TypeChat already retries, but my embeddings don't.

## Refactoring implementations

- Change inconsistent module names (Claude uses different naming style)
- Rewrite podcast parsing without regexes (low priority)
- Switch from Protocol to ABC

## Type checking stuff

- Fix need for `# type: ignore` comments (usually need to make some
  I-interface generic in actual message/index/etc.) I see 22 in typeagent/.

## Deletions

- A consistent approach to deletions. Deleting a message should remove
  all semrefs referencing it and all index entries reference those.
  Probably the only way is tombstoning (in sqlite can just delete rows)

## Questions

- Do the serialization data formats (which are TypedDicts, not Protocols):
  - Really belong in interfaces.py? [UMESH: No] [me: NO; TODO]
