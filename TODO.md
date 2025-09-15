# TODO for the Python knowpro port

# TODOs for new repo setup

- Merge newer changes from TypeAgent repo
- Vendor TypeChat (Python version)
- Update load_dotenv() to look for .env in current directory and going up (*plus* ts/.env)

# TODOs for fully implementing persistence through SQLite

## Now

- Switch to (agents.md)[https://agents.md]

- Vendor TypeChat

- Start practicing PyPI releases

- Scrutinize sqlite/reltermsindex.py
- Unify tests for storage APIs
- Review the new storage code more carefully, adding notes here
- Conversation id in conversation metadata table feels wrong
- Conversation metadata isn't written -- needs a separate call
- Improve test coverage for search, searchlang, query, sqlite
- Reduce code size
- Make coding style more uniform (e.g. docstrings)
- Document the highest-level API

## Also

- The aliases part of the related terms index is hard to understand because the
  relationship between term and alias feels reversed:
  We have dozens of aliases for "say", and these all show up as entries
  like (term="talk", alias="say"), (term="discuss", alias="say"), etc.
  My feeling (from the unix shell alias command) is that the term is "say"
  and the alias is "talk", "discuss", etc.
  (Not sure if the same is true for the fuzzy index, but I am confused there too.)
- Make (de)serialize methods async in interfaces.py if they might execute SQL statements

## Knowledge extraction pipeline

- Write a function that does the following:
  - Add a given list of messages to the end of the message collection
  - Extracts knowledge for all
  - Call the next function

- Write a function that adds a list of messages *and* a list of corresponding
  semantic refs, and then updates everything. This is somewhat complicated
  because we won't know the message ordinals/ids until they have been
  inserted, and ditto for the semantic refs.
  (Why do semantic refs contain their own ord/id anyway?)

## Maybe

- Flatten secondary indexes into Conversation (they are no longer optional)
- Split related terms index in two (aliases and fuzzy_index)
- Make the collection/index accessors in StorageProvider synchronous
  (the async work is all done in create())
- Replace the storage accessors with readonly @property functions
- Refactor memory and sqlite indexes to share more code
  (e.g. population and query logic)
- Store embeddings in message_index

## Lower priority

- Rework pyproject to separate build-time from runtime deps
  - Make some runtime deps optional (e.g. logfire, mcp)
  - Comment out pydantic-ai until we resume that work

- Try to avoid so many inline imports.
  Break cycles by moving things to their own file if necessary

# From Meeting 8/12/2025 morning

- Get rid of `__getitem__` in favor of get_item(), get_slice(), get_multiple() [**DONE**]
  - Also rename `__len__` to size() [**DONE**]
- Switch db API to async (even for in-memory); fix all related bugs [**DONE**]
- "Ordinals" ("ids") are sequential (ordered) but not contiguous
- So we can use auto-increment
- Fix all bugs related to that
- Flatten and reduce IConversation structure:
  - Message collection
  - SemanticRef collection
  - SemanticRef index
  - Property to SemanticRef index
  - Timestamp to TextRange
  - Terms to related terms
- Persist all of the above in the SQLite version
- Keep in-memory version (with some compromises) for comparison

# From Meeting 8/12/2025 afternoon

- Toss out character ordinals from TextLocation etc. [**DONE**]
  - Message ordinal must exist [**DONE**]
  - Chunk ordinal of end in range is 1 past last chunk in range
    (== standard Python slice conventions) [**DONE**]
  - TextRange is a half-open interval; end points past last chunk [**DONE**]
- Indexing (knowledge extraction) operates chunk by chunk
- TimeRange always points to a TextRange
- Always import VTT, helper to convert podcast to VTT format
- Rename "Ordinal" to "Id"

## From Meeting 8/13/2025 morning

- Keep "Conversation" as the top-level name; changing it isn't worth it
- Get rid of event handling API, move this into the front-line extractor
  - batching is up to the extractor
  - every batch is processed completely inside one transaction
  - extractor gets an exception when the transaction fails or is rolled back
- Move to a toplevel library (structured_rag or python?) and update
  toplevel README.md to advertise that.
- If pydantic AI doesn't pan out, vendor TypeChat
- See images of database schema and API proposals
- Example data types for which we ship extractors:
  - VTT (for Teams, YouTube, podcasts); podcast examples converted by unpublished example
  - Email (MIME messages); GMail example via adapter
  - Markdown; HTML via Markdown conversion
- Each of these has a textual format; it's the user's responsibility to provide that format
- Everything else is unofficial and undocumented

### Sqlite database schema details

**THIS HAS BEEN SUPERSEDED BY THE FILES IN THE `spec/` FOLDER**

Note: `*` means a column with an index

- ConversationMetadata
  - name_tag TEXT
  - schema_version TEXT
  - other per-conversation stuff

- Messages
  - * msg_id INTEGER UNIQUE PRIMARY KEY AUTOINCREMENT -- whatever
  - chunks NULL or JSON
  - chunkuri NULL or TEXT -- exactly one of chunks, chunkuri must be not-NULL
  - * timestamp NULL or TEXT -- in ISO format with Z timezone (rename to start_timestamp?)
  - * end_timestamp NULL or TEXT -- ditto
  - (no tomstone -- when deleting, save stuff in a separate Undo table)
  - tags JSON
  - metadata JSON
  - extra JSON -- for extra message fields that were serialized

  In-memory Message has get_chunks() method:
  - if chunks != None: return chunks
  - assert chunkuri
  - retrieve chunks using chunkuri  # this is up to the extractor to customize
  - set chunks
  - return chunks

  Creating an in-memory Message can provide both chunks and chunkuri
  - If chunkuri is set, chunks are not stored in the db

- SemanticRefs
  - * semref_id INTEGER UNIQUE PRIMARY KEY AUTOINCREMENT -- whatever
  - range -- not a field, split up into four
    - * start_msg_id INTEGER FOREIGN KEY
    - start_chunk_ord INTEGER
    - * end_msg_id INTEGER FOREIGN KEY -- Never NULL
    - end_chunk_ord INTEGER
  - ktype TEXT -- Choices: entity, action, topic, tag
  - knowledge JSON BLOB

- SemanticRefIndex -- a list of (term, semref_id) pairs
  - * term -- lowercased, not-unique/normalized
  - semref_id -- points to semref that contains this term

  To get all semrefs that mention a given term, SELECT * ... WHERE term = ?

  Future research: Store terms in separate table with term_id,
  make this table a tuple of (term_id, semref_id)?

- PropertyIndex
  - * prop_name TEXT
  - * value_str TEXT or NULL
  - score FLOAT
  - semref_id INTEGER

  Later add shredded columns for value_number (bool, int, float),
  value_quantity_amount, -number; all indexed?

- Timestamp is not a table, just an index on timestamp, end_timestamp in Messages

# Other stuff

## Eval-based improvements

- Collect eval outputs (one change at a time) **[DONE]**
  - Baseline=(
  - Schema in reversed order (TC change)
  - null -> undefined (TC change)
  - field names -> fieldNames (local change, add alias)
  - Move required fields in ActionTerm before optionals
  - Many others.

Then combine the ones that most improve effectiveless.  **[DONE]**
(Actually, we mostly found bugs this way. Only one schema change.)

### Left to do here

- Look more into why the search query schema is so instable
- Implement at least a subset of the @kpBlah commands in utool.py
- More debug options (turn on/off various debug prints dynamically)
- Unify ui.py and cmp [**done**]
- Try pydantic.ai again

## General: Look for things marked as incomplete in source

- `TODO` comments (too numerous)
- `raise NotImplementedError("TODO")` (three found indexing)

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
- Separate development dependencies from installation dependencies.
- Move test to tests
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

All of these stages are at least partially implemented
(though only the simplest form of answer generation),
so we have some end-to-end functionality.

The TODO items include (in no particular order):

- Implement token budgets -- may leave out messages, favoring only knowledge,
  if it answers the question.
- Fix handling of datetime range queries. [**DONE**]
- Use fallback query and other fallback stuff in search_conv*_w*_lang*. [**DONE**]
- Change the context to be text, including message texts and timestamps,
  rather than JSON (and especially not just semantic ref ordinals).
- Property translate time range filters. [**DONE**]
- Add message timestamp to context. [**DONE**]
- Move out of `demo` into `knowpro` what belongs there. [**DONE**]
- Complete the implementation of each stage (3b is missing a lot). [**DONE**]
- Split large contexts to avoid overflowing the answer generator's
  context buffer (4b).
- Fix all the TODOs left in the code.
- Redesign the whole pipeline now that I understand the archtecture better;
  notably make each stage its own function with simpler API.

# Older TODO action items

## Retries for embeddings

For robustness -- TypeChat already retries, but my embeddings don't.

## Refactoring implementations

- Change some inconsistent module names  [DONE -- if we are okay deviating from the TS module names]
- Rewrite podcast parsing without regexes (low priority)
- Switch from Protocol to ABC

## Type checking stuff

- fuzzy_index type mismatch (could VectorBase be made to match the type?)
- Fix need for `# type: ignore` comments (usually need to make some I-interface generic in actual message/index/etc.)

## Deletions

- Tombstones for message and semantic ref deletions.
- Support other deletions.

## Questions

- Do the serialization data formats (which are TypedDicts, not Protocols):
  - Really belong in interfaces.py? [UMESH: No] [me: TODO]
  - Need to have names starting with 'I'? [UMESH: No] [me: DONE] [TS hasn't changed yet]
  My answers for both are no, unless Steve explains why.
