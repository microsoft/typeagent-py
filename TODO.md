# TODO for the Python knowpro port

# From Meeting 8/12/2025 morning

- Get rid of `__getitem__` in favor of get_item(), get_slice(), get_multiple() [DONE]
  - Also rename `__len__` to size() [DONE]
- Switch db API to async (even for in-memory); fix all related bugs
- "Ordinals" ("ids") are sequential (ordered) but not contiguous
- So we can use auto-increment
- Fix all bugs related to that
- Flatten and reduce IConversation structure:
  - Message collection
  - SemanticRef collection
  - SemanticRef index
  - Property to SemanticRef index
  - Timestamp to TextRange
  - Terms to relatedterms
- Persist all of the above in the SQLite version
- Keep in-memory version (with some compromises) for comparison

# From Meeting 8/12/2025 afternoon

- Toss out character ordinals from TextLocation etc. [DONE]
  - Message ordinal must exist
  - Chunk ordinal of end in range is 1 past last chunk in range
    (== standard Python slice conventions)
  - TextRange is a half-open interval; end points past last chunk
- Indexing (knowledge extraction) operates chunk by chunk
- TimeRange always points to a TextRange
- Always import VTT, helper to convert podcast to VTT format
- Rename "Ordinal" to "Id"

# Other stuff

## Eval-based improvements

- Collect eval outputs (one change at a time) **[DONE]**
  - Baseline
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
