# TODO for the Python knowpro port

## Short-term plans and TODOs

- Collect eval outputs (one change at a time)
  - Baseline
  - Schema in reversed order (TC change)
  - null -> undefined (TC change)
  - field names -> fieldNames (local change, add alias)
  - Move required fields in ActionTerm before optionals

Then combine the ones that most improve effectiveless

- Look more into why the search query schema is so instable
- Implement at least a subset of the @kpBlah commands in ui.py
- More debug options (turn on/off various debug prints dynamically)
- Try pydantic.ai again

## General: Look for things marked as incomplete in source

- `TODO` comments
- `raise NotImplementedError` with TODO in arg or comment

## Cleanup:

- Remove a bunch of `XxxData` TypedDicts that can be dealt with using
  `deserialize_object` and `serialize_object`
- Catch and report `DeserializationError` better
- Sort out why `IConversation` needs two generic parameters;
  especially `TTermToSemanticRefIndex` is annoying. Can we do better?
- Unify or align or refactor `VectorBase` and `EmbeddingIndex`.

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

- Fix handling of datetime range queries.
- Use fallback query and other fallback stuff in search_conv*_w*_lang*.
- Change the context to be text, including message texts and timestamps,
  rather than JSON (and especially not just semantic ref ordinals).
- Property translate time range filters.
- Add message timestamp to context.
- Move out of `demo` into `knowpro` what belongs there.
- Complete the implementation of each stage (3b is missing a lot).
- Split large contexts to avoid overflowing the answer generator's
  context buffer (4b).
- Fix all the TODOs left in the code.
- Redesign the whole pipeline now that I understand the archtecture better.

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
