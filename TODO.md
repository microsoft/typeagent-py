# TODO for typeagent-py

## Organizing principle

Tasks are grouped by component/feature area, with priorities
P0-P3 (P0=higest, P3=lowest). Note that priorities were assigned by Copilot,
and I don't always agree (it often assigns P0 or P1 to complicated work items).

## Meta-TODO

Gradually move work items from here to repo Issues.

---

## Storage & Persistence

### P0 - Critical
- **[P0]** Fix all bugs related to ordinals/ids relying on starting at 0 and no gaps
- **[P0]** Conversation metadata isn't written -- needs a separate call
- **[P0]** Need embedding size and embedding name in metadata

### P1 - High Priority
- **[P1]** Scrutinize sqlite/reltermsindex.py
- **[P1]** Review the new storage code more carefully, adding notes
- **[P1]** Unify tests for storage APIs
- **[P1]** Conversation id in conversation metadata table feels wrong
- **[P1]** Make (de)serialize methods async in interfaces.py if they might execute SQL statements

### P2 - Medium Priority
- **[P2]** Refactor memory and sqlite indexes to share more code (e.g. population and query logic)
- **[P2]** Make the collection/index accessors in StorageProvider synchronous (the async work is all done in create())
- **[P2]** Replace the storage accessors with readonly @property functions
- **[P2]** "Ordinals" should be renamed to "Id" (tedious though)

### P3 - Low Priority
- **[P3]** Flatten secondary indexes into Conversation (they are no longer optional)
- **[P3]** Split related terms index in two (aliases and fuzzy_index)
- **[P3]** Remove message text index -- it doesn't appear to be used at all
- **[P3]** Implement consistent approach to deletions (tombstoning in sqlite, cascade delete semrefs and indexes)

---

## Query & Search Pipeline

### P1 - High Priority
- **[P1]** Look more into why the search query schema is so unstable
- **[P1]** Redesign the whole pipeline; make each stage its own function with simpler API
- **[P1]** Improve test coverage for search, searchlang, query, sqlite

### P2 - Medium Priority
- **[P2]** Implement token budgets for answer generation (may leave out messages, favoring only knowledge)
- **[P2]** Change the context to be text (message texts and timestamps), not JSON or semantic ref ordinals
- **[P2]** Split large contexts to avoid overflowing the answer generator's context buffer
- **[P2]** Implement at least some @-commands in query.py
- **[P2]** More debug options (turn on/off various debug prints dynamically)

---

## Type System & Interfaces

### P1 - High Priority
- **[P1]** Move TypedDicts out of interfaces.py (they don't belong there)
- **[P1]** Fix need for `# type: ignore` comments (22 in typeagent/) by making I-interfaces more generic

### P2 - Medium Priority
- **[P2]** Sort out why `IConversation` needs two generic parameters
- **[P2]** Simplify `TTermToSemanticRefIndex` generic parameter
- **[P2]** Tighten types: several places allow `None` and construct default instances; either disallow `None` or skip that functionality

---

## Serialization

### P2 - Medium Priority
- **[P2]** Remove a bunch of `XxxData` TypedDicts that can be dealt with using `deserialize_object` and `serialize_object`
- **[P2]** Catch and report `DeserializationError` better
- **[P2]** Look into whether Pydantic can do our (de)serialization (presumably faster?)

---

## Code Quality & Cleanup

### P1 - High Priority
- **[P1]** Make coding style more uniform (e.g. docstrings)
- **[P1]** Reduce code size

### P2 - Medium Priority
- **[P2]** Avoid most inline imports
- **[P2]** Break cycles by moving things to their own file if necessary
- **[P2]** Unify or align or refactor `VectorBase` and `EmbeddingIndex`
- **[P2]** Address `TODO` comments (too numerous)
- **[P2]** Address `raise NotImplementedError("TODO")` (five found)

### P3 - Low Priority
- **[P3]** Change inconsistent module names (Claude uses different naming style)
- **[P3]** Rewrite podcast parsing without regexes
- **[P3]** Switch from Protocol to ABC
- **[P3]** Reduce duplication between ingest_vtt.py and typeagent/transcripts/

---

## Testing

### P1 - High Priority
- **[P1]** Add new tests for newly added classes/methods/functions

### P2 - Medium Priority
- **[P2]** Review Copilot-generated tests for sanity and minimal mocking

---

## Documentation

### P1 - High Priority
- **[P1]** Document test/build/release process
- **[P1]** Document how to run evaluations (but don't share the data)

### P2 - Medium Priority
- **[P2]** Document low-level APIs (key parts used directly by high-level APIs, e.g. ConversationSettings)

---

## Development Infrastructure

### P3 - Low Priority
- **[P3]** Move `typeagent` into `src/`
- **[P3]** Move `test/` to `tests/`

---

## Features & Enhancements

### P2 - Medium Priority
- **[P2]** Use pydantic.ai for model drivers, to support non-openai models

---

## Architecture Decisions

### P2 - Medium Priority
- **[P2]** Flatten and reduce IConversation structure:
  - Message collection
  - SemanticRef collection
  - SemanticRef index
  - Property to SemanticRef index
  - Timestamp to TextRange
  - Terms to related terms
- **[P2]** Keep in-memory version (with some compromises) for comparison
