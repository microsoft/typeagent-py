# TODO for typeagent-py

## Organizing principle

Tasks are grouped by component/feature area, with priorities
P0-P3 (P0=higest, P3=lowest). Note that priorities were assigned by Copilot,
and I don't always agree (it often assigns P0 or P1 to complicated work items).

Each item also has a rough size estimate.

## Meta-TODO

Gradually move work items from here to repo Issues.

---

## Storage & Persistence

### P0 - Critical
- **[P0, large]** Fix all bugs related to ordinals/ids relying on starting at 0 and no gaps
- **[P0, small]** Need embedding size and embedding name in metadata

### P1 - High Priority
- **[P1, medium]** Scrutinize sqlite/reltermsindex.py
- **[P1, medium]** Review the new storage code more carefully, adding notes
- **[P1, large]** Unify tests for storage APIs
- **[P1, medium]** Make (de)serialize methods async in interfaces.py if they might execute SQL statements

### P2 - Medium Priority
- **[P2, large]** Refactor memory and sqlite indexes to share more code (e.g. population and query logic)
- **[P2, medium]** Make the collection/index accessors in StorageProvider synchronous (the async work is all done in create())
- **[P2, medium]** Replace the storage accessors with readonly @property functions
- **[P2, medium]** "Ordinals" should be renamed to "Id" (tedious though)

### P3 - Low Priority
- **[P3, medium]** Flatten secondary indexes into Conversation (they are no longer optional), reducing the structure to:
  - Message collection
  - SemanticRef collection
  - SemanticRef index
  - Property to SemanticRef index
  - Timestamp to TextRange
  - Terms to related terms
- **[P3, medium]** Split related terms index in two (aliases and fuzzy_index)
- **[P3, large]** Implement consistent approach to deletions (tombstoning in sqlite, cascade delete semrefs and indexes)

---

## Query & Search Pipeline

### P1 - High Priority
- **[P1, medium]** Look more into why the search query schema is so unstable
- **[P1, large]** Redesign the whole pipeline; make each stage its own function with simpler API
- **[P1, large]** Improve test coverage for search, searchlang, query, sqlite

### P2 - Medium Priority
- **[P2, medium]** Implement token budgets for answer generation (may leave out messages, favoring only knowledge)
- **[P2, medium]** Change answer context to be text (message texts and timestamps), not JSON or semantic ref ordinals
- **[P2, medium]** Split large answer contexts to avoid overflowing the answer generator's context buffer

---

## Type System & Interfaces

### P1 - High Priority
- **[P1, medium]** Move TypedDicts out of interfaces.py (they don't belong there)
- **[P1, medium]** Fix need for `# type: ignore` comments (22 in typeagent/) by making I-interfaces more generic

### P2 - Medium Priority
- **[P2, medium]** Sort out why `IConversation` needs two generic parameters
- **[P2, medium]** Simplify `TTermToSemanticRefIndex` generic parameter
- **[P2, medium]** Tighten types: several places allow `None` and construct default instances; either disallow `None` or skip that functionality

---

## Serialization

### P2 - Medium Priority
- **[P2, medium]** Remove a bunch of `XxxData` TypedDicts that can be dealt with using `deserialize_object` and `serialize_object`
- **[P2, small]** Catch and report `DeserializationError` better
- **[P2, medium]** Look into whether Pydantic can do our (de)serialization (presumably faster?)

---

## Code Quality & Cleanup

### P1 - High Priority
- **[P1, large]** Make coding style more uniform (e.g. docstrings)
- **[P1, large]** Reduce code size

### P2 - Medium Priority
- **[P2, small]** Avoid most inline imports
- **[P2, medium]** Break cycles by moving things to their own file if necessary
- **[P2, medium]** Unify or align or refactor `VectorBase` and `EmbeddingIndex`
- **[P2, medium]** Address `TODO` comments (too numerous)
- **[P2, medium]** Address `raise NotImplementedError("TODO")` (five found) -- implement it

### P3 - Low Priority
- **[P3, medium]** Change inconsistent module names (Claude uses different naming style)
- **[P3, medium]** Rewrite podcast parsing without regexes
- **[P3, medium]** Switch from Protocol to ABC
- **[P3, medium]** Reduce duplication between ingest_vtt.py and typeagent/transcripts/
- **[P3, small]** Rename `kplib.py` to something ending in `_schema.py`

---

## Testing

### P1 - High Priority
- **[P1, large]** Add new tests for newly added classes/methods/functions

### P2 - Medium Priority
- **[P2, medium]** Review Copilot-generated tests for sanity and minimal mocking

---

## Documentation

### P1 - High Priority
- **[P1, small]** Document test/build/release process
- **[P1, small]** Document how to run evaluations (but don't share the data)

### P2 - Medium Priority
- **[P2, large]** Document low-level APIs (key parts used directly by high-level APIs, e.g. ConversationSettings)

---

## Development Infrastructure

### P3 - Low Priority
- **[P3, small]** Move `typeagent` into `src/`
- **[P3, tiny]** Move `test/` to `tests/`

---

## Features & Enhancements

### P2 - Medium Priority
- **[P2, medium]** Use pydantic.ai for model drivers, to support non-openai models

---

## Architecture Decisions

### P2 - Medium Priority
- **[P2, medium]** Keep in-memory version (with some compromises) for comparison
