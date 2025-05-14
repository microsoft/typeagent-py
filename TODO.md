# TODO for the Python knowpro port

## Serialization and deserialization (basically done)

Do this before queries, since fully re-indexing takes too long.

- Serialization appears done (gotta check TODO comments)
- Deserialization ditto

### Still TODO:

- Remove a bunch of `XxxData` TypedDicts that can be dealt with using
  `deserialize_object` and `serialize_object`
- Catch and report DeserializationError better

## Queries and searches

So we can finally do some end-to-end testing.

STARTING THIS NOW.

UMESH:
> - query.ts
>   - MatchSearchTermExpr  [DONE]
>   - MatchPropertySearchTermExpr
> - collections.ts
>   - SemanticRefAccumulator is in collections.ts  [DONE]
> 
> Ignore code path "without indexes"

## Retries for embeddings

For robustness -- TypeChat already retries, but my embeddings don't.

## Refactoring implementations

- Change some inconsistent module names
- Rewrite podcast parsing without regexes

## Type checking stuff

- fuzzy_index type mismatch (could VectorBase be made to match the type?)
- Fix need for `# type: ignore` comments (usually need to make some I-interface generic in actual message/index/etc.)

## Projct layout

- Move typeagent into src and associated changes everywhere
- Move test to tests
- Configure testpaths in pyproject.toml

## Testing

- Review Copilot-generated tests for sanity and minimal mocking

## Low priority

- Tombstones for deletions.

## Maybe...

- Add docstrings etc.

## Questions

- Do the serialization data formats (which are TypedDicts, not Protocols):
  - Really belong in interfaces.py? [UMES: No] [me: TODO]
  - Need to have names starting with 'I'? [UMESH: No] [me: DONE]
  My answers for both are no, unless Steve explains why.


# EXPERIMENT: Copilot's comparison of `knowledge.ts` and `knowledge.py`

## 1. Functionality Differences

### `createKnowledgeExtractor` / `create_knowledge_extractor`
- **Both implementations create a knowledge extractor.**
  - **TypeScript:** Uses `createKnowledgeModel` and explicitly sets `mergeActionKnowledge` to `false`.
  - **Python:** Uses `convknowledge.create_typechat_model` and also sets `merge_action_knowledge` to `False`.

### `extractKnowledgeFromText` / `extract_knowledge_from_text`
- **Both extract knowledge from a single text input.**
  - **TypeScript:** Uses `knowledgeExtractor.extractWithRetry` with retries handled by `async.callWithRetry`.
  - **Python:** Calls `knowledge_extractor.extract` but lacks a retry mechanism (marked as a TODO).

### `extractKnowledgeFromTextBatch` / `extract_knowledge_from_text_batch`
- **Both process a batch of text inputs concurrently.**
  - **TypeScript:** Uses `asyncArray.mapAsync` for concurrency.
  - **Python:** Processes the batch sequentially in a loop (marked as a TODO to add concurrency).

### `mergeConcreteEntities` / `merge_concrete_entities`
- **Both merge a list of concrete entities.**
  - **TypeScript:** Fully implemented using `concreteToMergedEntities` and `mergedToConcreteEntity`.
  - **Python:** Not implemented (marked as a TODO).

### `mergeTopics` / `merge_topics`
- **Both merge a list of topics into a unique list.**
  - **TypeScript:** Uses a `Set` and returns the values as an array.
  - **Python:** Uses a `set` and returns the values as a list. It includes a TODO to preserve the order of first occurrence.

### `extractKnowledgeForTextBatchQ` / `extract_knowledge_for_text_batch_q`
- **Both extract knowledge for a batch of text inputs using a task queue.**
  - **TypeScript:** Fully implemented using `BatchTask` and `runInBatches`.
  - **Python:** Not implemented (marked as a TODO).

---

## 2. Concurrency Handling
- **TypeScript:**
  - Uses `asyncArray.mapAsync` for concurrent processing in `extractKnowledgeFromTextBatch`.
  - Uses `runInBatches` for task queue processing in `extractKnowledgeForTextBatchQ`.

- **Python:**
  - Sequentially processes text inputs in `extract_knowledge_from_text_batch`.
  - Concurrency and task queue handling are marked as TODOs.

---

## 3. Error Handling
- **TypeScript:**
  - Uses `async.callWithRetry` for retries in `extractKnowledgeFromText`.
  - Handles missing results in `extractKnowledgeForTextBatchQ` by returning an error result (`error("No result")`).

- **Python:**
  - Lacks retry handling in `extract_knowledge_from_text` (marked as a TODO).
  - Missing result handling in `extract_knowledge_for_text_batch_q` is marked as a TODO.

---

## 4. Implementation Status
- **TypeScript:**
  - Fully implemented for all functions.
  - Includes robust concurrency and error handling.

- **Python:**
  - Several functions are marked as TODOs:
    - `extract_knowledge_for_text_batch_q` is not implemented.
    - `merge_concrete_entities` is not implemented.
    - Retry handling in `extract_knowledge_from_text` is missing.
    - Concurrency in `extract_knowledge_from_text_batch` is missing.

---

## 5. Utility Functions
- **TypeScript:**
  - Includes utility functions like `concreteToMergedEntities`, `mergedToConcreteEntity`, `BatchTask`, and `runInBatches` for merging entities and task queue processing.

- **Python:**
  - Relies on placeholders and TODOs for similar functionality, such as `concrete_to_merged_entities` and `merged_to_concrete_entity`.

---

## 6. Code Style
- **TypeScript:**
  - Uses modern JavaScript/TypeScript features like optional chaining (`chatModel ??= createKnowledgeModel()`), `async/await`, and `Set`.

- **Python:**
  - Uses Python idioms like `set` for unique collections and type hints (`list[str]`, `Result[kplib.KnowledgeResponse]`).

---

## Summary of Key Differences
1. **Concurrency:** TypeScript has robust concurrency handling, while Python lacks it (marked as TODO).
2. **Retry Mechanism:** TypeScript implements retries, while Python does not (marked as TODO).
3. **Task Queue:** TypeScript implements task queue processing, while Python does not (marked as TODO).
4. **Entity Merging:** Fully implemented in TypeScript but missing in Python (marked as TODO).
5. **Error Handling:** TypeScript handles missing results gracefully, while Python lacks this functionality.

---
