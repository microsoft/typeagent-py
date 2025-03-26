# TODO for the Python knowpro port

## Big functionality

    ### Serialization and deserialization

    Do this before queries, since fully re-indexing takes too long.

    Q: Should we serialize settings? Or are they intentionally not serialized?

    ### Queries and searches

    So we can finally do some end-to-end testing.

    ### Retries for embeddings

    For robustness -- TypeChat already retries, but embeddings don't.

---

## Small functionality

- Implement Podcast._build_caches(), by implementing TermEmbeddingCache?
  - Umesh wants to redo this

## Refactoring implementations

- Move various create_blah() functions into  Blah.__init__()
- Change some inconsistent module names
- Rewrite podcast parsing without regexes

## Main/demo/test program support

- Unify various dotenv calls and make them search harder (relative to repo)

## Type checking stuff

- fuzzy_index type mismatch (could VectorBase be made to match the type?)
- Fix need for `# type: ignore` comments (usually need to make some I-interface generic in actual message/index/etc.)

## Low priority

- Tombstones for deletions.

## Maybe...

- Add docstrings etc.
