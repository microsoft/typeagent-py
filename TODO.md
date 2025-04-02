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

> UMESH:
> - query.ts
>   - MatchSearchTermExpr
>   - MatchPropertySearchTermExpr
> - SemanticRefAccumulator is in collection.ts
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

## Low priority

- Tombstones for deletions.

## Maybe...

- Add docstrings etc.

## Questions

- Do the serialization data formats (which are TypedDicts, not Protocols):
  - Really belong in interfaces.py? [UMES: No] [me: TODO]
  - Need to have names starting with 'I'? [UMESH: No] [me: DONE]
  My answers for both are no, unless Steve explains why.
