# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..aitools.embeddings import AsyncEmbeddingModel


class ScoredKey(NamedTuple):
    key: str
    score: float


class VectorBase:
    def __init__(self):
        self._model = AsyncEmbeddingModel()
        self._keys: list[str] = []  # Should have no duplicates
        self._vectors: NDArray[np.float32] = np.array([], dtype=np.float32).reshape(
            (0, 0)
        )
        self._embedding_cache: dict[str, NDArray[np.float32]] = {}

    def __len__(self) -> int:
        assert len(self._keys) == len(self._vectors), (self._keys, self._vectors)
        return len(self._keys)

    # Ensure an empty vectorbase is truthy.
    def __bool__(self) -> bool:
        return True

    def keys(self) -> set[str]:
        s = set(self._keys)
        assert len(s) == len(self._keys), "Duplicate key present"
        return s

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    async def get_embedding(self, key: str) -> NDArray[np.float32]:
        """Retrieve an embedding, using the cache."""
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        embedding = await self._model.get_embedding(key)
        self._embedding_cache[key] = embedding
        return embedding

    async def get_embeddings(self, keys: list[str]) -> NDArray[np.float32]:
        """Retrieve embeddings for multiple keys, using the cache."""
        embeddings = []
        missing_keys = []

        # Collect cached embeddings and identify missing keys
        for key in keys:
            if key in self._embedding_cache:
                embeddings.append(self._embedding_cache[key])
            else:
                embeddings.append(None)  # Placeholder for missing keys
                missing_keys.append(key)

        # Retrieve embeddings for missing keys
        if missing_keys:
            new_embeddings = await self._model.get_embeddings(missing_keys)
            for key, embedding in zip(missing_keys, new_embeddings):
                self._embedding_cache[key] = embedding

            # Replace placeholders with retrieved embeddings
            for i, key in enumerate(keys):
                if embeddings[i] is None:
                    embeddings[i] = self._embedding_cache[key]

        return np.array(embeddings, dtype=np.float32).reshape((len(keys), -1))

    async def fuzzy_lookup(
        self, key: str, max_hits: int = 10, min_score: float = 0.0
    ) -> list[ScoredKey]:
        if not len(self._vectors):
            return []

        embedding = await self.get_embedding(key)

        scores = np.dot(self._vectors, embedding)  # This does most of the work

        scored_keys = [
            ScoredKey(k, score)
            for score, k in zip(scores, self._keys)
            if score >= min_score
        ]
        scored_keys.sort(key=lambda x: x.score, reverse=True)
        return scored_keys[:max_hits]

    async def add_key(self, key: str) -> None:
        if key in self._keys:
            return

        embedding = await self.get_embedding(key)

        embedding = embedding.reshape((1, len(embedding)))
        if not len(self._vectors):
            self._vectors = embedding
        else:
            self._vectors = np.append(self._vectors, embedding, axis=0)
        self._keys.append(key)

    async def add_keys(self, keys: list[str]) -> None:
        s = set(keys)
        for key in self._keys:
            s.discard(key)
            if not s:
                return  # They were all already there
        keys = list(s)  # Not the original order, but that's fine

        embeddings = await self.get_embeddings(keys)

        self._vectors = np.append(self._vectors, embeddings, axis=0)
        self._keys.extend(keys)

    # TODO: Remove key[s]?


async def main():
    import dotenv, os, time

    epoch = time.time()

    def log(*args, end="\n"):
        stamp = f"{time.time()-epoch:7.3f}"
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, str) and "\n" in arg:
                args[i] = arg.replace("\n", f"\n{stamp}: ")
        print(f"{stamp}:", *args, end=end)

    def debugv(heading):
        log(f"{heading}: bool={bool(v)}, len={len(v)}, keys={v.keys()}")

    dotenv.load_dotenv(os.path.expanduser("~/TypeAgent/ts/.env"))
    v = VectorBase()
    debugv("\nEmpty vector base")

    words: list[str] = (
        "Mostly about multi-agent frameworks, "
        + "but also about answering questions about podcast transcripts."
    ).split()  # type: ignore  # pyscript complains about list[LiteralString] -> list[str]
    for word in words[:2]:
        log("\nAdding:", word)
        await v.add_key(word)
        assert word in v, f"{word} did not get added"
        debugv(word)

        log("Redundant adding:", word)
        await v.add_key(word)
        assert word in v, f"{word} disappeared"
        debugv(word)

    log("\nAdding all words")
    await v.add_keys(words)
    debugv("After adding all")

    log("\nChecking presence")
    for word in words:
        assert word in v, f"{word} not in v"
    log("All words are present")
    assert "foo" not in v, "foo is present but should not be"

    log("\nFuzzy lookups:")
    for word in words + ["pancakes", "hello world", "book", "author"]:
        neighbors = await v.fuzzy_lookup(word, max_hits=3)
        log(f"{word}:", [(nb.key, float(nb.score)) for nb in neighbors])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
