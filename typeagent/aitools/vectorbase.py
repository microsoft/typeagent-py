# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..aitools.embeddings import AsyncEmbeddingModel


class ScoredOrdinal(NamedTuple):
    ordinal: int
    score: float


class VectorBase:
    # TODO: pass TextEmbeddingIndexSettings to give the model and the embedding size.
    def __init__(self):  # TODO: settings: TextEmbeddingIndexSettings | None = None
        self._model = AsyncEmbeddingModel()
        self._vectors: NDArray[np.float32] = np.array([], dtype=np.float32).reshape(
            (0, 0)
        )
        self._embedding_cache: dict[str, NDArray[np.float32]] = {}

    def __len__(self) -> int:
        return len(self._vectors)

    # Needed because otherwise an empty index would be falsy.
    def __bool__(self) -> bool:
        return True

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
        if len(keys):
            return np.array(embeddings, dtype=np.float32).reshape((len(keys), -1))
        else:
            return np.array([], dtype=np.float32).reshape((0, 0))

    async def add_key(self, key: str) -> None:
        embedding = (await self.get_embedding(key)).reshape((1, -1))
        if not len(self._vectors):
            self._vectors = embedding
        else:
            self._vectors = np.append(self._vectors, embedding, axis=0)

    async def add_keys(self, keys: list[str]) -> None:
        embeddings = await self.get_embeddings(keys)
        if not len(self._vectors):
            self._vectors = embeddings
        else:
            self._vectors = np.append(self._vectors, embeddings, axis=0)

    async def fuzzy_lookup(
        self, key: str, max_hits: int | None = None, min_score: float | None = None
    ) -> list[ScoredOrdinal]:
        if not len(self._vectors):
            return []
        if max_hits is None:
            max_hits = 10
        if min_score is None:
            min_score = 0.0
        embedding = await self.get_embedding(key)
        scores = np.dot(self._vectors, embedding)  # This does most of the work
        scored_ordinals = [
            ScoredOrdinal(i, score)
            for i, score in enumerate(scores)
            if score >= min_score
        ]
        scored_ordinals.sort(key=lambda x: x.score, reverse=True)
        return scored_ordinals[:max_hits]

    def clear(self) -> None:
        size = len(self._vectors)
        self._vectors = np.array([], dtype=np.float32).reshape((0, 0))


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
        log(f"{heading}: bool={bool(v)}, len={len(v)}")

    dotenv.load_dotenv(os.path.expanduser("~/TypeAgent/ts/.env"))
    v = VectorBase()
    debugv("\nEmpty vector base")

    words: list[str] = (
        "Mostly about multi-agent frameworks, "
        + "but also about answering questions about podcast transcripts."
    ).split()  # type: ignore  # pyright complains about list[LiteralString] -> list[str]
    cut = 2
    for word in words[:cut]:
        log("\nAdding:", word)
        await v.add_key(word)
        scored_ordinals = await v.fuzzy_lookup(word, max_hits=1)
        assert (
            round(scored_ordinals[0].score, 5) == 1.0
        ), f"{word} scores {scored_ordinals[0]}"
        debugv(word)

    log("\nAdding remaining words")
    await v.add_keys(words[cut:])
    debugv("After adding all")

    log("\nChecking presence")
    for word in words:
        scored_ordinals = await v.fuzzy_lookup(word, max_hits=1)
        assert (
            round(scored_ordinals[0].score, 5) == 1.0
        ), f"{word} scores {scored_ordinals[0]}"
    log("All words are present")
    word = "pancakes"
    scored_ordinals = await v.fuzzy_lookup(word, max_hits=1)
    assert scored_ordinals[0].score < 0.7, f"{word} scores {scored_ordinals[0]}"

    log("\nFuzzy lookups:")
    for word in words + ["pancakes", "hello world", "book", "author"]:
        neighbors = await v.fuzzy_lookup(word, max_hits=3)
        log(f"{word}:", [(nb.ordinal, nb.score) for nb in neighbors])


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
