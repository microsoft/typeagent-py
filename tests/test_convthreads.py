# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for storage/memory/convthreads.py."""

import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.interfaces import TextLocation, TextRange, Thread
from typeagent.knowpro.interfaces_serialization import ConversationThreadData
from typeagent.storage.memory.convthreads import ConversationThreads


@pytest.fixture
def settings() -> TextEmbeddingIndexSettings:
    return TextEmbeddingIndexSettings(create_test_embedding_model())


@pytest.fixture
def threads(settings: TextEmbeddingIndexSettings) -> ConversationThreads:
    return ConversationThreads(settings)


def make_thread(description: str, start: int = 0, end: int = 1) -> Thread:
    return Thread(
        description=description,
        ranges=[
            TextRange(start=TextLocation(start), end=TextLocation(end)),
        ],
    )


@pytest.mark.asyncio
async def test_add_thread_appends(threads: ConversationThreads) -> None:
    await threads.add_thread(make_thread("topic one"))
    assert len(threads.threads) == 1
    assert threads.threads[0].description == "topic one"


@pytest.mark.asyncio
async def test_add_multiple_threads(threads: ConversationThreads) -> None:
    await threads.add_thread(make_thread("alpha"))
    await threads.add_thread(make_thread("beta"))
    await threads.add_thread(make_thread("gamma"))
    assert len(threads.threads) == 3


@pytest.mark.asyncio
async def test_clear_resets_state(threads: ConversationThreads) -> None:
    await threads.add_thread(make_thread("something"))
    threads.clear()
    assert len(threads.threads) == 0
    assert len(threads.vector_base) == 0


@pytest.mark.asyncio
async def test_build_index_rebuilds_from_threads(threads: ConversationThreads) -> None:
    # Manually add threads without building the vector index.
    t1 = make_thread("python programming")
    t2 = make_thread("data science")
    threads.threads.append(t1)
    threads.threads.append(t2)
    # build_index should embed all existing threads.
    await threads.build_index()
    assert len(threads.vector_base) == 2


@pytest.mark.asyncio
async def test_serialize_roundtrip(threads: ConversationThreads) -> None:
    await threads.add_thread(make_thread("episode one", 0, 5))
    await threads.add_thread(make_thread("episode two", 5, 10))

    data = threads.serialize()
    assert "threads" in data
    thread_list = data["threads"]
    assert thread_list is not None
    assert len(thread_list) == 2

    # Deserialize into a fresh instance.
    settings = TextEmbeddingIndexSettings(create_test_embedding_model())
    fresh = ConversationThreads(settings)
    fresh.deserialize(data)
    assert len(fresh.threads) == 2
    assert fresh.threads[0].description == "episode one"
    assert fresh.threads[1].description == "episode two"


@pytest.mark.asyncio
async def test_deserialize_empty_data(threads: ConversationThreads) -> None:
    data: ConversationThreadData = {}  # type: ignore[typeddict-item]
    threads.deserialize(data)
    assert len(threads.threads) == 0


@pytest.mark.asyncio
async def test_serialize_without_embeddings(threads: ConversationThreads) -> None:
    # Add a thread without going through add_thread (so no embedding yet).
    threads.threads.append(make_thread("bare thread"))
    data = threads.serialize()
    thread_list = data["threads"]
    assert thread_list is not None
    assert len(thread_list) == 1
    # Embedding may be None because vector_base has no entries for this slot.
    assert thread_list[0]["embedding"] is None or isinstance(
        thread_list[0]["embedding"], list
    )


@pytest.mark.asyncio
async def test_lookup_thread_returns_matches(threads: ConversationThreads) -> None:
    await threads.add_thread(make_thread("machine learning and AI"))
    await threads.add_thread(make_thread("cooking recipes"))
    results = await threads.lookup_thread("artificial intelligence")
    assert len(results) > 0
    assert results[0].thread_ordinal == 0  # ordinal of the matching thread


@pytest.mark.asyncio
async def test_lookup_thread_empty_index(threads: ConversationThreads) -> None:
    results = await threads.lookup_thread("anything")
    assert results == []
