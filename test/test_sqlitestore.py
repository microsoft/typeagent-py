# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import field
import os
import tempfile
from typing import Generator

import pytest
from pydantic.dataclasses import dataclass

from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.knowpro.interfaces import (
    IMessage,
    SemanticRef,
    TextLocation,
    TextRange,
    Topic,
)
from typeagent.knowpro.serialization import serialize_object, deserialize_object
from typeagent.storage.sqlitestore import (
    SqliteStorageProvider,
    SqliteMessageCollection,
    SqliteSemanticRefCollection,
)


# Dummy IMessage for testing
@dataclass
class DummyMessage(IMessage):
    text_chunks: list[str]
    tags: list[str] = field(default_factory=list)

    def get_knowledge(self) -> KnowledgeResponse:
        raise NotImplementedError("Should not be called")


def make_dummy_semantic_ref(ordinal: int = 0) -> SemanticRef:
    # Minimal valid Topic for knowledge
    topic = Topic(text="dummy_topic")
    # Minimal valid TextLocation and TextRange for range
    location = TextLocation(message_ordinal=0)
    text_range = TextRange(start=location)
    return SemanticRef(
        semantic_ref_ordinal=ordinal,
        range=text_range,
        knowledge=topic,
    )


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.asyncio
async def test_sqlite_storage_provider_message_collection(temp_db_path: str):
    provider = SqliteStorageProvider(temp_db_path)
    collection = await provider.get_message_collection(DummyMessage)
    assert collection.is_persistent
    assert await collection.size() == 0

    msg = DummyMessage(["hello"])
    await collection.append(msg)
    assert await collection.size() == 1
    # get_item and async iteration
    loaded = await collection.get_item(0)
    assert isinstance(loaded, DummyMessage)
    assert loaded.text_chunks == ["hello"]
    collection_list = [item async for item in collection]
    assert collection_list[0].text_chunks == ["hello"]
    await collection.append(DummyMessage(["world"]))
    await collection.append(DummyMessage(["foo", "bar"]))
    assert await collection.size() == 3
    # slice
    slice_result = await collection.get_slice(1, 3)
    assert [msg.text_chunks[0] for msg in slice_result] == [
        "world",
        "foo",
    ]
    # multiple get
    multiple_result = await collection.get_multiple([0, 2])
    assert [msg.text_chunks[0] for msg in multiple_result] == [
        "hello",
        "foo",
    ]


@pytest.mark.asyncio
async def test_sqlite_storage_provider_semantic_ref_collection(temp_db_path: str):
    provider = SqliteStorageProvider(temp_db_path)
    collection = await provider.get_semantic_ref_collection()
    assert collection.is_persistent
    assert await collection.size() == 0

    # Create a dummy SemanticRef
    ref = make_dummy_semantic_ref()

    await collection.append(ref)
    assert await collection.size() == 1
    loaded = await collection.get_item(0)
    assert isinstance(loaded, SemanticRef)
    assert loaded.semantic_ref_ordinal == 0
    collection_list = [item async for item in collection]
    assert collection_list[0].semantic_ref_ordinal == 0


@pytest.mark.asyncio
async def test_sqlite_message_collection_append_and_get(temp_db_path: str):
    db = SqliteStorageProvider(temp_db_path).get_db()
    store = SqliteMessageCollection(db, DummyMessage)
    msg = DummyMessage(["foo"])
    await store.append(msg)
    assert await store.size() == 1
    loaded = await store.get_item(0)
    assert loaded.text_chunks == ["foo"]
    with pytest.raises(IndexError):
        _ = await store.get_item(999)
    with pytest.raises(TypeError):
        _ = await store.get_item("bad")  # type: ignore  # Tests runtime behavior


@pytest.mark.asyncio
async def test_sqlite_message_collection_iter(temp_db_path: str):
    db = SqliteStorageProvider(temp_db_path).get_db()
    store = SqliteMessageCollection(db, DummyMessage)
    msgs = [DummyMessage([f"msg{i}"]) for i in range(3)]
    for m in msgs:
        await store.append(m)
    assert [m.text_chunks[0] async for m in store] == ["msg0", "msg1", "msg2"]


@pytest.mark.asyncio
async def test_sqlite_semantic_ref_collection_append_and_get(temp_db_path: str):
    db = SqliteStorageProvider(temp_db_path).get_db()
    collection = SqliteSemanticRefCollection(db)
    ref = make_dummy_semantic_ref(123)
    await collection.append(ref)
    assert await collection.size() == 1
    loaded = await collection.get_item(123)
    assert loaded.semantic_ref_ordinal == 123
    with pytest.raises(IndexError):
        _ = await collection.get_item(999)
    with pytest.raises(TypeError):
        _ = await collection.get_item("bad")  # type: ignore  # Tests runtime behavior


@pytest.mark.asyncio
async def test_sqlite_semantic_ref_collection_iter(temp_db_path: str):
    db = SqliteStorageProvider(temp_db_path).get_db()
    collection = SqliteSemanticRefCollection(db)
    refs = [make_dummy_semantic_ref(i) for i in range(2)]
    for r in refs:
        await collection.append(r)
    assert [r.semantic_ref_ordinal async for r in collection] == [0, 1]
