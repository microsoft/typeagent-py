# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
import os
import tempfile

import pytest

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
    DefaultSerializer,
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
        knowledge_type="topic",
        knowledge=topic,
    )


@pytest.fixture
def temp_db_path():
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_sqlite_storage_provider_message_collection(temp_db_path):
    provider = SqliteStorageProvider(temp_db_path)
    collection = provider.create_message_collection(DummyMessage)
    assert collection.is_persistent
    assert len(collection) == 0

    msg = DummyMessage(["hello"])
    collection.append(msg)
    assert len(collection) == 1
    # get_item and __iter__
    loaded = collection.get_item(0)
    assert isinstance(loaded, DummyMessage)
    assert loaded.text_chunks == ["hello"]
    assert list(collection)[0].text_chunks == ["hello"]
    collection.append(DummyMessage(["world"]))
    collection.append(DummyMessage(["foo", "bar"]))
    assert len(collection) == 3
    # slice
    assert [msg.text_chunks[0] for msg in collection.get_slice(1, 3)] == [
        "world",
        "foo",
    ]
    # multiple get
    assert [msg.text_chunks[0] for msg in collection.get_multiple([0, 2])] == [
        "hello",
        "foo",
    ]


def test_sqlite_storage_provider_semantic_ref_collection(temp_db_path):
    provider = SqliteStorageProvider(temp_db_path)
    collection = provider.create_semantic_ref_collection()
    assert collection.is_persistent
    assert len(collection) == 0

    # Create a dummy SemanticRef
    ref = make_dummy_semantic_ref()

    collection.append(ref)
    assert len(collection) == 1
    loaded = collection.get_item(0)
    assert isinstance(loaded, SemanticRef)
    assert loaded.semantic_ref_ordinal == 0
    assert list(collection)[0].semantic_ref_ordinal == 0


def test_default_serializer_roundtrip():
    serializer = DefaultSerializer(DummyMessage)
    msg = DummyMessage(["test"])
    json_obj = serializer.serialize(msg)
    assert isinstance(json_obj, dict)  # Should return a JSON object, not a string
    msg2 = serializer.deserialize(json_obj)
    assert isinstance(msg2, DummyMessage)
    assert msg2.text_chunks == ["test"]


def test_sqlite_message_collection_append_and_get(temp_db_path):
    db = SqliteStorageProvider(temp_db_path).get_db()
    serializer = DefaultSerializer(DummyMessage)
    store = SqliteMessageCollection(db, serializer)
    msg = DummyMessage(["foo"])
    store.append(msg)
    assert len(store) == 1
    loaded = store.get_item(0)
    assert loaded.text_chunks == ["foo"]
    with pytest.raises(IndexError):
        _ = store.get_item(999)
    with pytest.raises(TypeError):
        _ = store.get_item("bad")  # type: ignore  # Tests runtime behavior


def test_sqlite_message_collection_iter(temp_db_path):
    db = SqliteStorageProvider(temp_db_path).get_db()
    serializer = DefaultSerializer(DummyMessage)
    store = SqliteMessageCollection(db, serializer)
    msgs = [DummyMessage([f"msg{i}"]) for i in range(3)]
    for m in msgs:
        store.append(m)
    assert [m.text_chunks[0] for m in store] == ["msg0", "msg1", "msg2"]


def test_sqlite_semantic_ref_collection_append_and_get(temp_db_path):
    db = SqliteStorageProvider(temp_db_path).get_db()
    collection = SqliteSemanticRefCollection(db)
    ref = make_dummy_semantic_ref(123)
    collection.append(ref)
    assert len(collection) == 1
    loaded = collection.get_item(123)
    assert loaded.semantic_ref_ordinal == 123
    with pytest.raises(IndexError):
        _ = collection.get_item(999)
    with pytest.raises(TypeError):
        _ = collection.get_item("bad")  # type: ignore  # Tests runtime behavior


def test_sqlite_semantic_ref_collection_iter(temp_db_path):
    db = SqliteStorageProvider(temp_db_path).get_db()
    collection = SqliteSemanticRefCollection(db)
    refs = [make_dummy_semantic_ref(i) for i in range(2)]
    for r in refs:
        collection.append(r)
    assert [r.semantic_ref_ordinal for r in collection] == [0, 1]
