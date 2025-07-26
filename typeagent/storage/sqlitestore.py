# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sqlite3
import typing

from ..knowpro.storage import MemoryStorageProvider
from ..knowpro import interfaces
from ..knowpro import serialization


MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS Messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    msgdata TEXT NOT NULL
);
"""

SEMANTIC_REFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS SemanticRefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    srdata TEXT NOT NULL
);
"""


class DefaultSerializer[TMessage: interfaces.IMessage](interfaces.JsonSerializer):
    def __init__(self, cls: type[TMessage]):
        self.cls = cls

    def serialize(self, value: TMessage) -> str:
        return json.dumps(serialization.serialize_object(value))

    def deserialize(self, data: str) -> TMessage:
        return serialization.deserialize_object(self.cls, json.loads(data))


class SqliteMessageCollection[TMessage: interfaces.IMessage](
    interfaces.IMessageCollection
):
    def __init__(
        self, db: sqlite3.Connection, serializer: interfaces.JsonSerializer[TMessage]
    ):
        self.db = db
        self._deserialize = serializer.deserialize
        self._serialize = serializer.serialize

    @property
    def is_persistent(self) -> bool:
        return True

    def __len__(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Messages")
        return cursor.fetchone()[0]

    def __iter__(self) -> typing.Iterator[TMessage]:
        cursor = self.db.cursor()
        cursor.execute("SELECT msgdata FROM Messages")
        for row in cursor:
            yield self._deserialize(row[0])

    @typing.overload
    def __getitem__(self, arg: int) -> TMessage: ...
    @typing.overload
    def __getitem__(self, arg: slice) -> list[TMessage]: ...
    @typing.overload
    def __getitem__(self, arg: list[int]) -> list[TMessage]: ...

    def __getitem__(self, arg: int | list[int] | slice) -> TMessage | list[TMessage]:
        cursor = self.db.cursor()
        if isinstance(arg, int):
            cursor.execute("SELECT msgdata FROM Messages WHERE id = ?", (arg,))
            row = cursor.fetchone()
            if row:
                return self._deserialize(row[0])
            raise IndexError("Message not found")
        elif isinstance(arg, list):
            # TODO: Do we really want to support this?
            # If so, we should probably try to optimize it.
            return [self[i] for i in arg]
        elif isinstance(arg, slice):
            start, stop, step = arg.indices(999_999_999)  # Avoid len()
            if step not in (None, 1):
                raise ValueError("Slice step must be 1")
            if stop <= start:
                return []
            cursor.execute(
                "SELECT msgdata FROM Messages WHERE id >= ? AND id < ?",
                (start, stop),
            )
            rows = cursor.fetchall()
            res = [self._deserialize(row[0]) for row in rows]
            return res
        else:
            raise TypeError("Index must be an int, list or slice")

    def append(self, item: TMessage) -> None:
        cursor = self.db.cursor()
        serialized_message = self._serialize(item)
        cursor.execute(
            "INSERT INTO Messages (id, msgdata) VALUES (?, ?)",
            (len(self), serialized_message),
        )

    def extend(self, items: typing.Iterable[TMessage]) -> None:
        self.db.commit()
        cursor = self.db.cursor()
        for ord, item in enumerate(items, len(self)):
            serialized_message = self._serialize(item)
            cursor.execute(
                "INSERT INTO Messages (id, msgdata) VALUES (?, ?)",
                (ord, serialized_message),
            )
        self.db.commit()


class SqliteSemanticRefCollection(interfaces.ISemanticRefCollection):
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def _serialize(self, sem_ref: interfaces.SemanticRef) -> str:
        return json.dumps(sem_ref.serialize())

    def _deserialize(self, data: str) -> interfaces.SemanticRef:
        return interfaces.SemanticRef.deserialize(json.loads(data))

    @property
    def is_persistent(self) -> bool:
        return True

    def __len__(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM SemanticRefs")
        return cursor.fetchone()[0]

    def __iter__(self) -> typing.Iterator[interfaces.SemanticRef]:
        cursor = self.db.cursor()
        cursor.execute("SELECT srdata FROM SemanticRefs")
        for row in cursor:
            yield self._deserialize(row[0])

    # NOTE: Indexing and slicing are weird since unique ids start at 1.
    @typing.overload
    def __getitem__(self, arg: int) -> interfaces.SemanticRef: ...
    @typing.overload
    def __getitem__(self, arg: slice) -> list[interfaces.SemanticRef]: ...
    @typing.overload
    def __getitem__(self, arg: list[int]) -> list[interfaces.SemanticRef]: ...

    def __getitem__(
        self, arg: int | list[int] | slice
    ) -> interfaces.SemanticRef | list[interfaces.SemanticRef]:
        cursor = self.db.cursor()
        if isinstance(arg, int):
            cursor.execute("SELECT srdata FROM SemanticRefs WHERE id = ?", (arg,))
            row = cursor.fetchone()
            if row:
                return self._deserialize(row[0])
            raise IndexError("SemanticRef not found")
        elif isinstance(arg, list):
            # TODO: Do we really want to support this?
            # If so, we should probably try to optimize it.
            return [self[i] for i in arg]
        elif isinstance(arg, slice):
            start, stop, step = arg.indices(999_999_999)  # Avoid len()
            if step not in (None, 1):
                raise ValueError("Slice step must be 1")
            if stop <= start:
                return []
            cursor.execute(
                "SELECT srdata FROM SemanticRefs WHERE id >= ? AND id < ?",
                (start, stop),
            )
            return [self._deserialize(row[0]) for row in cursor.fetchall()]
        else:
            raise TypeError("Index must be an int, list or slice")

    def append(self, item: interfaces.SemanticRef) -> None:
        cursor = self.db.cursor()
        serialized_message = self._serialize(item)
        cursor.execute(
            "INSERT INTO SemanticRefs (id, srdata) VALUES (?, ?)",
            (item.semantic_ref_ordinal, serialized_message),
        )
        self.db.commit()

    def extend(self, items: typing.Iterable[interfaces.SemanticRef]) -> None:
        cursor = self.db.cursor()
        for item in items:
            serialized_message = self._serialize(item)
            cursor.execute(
                "INSERT INTO SemanticRefs (id, srdata) VALUES (?, ?)",
                (item.semantic_ref_ordinal, serialized_message),
            )
        self.db.commit()


class SqliteStorageProvider(interfaces.IStorageProvider):
    """A storage provider backed by SQLite.

    NOTE: You can create only one message collection
    and one semantic ref collection per provider.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db: sqlite3.Connection | None = None

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None

    def get_db(self) -> sqlite3.Connection:
        if self.db is None:
            self.db = sqlite3.connect(self.db_path)
            self.db.execute(MESSAGES_SCHEMA)
            self.db.execute(SEMANTIC_REFS_SCHEMA)
            self.db.commit()
        return self.db

    def create_message_collection[TMessage: interfaces.IMessage](
        self,
        serializer: interfaces.JsonSerializer[TMessage] | type[TMessage] | None = None,
    ) -> SqliteMessageCollection[TMessage]:
        if serializer is None:
            raise ValueError("serializer must not be None")
        if not isinstance(serializer, interfaces.JsonSerializer):
            serializer = DefaultSerializer[TMessage](serializer)
        return SqliteMessageCollection[TMessage](self.get_db(), serializer)

    def create_semantic_ref_collection(self) -> interfaces.ISemanticRefCollection:
        return SqliteSemanticRefCollection(self.get_db())


def get_storage_provider(dbname: str | None = None) -> interfaces.IStorageProvider:
    if dbname is None:
        return MemoryStorageProvider()
    else:
        return SqliteStorageProvider(dbname)
