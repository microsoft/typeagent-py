# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import sqlite3
import typing

from ..knowpro import interfaces
from ..knowpro import serialization
from ..knowpro.semrefindex import TermToSemanticRefIndex
from ..knowpro.propindex import PropertyIndex
from ..knowpro.timestampindex import TimestampToTextRangeIndex
from ..knowpro.messageindex import MessageTextIndex, MessageTextIndexSettings
from ..knowpro.reltermsindex import RelatedTermsIndex, RelatedTermIndexSettings
from ..knowpro.convthreads import ConversationThreads


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

    def serialize(self, value: TMessage) -> dict[str, typing.Any] | list[typing.Any]:
        return serialization.serialize_object(value)

    def deserialize(self, data: dict[str, typing.Any] | list[typing.Any]) -> TMessage:
        return serialization.deserialize_object(self.cls, data)


class SqliteMessageCollection[TMessage: interfaces.IMessage](
    interfaces.IMessageCollection
):
    def __init__(
        self, db: sqlite3.Connection, serializer: interfaces.JsonSerializer[TMessage]
    ):
        self.db = db
        self._deserialize_message = serializer.deserialize
        self._serialize_message = serializer.serialize

    @property
    def is_persistent(self) -> bool:
        return True

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Messages")
        return cursor.fetchone()[0]

    def __aiter__(self) -> typing.AsyncIterator[TMessage]:
        return self._async_iterator()

    async def _async_iterator(self) -> typing.AsyncIterator[TMessage]:
        cursor = self.db.cursor()
        cursor.execute("SELECT msgdata FROM Messages")
        for row in cursor:
            json_data = json.loads(row[0])
            yield self._deserialize_message(json_data)
            # Potentially add await asyncio.sleep(0) here to yield control

    async def get_item(self, arg: int) -> TMessage:
        if not isinstance(arg, int):
            raise TypeError(f"Index must be an int, not {type(arg).__name__}")
        cursor = self.db.cursor()
        cursor.execute("SELECT msgdata FROM Messages WHERE id = ?", (arg,))
        row = cursor.fetchone()
        if row:
            json_data = json.loads(row[0])
            return self._deserialize_message(json_data)
        raise IndexError("Message not found")

    async def get_slice(self, start: int, stop: int) -> list[TMessage]:
        if stop <= start:
            return []
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT msgdata FROM Messages WHERE id >= ? AND id < ?",
            (start, stop),
        )
        rows = cursor.fetchall()
        return [self._deserialize_message(json.loads(row[0])) for row in rows]

    async def get_multiple(self, arg: list[int]) -> list[TMessage]:
        results = []
        for i in arg:
            results.append(await self.get_item(i))
        return results

    async def append(self, item: TMessage) -> None:
        cursor = self.db.cursor()
        json_obj = self._serialize_message(item)
        serialized_message = json.dumps(json_obj)
        cursor.execute(
            "INSERT INTO Messages (id, msgdata) VALUES (?, ?)",
            (await self.size(), serialized_message),
        )

    async def extend(self, items: typing.Iterable[TMessage]) -> None:
        self.db.commit()
        cursor = self.db.cursor()
        current_size = await self.size()
        for ord, item in enumerate(items, current_size):
            json_obj = self._serialize_message(item)
            serialized_message = json.dumps(json_obj)
            cursor.execute(
                "INSERT INTO Messages (id, msgdata) VALUES (?, ?)",
                (ord, serialized_message),
            )
        self.db.commit()


class SqliteSemanticRefCollection(interfaces.ISemanticRefCollection):
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def _serialize(self, sem_ref: interfaces.SemanticRef) -> dict[str, typing.Any]:
        return sem_ref.serialize()  # type: ignore[return-value]

    def _deserialize(self, data: dict[str, typing.Any]) -> interfaces.SemanticRef:
        return interfaces.SemanticRef.deserialize(data)  # type: ignore[arg-type]

    @property
    def is_persistent(self) -> bool:
        return True

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM SemanticRefs")
        return cursor.fetchone()[0]

    async def __aiter__(self) -> typing.AsyncIterator[interfaces.SemanticRef]:
        cursor = self.db.cursor()
        cursor.execute("SELECT srdata FROM SemanticRefs")
        for row in cursor:
            json_obj = json.loads(row[0])
            yield self._deserialize(json_obj)

    async def get_item(self, arg: int) -> interfaces.SemanticRef:
        if not isinstance(arg, int):
            raise TypeError(f"Index must be an int, not {type(arg).__name__}")
        cursor = self.db.cursor()
        cursor.execute("SELECT srdata FROM SemanticRefs WHERE id = ?", (arg,))
        row = cursor.fetchone()
        if row:
            json_obj = json.loads(row[0])
            return self._deserialize(json_obj)
        raise IndexError("SemanticRef not found")

    async def get_slice(self, start: int, stop: int) -> list[interfaces.SemanticRef]:
        if stop <= start:
            return []
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT srdata FROM SemanticRefs WHERE id >= ? AND id < ?",
            (start, stop),
        )
        return [self._deserialize(json.loads(row[0])) for row in cursor.fetchall()]

    async def get_multiple(self, arg: list[int]) -> list[interfaces.SemanticRef]:
        # TODO: Do we really want to support this?
        # If so, we should probably try to optimize it.
        results = []
        for i in arg:
            results.append(await self.get_item(i))
        return results

    async def append(self, item: interfaces.SemanticRef) -> None:
        cursor = self.db.cursor()
        json_obj = self._serialize(item)
        serialized_message = json.dumps(json_obj)
        cursor.execute(
            "INSERT INTO SemanticRefs (id, srdata) VALUES (?, ?)",
            (item.semantic_ref_ordinal, serialized_message),
        )
        self.db.commit()

    async def extend(self, items: typing.Iterable[interfaces.SemanticRef]) -> None:
        cursor = self.db.cursor()
        for item in items:
            json_obj = self._serialize(item)
            serialized_message = json.dumps(json_obj)
            cursor.execute(
                "INSERT INTO SemanticRefs (id, srdata) VALUES (?, ?)",
                (item.semantic_ref_ordinal, serialized_message),
            )
        self.db.commit()


class SqliteStorageProvider[TMessage: interfaces.IMessage](
    interfaces.IStorageProvider[TMessage]
):
    """A storage provider backed by SQLite.

    NOTE: You can create only one message collection
    and one semantic ref collection per provider.
    For now, indexes are stored in memory (not persisted to SQLite).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db: sqlite3.Connection | None = None
        # In-memory indexes for now - TODO: persist to SQLite
        self._conversation_index: TermToSemanticRefIndex | None = None
        self._property_index: PropertyIndex | None = None
        self._timestamp_index: TimestampToTextRangeIndex | None = None
        self._message_text_index: interfaces.IMessageTextIndex[TMessage] | None = None
        self._related_terms_index: RelatedTermsIndex | None = None
        self._conversation_threads: ConversationThreads | None = None

    @classmethod
    async def create(
        cls,
        message_text_settings: MessageTextIndexSettings,
        related_terms_settings: RelatedTermIndexSettings,
        db_path: str,
    ) -> "SqliteStorageProvider[TMessage]":
        """Create and initialize a SqliteStorageProvider with all indexes."""
        instance = cls(db_path)
        # Initialize all indexes to ensure they exist in memory
        instance._conversation_index = TermToSemanticRefIndex()
        instance._property_index = PropertyIndex()
        instance._timestamp_index = TimestampToTextRangeIndex()

        # Use the provided settings instead of creating new ones
        instance._message_text_index = MessageTextIndex(message_text_settings)
        instance._related_terms_index = RelatedTermsIndex(related_terms_settings)
        instance._conversation_threads = ConversationThreads(
            related_terms_settings.embedding_index_settings
        )
        return instance

    async def close(self) -> None:
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

    async def get_message_collection(
        self,
        serializer: interfaces.JsonSerializer[TMessage] | type[TMessage],
    ) -> SqliteMessageCollection[TMessage]:
        if not isinstance(serializer, interfaces.JsonSerializer):
            serializer = DefaultSerializer[TMessage](serializer)
        return SqliteMessageCollection[TMessage](self.get_db(), serializer)

    async def get_semantic_ref_collection(self) -> interfaces.ISemanticRefCollection:
        return SqliteSemanticRefCollection(self.get_db())

    # Index getter methods
    async def get_conversation_index(self) -> interfaces.ITermToSemanticRefIndex:
        assert (
            self._conversation_index is not None
        ), "Use SqliteStorageProvider.create() to create an initialized instance"
        return self._conversation_index

    async def get_property_index(self) -> interfaces.IPropertyToSemanticRefIndex:
        assert (
            self._property_index is not None
        ), "Use SqliteStorageProvider.create() to create an initialized instance"
        return self._property_index

    async def get_timestamp_index(self) -> interfaces.ITimestampToTextRangeIndex:
        assert (
            self._timestamp_index is not None
        ), "Use SqliteStorageProvider.create() to create an initialized instance"
        return self._timestamp_index

    async def get_message_text_index(self) -> interfaces.IMessageTextIndex[TMessage]:
        assert (
            self._message_text_index is not None
        ), "Use SqliteStorageProvider.create() to create an initialized instance"
        return self._message_text_index

    async def get_related_terms_index(self) -> interfaces.ITermToRelatedTermsIndex:
        assert (
            self._related_terms_index is not None
        ), "Use SqliteStorageProvider.create() to create an initialized instance"
        return self._related_terms_index

    async def get_conversation_threads(self) -> interfaces.IConversationThreads:
        assert (
            self._conversation_threads is not None
        ), "Use SqliteStorageProvider.create() to create an initialized instance"
        return self._conversation_threads
