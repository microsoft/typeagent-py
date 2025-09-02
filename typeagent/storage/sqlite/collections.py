# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based collection implementations."""

import json
import sqlite3
import typing

from .schema import ShreddedMessage, ShreddedSemanticRef
from ...knowpro import interfaces
from ...knowpro import serialization


class SqliteMessageCollection[TMessage: interfaces.IMessage](
    interfaces.IMessageCollection[TMessage]
):
    """SQLite-backed message collection."""

    def __init__(
        self, db: sqlite3.Connection, message_type: type[TMessage] | None = None
    ):
        self.db = db
        self.message_type = message_type

    @property
    def is_persistent(self) -> bool:
        return True

    async def size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM Messages")
        return cursor.fetchone()[0]

    def __aiter__(self) -> typing.AsyncGenerator[TMessage, None]:
        return self._async_iterator()

    async def _async_iterator(self) -> typing.AsyncGenerator[TMessage, None]:
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
            FROM Messages ORDER BY msg_id
            """
        )
        for row in cursor:
            message = self._deserialize_message_from_row(row)
            yield message

    def _deserialize_message_from_row(self, row: ShreddedMessage) -> TMessage:
        """Rehydrate a message from database row columns."""
        (
            chunks_json,
            chunk_uri,
            start_timestamp,
            tags_json,
            metadata_json,
            extra_json,
        ) = row

        # Parse JSON fields and build a JSON object using camelCase.
        message_data = json.loads(extra_json) if extra_json else {}
        message_data["textChunks"] = json.loads(chunks_json) if chunks_json else []
        message_data["timestamp"] = start_timestamp
        message_data["tags"] = json.loads(tags_json) if tags_json else []
        message_data["metadata"] = json.loads(metadata_json) if metadata_json else {}

        # The serialization.deserialize_object will convert to snake_case Python attributes.
        if self.message_type is None:
            raise ValueError(
                "Deserialization requires message_type passed to SqliteMessageCollection"
            )
        return serialization.deserialize_object(self.message_type, message_data)

    def _serialize_message_to_row(self, message: TMessage) -> ShreddedMessage:
        """Shred a message object into database columns."""
        # Serialize the message to JSON first (this uses camelCase)
        message_data = serialization.serialize_object(message)

        # Extract shredded fields (JSON uses camelCase)
        chunks_json = json.dumps(message_data.pop("textChunks", []))
        chunk_uri = None  # For now, we're not using chunk URIs
        start_timestamp = message_data.pop("timestamp", None)
        tags_json = json.dumps(message_data.pop("tags", []))
        metadata_json = json.dumps(message_data.pop("metadata", {}))

        # What's left in message_data becomes 'extra'.
        extra_json = json.dumps(message_data) if message_data else None

        return (
            chunks_json,
            chunk_uri,
            start_timestamp,
            tags_json,
            metadata_json,
            extra_json,
        )

    async def get_item(self, arg: int) -> TMessage:
        if not isinstance(arg, int):
            raise TypeError(f"Index must be an int, not {type(arg).__name__}")
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
            FROM Messages WHERE msg_id = ?
            """,
            (arg,),
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_message_from_row(row)
        raise IndexError("Message not found")

    async def get_slice(self, start: int, stop: int) -> list[TMessage]:
        if stop <= start:
            return []
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
            FROM Messages WHERE msg_id >= ? AND msg_id < ? ORDER BY msg_id
            """,
            (start, stop),
        )
        rows = cursor.fetchall()
        return [self._deserialize_message_from_row(row) for row in rows]

    async def get_multiple(self, arg: list[int]) -> list[TMessage]:
        results = []
        for i in arg:
            results.append(await self.get_item(i))
        return results

    async def append(self, item: TMessage) -> None:
        with self.db:
            cursor = self.db.cursor()
            (
                chunks_json,
                chunk_uri,
                start_timestamp,
                tags_json,
                metadata_json,
                extra_json,
            ) = self._serialize_message_to_row(item)
            # Use the current size as the ID to maintain 0-based indexing like the old implementation
            msg_id = await self.size()
            cursor.execute(
                """
                INSERT INTO Messages (msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg_id,
                    chunks_json,
                    chunk_uri,
                    start_timestamp,
                    tags_json,
                    metadata_json,
                    extra_json,
                ),
            )

    async def extend(self, items: typing.Iterable[TMessage]) -> None:
        with self.db:
            cursor = self.db.cursor()
            current_size = await self.size()
            for msg_id, item in enumerate(items, current_size):
                (
                    chunks_json,
                    chunk_uri,
                    start_timestamp,
                    tags_json,
                    metadata_json,
                    extra_json,
                ) = self._serialize_message_to_row(item)
                cursor.execute(
                    """
                    INSERT INTO Messages (msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        msg_id,
                        chunks_json,
                        chunk_uri,
                        start_timestamp,
                        tags_json,
                        metadata_json,
                        extra_json,
                    ),
                )


class SqliteSemanticRefCollection(interfaces.ISemanticRefCollection):
    """SQLite-backed semantic reference collection."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def _deserialize_semantic_ref_from_row(
        self, row: ShreddedSemanticRef
    ) -> interfaces.SemanticRef:
        """Deserialize a semantic ref from database row columns."""
        semref_id, range_json, knowledge_type, knowledge_json = row

        # Build semantic ref data using camelCase (JSON format)
        semantic_ref_data = interfaces.SemanticRefData(
            semanticRefOrdinal=semref_id,
            range=json.loads(range_json),
            knowledgeType=knowledge_type,  # type: ignore
            knowledge=json.loads(knowledge_json),
        )

        return interfaces.SemanticRef.deserialize(semantic_ref_data)

    def _serialize_semantic_ref_to_row(
        self, semantic_ref: interfaces.SemanticRef
    ) -> ShreddedSemanticRef:
        """Serialize a semantic ref object into database columns."""
        # Serialize the semantic ref to JSON first (this uses camelCase)
        semantic_ref_data = semantic_ref.serialize()

        # Extract shredded fields (JSON uses camelCase)
        semref_id = semantic_ref_data["semanticRefOrdinal"]
        range_json = json.dumps(semantic_ref_data["range"])
        knowledge_type = semantic_ref_data["knowledgeType"]
        knowledge_json = json.dumps(semantic_ref_data["knowledge"])

        return (semref_id, range_json, knowledge_type, knowledge_json)

    @property
    def is_persistent(self) -> bool:
        return True

    async def size(self) -> int:
        return self._size()

    def _size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute("SELECT COUNT(*) FROM SemanticRefs")
        return cursor.fetchone()[0]

    async def __aiter__(self) -> typing.AsyncGenerator[interfaces.SemanticRef, None]:
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT semref_id, range_json, knowledge_type, knowledge_json
            FROM SemanticRefs ORDER BY semref_id
            """
        )
        for row in cursor:
            yield self._deserialize_semantic_ref_from_row(row)

    async def get_item(self, arg: int) -> interfaces.SemanticRef:
        if not isinstance(arg, int):
            raise TypeError(f"Index must be an int, not {type(arg).__name__}")
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT semref_id, range_json, knowledge_type, knowledge_json
            FROM SemanticRefs WHERE semref_id = ?
            """,
            (arg,),
        )
        row = cursor.fetchone()
        if row:
            return self._deserialize_semantic_ref_from_row(row)
        raise IndexError("SemanticRef not found")

    async def get_slice(self, start: int, stop: int) -> list[interfaces.SemanticRef]:
        if stop <= start:
            return []
        cursor = self.db.cursor()
        cursor.execute(
            """
            SELECT semref_id, range_json, knowledge_type, knowledge_json
            FROM SemanticRefs WHERE semref_id >= ? AND semref_id < ?
            ORDER BY semref_id
            """,
            (start, stop),
        )
        rows = cursor.fetchall()
        return [self._deserialize_semantic_ref_from_row(row) for row in rows]

    async def get_multiple(self, arg: list[int]) -> list[interfaces.SemanticRef]:
        # TODO: Do we really want to support this?
        # If so, we should probably try to optimize it.
        results = []
        for i in arg:
            results.append(await self.get_item(i))
        return results

    async def append(self, item: interfaces.SemanticRef) -> None:
        with self.db:
            cursor = self.db.cursor()
            semref_id, range_json, knowledge_type, knowledge_json = (
                self._serialize_semantic_ref_to_row(item)
            )
            cursor.execute(
                """
                INSERT INTO SemanticRefs (semref_id, range_json, knowledge_type, knowledge_json)
                VALUES (?, ?, ?, ?)
                """,
                (semref_id, range_json, knowledge_type, knowledge_json),
            )

    async def extend(self, items: typing.Iterable[interfaces.SemanticRef]) -> None:
        with self.db:
            cursor = self.db.cursor()
            for item in items:
                semref_id, range_json, knowledge_type, knowledge_json = (
                    self._serialize_semantic_ref_to_row(item)
                )
                cursor.execute(
                    """
                    INSERT INTO SemanticRefs (semref_id, range_json, knowledge_type, knowledge_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (semref_id, range_json, knowledge_type, knowledge_json),
                )
