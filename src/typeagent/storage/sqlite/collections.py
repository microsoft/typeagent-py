# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based collection implementations."""

import json
import sqlite3
import typing

from ...knowpro import interfaces, serialization
from .schema import ShreddedMessage, ShreddedSemanticRef


class SqliteMessageCollection[TMessage: interfaces.IMessage](
    interfaces.IMessageCollection[TMessage]
):
    """SQLite-backed message collection."""

    def __init__(
        self,
        db: sqlite3.Connection,
        message_type: type[TMessage] | None = None,
        message_text_index: "interfaces.IMessageTextIndex[TMessage] | None" = None,
    ):
        self.db = db
        self.message_type = message_type
        self.message_text_index = message_text_index

    def set_message_text_index(
        self, message_text_index: "interfaces.IMessageTextIndex[TMessage]"
    ) -> None:
        """Set the message text index for automatic indexing of new messages."""
        self.message_text_index = message_text_index

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
        cursor.execute("""
            SELECT chunks, chunk_uri, start_timestamp, tags, metadata, extra
            FROM Messages ORDER BY msg_id
            """)
        for row in cursor:
            message = self._deserialize_message_from_row(row)
            yield message

    def _deserialize_message_from_row(self, row: ShreddedMessage) -> TMessage:
        """Rehydrate a message from database row columns."""
        (
            chunks_json,
            _chunk_uri,
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
        size = await self.size()
        if not all((0 <= i < size) for i in arg):
            raise IndexError("One or more Message indices are out of bounds")
        if len(arg) < 2:
            return [await self.get_item(i) for i in arg]
        cursor = self.db.cursor()
        cursor.execute(
            f"""
            SELECT msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra
            FROM Messages WHERE msg_id IN {tuple(arg)}
            """,
        )
        rows = cursor.fetchall()
        rowdict = {}
        for row in rows:
            msg_id, row = row[0], row[1:]
            rowdict[msg_id] = row
        assert set(rowdict) == set(arg)
        return [self._deserialize_message_from_row(rowdict[i]) for i in arg]

    async def append(self, item: TMessage) -> None:
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

        # Also add to message text index if available
        if self.message_text_index is not None:
            await self.message_text_index.add_messages_starting_at(msg_id, [item])

    async def extend(self, items: typing.Iterable[TMessage]) -> None:
        items_list = list(items)  # Convert to list to iterate twice
        if not items_list:
            return

        # Get the starting ordinal before adding any messages
        current_size = await self.size()

        # Prepare all insertion data for bulk operation
        insertion_data = []
        for msg_id, item in enumerate(items_list, current_size):
            (
                chunks_json,
                chunk_uri,
                start_timestamp,
                tags_json,
                metadata_json,
                extra_json,
            ) = self._serialize_message_to_row(item)
            insertion_data.append(
                (
                    msg_id,
                    chunks_json,
                    chunk_uri,
                    start_timestamp,
                    tags_json,
                    metadata_json,
                    extra_json,
                )
            )

        # Bulk insert all messages
        cursor = self.db.cursor()
        if insertion_data:
            cursor.executemany(
                """
                INSERT INTO Messages (msg_id, chunks, chunk_uri, start_timestamp, tags, metadata, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                insertion_data,
            )

        # Also add to message text index if available
        if self.message_text_index is not None:
            await self.message_text_index.add_messages_starting_at(
                current_size, items_list
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
        cursor.execute("""
            SELECT semref_id, range_json, knowledge_type, knowledge_json
            FROM SemanticRefs ORDER BY semref_id
            """)
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
        size = await self.size()
        if not all((0 <= i < size) for i in arg):
            raise IndexError("One or more SemanticRef indices are out of bounds")
        if len(arg) < 2:
            return [await self.get_item(ordinal) for ordinal in arg]
        cursor = self.db.cursor()
        cursor.execute(f"""
            SELECT semref_id, range_json, knowledge_type, knowledge_json
            FROM SemanticRefs WHERE semref_id IN {tuple(arg)}
            """)
        rows = cursor.fetchall()
        rowdict = {row[0]: row for row in rows}
        assert set(rowdict) == set(arg)
        return [self._deserialize_semantic_ref_from_row(rowdict[ordl]) for ordl in arg]

    async def append(self, item: interfaces.SemanticRef) -> None:
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
        items_list = list(items)
        if not items_list:
            return

        # Prepare all insertion data for bulk operation
        insertion_data = []
        for item in items_list:
            semref_id, range_json, knowledge_type, knowledge_json = (
                self._serialize_semantic_ref_to_row(item)
            )
            insertion_data.append(
                (semref_id, range_json, knowledge_type, knowledge_json)
            )

        # Bulk insert all semantic refs
        cursor = self.db.cursor()
        if insertion_data:
            cursor.executemany(
                """
                INSERT INTO SemanticRefs (semref_id, range_json, knowledge_type, knowledge_json)
                VALUES (?, ?, ?, ?)
                """,
                insertion_data,
            )
