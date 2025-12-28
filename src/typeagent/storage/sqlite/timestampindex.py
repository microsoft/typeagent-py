# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based timestamp index implementation."""

import sqlite3

from ...knowpro import interfaces
from ...knowpro.universal_message import format_timestamp_utc


class SqliteTimestampToTextRangeIndex(interfaces.ITimestampToTextRangeIndex):
    """SQL-based timestamp index that queries Messages table directly."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    async def size(self) -> int:
        return self._size()

    def _size(self) -> int:
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM Messages WHERE start_timestamp IS NOT NULL"
        )
        return cursor.fetchone()[0]

    async def add_timestamp(
        self, message_ordinal: interfaces.MessageOrdinal, timestamp: str
    ) -> bool:
        return self._add_timestamp(message_ordinal, timestamp)

    def _add_timestamp(
        self, message_ordinal: interfaces.MessageOrdinal, timestamp: str
    ) -> bool:
        """Add timestamp to Messages table start_timestamp column."""
        cursor = self.db.cursor()
        cursor.execute(
            "UPDATE Messages SET start_timestamp = ? WHERE msg_id = ?",
            (timestamp, message_ordinal),
        )
        return cursor.rowcount > 0

    async def get_timestamp_ranges(
        self, start_timestamp: str, end_timestamp: str | None = None
    ) -> list[interfaces.TimestampedTextRange]:
        """Get timestamp ranges from Messages table."""
        cursor = self.db.cursor()

        if end_timestamp is None:
            # Single timestamp query
            cursor.execute(
                """
                SELECT msg_id, start_timestamp
                FROM Messages
                WHERE start_timestamp = ?
                ORDER BY msg_id
                """,
                (start_timestamp,),
            )
        else:
            # Range query
            cursor.execute(
                """
                SELECT msg_id, start_timestamp
                FROM Messages
                WHERE start_timestamp >= ? AND start_timestamp <= ?
                ORDER BY msg_id
                """,
                (start_timestamp, end_timestamp),
            )

        results = []
        for msg_id, timestamp in cursor.fetchall():
            # Create text range for message
            from ...knowpro.interfaces import TextLocation, TextRange

            text_range = TextRange(
                start=TextLocation(message_ordinal=msg_id, chunk_ordinal=0)
            )
            results.append(
                interfaces.TimestampedTextRange(range=text_range, timestamp=timestamp)
            )

        return results

    async def add_timestamps(
        self, message_timestamps: list[tuple[interfaces.MessageOrdinal, str]]
    ) -> None:
        """Add multiple timestamps."""
        cursor = self.db.cursor()
        for message_ordinal, timestamp in message_timestamps:
            cursor.execute(
                "UPDATE Messages SET start_timestamp = ? WHERE msg_id = ?",
                (timestamp, message_ordinal),
            )

    async def lookup_range(
        self, date_range: interfaces.DateRange
    ) -> list[interfaces.TimestampedTextRange]:
        """Lookup messages in a date range."""
        cursor = self.db.cursor()

        # Convert datetime objects to ISO format strings with Z suffix
        start_timestamp = format_timestamp_utc(date_range.start)
        end_timestamp = format_timestamp_utc(date_range.end) if date_range.end else None

        if date_range.end is None:
            # Point query
            cursor.execute(
                """
                SELECT msg_id, start_timestamp, chunks
                FROM Messages
                WHERE start_timestamp = ?
                ORDER BY msg_id
                """,
                (start_timestamp,),
            )
        else:
            # Range query
            cursor.execute(
                """
                SELECT msg_id, start_timestamp, chunks
                FROM Messages
                WHERE start_timestamp >= ? AND start_timestamp < ?
                ORDER BY msg_id
                """,
                (start_timestamp, end_timestamp),
            )

        results = []
        for msg_id, timestamp, chunks in cursor.fetchall():
            text_location = interfaces.TextLocation(
                message_ordinal=msg_id, chunk_ordinal=0
            )
            text_range = interfaces.TextRange(
                start=text_location, end=None  # Point range
            )
            results.append(
                interfaces.TimestampedTextRange(timestamp=timestamp, range=text_range)
            )

        return results
