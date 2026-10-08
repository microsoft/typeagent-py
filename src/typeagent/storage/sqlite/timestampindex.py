# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite-based timestamp index implementation."""

import sqlite3

from ...knowpro.interfaces import (
    DateRange,
    ITimestampToTextRangeIndex,
    MessageOrdinal,
    TextLocation,
    TextRange,
    TimestampedTextRange,
)
from ...knowpro.universal_message import format_timestamp_utc
from .schema import NORMALIZED_TIMESTAMP_SQL


class SqliteTimestampToTextRangeIndex(ITimestampToTextRangeIndex):
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
        self, message_ordinal: MessageOrdinal, timestamp: str
    ) -> bool:
        return self._add_timestamp(message_ordinal, timestamp)

    def _add_timestamp(self, message_ordinal: MessageOrdinal, timestamp: str) -> bool:
        """Add timestamp to Messages table start_timestamp column."""
        cursor = self.db.cursor()
        cursor.execute(
            "UPDATE Messages SET start_timestamp = ? WHERE msg_id = ?",
            (timestamp, message_ordinal),
        )
        return cursor.rowcount > 0

    async def get_timestamp_ranges(
        self, start_timestamp: str, end_timestamp: str | None = None
    ) -> list[TimestampedTextRange]:
        """Get timestamp ranges from Messages table."""
        cursor = self.db.cursor()

        norm_col = NORMALIZED_TIMESTAMP_SQL.format(value="start_timestamp")
        norm_arg = NORMALIZED_TIMESTAMP_SQL.format(value="?")
        if end_timestamp is None:
            # Single timestamp query
            cursor.execute(
                f"""
                SELECT msg_id, start_timestamp
                FROM Messages
                WHERE {norm_col} = {norm_arg}
                ORDER BY msg_id
                """,
                (start_timestamp,),
            )
        else:
            # Range query (inclusive)
            cursor.execute(
                f"""
                SELECT msg_id, start_timestamp
                FROM Messages
                WHERE {norm_col} >= {norm_arg} AND {norm_col} <= {norm_arg}
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
            results.append(TimestampedTextRange(range=text_range, timestamp=timestamp))

        return results

    async def add_timestamps(
        self, message_timestamps: list[tuple[MessageOrdinal, str]]
    ) -> None:
        """Add multiple timestamps."""
        if not message_timestamps:
            return
        cursor = self.db.cursor()
        cursor.executemany(
            "UPDATE Messages SET start_timestamp = ? WHERE msg_id = ?",
            [(ts, ordinal) for ordinal, ts in message_timestamps],
        )

    async def lookup_range(self, date_range: DateRange) -> list[TimestampedTextRange]:
        """Lookup messages in a date range."""
        cursor = self.db.cursor()

        # Stored timestamps may have any UTC offset and any fractional precision,
        # so raw strings don't sort chronologically. Compare via strftime(), which
        # converts to UTC and renders a fixed-width value (also for existing rows).
        start_timestamp = format_timestamp_utc(date_range.start)
        end_timestamp = format_timestamp_utc(date_range.end) if date_range.end else None

        norm_col = NORMALIZED_TIMESTAMP_SQL.format(value="start_timestamp")
        norm_arg = NORMALIZED_TIMESTAMP_SQL.format(value="?")
        if date_range.end is None:
            # Point query
            cursor.execute(
                f"""
                SELECT msg_id, start_timestamp, chunks
                FROM Messages
                WHERE {norm_col} = {norm_arg}
                ORDER BY msg_id
                """,
                (start_timestamp,),
            )
        else:
            # Range query
            cursor.execute(
                f"""
                SELECT msg_id, start_timestamp, chunks
                FROM Messages
                WHERE {norm_col} >= {norm_arg} AND {norm_col} < {norm_arg}
                ORDER BY msg_id
                """,
                (start_timestamp, end_timestamp),
            )

        results = []
        for msg_id, timestamp, _chunks in cursor.fetchall():
            text_location = TextLocation(message_ordinal=msg_id, chunk_ordinal=0)
            text_range = TextRange(start=text_location, end=None)  # Point range
            results.append(TimestampedTextRange(timestamp=timestamp, range=text_range))

        return results
