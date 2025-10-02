# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Timestamp-to-text-range in-memory index (pre-SQLite prep).
#
# Contract (stable regardless of backing store):
# - add_timestamp(s) accepts ISO 8601 timestamps that are lexicographically sortable
#   (Datetime.isoformat). Missing/None timestamps are ignored.
# - lookup_range(DateRange) returns items whose ISO timestamp t satisfies
#   start <= t < end (end is exclusive). If end is None, treat as a point
#   query with end = start + epsilon.
# - Results are sorted ascending by timestamp; stability across runs is expected.
#
# SQLite plan (no behavior change now):
# - This in-memory structure will be replaced by direct queries over a Messages table
#   with a timestamp column (or start/end timestamps if ranges are later needed).
# - The public methods and semantics here define the contract for the future provider
#   implementation; callers should not rely on internal list layout or mutability.


import bisect
from collections.abc import AsyncIterable, Callable
from typing import Any

from ...knowpro.interfaces import (
    DateRange,
    Datetime,
    IConversation,
    IMessage,
    ITimestampToTextRangeIndex,
    MessageOrdinal,
    TimestampedTextRange,
)
from ...knowpro.messageutils import text_range_from_message_chunk


class TimestampToTextRangeIndex(ITimestampToTextRangeIndex):
    # In-memory implementation of ITimestampToTextRangeIndex.
    #
    # Notes for SQLite implementation:
    # - add_timestamp(s): will translate to inserting/updating rows in the Messages
    #   storage (or a dedicated index table) keyed by message ordinal with an ISO
    #   timestamp column indexed for range scans.
    # - lookup_range(): will map to a single indexed range query on the timestamp
    #   column and project the corresponding text ranges.
    def __init__(self):
        self._ranges: list[TimestampedTextRange] = []

    async def size(self) -> int:
        return self._size()

    def _size(self) -> int:
        return len(self._ranges)

    async def lookup_range(self, date_range: DateRange) -> list[TimestampedTextRange]:
        return self._lookup_range(date_range)

    def _lookup_range(self, date_range: DateRange) -> list[TimestampedTextRange]:
        start_at = date_range.start.isoformat()
        stop_at = None if date_range.end is None else date_range.end.isoformat()
        return get_in_range(
            self._ranges,
            start_at,
            stop_at,
            key=lambda x: x.timestamp,
        )

    async def add_timestamp(
        self,
        message_ordinal: MessageOrdinal,
        timestamp: str,
    ) -> bool:
        return self._add_timestamp(message_ordinal, timestamp)

    def _add_timestamp(
        self,
        message_ordinal: MessageOrdinal,
        timestamp: str,
    ) -> bool:
        return self._insert_timestamp(message_ordinal, timestamp, True)

    async def add_timestamps(
        self,
        message_timestamps: list[tuple[MessageOrdinal, str]],
    ) -> None:
        self._add_timestamps(message_timestamps)

    def _add_timestamps(
        self,
        message_timestamps: list[tuple[MessageOrdinal, str]],
    ) -> None:
        for message_ordinal, timestamp in message_timestamps:
            self._insert_timestamp(message_ordinal, timestamp, False)
        self._ranges.sort(key=lambda x: x.timestamp)

    def _insert_timestamp(
        self,
        message_ordinal: MessageOrdinal,
        timestamp: str | None,
        in_order: bool,
    ) -> bool:
        if not timestamp:
            return False
        timestamp_datetime = Datetime.fromisoformat(timestamp)
        entry: TimestampedTextRange = TimestampedTextRange(
            range=text_range_from_message_chunk(message_ordinal),
            # This string is formatted to be lexically sortable.
            timestamp=timestamp_datetime.isoformat(),
        )
        if in_order:
            where = bisect.bisect_left(
                self._ranges, entry.timestamp, key=lambda x: x.timestamp
            )
            self._ranges.insert(where, entry)
        else:
            self._ranges.append(entry)
        return True


def get_in_range[T, S: Any](
    values: list[T],
    start_at: S,
    stop_at: S | None,
    key: Callable[[T], S],
) -> list[T]:
    # Return the sublist of values with key in [start_at, stop_at), sorted.
    # Details:
    # - End is exclusive: values with key == stop_at are not returned.
    # - If stop_at is None, treat as a point query with end = start_at + epsilon.
    # - Requires that values are already sorted by the provided key.
    istart = bisect.bisect_left(values, start_at, key=key)
    if istart == len(values):
        return []
    if stop_at is None:
        # Point query: include only items exactly equal to start_at
        istop = bisect.bisect_right(values, start_at, istart, key=key)
        return values[istart:istop]
    # End-exclusive: do not include items with key == stop_at
    istop = bisect.bisect_left(values, stop_at, istart, key=key)
    return values[istart:istop]


async def build_timestamp_index(conversation: IConversation) -> None:
    if conversation.messages is not None and conversation.secondary_indexes is not None:
        # There's nothing to do if there are no messages
        if await conversation.messages.size() == 0:
            return

        # There's nothing to do for persistent collections; the timestamp index
        # is created implicitly (as an index over the message collection)
        if conversation.messages.is_persistent:
            return

        # Caller must have established the timestamp index
        assert conversation.secondary_indexes.timestamp_index is not None

        await add_to_timestamp_index(
            conversation.secondary_indexes.timestamp_index,
            conversation.messages,
            0,
        )


async def add_to_timestamp_index(
    timestamp_index: ITimestampToTextRangeIndex,
    messages: AsyncIterable[IMessage],
    base_message_ordinal: int,
) -> None:
    message_timestamps: list[tuple[int, str]] = []
    i = 0
    async for message in messages:
        timestamp = message.timestamp
        if timestamp:
            message_timestamps.append((base_message_ordinal + i, timestamp))
        i += 1
    await timestamp_index.add_timestamps(message_timestamps)
