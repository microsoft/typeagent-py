# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import bisect
from collections.abc import Iterable
from typing import Any, Callable

from . import convindex
from .interfaces import (
    DateRange,
    Datetime,
    IConversation,
    IMessage,
    ITimestampToTextRangeIndex,
    ListIndexingResult,
    MessageOrdinal,
    TimestampedTextRange,
)


class TimestampToTextRangeIndex(ITimestampToTextRangeIndex):
    def __init__(self):
        self._ranges: list[TimestampedTextRange] = []

    def lookup_range(self, date_range: DateRange):
        start_at = date_range.start.isoformat()
        stop_at = None if date_range.end is None else date_range.end.isoformat()
        return get_in_range(
            self._ranges,
            start_at,
            stop_at,
            key=lambda x: x.timestamp,
        )

    def add_timestamp(
        self,
        message_ordinal: MessageOrdinal,
        timestamp: str,
    ) -> bool:
        return self._insert_timestamp(message_ordinal, timestamp, True)

    def add_timestamps(
        self,
        message_timestamps: list[tuple[MessageOrdinal, str]],
    ) -> ListIndexingResult:
        for message_ordinal, timestamp in message_timestamps:
            self._insert_timestamp(message_ordinal, timestamp, False)
        self._ranges.sort(key=lambda x: x.timestamp)
        return ListIndexingResult(len(message_timestamps))

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
            range=convindex.text_range_from_message_chunk(message_ordinal),
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
    istart = bisect.bisect_left(values, start_at, key=key)
    if istart == len(values):
        return []
    if stop_at is None:
        return values[istart:]
    istop = bisect.bisect_right(values, stop_at, istart, key=key)
    # If istop has a value that matches the range, use it.
    if istop < len(values) and key(values[istop]) == stop_at:
        return values[istart : istop + 1]
    else:
        return values[istart:istop]


def build_timestamp_index(conversation: IConversation) -> ListIndexingResult:
    if conversation.messages and conversation.secondary_indexes:
        if conversation.secondary_indexes.timestamp_index is None:
            conversation.secondary_indexes.timestamp_index = TimestampToTextRangeIndex()
        return add_to_timestamp_index(
            conversation.secondary_indexes.timestamp_index,
            conversation.messages,
            0,
        )
    return ListIndexingResult(0)


def add_to_timestamp_index(
    timestamp_index: ITimestampToTextRangeIndex,
    messages: Iterable[IMessage],
    base_message_ordinal: int,
) -> ListIndexingResult:
    message_timestamps: list[tuple[int, str]] = []
    for i, message in enumerate(messages):
        timestamp = message.timestamp
        if timestamp:
            message_timestamps.append((base_message_ordinal + i, timestamp))
    return timestamp_index.add_timestamps(message_timestamps)
