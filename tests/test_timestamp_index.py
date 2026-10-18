# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import timedelta, timezone

import pytest

from typeagent.knowpro.interfaces import DateRange, Datetime
from typeagent.storage.memory.timestamp_index import TimestampToTextRangeIndex


async def make_index(ts: list[str]) -> TimestampToTextRangeIndex:
    idx = TimestampToTextRangeIndex()
    # Add as (message_ordinal, timestamp)
    await idx.add_timestamps(list(enumerate(ts)))
    return idx


def to_ts_list(entries):
    return [e.timestamp for e in entries]


@pytest.mark.asyncio
async def test_lookup_range_half_open_and_point_query():
    # Stored timestamps are normalized to UTC with a "Z" suffix (naive == UTC).
    # Three sequential timestamps
    t0 = "2025-01-01T00:00:00"
    t1 = "2025-01-01T01:00:00"
    t2 = "2025-01-01T02:00:00"
    idx = await make_index([t0, t1, t2])

    # [t0, t1) includes t0, excludes t1
    dr = DateRange(start=Datetime.fromisoformat(t0), end=Datetime.fromisoformat(t1))
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t0 + ".000000Z"]

    # [t0, t2) includes t0 and t1, excludes t2
    dr = DateRange(start=Datetime.fromisoformat(t0), end=Datetime.fromisoformat(t2))
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t0 + ".000000Z", t1 + ".000000Z"]

    # [t1, t2) includes only t1
    dr = DateRange(start=Datetime.fromisoformat(t1), end=Datetime.fromisoformat(t2))
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t1 + ".000000Z"]

    # Point query: end=None means [t1, t1+epsilon) -> exactly t1
    dr = DateRange(start=Datetime.fromisoformat(t1), end=None)
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t1 + ".000000Z"]

    # Point query at t2 returns [t2]
    dr = DateRange(start=Datetime.fromisoformat(t2), end=None)
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t2 + ".000000Z"]

    # Point query at a time not present returns []
    tmid = "2025-01-01T00:30:00"
    dr = DateRange(start=Datetime.fromisoformat(tmid), end=None)
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == []


@pytest.mark.asyncio
async def test_lookup_range_mixed_utc_offsets():
    # 12:00+05:00 == 07:00 UTC; 10:00-08:00 == 18:00 UTC
    idx = await make_index(["2024-01-01T12:00:00+05:00", "2024-01-01T10:00:00-08:00"])
    utc = timezone.utc

    dr = DateRange(
        start=Datetime(2024, 1, 1, 6, tzinfo=utc),
        end=Datetime(2024, 1, 1, 8, tzinfo=utc),
    )
    assert [e.range.start.message_ordinal for e in await idx.lookup_range(dr)] == [0]

    dr = DateRange(
        start=Datetime(2024, 1, 1, 6, tzinfo=utc),
        end=Datetime(2024, 1, 1, 19, tzinfo=utc),
    )
    assert [e.range.start.message_ordinal for e in await idx.lookup_range(dr)] == [0, 1]

    # Query bound with a non-UTC offset; point query on the exact instant.
    plus5 = timezone(timedelta(hours=5))
    dr = DateRange(start=Datetime(2024, 1, 1, 12, tzinfo=plus5), end=None)
    assert [e.range.start.message_ordinal for e in await idx.lookup_range(dr)] == [0]


@pytest.mark.asyncio
async def test_add_timestamp_incremental_keeps_chronological_order():
    idx = TimestampToTextRangeIndex()
    await idx.add_timestamp(0, "2024-01-01T12:00:00+05:00")  # 07:00Z
    await idx.add_timestamp(1, "2024-01-01T08:00:00+00:00")  # 08:00Z
    await idx.add_timestamp(2, "2024-01-01T01:00:00-08:00")  # 09:00Z
    dr = DateRange(
        start=Datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=Datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert [e.range.start.message_ordinal for e in await idx.lookup_range(dr)] == [
        0,
        1,
        2,
    ]


@pytest.mark.asyncio
async def test_naive_timestamp_on_aware_range_start_is_included():
    idx = await make_index(["2024-01-01T00:00:00"])
    dr = DateRange(
        start=Datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=Datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
    )
    assert len(await idx.lookup_range(dr)) == 1


@pytest.mark.asyncio
async def test_fractional_seconds_sort_chronologically():
    idx = await make_index(["2024-01-01T12:00:00", "2024-01-01T12:00:00.5"])
    dr = DateRange(
        start=Datetime(2024, 1, 1, 12, tzinfo=timezone.utc),
        end=Datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc),
    )
    assert [e.range.start.message_ordinal for e in await idx.lookup_range(dr)] == [0, 1]
    dr = DateRange(
        start=Datetime(2024, 1, 1, 12, 0, 0, 250000, tzinfo=timezone.utc), end=None
    )
    assert await idx.lookup_range(dr) == []
