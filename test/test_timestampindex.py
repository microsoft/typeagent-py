# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typeagent.storage.memory.timestampindex import TimestampToTextRangeIndex
from typeagent.knowpro.interfaces import DateRange, Datetime, TextLocation, TextRange


async def make_index(ts: list[str]) -> TimestampToTextRangeIndex:
    idx = TimestampToTextRangeIndex()
    # Add as (message_ordinal, timestamp)
    await idx.add_timestamps(list(enumerate(ts)))
    return idx


def to_ts_list(entries):
    return [e.timestamp for e in entries]


@pytest.mark.asyncio
async def test_lookup_range_half_open_and_point_query():
    # Three sequential timestamps
    t0 = "2025-01-01T00:00:00"
    t1 = "2025-01-01T01:00:00"
    t2 = "2025-01-01T02:00:00"
    idx = await make_index([t0, t1, t2])

    # [t0, t1) includes t0, excludes t1
    dr = DateRange(start=Datetime.fromisoformat(t0), end=Datetime.fromisoformat(t1))
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t0]

    # [t0, t2) includes t0 and t1, excludes t2
    dr = DateRange(start=Datetime.fromisoformat(t0), end=Datetime.fromisoformat(t2))
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t0, t1]

    # [t1, t2) includes only t1
    dr = DateRange(start=Datetime.fromisoformat(t1), end=Datetime.fromisoformat(t2))
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t1]

    # Point query: end=None means [t1, t1+epsilon) -> exactly t1
    dr = DateRange(start=Datetime.fromisoformat(t1), end=None)
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t1]

    # Point query at t2 returns [t2]
    dr = DateRange(start=Datetime.fromisoformat(t2), end=None)
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == [t2]

    # Point query at a time not present returns []
    tmid = "2025-01-01T00:30:00"
    dr = DateRange(start=Datetime.fromisoformat(tmid), end=None)
    results = await idx.lookup_range(dr)
    assert to_ts_list(results) == []
