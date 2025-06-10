# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pydantic.dataclasses import dataclass
from typing import Annotated
from typing_extensions import Doc


@dataclass
class DateVal:
    day: int
    month: int
    year: int


@dataclass
class TimeVal:
    hour: Annotated[int, Doc("In 24 hour form")]
    minute: int
    seconds: int


@dataclass
class DateTime:
    date: DateVal
    time: TimeVal | None = None


@dataclass
class DateTimeRange:
    start_date: DateTime
    stop_date: DateTime | None = None
