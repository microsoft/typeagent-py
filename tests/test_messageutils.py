# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.knowpro.interfaces import TextLocation, TextRange
from typeagent.knowpro.messageutils import (
    text_range_from_message_chunk,
)


class TestTextRangeFromMessageChunk:
    def test_default_chunk_ordinal(self) -> None:
        tr = text_range_from_message_chunk(message_ordinal=3)
        assert tr.start == TextLocation(3, 0)
        assert tr.end is None

    def test_explicit_chunk_ordinal(self) -> None:
        tr = text_range_from_message_chunk(message_ordinal=5, chunk_ordinal=2)
        assert tr.start == TextLocation(5, 2)
        assert tr.end is None

    def test_returns_text_range(self) -> None:
        tr = text_range_from_message_chunk(0)
        assert isinstance(tr, TextRange)
