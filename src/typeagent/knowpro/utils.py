# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for the knowpro package."""

from collections.abc import AsyncIterable

from .interfaces import MessageOrdinal, TextLocation, TextRange


async def aenumerate[T](aiterable: AsyncIterable[T], start: int = 0):
    i = start
    async for item in aiterable:
        yield i, item
        i += 1


def text_range_from_message_chunk(
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> TextRange:
    """Create a TextRange from message and chunk ordinals."""
    return TextRange(
        start=TextLocation(message_ordinal, chunk_ordinal),
        end=None,
    )
