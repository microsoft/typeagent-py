# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for the knowpro package."""

from .interfaces import MessageOrdinal, TextLocation, TextRange


def text_range_from_message_chunk(
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> TextRange:
    """Create a TextRange from message and chunk ordinals."""
    return TextRange(
        start=TextLocation(message_ordinal, chunk_ordinal),
        end=None,
    )
