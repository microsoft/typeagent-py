# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Message utility functions for the knowpro package."""

from .interfaces import (
    IMessage,
    MessageOrdinal,
    TextLocation,
    TextRange,
)


def text_range_from_message_chunk(
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> TextRange:
    """Create a TextRange from message and chunk ordinals."""
    return TextRange(
        start=TextLocation(message_ordinal, chunk_ordinal),
        end=None,
    )


def get_all_message_chunk_locations[TMessage: IMessage](
    messages: list[TMessage],
    message_ordinal_start_at: MessageOrdinal,
) -> list[TextLocation]:
    """
    Get a flat list of all message chunk locations from a list of messages.

    Args:
        messages: List of messages to process
        message_ordinal_start_at: Starting message ordinal (ordinal of first message in list)

    Returns:
        Flat list of TextLocation objects, one per message chunk
    """
    locations: list[TextLocation] = []
    for idx, message in enumerate(messages):
        message_ordinal = message_ordinal_start_at + idx
        for chunk_ordinal in range(len(message.text_chunks)):
            locations.append(
                TextLocation(
                    message_ordinal=message_ordinal,
                    chunk_ordinal=chunk_ordinal,
                )
            )
    return locations
