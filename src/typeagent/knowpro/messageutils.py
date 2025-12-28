# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Message utility functions for the knowpro package."""

from .interfaces import (
    IMessage,
    IMessageCollection,
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


async def get_message_chunk_batch[TMessage: IMessage](
    messages: IMessageCollection[TMessage],
    message_ordinal_start_at: MessageOrdinal,
    batch_size: int,
) -> list[list[TextLocation]]:
    """
    Get batches of message chunk locations for processing.

    Args:
        messages: Collection of messages to process
        message_ordinal_start_at: Starting message ordinal
        batch_size: Number of message chunks per batch

    Yields:
        Lists of TextLocation objects, each representing a message chunk
    """
    batches: list[list[TextLocation]] = []
    current_batch: list[TextLocation] = []

    message_ordinal = message_ordinal_start_at
    async for message in messages:
        if message_ordinal < message_ordinal_start_at:
            message_ordinal += 1
            continue

        # Process each text chunk in the message
        for chunk_ordinal in range(len(message.text_chunks)):
            text_location = TextLocation(
                message_ordinal=message_ordinal,
                chunk_ordinal=chunk_ordinal,
            )
            current_batch.append(text_location)

            # When batch is full, yield it and start a new one
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

        message_ordinal += 1

    # Don't forget the last batch if it has items
    if current_batch:
        batches.append(current_batch)

    return batches


def get_message_chunk_batch_from_list[TMessage: IMessage](
    messages: list[TMessage],
    message_ordinal_start_at: MessageOrdinal,
    batch_size: int,
) -> list[list[TextLocation]]:
    """
    Get batches of message chunk locations for processing from a list of messages.

    Args:
        messages: List of messages to process
        message_ordinal_start_at: Starting message ordinal (ordinal of first message in list)
        batch_size: Number of message chunks per batch

    Returns:
        Lists of TextLocation objects, each representing a message chunk
    """
    batches: list[list[TextLocation]] = []
    current_batch: list[TextLocation] = []

    for idx, message in enumerate(messages):
        message_ordinal = message_ordinal_start_at + idx

        # Process each text chunk in the message
        for chunk_ordinal in range(len(message.text_chunks)):
            text_location = TextLocation(
                message_ordinal=message_ordinal,
                chunk_ordinal=chunk_ordinal,
            )
            current_batch.append(text_location)

            # When batch is full, yield it and start a new one
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

    # Don't forget the last batch if it has items
    if current_batch:
        batches.append(current_batch)

    return batches
