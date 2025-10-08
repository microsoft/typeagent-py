# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
from datetime import timedelta

from ..knowpro.convsettings import ConversationSettings
from ..knowpro.interfaces import Datetime
from ..knowpro.universal_message import (
    UNIX_EPOCH,
    format_timestamp_utc,
)
from ..storage.utils import create_storage_provider
from .podcast import Podcast, PodcastMessage, PodcastMessageMeta


async def ingest_podcast(
    transcript_file_path: str,
    settings: ConversationSettings,
    podcast_name: str | None = None,
    start_date: Datetime | None = None,
    length_minutes: float = 60.0,
    dbname: str | None = None,
) -> Podcast:
    """
    Ingest a podcast transcript file into a Podcast object.

    Args:
        transcript_file_path: Path to the transcript file
        settings: Conversation settings
        podcast_name: Name for the podcast (defaults to filename)
        start_date: Base datetime for timestamp generation.
                   If None, uses Unix epoch (1970-01-01 00:00:00 UTC),
                   preserving relative timing while signaling that the actual
                   date is unknown (Unix "timestamp left at zero" convention).
        length_minutes: Total length of podcast in minutes (for proportional timestamp allocation)
        dbname: Database name

    Returns:
        Podcast object with imported data
    """
    with open(transcript_file_path, "r") as f:
        transcript_lines = f.readlines()
    if not podcast_name:
        podcast_name = os.path.splitext(os.path.basename(transcript_file_path))[0]

    # Use Unix epoch if no start_date provided (Easter egg!)
    base_date = start_date if start_date is not None else UNIX_EPOCH

    # TODO: Don't use a regex, just basic string stuff
    regex = r"""(?x)                  # Enable verbose regex syntax
        ^
        (?:                           # Optional speaker part
            \s*                       # Optional leading whitespace
            (?P<speaker>              # Capture group for speaker
                [A-Z0-9]+             # One or more uppercase letters/digits
                (?:\s+[A-Z0-9]+)*     # Optional additional words
            )
            \s*                       # Optional whitespace after speaker
            :                         # Colon separator
            \s*                       # Optional whitespace after colon
        )?
        (?P<speech>(?:.*\S)?)         # Capture the rest as speech (ending in non-whitespace)
        \s*                           # Optional trailing whitespace
        $
    """
    turn_parse_regex = re.compile(regex)
    participants: set[str] = set()

    cur_msg: PodcastMessage | None = None
    msgs: list[PodcastMessage] = []
    for line in transcript_lines:
        match = turn_parse_regex.match(line)
        if match:
            speaker = match.group("speaker")
            if speaker:
                speaker = speaker.lower()
            speech = match.group("speech")
            if not (speaker or speech):
                continue
            if cur_msg:
                if not speaker:
                    cur_msg.add_content("\n" + speech)
                else:
                    msgs.append(cur_msg)
                    cur_msg = None
            if not cur_msg:
                if speaker:
                    participants.add(speaker)
                metadata = PodcastMessageMeta(speaker=speaker, recipients=[])
                cur_msg = PodcastMessage([speech], metadata)
    if cur_msg:
        msgs.append(cur_msg)

    assign_message_listeners(msgs, participants)

    # Assign timestamps proportionally based on message length
    assign_timestamps_proportionally(msgs, base_date, length_minutes)

    provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        dbname,
        PodcastMessage,
    )
    settings.storage_provider = provider
    msg_coll = await provider.get_message_collection()
    semref_coll = await provider.get_semantic_ref_collection()
    if await msg_coll.size() or await semref_coll.size():
        raise RuntimeError(f"{dbname!r} already has messages or semantic refs.")

    pod = await Podcast.create(
        settings,
        name=podcast_name,
        tags=[podcast_name],
    )

    # Add messages with indexing to build embeddings
    await pod.add_messages_with_indexing(msgs)

    return pod


def assign_message_listeners(
    msgs: list[PodcastMessage],
    participants: set[str],
) -> None:
    """Assign listeners (recipients) to each message - all participants except the speaker."""
    for msg in msgs:
        if msg.metadata.speaker:
            listeners = [p for p in participants if p != msg.metadata.speaker]
            msg.metadata.recipients = listeners


def assign_timestamps_proportionally(
    msgs: list[PodcastMessage],
    base_date: Datetime,
    length_minutes: float,
) -> None:
    """
    Assign timestamps to messages proportionally based on their text length.

    This is used for podcasts where we don't have exact timing data like WebVTT,
    so we allocate time proportionally based on how much text each speaker said.
    """
    if not msgs:
        return

    # Calculate total text length
    message_lengths = [sum(len(chunk) for chunk in msg.text_chunks) for msg in msgs]
    total_length = sum(message_lengths)

    if total_length == 0:
        # Edge case: no text, just assign all to start time
        timestamp = format_timestamp_utc(base_date)
        for msg in msgs:
            msg.timestamp = timestamp
        return

    # Calculate seconds per character
    total_seconds = length_minutes * 60.0
    seconds_per_char = total_seconds / total_length

    # Assign timestamps
    current_offset = 0.0
    for msg, length in zip(msgs, message_lengths):
        msg.timestamp = format_timestamp_utc(
            base_date + timedelta(seconds=current_offset)
        )
        current_offset += seconds_per_char * length
