# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re

from ..knowpro.importing import ConversationSettings
from ..knowpro.interfaces import Datetime, IMessageCollection
from ..knowpro.storage import MessageCollection
from .podcast import Podcast, PodcastMessage


def import_podcast(
    transcript_file_path: str,
    podcast_name: str | None = None,
    start_date: Datetime | None = None,
    length_minutes: float = 60.0,
    settings: ConversationSettings | None = None,
) -> Podcast:
    with open(transcript_file_path, "r") as f:
        transcript_lines = f.readlines()
    if not podcast_name:
        podcast_name = os.path.splitext(os.path.basename(transcript_file_path))[0]
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
    msgs = MessageCollection[PodcastMessage]()
    cur_msg: PodcastMessage | None = None
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
                cur_msg = PodcastMessage(speaker, [], [speech])
    if cur_msg:
        msgs.append(cur_msg)

    assign_message_listeners(msgs, participants)

    pod = Podcast(
        podcast_name, msgs, [podcast_name], settings=settings or ConversationSettings()
    )
    if start_date:
        pod.generate_timestamps(start_date, length_minutes)
    # TODO: Add more tags.
    return pod


def assign_message_listeners(
    msgs: IMessageCollection[PodcastMessage],
    participants: set[str],
) -> None:
    for msg in msgs:
        if msg.speaker:
            listeners = [p for p in participants if p != msg.speaker]
            msg.listeners = listeners
