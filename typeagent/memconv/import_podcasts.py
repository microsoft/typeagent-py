# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from datetime import datetime as Datetime, timedelta as Timedelta
import os
import re
from typing import Any, Sequence

from ..knowpro import convindex, interfaces, kplib


@dataclass
class PodcastMessageBase(interfaces.IKnowledgeSource):
    """Base class for podcast messages."""

    speaker: str
    listeners: list[str] = field(init=False, default_factory=list)

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        if not self.speaker:
            return kplib.KnowledgeResponse(
                entities=[],
                actions=[],
                inverse_actions=[],
                topics=[],
            )
        else:
            entities: list[kplib.ConcreteEntity] = []
            entities.append(
                kplib.ConcreteEntity(
                    name=self.speaker,
                    type=["person"],
                )
            )
            listener_entities = [
                kplib.ConcreteEntity(
                    name=listener,
                    type=["person"],
                )
                for listener in self.listeners
            ]
            entities.extend(listener_entities)
            actions = [
                kplib.Action(
                    verbs=["say"],
                    verb_tense="past",
                    subject_entity_name=self.speaker,
                    object_entity_name=listener,
                    indirect_object_entity_name="none",
                )
                for listener in self.listeners
            ]
            return kplib.KnowledgeResponse(
                entities=entities,
                actions=actions,
                inverse_actions=[],
                topics=[],
            )


@dataclass
class PodcastMessage(interfaces.IMessage, PodcastMessageBase):
    text_chunks: list[str]
    tags: list[str] = field(default_factory=list)
    timestamp: str | None = None

    def add_timestamp(self, timestamp: str) -> None:
        self.timestamp = timestamp

    def add_content(self, content: str) -> None:
        self.text_chunks[0] += content


@dataclass
class Podcast(interfaces.IConversation[PodcastMessage]):
    # Instance variables not passed to `__init__()`.
    # TODO
    # settings: ConversationSettings = field(
    #     init=False, default_factory=create_conversation_settings
    # )
    semantic_ref_index: convindex.ITermToSemanticRefIndex | None = field(
        init=False, default_factory=convindex.ConversationIndex
    )
    # TODO
    secondary_indexes: interfaces.IConversationSecondaryIndexes | None = field(
        init=False, default=None
    )
    # Work in progress.
    # message_index: TextToTextLocationIndexFuzzy | None = None

    # __init__() parameters, in that order (via `@dataclass`).
    name_tag: str = field(default="")
    messages: list[PodcastMessage] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    semantic_refs: list[interfaces.SemanticRef] | None = field(default_factory=list)

    def add_metadata_to_index(self) -> None:
        if self.semantic_ref_index:
            assert self.semantic_refs is not None
            convindex.add_metadata_to_index(
                self.messages,
                self.semantic_refs,
                self.semantic_ref_index,
            )

    def generate_timestamps(
        self, start_date: Datetime, length_minutes: float = 60.0
    ) -> None:
        timestamp_messages(
            self.messages, start_date, start_date + Timedelta(minutes=length_minutes)
        )

    async def build_index(
        self,
        event_handler: interfaces.IndexingEventHandlers | None = None,
    ) -> interfaces.IndexingResults:
        self.add_metadata_to_index()
        result = await convindex.build_conversation_index(self, event_handler)
        # TODO
        # if not result.error:
        #     # build_conversation_index now automatically builds standard secondary indexes
        #     # Pass false to build podcast specific secondary indexes only
        #     await self.build_secondary_indexes(False)
        #     await self.secondary_indexes.threads.build_index()
        return result

    # TODO: Methods about serialization, file I/O, and indexing


def assign_message_listeners(
    msgs: Sequence[PodcastMessage],
    participants: set[str],
) -> None:
    for msg in msgs:
        if msg.speaker:
            listeners = [p for p in participants if p != msg.speaker]
            msg.listeners = listeners


# NOTE: Doesn't need to be async (Python file I/O is synchronous)
def import_podcast(
    transcript_file_path: str,
    podcast_name: str | None = None,
    start_date: Datetime | None = None,
    length_minutes: float = 60.0,
) -> Podcast:
    with open(transcript_file_path, "r") as f:
        transcript_lines = f.readlines()
    if not podcast_name:
        podcast_name = os.path.splitext(os.path.basename(transcript_file_path))[0]
    transcript_lines = [line.rstrip() for line in transcript_lines if line.strip()]
    turn_parse_regex = re.compile(r"^(?P<speaker>[A-Z0-9 ]+:)?(?P<speech>.*)$")
    participants: set[str] = set()
    msgs: list[PodcastMessage] = []
    cur_msg: PodcastMessage | None = None
    for line in transcript_lines:
        match = turn_parse_regex.match(line)
        if match:
            speaker = match.group("speaker")
            speech = match.group("speech")
            if cur_msg:
                if not speaker:
                    cur_msg.add_content("\n" + speech)
                else:
                    msgs.append(cur_msg)
                    cur_msg = None
            if not cur_msg:
                if speaker:
                    speaker = speaker.strip()
                    if speaker.endswith(":"):
                        speaker = speaker[:-1]
                    speaker = speaker.lower()  # TODO: locale
                    participants.add(speaker)
                cur_msg = PodcastMessage(speaker, [speech])
    if cur_msg:
        msgs.append(cur_msg)
    assign_message_listeners(msgs, participants)
    pod = Podcast(podcast_name, msgs, [podcast_name])
    if start_date:
        pod.generate_timestamps(start_date, length_minutes)
    # TODO: Add more tags.
    return pod


# Text (such as a transcript) can be collected over a time range.
# This text can be partitioned into blocks.
# However, timestamps for individual blocks are not available.
# Assigns individual timestamps to blocks proportional to their lengths.
# @param messages The messages to assign timestamps to
# @param start_time
# @param end_time
def timestamp_messages(
    messages: Sequence[interfaces.IMessage],
    start_time: Datetime,
    end_time: Datetime,
) -> None:
    # ticks ~~ posix timestamp
    start_ticks = start_time.timestamp()
    ticks_length = end_time.timestamp() - start_ticks
    if ticks_length <= 0:
        raise RuntimeError(f"{start_time} is not < {end_time}")
    message_lengths = [sum(len(chunk) for chunk in m.text_chunks)
                       for m in messages]
    text_length = sum(message_lengths)
    ticks_per_char = ticks_length / text_length
    for message, length in zip(messages, message_lengths):
        message.timestamp = Datetime.fromtimestamp(start_ticks).isoformat()
        start_ticks += ticks_per_char * length
