# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from datetime import datetime as Datetime, timedelta as Timedelta
import os
import re
from typing import cast, Sequence

from ..knowpro.importing import ConversationSettings, create_conversation_settings

from ..knowpro import convindex, interfaces, kplib


@dataclass
class PodcastMessageBase(interfaces.IKnowledgeSource):
    """Base class for podcast messages."""

    speaker: str
    listeners: list[str]

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


# TODO: Need a concrete implementation of IConversationSecondaryIndexes
@dataclass
class Podcast(
    interfaces.IConversation[
        PodcastMessage,
        convindex.ConversationIndex,
        interfaces.IConversationSecondaryIndexes,
    ]
):
    # Instance variables not passed to `__init__()`.
    settings: ConversationSettings = field(
        init=False, default_factory=create_conversation_settings
    )
    semantic_ref_index: convindex.ConversationIndex | None = field(
        init=False, default_factory=convindex.ConversationIndex
    )
    secondary_indexes: interfaces.IConversationSecondaryIndexes | None = field(
        init=False, default=None
    )

    # __init__() parameters, in that order (via `@dataclass`).
    name_tag: str = field(default="")
    messages: list[PodcastMessage] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    semantic_refs: list[interfaces.SemanticRef] | None = field(default_factory=list)

    def add_metadata_to_index(self) -> None:
        if self.semantic_ref_index is not None:
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
        result = await convindex.build_conversation_index(
            self, self.settings, event_handler
        )
        # TODO: implement secondary indexes
        # # build_conversation_index now automatically builds standard secondary indexes.
        # # Pass false to build podcast specific secondary indexes only.
        # await self.build_transient_secondary_indexes(False)
        # await self.secondary_indexes.threads.build_index()
        return result

    async def serialize(self) -> dict:
        data = {
            "name_tag": self.name_tag,
            "messages": self.messages,
            "tags": self.tags,
            "semantic_refs": self.semantic_refs,
            "semantic_index_data": (
                self.semantic_ref_index.serialize() if self.semantic_ref_index else None
            ),
            # TODO: set these only if things aren't None
            # "related_terms_index_data": self.secondary_indexes.term_to_related_terms_index.serialize(),
            # "thread_data": self.secondary_indexes.threads.serialize(),
            # "message_index_data": self.secondary_indexes.message_index.serialize(),
        }
        return data

    async def deserialize(self, podcast_data: dict) -> None:
        self.name_tag = podcast_data["name_tag"]
        self.messages = []
        for m in podcast_data["messages"]:
            msg = PodcastMessage(
                m["speaker"],
                m["listeners"],
                m["text_chunks"],
                m["tags"],
                m["timestamp"],
            )
            self.messages.append(msg)
        self.semantic_refs = podcast_data["semantic_refs"]
        self.tags = podcast_data["tags"]

        if podcast_data.get("semantic_index_data"):
            self.semantic_ref_index = convindex.ConversationIndex(
                podcast_data["semantic_index_data"]
            )
        # if podcast_data.get("related_terms_index_data"):
        #     self.secondary_indexes.term_to_related_terms_index.deserialize(podcast_data["related_terms_index_data"])
        # if podcast_data.get("thread_data"):
        #     self.secondary_indexes.threads = ConversationThreads(self.settings.thread_settings)
        #     self.secondary_indexes.threads.deserialize(podcast_data["thread_data"])
        # if podcast_data.get("message_index_data"):
        #     self.secondary_indexes.message_index = MessageTextIndex(self.settings.message_text_index_settings)
        #     self.secondary_indexes.message_index.deserialize(podcast_data["message_index_data"])
        # await self.build_transient_secondary_indexes(True)

    # TODO: Implement write_conversation_data_to_file, read_conversation_data_from_file
    # async def write_to_file(self, dir_path: str, base_file_name: str) -> None:
    #     data = await self.serialize()
    #     await write_conversation_data_to_file(data, dir_path, base_file_name)

    # @staticmethod
    # async def read_from_file(dir_path: str, base_file_name: str) -> Optional["Podcast"]:
    #     podcast = Podcast()
    #     embedding_size = (
    #         podcast.settings.related_term_index_settings.embedding_index_settings.embedding_size
    #         if podcast.settings.related_term_index_settings.embedding_index_settings
    #         else None
    #     )
    #     data = await read_conversation_data_from_file(dir_path, base_file_name, embedding_size)
    #     if data:
    #         await podcast.deserialize(data)
    #     return podcast

    # TODO: Stuff about secondary indexes


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
    msgs: list[PodcastMessage] = []
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
    pod = Podcast(podcast_name, msgs, [podcast_name])
    if start_date:
        pod.generate_timestamps(start_date, length_minutes)
    # TODO: Add more tags.
    return pod


def assign_message_listeners(
    msgs: Sequence[PodcastMessage],
    participants: set[str],
) -> None:
    for msg in msgs:
        if msg.speaker:
            listeners = [p for p in participants if p != msg.speaker]
            msg.listeners = listeners


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
    start = start_time.timestamp()
    duration = end_time.timestamp() - start
    if duration <= 0:
        raise RuntimeError(f"{start_time} is not < {end_time}")
    message_lengths = [sum(len(chunk) for chunk in m.text_chunks) for m in messages]
    text_length = sum(message_lengths)
    seconds_per_char = duration / text_length
    for message, length in zip(messages, message_lengths):
        message.timestamp = Datetime.fromtimestamp(start).isoformat()
        start += seconds_per_char * length
