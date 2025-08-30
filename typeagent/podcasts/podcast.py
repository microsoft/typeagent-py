# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
import json
import os
from typing import TypedDict, cast

from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field, AliasChoices

from ..knowpro import semrefindex, kplib, secindex
from ..knowpro.field_helpers import CamelCaseField
from ..knowpro.convthreads import ConversationThreads
from ..knowpro.convsettings import ConversationSettings
from ..knowpro.interfaces import (
    ConversationDataWithIndexes,
    Datetime,
    ICollection,
    IConversation,
    IConversationSecondaryIndexes,
    IKnowledgeSource,
    IMessage,
    IMessageCollection,
    IMessageMetadata,
    ISemanticRefCollection,
    IStorageProvider,
    MessageOrdinal,
    SemanticRef,
    Term,
    Timedelta,
)
from ..knowpro.messageindex import MessageTextIndex
from ..knowpro.reltermsindex import TermToRelatedTermsMap
from ..storage.utils import create_storage_provider
from ..knowpro import serialization
from ..knowpro.collections import (
    MemoryMessageCollection,
    MemorySemanticRefCollection,
)


@pydantic_dataclass
class PodcastMessageMeta(IKnowledgeSource, IMessageMetadata):
    """Metadata class (!= metaclass) for podcast messages."""

    speaker: str | None = None
    listeners: list[str] = Field(default_factory=list)

    @property
    def source(self) -> str | None:  # type: ignore[reportIncompatibleVariableOverride]
        return self.speaker

    @property
    def dest(self) -> str | list[str] | None:  # type: ignore[reportIncompatibleVariableOverride]
        return self.listeners

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
                # TODO: Also create inverse actions.
                inverse_actions=[],
                topics=[],
            )


class PodcastMessageMetaData(TypedDict):
    speaker: str | None
    listeners: list[str]


class PodcastMessageData(TypedDict):
    metadata: PodcastMessageMetaData
    textChunks: list[str]
    tags: list[str]
    timestamp: str | None


@pydantic_dataclass
class PodcastMessage(IMessage):
    text_chunks: list[str] = CamelCaseField("The text chunks of the podcast message")
    metadata: PodcastMessageMeta = CamelCaseField(
        "Metadata associated with the podcast message"
    )
    tags: list[str] = CamelCaseField(
        "Tags associated with the message", default_factory=list
    )
    timestamp: str | None = None

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        return self.metadata.get_knowledge()

    def add_timestamp(self, timestamp: str) -> None:
        self.timestamp = timestamp

    def add_content(self, content: str) -> None:
        self.text_chunks[0] += content

    def serialize(self) -> PodcastMessageData:
        return self.__pydantic_serializer__.to_python(self, by_alias=True)  # type: ignore

    @staticmethod
    def deserialize(message_data: PodcastMessageData) -> "PodcastMessage":
        return PodcastMessage.__pydantic_validator__.validate_python(message_data)  # type: ignore


class PodcastData(ConversationDataWithIndexes[PodcastMessageData]):
    pass


@dataclass
class Podcast(IConversation[PodcastMessage, semrefindex.TermToSemanticRefIndex]):
    settings: ConversationSettings
    name_tag: str
    messages: IMessageCollection[PodcastMessage]
    semantic_refs: ISemanticRefCollection
    tags: list[str]
    semantic_ref_index: semrefindex.TermToSemanticRefIndex
    secondary_indexes: IConversationSecondaryIndexes[PodcastMessage] | None

    @classmethod
    async def create(
        cls,
        settings: ConversationSettings,
        name_tag: str | None = None,
        messages: IMessageCollection[PodcastMessage] | None = None,
        semantic_refs: ISemanticRefCollection | None = None,
        semantic_ref_index: semrefindex.TermToSemanticRefIndex | None = None,
        tags: list[str] | None = None,
        secondary_indexes: IConversationSecondaryIndexes[PodcastMessage] | None = None,
    ) -> "Podcast":
        """Create a fully initialized Podcast instance."""
        storage_provider = await settings.get_storage_provider()
        return cls(
            settings,
            name_tag or "",
            messages or await storage_provider.get_message_collection(),
            semantic_refs or await storage_provider.get_semantic_ref_collection(),
            tags if tags is not None else [],
            semantic_ref_index or await storage_provider.get_semantic_ref_index(),  # type: ignore
            secondary_indexes
            or await secindex.ConversationSecondaryIndexes.create(
                storage_provider, settings.related_term_index_settings
            ),
        )

    def _get_secondary_indexes(self) -> IConversationSecondaryIndexes[PodcastMessage]:
        """Get secondary indexes, asserting they are initialized."""
        assert (
            self.secondary_indexes is not None
        ), "Use await Podcast.create() to create an initialized instance"
        return self.secondary_indexes

    async def add_metadata_to_index(self) -> None:
        if self.semantic_ref_index is not None:
            assert self.semantic_refs is not None
            await semrefindex.add_metadata_to_index(
                self.messages,
                self.semantic_refs,
                self.semantic_ref_index,
            )

    async def generate_timestamps(
        self, start_date: Datetime, length_minutes: float = 60.0
    ) -> None:
        await timestamp_messages(
            self.messages, start_date, start_date + Timedelta(minutes=length_minutes)
        )

    async def build_index(
        self,
    ) -> None:
        await self.add_metadata_to_index()
        assert (
            self.settings is not None
        ), "Settings must be initialized before building index"
        await semrefindex.build_conversation_index(self, self.settings)
        # build_conversation_index automatically builds standard secondary indexes.
        # Pass false here to build podcast specific secondary indexes only.
        await self._build_transient_secondary_indexes(False)
        if self.secondary_indexes is not None:
            if self.secondary_indexes.threads is not None:
                await self.secondary_indexes.threads.build_index()  # type: ignore  # TODO

    async def serialize(self) -> PodcastData:
        data = PodcastData(
            nameTag=self.name_tag,
            messages=[m.serialize() async for m in self.messages],
            tags=self.tags,
            semanticRefs=(
                [r.serialize() async for r in self.semantic_refs]
                if self.semantic_refs is not None
                else None
            ),
        )
        # Set the rest only if they aren't None.
        if self.semantic_ref_index is not None:
            data["semanticIndexData"] = self.semantic_ref_index.serialize()

        secondary_indexes = self._get_secondary_indexes()
        if secondary_indexes.term_to_related_terms_index is not None:
            data["relatedTermsIndexData"] = (
                await secondary_indexes.term_to_related_terms_index.serialize()
            )
        if secondary_indexes.threads:
            data["threadData"] = secondary_indexes.threads.serialize()
        if secondary_indexes.message_index is not None:
            data["messageIndexData"] = secondary_indexes.message_index.serialize()
        return data

    async def write_to_file(self, filename: str) -> None:
        data = await self.serialize()
        serialization.write_conversation_data_to_file(data, filename)

    async def deserialize(
        self, podcast_data: ConversationDataWithIndexes[PodcastMessageData]
    ) -> None:
        if await self.messages.size() or (
            self.semantic_refs is not None and await self.semantic_refs.size()
        ):
            raise RuntimeError("Cannot deserialize into a non-empty Podcast.")

        self.name_tag = podcast_data["nameTag"]

        for message_data in podcast_data["messages"]:
            msg = PodcastMessage.deserialize(message_data)
            await self.messages.append(msg)

        semantic_refs_data = podcast_data.get("semanticRefs")
        if semantic_refs_data is not None:
            if self.semantic_refs is None:
                self.semantic_refs = MemorySemanticRefCollection()
            semrefs = [SemanticRef.deserialize(r) for r in semantic_refs_data]
            await self.semantic_refs.extend(semrefs)

        self.tags = podcast_data["tags"]

        semantic_index_data = podcast_data.get("semanticIndexData")
        if semantic_index_data is not None:
            if self.semantic_ref_index is not None:
                # Use the existing index and deserialize into it (for persistent storage)
                self.semantic_ref_index.deserialize(semantic_index_data)
            else:
                # Direct construction for in-memory storage
                self.semantic_ref_index = semrefindex.TermToSemanticRefIndex(  # type: ignore  # TODO
                    semantic_index_data
                )

        related_terms_index_data = podcast_data.get("relatedTermsIndexData")
        if related_terms_index_data is not None:
            secondary_indexes = self._get_secondary_indexes()
            term_to_related_terms_index = secondary_indexes.term_to_related_terms_index
            if term_to_related_terms_index is not None:
                # Assert empty before deserializing
                assert (
                    await term_to_related_terms_index.aliases.is_empty()
                ), "Term to related terms index must be empty before deserializing"
                await term_to_related_terms_index.deserialize(related_terms_index_data)

        thread_data = podcast_data.get("threadData")
        if thread_data is not None:
            assert (
                self.settings is not None
            ), "Settings must be initialized for deserialization"
            secondary_indexes = self._get_secondary_indexes()
            secondary_indexes.threads = ConversationThreads(
                self.settings.thread_settings
            )
            secondary_indexes.threads.deserialize(thread_data)

        message_index_data = podcast_data.get("messageIndexData")
        if message_index_data is not None:
            secondary_indexes = self._get_secondary_indexes()
            # Assert the message index is empty before deserializing
            assert (
                secondary_indexes.message_index is not None
            ), "Message index should be initialized"

            if isinstance(secondary_indexes.message_index, MessageTextIndex):
                index_size = await secondary_indexes.message_index.size()
                assert (
                    index_size == 0
                ), "Message index must be empty before deserializing"
            secondary_indexes.message_index.deserialize(message_index_data)

        await self._build_transient_secondary_indexes(True)

    @staticmethod
    async def read_from_file(
        filename_prefix: str,
        settings: ConversationSettings,
        dbname: str | None = None,
    ) -> "Podcast":
        embedding_size = settings.embedding_model.embedding_size
        data = serialization.read_conversation_data_from_file(
            filename_prefix, embedding_size
        )

        provider = await settings.get_storage_provider()
        msgs = await provider.get_message_collection()
        semrefs = await provider.get_semantic_ref_collection()
        if await msgs.size() or await semrefs.size():
            raise RuntimeError(
                f"Database {dbname!r} already has messages or semantic refs."
            )
        podcast = await Podcast.create(settings, messages=msgs, semantic_refs=semrefs)
        await podcast.deserialize(data)
        return podcast

    async def _build_transient_secondary_indexes(self, build_all: bool) -> None:
        # Secondary indexes are already initialized via create() factory method
        if build_all:
            await secindex.build_transient_secondary_indexes(self, self.settings)
        await self._build_participant_aliases()
        await self._add_synonyms()

    async def _build_participant_aliases(self) -> None:
        secondary_indexes = self._get_secondary_indexes()
        term_to_related_terms_index = secondary_indexes.term_to_related_terms_index
        assert term_to_related_terms_index is not None
        aliases = term_to_related_terms_index.aliases
        await aliases.clear()
        name_to_alias_map = await self._collect_participant_aliases()
        for name in name_to_alias_map.keys():
            related_terms: list[Term] = [
                Term(text=alias) for alias in name_to_alias_map[name]
            ]
            await aliases.add_related_term(name, related_terms)

    async def _add_synonyms(self) -> None:
        secondary_indexes = self._get_secondary_indexes()
        assert secondary_indexes.term_to_related_terms_index is not None
        aliases = cast(
            TermToRelatedTermsMap,
            secondary_indexes.term_to_related_terms_index.aliases,
        )
        synonym_file = os.path.join(os.path.dirname(__file__), "podcastVerbs.json")
        with open(synonym_file) as f:
            data: list[dict] = json.load(f)
        if data:
            for obj in data:
                text = obj.get("term")
                synonyms = obj.get("relatedTerms")
                if text and synonyms:
                    related_term = Term(text=text.lower())
                    for synonym in synonyms:
                        await aliases.add_related_term(
                            synonym.lower(),
                            related_term,
                        )

    async def _collect_participant_aliases(self) -> dict[str, set[str]]:

        aliases: dict[str, set[str]] = {}

        def collect_name(participant_name: str | None):
            if participant_name:
                participant_name = participant_name.lower()
                parsed_name = split_participant_name(participant_name)
                if parsed_name and parsed_name.first_name and parsed_name.last_name:
                    # If participant_name is a full name, associate first_name with the full name.
                    aliases.setdefault(parsed_name.first_name, set()).add(
                        participant_name
                    )
                    aliases.setdefault(participant_name, set()).add(
                        parsed_name.first_name
                    )

        async for message in self.messages:
            collect_name(message.metadata.speaker)
            for listener in message.metadata.listeners:
                collect_name(listener)

        return aliases


# Text (such as a transcript) can be collected over a time range.
# This text can be partitioned into blocks.
# However, timestamps for individual blocks are not available.
# Assigns individual timestamps to blocks proportional to their lengths.
async def timestamp_messages(
    messages: ICollection[PodcastMessage, MessageOrdinal],
    start_time: Datetime,
    end_time: Datetime,
) -> None:
    start = start_time.timestamp()
    duration = end_time.timestamp() - start
    if duration <= 0:
        raise RuntimeError(f"{start_time} is not < {end_time}")
    message_lengths = [
        sum(len(chunk) for chunk in m.text_chunks) async for m in messages
    ]
    text_length = sum(message_lengths)
    seconds_per_char = duration / text_length
    messages_list = [m async for m in messages]
    for message, length in zip(messages_list, message_lengths):
        message.timestamp = Datetime.fromtimestamp(start).isoformat()
        start += seconds_per_char * length


@dataclass
class ParticipantName:
    first_name: str
    last_name: str | None = None
    middle_name: str | None = None


def split_participant_name(full_name: str) -> ParticipantName | None:
    parts = full_name.split(None, 2)
    match len(parts):
        case 0:
            return None
        case 1:
            return ParticipantName(first_name=parts[0])
        case 2:
            return ParticipantName(first_name=parts[0], last_name=parts[1])
        case 3:
            if parts[1].lower() == "van":
                parts[1:] = [f"{parts[1]} {parts[2]}"]
                return ParticipantName(first_name=parts[0], last_name=parts[1])
            last_name = " ".join(parts[2].split())
            return ParticipantName(
                first_name=parts[0], middle_name=parts[1], last_name=last_name
            )
        case _:
            assert False, "SHOULD BE UNREACHABLE: Full name has too many parts"
