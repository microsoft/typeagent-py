# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
import json
import os
from typing import TypedDict, cast, Any

import numpy as np
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic import Field, AliasChoices

from ..aitools.embeddings import NormalizedEmbeddings
from ..storage.memory import semrefindex
from ..knowpro import kplib, secindex
from ..knowpro.conversation_base import ConversationBase
from ..knowpro.field_helpers import CamelCaseField
from ..storage.memory.convthreads import ConversationThreads
from ..knowpro.convsettings import ConversationSettings
from ..knowpro.interfaces import (
    AddMessagesResult,
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
    ITermToSemanticRefIndex,
    IndexingStartPoints,
    MessageOrdinal,
    SemanticRef,
    Term,
    Timedelta,
)
from ..storage.memory.messageindex import MessageTextIndex
from ..storage.memory.reltermsindex import TermToRelatedTermsMap
from ..storage.utils import create_storage_provider
from ..knowpro import serialization
from ..storage.memory.collections import (
    MemoryMessageCollection,
    MemorySemanticRefCollection,
)
from ..knowpro.universal_message import (
    ConversationMessage,
    ConversationMessageMeta,
)

# Type aliases for backward compatibility
PodcastMessage = ConversationMessage
PodcastMessageMeta = ConversationMessageMeta


# TypedDict for serialization (kept for backward compatibility with saved files)
class PodcastMessageMetaData(TypedDict):
    speaker: str | None
    recipients: list[str]  # Updated from 'listeners' to match ConversationMessageMeta


class PodcastMessageData(TypedDict):
    metadata: PodcastMessageMetaData
    textChunks: list[str]
    tags: list[str]
    timestamp: str | None


class PodcastData(ConversationDataWithIndexes[PodcastMessageData]):
    pass


class Podcast(ConversationBase[PodcastMessage]):
    """Podcast conversation with incremental indexing support."""

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
        data["semanticIndexData"] = await self.semantic_ref_index.serialize()

        secondary_indexes = self._get_secondary_indexes()
        if secondary_indexes.term_to_related_terms_index is not None:
            data["relatedTermsIndexData"] = (
                await secondary_indexes.term_to_related_terms_index.serialize()
            )
        if secondary_indexes.threads:
            data["threadData"] = secondary_indexes.threads.serialize()
        if secondary_indexes.message_index is not None:
            data["messageIndexData"] = await secondary_indexes.message_index.serialize()
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

        message_list = [PodcastMessage.deserialize(m) for m in podcast_data["messages"]]
        await self.messages.extend(message_list)

        semantic_refs_data = podcast_data.get("semanticRefs")
        if semantic_refs_data is not None:
            semrefs = [SemanticRef.deserialize(r) for r in semantic_refs_data]
            await self.semantic_refs.extend(semrefs)

        self.tags = podcast_data["tags"]

        semantic_index_data = podcast_data.get("semanticIndexData")
        if semantic_index_data is not None:
            await self.semantic_ref_index.deserialize(semantic_index_data)

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
            await secondary_indexes.message_index.deserialize(message_index_data)

        # Don't rebuild aliases/synonyms since they were deserialized from relatedTermsIndexData
        # Only build transient indexes that aren't serialized
        if related_terms_index_data is None:
            # If related terms weren't deserialized, build them
            await self._build_participant_aliases()
            await self._add_synonyms()

        # Always build other transient indexes
        await secindex.build_transient_secondary_indexes(self, self.settings)

    @staticmethod
    def _read_conversation_data_from_file(
        filename_prefix: str, embedding_size: int
    ) -> ConversationDataWithIndexes[Any]:
        """Read podcast conversation data from files. No exceptions are caught; they just bubble out."""
        with open(filename_prefix + "_data.json", "r", encoding="utf-8") as f:
            json_data: serialization.ConversationJsonData[PodcastMessageData] = (
                json.load(f)
            )
        embeddings_list: list[NormalizedEmbeddings] | None = None
        if embedding_size:
            with open(filename_prefix + "_embeddings.bin", "rb") as f:
                embeddings = np.fromfile(f, dtype=np.float32).reshape(
                    (-1, embedding_size)
                )
                embeddings_list = [embeddings]
        else:
            print(
                "Warning: not reading embeddings file because size is {embedding_size}"
            )
            embeddings_list = None
        file_data = serialization.ConversationFileData(
            jsonData=json_data,
            binaryData=serialization.ConversationBinaryData(
                embeddingsList=embeddings_list
            ),
        )
        if json_data.get("fileHeader") is None:
            json_data["fileHeader"] = serialization.create_file_header()
        return serialization.from_conversation_file_data(file_data)

    @staticmethod
    async def read_from_file(
        filename_prefix: str,
        settings: ConversationSettings,
        dbname: str | None = None,
    ) -> "Podcast":
        embedding_size = settings.embedding_model.embedding_size
        data = Podcast._read_conversation_data_from_file(
            filename_prefix, embedding_size
        )

        provider = await settings.get_storage_provider()
        msgs = await provider.get_message_collection()
        semrefs = await provider.get_semantic_ref_collection()
        if await msgs.size() or await semrefs.size():
            raise RuntimeError(
                f"Database {dbname!r} already has messages or semantic refs."
            )
        podcast = await Podcast.create(settings)
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
        aliases = secondary_indexes.term_to_related_terms_index.aliases
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
                        await aliases.add_related_term(synonym.lower(), related_term)

    async def _collect_participant_aliases(self) -> dict[str, set[str]]:

        aliases: dict[str, set[str]] = {}

        def collect_name(participant_name: str | None):
            if not participant_name:
                return
            participant_name = participant_name.lower()
            parsed_name = split_participant_name(participant_name)
            if parsed_name and parsed_name.first_name and parsed_name.last_name:
                # If participant_name is a full name, associate first_name with the full name.
                aliases.setdefault(parsed_name.first_name, set()).add(participant_name)
                # And also the reverse.
                aliases.setdefault(participant_name, set()).add(parsed_name.first_name)

        async for message in self.messages:
            collect_name(message.metadata.speaker)
            for recipient in message.metadata.recipients:
                collect_name(recipient)

        return aliases


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
