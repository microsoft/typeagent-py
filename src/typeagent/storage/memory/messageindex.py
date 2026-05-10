# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Callable, Iterable

from ...aitools.embeddings import NormalizedEmbedding
from ...knowpro.convsettings import MessageTextIndexSettings
from ...knowpro.interfaces import (
    IConversation,
    IMessage,
    IMessageTextIndex,
    IStorageProvider,
    ITermToSemanticRefIndex,
    MessageOrdinal,
    MessageTextIndexData,
    ScoredMessageOrdinal,
    TextLocation,
)
from ...knowpro.textlocindex import ScoredTextLocation, TextToTextLocationIndex


async def build_message_index[
    TMessage: IMessage,
    TTermToSemanticRefIndex: ITermToSemanticRefIndex,
](
    conversation: IConversation[TMessage, TTermToSemanticRefIndex],
    storage_provider: IStorageProvider[TMessage],
) -> None:
    csi = conversation.secondary_indexes
    if csi is None:
        return
    if csi.message_index is None:
        csi.message_index = storage_provider.message_text_index
    messages = conversation.messages
    # Convert collection to list for add_messages
    messages_list = await messages.get_slice(0, await messages.size())
    await csi.message_index.add_messages(messages_list)


class IMessageTextEmbeddingIndex(IMessageTextIndex):
    async def generate_embedding(self, text: str) -> NormalizedEmbedding: ...

    async def lookup_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        max_matches: int | None = None,
        threshold_score: float | None = None,
        predicate: Callable[[MessageOrdinal], bool] | None = None,
    ) -> list[ScoredMessageOrdinal]: ...

    async def lookup_in_subset_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        ordinals_to_search: list[MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]: ...


class MessageTextIndex(IMessageTextEmbeddingIndex):
    def __init__(self, settings: MessageTextIndexSettings):
        self.settings = settings
        self.text_location_index = TextToTextLocationIndex(
            settings.embedding_index_settings
        )

    async def size(self) -> int:
        return await self.text_location_index.size()

    async def is_empty(self) -> bool:
        return await self.text_location_index.is_empty()

    async def add_messages[TMessage: IMessage](
        self,
        messages: Iterable[TMessage],
    ) -> None:
        base_message_ordinal: MessageOrdinal = await self.text_location_index.size()
        message_list = list(messages)
        if not message_list:
            return

        chunk_texts: list[str] = []
        for message in message_list:
            chunk_texts.extend(message.text_chunks)

        chunk_embeddings = await self.text_location_index.generate_embeddings(
            chunk_texts,
            cache=True,
        )
        await self.add_messages_starting_at_with_embeddings(
            base_message_ordinal,
            message_list,
            chunk_embeddings,
        )

    async def add_messages_starting_at[TMessage: IMessage](
        self,
        start_message_ordinal: int,
        messages: list[TMessage],
    ) -> None:
        """Add messages to the index starting at the given ordinal."""
        chunk_texts: list[str] = []
        for message in messages:
            chunk_texts.extend(message.text_chunks)

        chunk_embeddings = await self.text_location_index.generate_embeddings(
            chunk_texts,
            cache=True,
        )
        await self.add_messages_starting_at_with_embeddings(
            start_message_ordinal,
            messages,
            chunk_embeddings,
        )

    async def add_messages_starting_at_with_embeddings[TMessage: IMessage](
        self,
        start_message_ordinal: int,
        messages: list[TMessage],
        chunk_embeddings: list[NormalizedEmbedding],
    ) -> None:
        """Add messages starting at an ordinal using precomputed chunk embeddings."""
        text_locations: list[TextLocation] = []
        for idx, message in enumerate(messages):
            msg_ord = start_message_ordinal + idx
            for chunk_ord, _chunk in enumerate(message.text_chunks):
                text_locations.append(TextLocation(msg_ord, chunk_ord))

        if len(text_locations) != len(chunk_embeddings):
            raise ValueError(
                "messages and chunk_embeddings produced different chunk counts: "
                f"{len(text_locations)} != {len(chunk_embeddings)}"
            )

        await self.text_location_index.add_text_locations_with_embeddings(
            text_locations,
            chunk_embeddings,
        )

    async def lookup_messages(
        self,
        message_text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]:
        max_matches = max_matches or self.settings.embedding_index_settings.max_matches
        threshold_score = (
            threshold_score or self.settings.embedding_index_settings.min_score
        )
        scored_text_locations = await self.text_location_index.lookup_text(
            message_text, max_matches, threshold_score
        )
        return self.to_scored_message_ordinals(scored_text_locations)

    async def lookup_messages_in_subset(
        self,
        message_text: str,
        ordinals_to_search: list[MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]:
        scored_text_locations = await self.text_location_index.lookup_text_in_subset(
            message_text, ordinals_to_search, max_matches, threshold_score
        )
        return self.to_scored_message_ordinals(scored_text_locations)

    async def generate_embedding(self, text: str) -> NormalizedEmbedding:
        # Note: if you rename generate_embedding, be sure to also fix is_message_text_embedding_index.
        # TODO: Retries?
        # TODO: Find a prettier API to get an embedding rather than using _vector_base?
        return await self.text_location_index.generate_embedding(text)

    async def lookup_in_subset_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        ordinals_to_search: list[MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]:
        scored_text_locations = self.text_location_index.lookup_in_subset_by_embedding(
            text_embedding, ordinals_to_search, max_matches, threshold_score
        )
        return self.to_scored_message_ordinals(scored_text_locations)

    def to_scored_message_ordinals(
        self, scored_locations: list[ScoredTextLocation]
    ) -> list[ScoredMessageOrdinal]:
        matches: dict[MessageOrdinal, ScoredMessageOrdinal] = {}

        for sl in scored_locations:
            value = sl.text_location.message_ordinal
            score = sl.score
            match = matches.get(value)
            if match is None:
                matches[value] = ScoredMessageOrdinal(value, score)
            else:
                match.score = max(score, match.score)

        return [
            ScoredMessageOrdinal(
                match.message_ordinal,
                match.score,
            )
            for match in sorted(
                matches.values(), key=lambda match: match.score, reverse=True
            )
        ]

    async def serialize(self) -> MessageTextIndexData:
        return MessageTextIndexData(
            indexData=self.text_location_index.serialize(),
        )

    async def deserialize(self, data: MessageTextIndexData) -> None:
        index_data = data.get("indexData")
        if index_data is None:
            return
        self.text_location_index.deserialize(index_data)
