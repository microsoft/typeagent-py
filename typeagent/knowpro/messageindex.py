# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Iterable
from typing import Callable

from ..aitools.embeddings import NormalizedEmbedding
from .importing import MessageTextIndexSettings
from .interfaces import (
    IConversation,
    IMessage,
    IMessageTextIndex,
    MessageTextIndexData,
    ITermToSemanticRefIndex,
    IndexingEventHandlers,
    ListIndexingResult,
    MessageOrdinal,
    ScoredMessageOrdinal,
    TextLocation,
)
from .textlocindex import ScoredTextLocation, TextToTextLocationIndex


async def build_message_index[
    TMessage: IMessage,
    TTermToSemanticRefIndex: ITermToSemanticRefIndex,
](
    conversation: IConversation[TMessage, TTermToSemanticRefIndex],
    settings: MessageTextIndexSettings | None = None,
    event_handler: IndexingEventHandlers | None = None,
) -> ListIndexingResult:
    csi = conversation.secondary_indexes
    if csi is None:
        return ListIndexingResult(0)
    if csi.message_index is None:
        csi.message_index = MessageTextIndex(settings)
    messages = conversation.messages
    return await csi.message_index.add_messages(messages, event_handler)


class IMessageTextEmbeddingIndex(IMessageTextIndex):
    def __len__(self) -> int:
        raise NotImplementedError

    async def generate_embedding(self, text: str) -> NormalizedEmbedding:
        raise NotImplementedError

    def lookup_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        max_matches: int | None = None,
        threshold_score: float | None = None,
        predicate: Callable[[MessageOrdinal], bool] | None = None,
    ) -> list[ScoredMessageOrdinal]:
        raise NotImplementedError

    def lookup_in_subset_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        ordinals_to_search: list[MessageOrdinal],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredMessageOrdinal]:
        raise NotImplementedError


class MessageTextIndex(IMessageTextEmbeddingIndex):
    def __init__(self, settings: MessageTextIndexSettings | None = None):
        if settings is None:
            settings = MessageTextIndexSettings()
        self.settings = settings
        self.text_location_index = TextToTextLocationIndex(
            settings.embedding_index_settings
        )

    def __len__(self) -> int:
        return len(self.text_location_index)

    def __bool__(self) -> bool:
        return True

    async def add_messages[TMessage: IMessage](
        self,
        messages: Iterable[TMessage],
        event_handler: IndexingEventHandlers | None = None,
    ) -> ListIndexingResult:
        base_message_ordinal: MessageOrdinal = len(self.text_location_index)
        all_chunks: list[tuple[str, TextLocation]] = []
        # Collect everything so we can batch efficiently.
        for message_ordinal, message in enumerate(messages, base_message_ordinal):
            for chunk_ordinal, chunk in enumerate(message.text_chunks):
                all_chunks.append((chunk, TextLocation(message_ordinal, chunk_ordinal)))
        return await self.text_location_index.add_text_locations(
            all_chunks, event_handler
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
        return await self.text_location_index.generate_embedding(text)  # type: ignore  # Must use internal.

    def lookup_in_subset_by_embedding(
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

    def serialize(self) -> MessageTextIndexData:
        return MessageTextIndexData(
            indexData=self.text_location_index.serialize(),
        )

    def deserialize(self, data: MessageTextIndexData) -> None:
        index_data = data.get("indexData")
        if index_data is None:
            return
        self.text_location_index.deserialize(index_data)
