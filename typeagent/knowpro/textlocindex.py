# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

from ..aitools.embeddings import NormalizedEmbedding
from ..aitools.vectorbase import VectorBase

from .fuzzyindex import ScoredInt, EmbeddingIndex
from .importing import TextEmbeddingIndexSettings
import asyncio
from typing import Sequence
from .interfaces import (
    TextToTextLocationIndexData,
    IndexingEventHandlers,
    ListIndexingResult,
    TextLocation,
)


@dataclass
class ScoredTextLocation:
    text_location: TextLocation
    score: float


class ITextToTextLocationIndex(Protocol):
    async def add_text_location(
        self, text: str, text_location: TextLocation
    ) -> ListIndexingResult:
        raise NotImplementedError

    async def add_text_locations(
        self,
        text_and_locations: list[tuple[str, TextLocation]],
        event_handler: IndexingEventHandlers | None = None,
    ) -> ListIndexingResult:
        raise NotImplementedError

    async def lookup_text(
        self,
        text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredTextLocation]:
        raise NotImplementedError

    def serialize(self) -> TextToTextLocationIndexData:
        raise NotImplementedError

    def deserialize(self, data: TextToTextLocationIndexData) -> None:
        raise NotImplementedError


class TextToTextLocationIndex(ITextToTextLocationIndex):
    def __init__(self, settings: TextEmbeddingIndexSettings):
        self._text_locations: list[TextLocation] = []
        self._embedding_index: EmbeddingIndex = EmbeddingIndex(settings=settings)
        self._settings = settings

    def __len__(self) -> int:
        return len(self._embedding_index)

    def __bool__(self) -> bool:
        return True

    def get(self, pos: int, default: TextLocation | None = None) -> TextLocation | None:
        size = len(self._text_locations)
        if 0 <= pos < size:
            return self._text_locations[pos]
        return default

    async def add_text_location(
        self, text: str, text_location: TextLocation
    ) -> ListIndexingResult:
        return await self.add_text_locations([(text, text_location)])

    async def add_text_locations(
        self,
        text_and_locations: list[tuple[str, TextLocation]],
        event_handler: IndexingEventHandlers | None = None,
    ) -> ListIndexingResult:
        await self._embedding_index.add_texts([text for text, _ in text_and_locations])
        self._text_locations.extend([loc for _, loc in text_and_locations])
        return ListIndexingResult(number_completed=len(text_and_locations))

    async def lookup_text(
        self,
        text: str,
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredTextLocation]:
        embedding = await self.generate_embedding(text)
        matches = self._embedding_index.get_indexes_of_nearest(
            embedding,
            max_matches=max_matches,
            min_score=threshold_score if threshold_score is not None else 0.85,
        )
        return [
            ScoredTextLocation(self._text_locations[match.item], match.score)
            for match in matches
        ]

    async def lookup_text_in_subset(
        self,
        text: str,
        ordinals_to_search: list[int],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredTextLocation]:
        embedding = await self.generate_embedding(text)
        matches = self._embedding_index.get_indexes_of_nearest_in_subset(
            embedding,
            ordinals_to_search,
            max_matches,
            threshold_score,
        )
        return [
            ScoredTextLocation(self._text_locations[match.item], match.score)
            for match in matches
        ]

    async def generate_embedding(
        self, text: str, cache: bool = True
    ) -> NormalizedEmbedding:
        return await self._embedding_index.get_embedding(text, cache)

    def lookup_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        max_matches: int | None = None,
        threshold_score: float | None = None,
        predicate: Callable[[int], bool] | None = None,
    ) -> list[ScoredTextLocation]:
        matches = self._embedding_index.get_indexes_of_nearest(
            text_embedding,
            max_matches,
            threshold_score,
            predicate,
        )
        return self.to_scored_locations(matches)

    def lookup_in_subset_by_embedding(
        self,
        text_embedding: NormalizedEmbedding,
        ordinals_to_match: list[int],
        max_matches: int | None = None,
        threshold_score: float | None = None,
    ) -> list[ScoredTextLocation]:
        matches = self._embedding_index.get_indexes_of_nearest_in_subset(
            text_embedding,
            ordinals_to_match,
            max_matches,
            threshold_score,
        )
        return self.to_scored_locations(matches)

    def to_scored_locations(self, matches: list[ScoredInt]) -> list[ScoredTextLocation]:
        return [
            ScoredTextLocation(self._text_locations[match.item], match.score)
            for match in matches
        ]

    def clear(self) -> None:
        self._text_locations = []
        self._embedding_index.clear()

    def serialize(self) -> TextToTextLocationIndexData:
        return TextToTextLocationIndexData(
            textLocations=[loc.serialize() for loc in self._text_locations],
            embeddings=self._embedding_index.serialize(),
        )

    def deserialize(self, data: TextToTextLocationIndexData) -> None:
        self._text_locations.clear()
        self._embedding_index.clear()
        text_locations = data["textLocations"]
        embeddings = data["embeddings"]

        if embeddings is None:
            raise ValueError("No embeddings found")
        if len(text_locations) != len(embeddings):
            raise ValueError(
                f"TextToTextLocationIndexData corrupt. textLocation.length {len(text_locations)} != {len(embeddings)}"
            )

        self._text_locations = [TextLocation.deserialize(loc) for loc in text_locations]
        self._embedding_index.deserialize(embeddings)
