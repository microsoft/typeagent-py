# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import Callable

from .interfaces import (
    # Interfaces.
    IConversation,
    IMessage,
    ITermToSemanticRefIndex,
    ITermToSemanticRefIndexData,
    # Other imports.
    IndexingEventHandlers,
    IndexingResults,
    Knowledge,
    KnowledgeType,
    MessageOrdinal,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    TextLocation,
    TextRange,
)


def text_range_from_location(
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> TextRange:
    return TextRange(
        start=TextLocation(message_ordinal, chunk_ordinal),
        end=None,
    )


type KnowledgeValidator = Callable[
    [
        KnowledgeType,  # knowledge_type
        Knowledge,  # knowledge
    ],
    bool,
]


def add_metadata_to_index[TMessage: IMessage](
    messages: list[TMessage],
    semantic_refs: list[SemanticRef],
    semantic_ref_index: ITermToSemanticRefIndex,
    knowledge_validator: KnowledgeValidator | None = None,
) -> None:
    raise NotImplementedError


@dataclass
class ConversationIndex(ITermToSemanticRefIndex):
    _map: dict[str, list[ScoredSemanticRefOrdinal]]

    def __init__(self, data: ITermToSemanticRefIndexData | None = None):
        self._map = {}
        if data:
            self.deserialize(data)

    # TODO: More methods

    def deserialize(self, data: ITermToSemanticRefIndexData) -> None:
        raise NotImplementedError


# ...


async def build_conversation_index(
    conversation: IConversation,
    event_handler: IndexingEventHandlers | None = None,
) -> IndexingResults:
    raise NotImplementedError
