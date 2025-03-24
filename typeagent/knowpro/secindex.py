# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from .messageindex import MessageTextIndex, build_message_index
from .relatedtermsindex import RelatedTermsIndex, build_related_terms_index

from . import convthreads
from .importing import ConversationSettings, RelatedTermIndexSettings
from .interfaces import (
    IConversation,
    IConversationSecondaryIndexes,
    IMessage,
    ITermToSemanticRefIndex,
    IndexingEventHandlers,
    SecondaryIndexingResults,
    TextIndexingResult,
    TextLocation,
)
from .timestampindex import TimestampToTextRangeIndex, build_timestamp_index
from .propindex import PropertyIndex, build_property_index


class ConversationSecondaryIndexes(IConversationSecondaryIndexes):
    def __init__(self, settings: RelatedTermIndexSettings | None = None):
        if settings is None:
            settings = RelatedTermIndexSettings()
        self.property_to_semantic_ref_index = PropertyIndex()
        self.timestamp_index = TimestampToTextRangeIndex()
        self.term_to_related_terms_index: RelatedTermsIndex = RelatedTermsIndex(settings)  # type: ignore  # TODO


async def build_secondary_indexes[
    TM: IMessage,
    TT: ITermToSemanticRefIndex,
](
    conversation: IConversation[TM, TT, ConversationSecondaryIndexes],
    conversation_settings: ConversationSettings,
    event_handler: IndexingEventHandlers | None,
) -> SecondaryIndexingResults:
    if conversation.secondary_indexes is None:
        conversation.secondary_indexes = ConversationSecondaryIndexes()
    result: SecondaryIndexingResults = build_transient_secondary_indexes(
        conversation,  # type: ignore  # TODO
    )
    result.related_terms = await build_related_terms_index(
        conversation, conversation_settings, event_handler
    )
    if result.related_terms is not None and not result.related_terms.error:
        res = await build_message_index(
            conversation,
            conversation_settings.message_text_index_settings,
            event_handler,
        )
        result.message = TextIndexingResult(TextLocation(res.number_completed))

    return result


def build_transient_secondary_indexes[
    TM: IMessage, TT: ITermToSemanticRefIndex, TC: IConversationSecondaryIndexes
](
    conversation: IConversation[TM, TT, TC],
) -> SecondaryIndexingResults:
    if conversation.secondary_indexes is None:
        conversation.secondary_indexes = ConversationSecondaryIndexes()  # type: ignore  # TODO
    result = SecondaryIndexingResults()
    result.properties = build_property_index(conversation)
    result.timestamps = build_timestamp_index(conversation)
    return result
