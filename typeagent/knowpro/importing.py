# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass


@dataclass
class RelatedTermIndexSettings:
    pass  # TODO


@dataclass
class TextEmbeddingIndexSettings:
    pass  # TODO


@dataclass
class MessageTextIndexSettings:
    pass  # TODO


@dataclass
class ConversationSettings:
    relatedTermIndexSettings: RelatedTermIndexSettings
    threadSettings: TextEmbeddingIndexSettings
    messageTextIndexSettings: MessageTextIndexSettings


def create_conversation_settings() -> ConversationSettings:
    # TODO
    return ConversationSettings(
        RelatedTermIndexSettings(),
        TextEmbeddingIndexSettings(),
        MessageTextIndexSettings(),
    )
