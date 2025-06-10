# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import cast
import pytest

from typeagent.knowpro.importing import RelatedTermIndexSettings, ConversationSettings
from typeagent.knowpro.interfaces import (
    DeletionInfo,
    IConversation,
    IMessage,
    ListIndexingResult,
    SecondaryIndexingResults,
    TextIndexingResult,
    TextLocation,
)
from typeagent.knowpro import kplib
from typeagent.knowpro.propindex import PropertyIndex
from typeagent.knowpro.reltermsindex import RelatedTermsIndex
from typeagent.knowpro.secindex import (
    ConversationSecondaryIndexes,
    build_secondary_indexes,
    build_transient_secondary_indexes,
)
from typeagent.knowpro.storage import MessageCollection, SemanticRefCollection
from typeagent.knowpro.timestampindex import TimestampToTextRangeIndex

from fixtures import needs_auth  # type: ignore  # Yes it is used!


class SimpleMessage(IMessage):
    """A simple implementation of IMessage for testing purposes."""

    def __init__(self, text: str):
        self.text_chunks: list[str] = [text]
        self.timestamp: str | None = None
        self.tags: list[str] = []
        self.deletion_info: DeletionInfo | None = None

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        raise NotImplementedError


class SimpleConversation(IConversation):
    """A simple implementation of IConversation for testing purposes."""

    def __init__(self):
        self.name_tag = "SimpleConversation"
        self.tags = []
        self.messages = MessageCollection[SimpleMessage]()
        self.semantic_refs = SemanticRefCollection()
        self.semantic_ref_index = None
        self.secondary_indexes = None


@pytest.fixture
def simple_conversation():
    return SimpleConversation()


@pytest.fixture
def conversation_settings(needs_auth):
    return ConversationSettings()


def test_conversation_secondary_indexes_initialization(needs_auth):
    """Test initialization of ConversationSecondaryIndexes."""
    indexes = ConversationSecondaryIndexes()
    assert isinstance(indexes.property_to_semantic_ref_index, PropertyIndex)
    assert isinstance(indexes.timestamp_index, TimestampToTextRangeIndex)
    assert isinstance(indexes.term_to_related_terms_index, RelatedTermsIndex)

    # Test with custom settings
    settings = RelatedTermIndexSettings()
    indexes_with_settings = ConversationSecondaryIndexes(settings)
    assert (
        cast(
            RelatedTermsIndex, indexes_with_settings.term_to_related_terms_index
        ).settings
        == settings
    )


@pytest.mark.asyncio
async def test_build_secondary_indexes(simple_conversation, conversation_settings):
    """Test building secondary indexes asynchronously."""
    # Add some dummy data to the conversation
    simple_conversation.messages.append(SimpleMessage("Message 1"))
    simple_conversation.messages.append(SimpleMessage("Message 2"))

    result = await build_secondary_indexes(
        simple_conversation, conversation_settings, None
    )

    assert isinstance(result, SecondaryIndexingResults)
    assert result.related_terms is not None
    assert isinstance(result.message, TextIndexingResult)
    assert result.message.completed_upto == TextLocation(
        len(simple_conversation.messages)
    )


def test_build_transient_secondary_indexes(simple_conversation, needs_auth):
    """Test building transient secondary indexes."""
    # Add some dummy data to the conversation
    simple_conversation.messages.append(SimpleMessage("Message 1"))
    simple_conversation.messages.append(SimpleMessage("Message 2"))

    result = build_transient_secondary_indexes(simple_conversation)

    assert isinstance(result, SecondaryIndexingResults)
    assert result.properties is not None
    assert result.timestamps is not None
    assert isinstance(result.properties, ListIndexingResult)
    assert isinstance(result.timestamps, ListIndexingResult)
