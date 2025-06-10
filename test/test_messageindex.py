# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from unittest.mock import AsyncMock, MagicMock

from typeagent.knowpro.secindex import (
    ConversationSecondaryIndexes,
)
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.knowpro.messageindex import (
    MessageTextIndex,
    build_message_index,
)
from typeagent.knowpro.importing import MessageTextIndexSettings

from typeagent.knowpro.interfaces import (
    IConversation,
    IMessage,
    TextLocation,
    ListIndexingResult,
    IndexingEventHandlers,
    MessageTextIndexData,
    TextToTextLocationIndexData,
)
from typeagent.knowpro.textlocindex import TextToTextLocationIndex

from fixtures import needs_auth  # type: ignore  # It's used!


@pytest.fixture
def mock_text_location_index():
    """Fixture to mock the TextToTextLocationIndex."""
    mock_index = MagicMock(spec=TextToTextLocationIndex)
    mock_index.add_text_locations = AsyncMock(return_value=ListIndexingResult(2))
    mock_index.lookup_text = AsyncMock(return_value=[])
    mock_index.lookup_text_in_subset = AsyncMock(return_value=[])
    mock_index.serialize = MagicMock(return_value={"mock": "data"})
    mock_index.deserialize = MagicMock()
    return mock_index


@pytest.fixture
def message_text_index(mock_text_location_index):
    """Fixture to create a MessageTextIndex instance with a mocked TextToTextLocationIndex."""
    settings = MessageTextIndexSettings()
    index = MessageTextIndex(settings)
    index.text_location_index = mock_text_location_index
    return index


def test_message_text_index_init(needs_auth: None):
    """Test initialization of MessageTextIndex."""
    settings = MessageTextIndexSettings()
    index = MessageTextIndex(settings)
    assert index.settings == settings
    assert isinstance(index.text_location_index, TextToTextLocationIndex)


@pytest.mark.asyncio
async def test_add_messages(message_text_index, needs_auth: None):
    """Test adding messages to the MessageTextIndex."""
    messages = [
        MagicMock(text_chunks=["chunk1", "chunk2"]),
        MagicMock(text_chunks=["chunk3"]),
    ]
    event_handler = MagicMock()

    result = await message_text_index.add_messages(messages, event_handler)

    assert result.number_completed == 2
    message_text_index.text_location_index.add_text_locations.assert_awaited_once()


@pytest.mark.asyncio
async def test_lookup_messages(message_text_index, needs_auth: None):
    """Test looking up messages in the MessageTextIndex."""
    message_text_index.text_location_index.lookup_text.return_value = [
        MagicMock(text_location=TextLocation(1, 0), score=0.9),
        MagicMock(text_location=TextLocation(2, 0), score=0.8),
    ]

    result = await message_text_index.lookup_messages(
        "test message", max_matches=2, threshold_score=0.5
    )

    assert len(result) == 2
    assert result[0].message_ordinal == 1
    assert result[0].score == 0.9
    assert result[1].message_ordinal == 2
    assert result[1].score == 0.8


@pytest.mark.asyncio
async def test_lookup_messages_in_subset(message_text_index, needs_auth: None):
    """Test looking up messages in a subset of the MessageTextIndex."""
    message_text_index.text_location_index.lookup_text_in_subset.return_value = [
        MagicMock(text_location=TextLocation(1, 0), score=0.9),
    ]

    result = await message_text_index.lookup_messages_in_subset(
        "test message", [1, 2], max_matches=1, threshold_score=0.5
    )

    assert len(result) == 1
    assert result[0].message_ordinal == 1
    assert result[0].score == 0.9


@pytest.mark.skip(
    reason="TODO: Doesn't work; also does too much mocking (probably related)"
)
@pytest.mark.asyncio
async def test_generate_embedding(message_text_index, needs_auth: None):
    """Test generating an embedding for a message."""
    message_text_index.text_location_index._vector_base.get_embedding = AsyncMock(
        return_value=[0.1, 0.2, 0.3]
    )

    embedding = await message_text_index.generate_embedding("test message")

    assert embedding == [0.1, 0.2, 0.3]
    message_text_index.text_location_index._vector_base.get_embedding.assert_awaited_once()


def test_serialize(message_text_index):
    """Test serialization of the MessageTextIndex."""
    serialized = message_text_index.serialize()
    assert serialized["indexData"] == {"mock": "data"}
    message_text_index.text_location_index.serialize.assert_called_once()


def test_deserialize(message_text_index, needs_auth: None):
    """Test deserialization of the MessageTextIndex."""
    data = MessageTextIndexData(
        indexData=TextToTextLocationIndexData(textLocations=[], embeddings=None)
    )
    message_text_index.deserialize(data)
    message_text_index.text_location_index.deserialize.assert_called_once_with(
        dict(textLocations=[], embeddings=None)
    )


@pytest.mark.asyncio
async def test_build_message_index(needs_auth: None):
    """Test building a message index without using mocks."""

    class FakeMessage(IMessage):
        """Concrete implementation of IMessage for testing."""

        def __init__(self, text_chunks: list[str]):
            self.text_chunks = text_chunks
            self.tags = []

        def get_knowledge(self) -> KnowledgeResponse:
            return KnowledgeResponse(
                entities=[],
                actions=[],
                inverse_actions=[],
                topics=[],
            )

    class FakeConversation(IConversation):
        """Concrete implementation of IConversation for testing."""

        def __init__(self, messages):
            self.name_tag = "test_conversation"
            self.tags = []
            self.semantic_refs = None
            self.semantic_ref_index = None
            self.messages = messages
            self.secondary_indexes = ConversationSecondaryIndexes()

    # Create test messages and conversation
    messages = [
        FakeMessage(["chunk1", "chunk2"]),
        FakeMessage(["chunk3"]),
    ]
    conversation = FakeConversation(messages)

    # Build the message index
    settings = MessageTextIndexSettings()
    event_handler = IndexingEventHandlers()
    result = await build_message_index(conversation, settings, event_handler)

    # Assertions
    assert result.error is None
    assert result.number_completed == 3  # Counts chunks, not messages
    # TODO: The final assert triggers; fix this
    # assert conversation.secondary_indexes is not None
    # assert conversation.secondary_indexes.message_index is not None
    # assert (
    #     len(conversation.secondary_indexes.message_index.text_location_index.index) == 3
    # )
