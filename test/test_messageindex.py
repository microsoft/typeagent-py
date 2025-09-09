# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import cast

from fixtures import FakeConversation, FakeMessage
from typeagent.storage.memory.messageindex import (
    MessageTextIndex,
    build_message_index,
    IMessageTextEmbeddingIndex,
)
from typeagent.knowpro.convsettings import MessageTextIndexSettings

from typeagent.knowpro.interfaces import (
    MessageTextIndexData,
    TextLocation,
    TextToTextLocationIndexData,
)
from typeagent.storage.memory import (
    MemoryStorageProvider,
    MemoryMessageCollection,
)
from typeagent.knowpro.textlocindex import TextToTextLocationIndex

from fixtures import needs_auth


@pytest.fixture
def mock_text_location_index() -> MagicMock:
    """Fixture to mock the TextToTextLocationIndex."""
    mock_index = MagicMock(spec=TextToTextLocationIndex)
    # Empty index, so first message starts at ordinal 0
    mock_index.size = AsyncMock(return_value=0)
    mock_index.add_text_locations = AsyncMock(return_value=None)
    mock_index.lookup_text = AsyncMock(return_value=[])
    mock_index.lookup_text_in_subset = AsyncMock(return_value=[])
    mock_index.serialize = MagicMock(return_value={"mock": "data"})
    mock_index.deserialize = MagicMock()
    return mock_index


@pytest.fixture
def message_text_index(
    mock_text_location_index: MagicMock,
) -> IMessageTextEmbeddingIndex:
    """Fixture to create a MessageTextIndex instance with a mocked TextToTextLocationIndex."""
    from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
    from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings

    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(test_model)
    settings = MessageTextIndexSettings(embedding_settings)
    index = MessageTextIndex(settings)
    index.text_location_index = mock_text_location_index
    return index


def test_message_text_index_init(needs_auth: None):
    """Test initialization of MessageTextIndex."""
    from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
    from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings

    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(test_model)
    settings = MessageTextIndexSettings(embedding_settings)
    index = MessageTextIndex(settings)
    assert index.settings == settings
    assert isinstance(index.text_location_index, TextToTextLocationIndex)


@pytest.mark.asyncio
async def test_add_messages(
    message_text_index: IMessageTextEmbeddingIndex, needs_auth: None
):
    """Test adding messages to the MessageTextIndex."""
    messages = [
        MagicMock(text_chunks=["chunk1", "chunk2"]),
        MagicMock(text_chunks=["chunk3"]),
    ]

    await message_text_index.add_messages(messages)

    # Check that add_text_locations was called with the expected text and location data
    mock_text_loc_index = cast(
        MagicMock, cast(MessageTextIndex, message_text_index).text_location_index
    )
    call_args = mock_text_loc_index.add_text_locations.call_args
    assert call_args is not None
    text_and_locations = call_args[0][0]  # First positional argument
    assert (
        len(text_and_locations) == 3
    )  # Two chunks from first message, one from second
    assert text_and_locations[0] == (
        "chunk1",
        TextLocation(0, 0),
    )  # First message starts at ordinal 0
    assert text_and_locations[1] == ("chunk2", TextLocation(0, 1))
    assert text_and_locations[2] == (
        "chunk3",
        TextLocation(1, 0),
    )  # Second message at ordinal 1


@pytest.mark.asyncio
async def test_lookup_messages(message_text_index: IMessageTextEmbeddingIndex):
    """Test looking up messages in the MessageTextIndex."""
    mock_text_loc_index = cast(
        MagicMock, cast(MessageTextIndex, message_text_index).text_location_index
    )
    mock_text_loc_index.lookup_text.return_value = [
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
async def test_lookup_messages_in_subset(
    message_text_index: IMessageTextEmbeddingIndex,
):
    """Test looking up messages in a subset of the MessageTextIndex."""
    mock_text_loc_index = cast(
        MagicMock, cast(MessageTextIndex, message_text_index).text_location_index
    )
    mock_text_loc_index.lookup_text_in_subset.return_value = [
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
async def test_generate_embedding(message_text_index: IMessageTextEmbeddingIndex):
    """Test generating an embedding for a message."""
    mock_text_loc_index = cast(
        MagicMock, cast(MessageTextIndex, message_text_index).text_location_index
    )
    mock_text_loc_index._vector_base.get_embedding = AsyncMock(
        return_value=[0.1, 0.2, 0.3]
    )

    embedding = await message_text_index.generate_embedding("test message")

    assert embedding == [0.1, 0.2, 0.3]
    mock_text_loc_index._vector_base.get_embedding.assert_awaited_once()


@pytest.mark.asyncio
async def test_serialize(message_text_index: IMessageTextEmbeddingIndex):
    """Test serialization of the MessageTextIndex."""
    serialized = await message_text_index.serialize()
    assert serialized.get("indexData") == {"mock": "data"}
    mock_text_loc_index = cast(
        MagicMock, cast(MessageTextIndex, message_text_index).text_location_index
    )
    mock_text_loc_index.serialize.assert_called_once()


@pytest.mark.asyncio
async def test_deserialize(message_text_index: IMessageTextEmbeddingIndex):
    """Test deserialization of the MessageTextIndex."""
    data = MessageTextIndexData(
        indexData=TextToTextLocationIndexData(textLocations=[], embeddings=None)
    )
    await message_text_index.deserialize(data)
    mock_text_loc_index = cast(
        MagicMock, cast(MessageTextIndex, message_text_index).text_location_index
    )
    mock_text_loc_index.deserialize.assert_called_once_with(
        dict(textLocations=[], embeddings=None)
    )


@pytest.mark.asyncio
async def test_build_message_index(needs_auth: None):
    """Test building a message index without using mocks."""

    # Create test messages and conversation
    messages = [
        FakeMessage(["chunk1", "chunk2"]),
        FakeMessage(["chunk3"]),
    ]

    # Create storage provider asynchronously
    from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
    from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
    from typeagent.knowpro.convsettings import (
        MessageTextIndexSettings,
        RelatedTermIndexSettings,
    )

    test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
    embedding_settings = TextEmbeddingIndexSettings(test_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    storage_provider = MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )
    conversation = FakeConversation(
        messages=messages, storage_provider=storage_provider, has_secondary_indexes=True
    )

    # Build the message index
    # Pass the storage provider instead of settings
    storage_provider = await conversation.settings.get_storage_provider()
    await build_message_index(conversation, storage_provider)

    # TODO: The final assert triggers; fix this
    # assert conversation.secondary_indexes is not None
    # assert conversation.secondary_indexes.message_index is not None
    # assert (
    #     len(conversation.secondary_indexes.message_index.text_location_index.index) == 3
    # )
