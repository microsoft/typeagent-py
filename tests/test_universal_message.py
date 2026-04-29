# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import timedelta

from typeagent.knowpro.universal_message import (
    ConversationMessage,
    ConversationMessageData,
    ConversationMessageMeta,
    format_timestamp_utc,
    UNIX_EPOCH,
)


def test_conversation_message_deserialize_listeners() -> None:
    timestamp = format_timestamp_utc(UNIX_EPOCH + timedelta(seconds=1))
    message_data: ConversationMessageData = {
        "textChunks": ["Hello"],
        "metadata": {"speaker": "alice", "listeners": ["bob"]},
        "tags": [],
        "timestamp": timestamp,
    }

    message = ConversationMessage.deserialize(message_data)

    assert message.metadata.speaker == "alice"
    assert message.metadata.recipients == ["bob"]


def test_conversation_message_serialize_listeners() -> None:
    timestamp = format_timestamp_utc(UNIX_EPOCH + timedelta(seconds=2))
    message = ConversationMessage(
        text_chunks=["Hi"],
        metadata=ConversationMessageMeta(speaker="alice", recipients=["bob"]),
        tags=[],
        timestamp=timestamp,
    )

    serialized = message.serialize()

    assert serialized["metadata"]["listeners"] == ["bob"]
    assert "recipients" not in serialized["metadata"]
