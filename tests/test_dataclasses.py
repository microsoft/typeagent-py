# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Tests for the camelCase-aliasing dataclass decorator."""

import pytest

from pydantic import ConfigDict, ValidationError

from typeagent.knowpro import serialization
from typeagent.knowpro.dataclasses import dataclass
from typeagent.knowpro.universal_message import (
    ConversationMessage,
    ConversationMessageMeta,
)


@dataclass
class Sample:
    plain: int
    two_words: str = "x"


def dump(obj: object) -> dict:
    return obj.__pydantic_serializer__.to_python(obj, by_alias=True)  # type: ignore[attr-defined]


def load[T](cls: type[T], data: dict) -> T:
    return cls.__pydantic_validator__.validate_python(data)  # type: ignore[attr-defined]


def test_constructor_uses_python_names() -> None:
    assert Sample(1, two_words="y") == Sample(plain=1, two_words="y")


def test_output_is_camel_case() -> None:
    assert dump(Sample(1, "y")) == {"plain": 1, "twoWords": "y"}


@pytest.mark.parametrize("key", ["two_words", "twoWords"])
def test_input_accepts_both_spellings(key: str) -> None:
    assert load(Sample, {"plain": 1, key: "y"}) == Sample(1, "y")


def test_caller_config_is_merged() -> None:
    @dataclass(config=ConfigDict(extra="forbid"))
    class Strict:
        two_words: str

    assert dump(Strict("y")) == {"twoWords": "y"}
    with pytest.raises(ValidationError):
        load(Strict, {"twoWords": "y", "other": 1})


def test_explicit_alias_wins() -> None:
    meta = ConversationMessageMeta(speaker="a", recipients=["b"])
    data = serialization.serialize_object(meta)
    assert data["listeners"] == ["b"]
    assert "recipients" not in data
    assert serialization.deserialize_object(ConversationMessageMeta, data) == meta
    assert (
        serialization.deserialize_object(
            ConversationMessageMeta, {"speaker": "a", "recipients": ["b"]}
        )
        == meta
    )


def test_message_metadata_is_required() -> None:
    with pytest.raises(serialization.DeserializationError, match="metadata"):
        serialization.deserialize_object(ConversationMessage, {"textChunks": ["hi"]})


def test_message_source_id_round_trip() -> None:
    meta = ConversationMessageMeta(speaker="a")
    msg = ConversationMessage(text_chunks=["hi"], metadata=meta, source_id="s1")
    data = serialization.serialize_object(msg)
    assert data["sourceId"] == "s1"
    assert "source_id" not in data
    assert serialization.deserialize_object(ConversationMessage, data) == msg

    # Data written before the alias generator used the Python name.
    data["source_id"] = data.pop("sourceId")
    assert serialization.deserialize_object(ConversationMessage, data) == msg
