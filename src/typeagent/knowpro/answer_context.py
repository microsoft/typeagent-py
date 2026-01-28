# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Answer context utilities and chunking helpers."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any

from .answer_context_schema import AnswerContext, RelevantKnowledge, RelevantMessage


@dataclass
class AnswerContextOptions:
    entities_top_k: int | None = None
    topics_top_k: int | None = None
    messages_top_k: int | None = None
    chunking: bool | None = None
    debug: bool | None = None


def answer_context_to_string(context: AnswerContext, spaces: int | None = 0) -> str:
    json_parts: list[str] = ["{\n"]
    property_count = 0

    if context.entities:
        json_parts.append(
            _add_property("entities", context.entities, property_count, spaces)
        )
        property_count += 1
    if context.topics:
        json_parts.append(
            _add_property("topics", context.topics, property_count, spaces)
        )
        property_count += 1
    if context.messages:
        json_parts.append(
            _add_property("messages", context.messages, property_count, spaces)
        )
        property_count += 1

    json_parts.append("\n}")
    return "".join(json_parts)


def _add_property(name: str, value: Any, count: int, spaces: int | None) -> str:
    text = ""
    if count > 0:
        text += ",\n"
    text += f'"{name}": {json_stringify_for_prompt(value, spaces)}'
    return text


class AnswerContextChunkBuilder:
    def __init__(self, context: AnswerContext, max_chars_per_chunk: int) -> None:
        self.context = context
        self.max_chars_per_chunk = max_chars_per_chunk
        self.current_chunk = AnswerContext()
        self.current_chunk_char_count = 0

    def get_chunks(
        self, include_knowledge: bool = True, include_messages: bool = True
    ) -> Iterator[AnswerContext]:
        self._new_chunk()
        if include_knowledge:
            for chunk in self._chunk_knowledge(self.context.entities, "entities"):
                yield chunk
            for chunk in self._chunk_knowledge(self.context.topics, "topics"):
                yield chunk
            if self.current_chunk_char_count > 0:
                yield self.current_chunk
                self._new_chunk()
        if include_messages:
            for chunk in self._chunk_messages():
                yield chunk
        if self.current_chunk_char_count > 0:
            yield self.current_chunk

    def _chunk_knowledge(
        self, knowledge: list[RelevantKnowledge] | None, type_name: str
    ) -> Iterator[AnswerContext]:
        if knowledge:
            for item in knowledge:
                completed_chunk = self._add_to_current_chunk(item, type_name)
                if completed_chunk is not None:
                    yield completed_chunk

    def _chunk_messages(self) -> Iterator[AnswerContext]:
        if self.context.messages:
            for message in self.context.messages:
                if not message.message_text:
                    continue
                message_chunks = split_large_text_into_chunks(
                    message.message_text, self.max_chars_per_chunk
                )
                for msg_chunk in message_chunks:
                    chunk_message = RelevantMessage(
                        from_=message.from_,
                        to=message.to,
                        timestamp=message.timestamp,
                        message_text=msg_chunk,
                    )
                    completed_chunk = self._add_to_current_chunk(
                        chunk_message, "messages"
                    )
                    if completed_chunk is not None:
                        yield completed_chunk

    def _add_to_current_chunk(self, item: Any, type_name: str) -> AnswerContext | None:
        item_string = json_stringify_for_prompt(item)
        item_size = len(item_string)
        if (
            self.current_chunk_char_count + item_size > self.max_chars_per_chunk
            and self.current_chunk_char_count > 0
        ):
            completed_chunk = self.current_chunk
            self._new_chunk()
            self._append_item(item, type_name, item_size)
            return completed_chunk
        self._append_item(item, type_name, item_size)
        return None

    def _append_item(self, item: Any, type_name: str, item_size: int) -> None:
        if getattr(self.current_chunk, type_name) is None:
            setattr(self.current_chunk, type_name, [])
        getattr(self.current_chunk, type_name).append(item)
        self.current_chunk_char_count += item_size

    def _new_chunk(self) -> None:
        self.current_chunk = AnswerContext()
        self.current_chunk_char_count = 0


def json_stringify_for_prompt(value: Any, spaces: int | None = None) -> str:
    serializable = _to_prompt_value(value)
    return json.dumps(
        serializable,
        ensure_ascii=False,
        indent=spaces,
        separators=(",", ":") if spaces is None else None,
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def split_large_text_into_chunks(
    text: str | list[str], max_chars_per_chunk: int
) -> list[str]:
    if isinstance(text, list):
        chunks: list[str] = []
        for part in text:
            chunks.extend(split_large_text_into_chunks(part, max_chars_per_chunk))
        return chunks
    if len(text) <= max_chars_per_chunk:
        return [text]
    return list(_split_text_by_paragraph(text, max_chars_per_chunk))


def _split_text_by_paragraph(text: str, max_chars_per_chunk: int) -> Iterable[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text[:max_chars_per_chunk]]
    return _merge_chunks(paragraphs, "\n\n", max_chars_per_chunk)


def _merge_chunks(
    chunks: Iterable[str], separator: str, max_chars_per_chunk: int
) -> Iterable[str]:
    current = ""
    for new_chunk in chunks:
        if len(new_chunk) > max_chars_per_chunk:
            new_chunk = new_chunk[:max_chars_per_chunk]
        if (
            current
            and len(current) + len(new_chunk) + len(separator) > max_chars_per_chunk
        ):
            yield current
            current = new_chunk
        else:
            current = new_chunk if not current else f"{current}{separator}{new_chunk}"
    if current:
        yield current


def _to_prompt_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__pydantic_serializer__"):
        return value.__pydantic_serializer__.to_python(  # type: ignore[attr-defined]
            value, by_alias=True, exclude_none=True
        )
    if hasattr(value, "__annotations__"):
        data: dict[str, Any] = {}
        for key in value.__annotations__:
            item = getattr(value, key, None)
            if item is not None:
                data[key] = _to_prompt_value(item)
        return data
    if isinstance(value, dict):
        return {
            key: _to_prompt_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_to_prompt_value(item) for item in value]
    return value
