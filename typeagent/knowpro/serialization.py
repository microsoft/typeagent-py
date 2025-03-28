# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
import json
from typing import Any, NotRequired, overload, TypedDict

from ..aitools.embeddings import NormalizedEmbeddings
from .interfaces import (
    IConversationDataWithIndexes,
    Knowledge,
    KnowledgeData,
    KnowledgeType,
)

import numpy as np

DATA_FILE_SUFFIX = "_data.json"
EMBEDDING_FILE_SUFFIX = "_embeddings.bin"


class FileHeader(TypedDict):
    version: str


def create_file_header() -> FileHeader:
    return FileHeader(version="0.1")


class EmbeddingFileHeader(TypedDict):
    relatedCount: NotRequired[int | None]
    messageCount: NotRequired[int | None]


class EmbeddingData(TypedDict):
    embeddings: NormalizedEmbeddings | None


class ConversationJsonData(IConversationDataWithIndexes):
    fileHeader: NotRequired[FileHeader | None]
    embeddingFileHeader: NotRequired[EmbeddingFileHeader | None]


class ConversationBinaryData(TypedDict):
    embeddings: NotRequired[bytearray | None]


class ConversationFileData(TypedDict):
    # This data goes into a JSON text file
    jsonData: ConversationJsonData
    # This goes into a single binary file
    binaryData: ConversationBinaryData


def write_conversation_data_to_file(
    conversation_data: IConversationDataWithIndexes,
    filename: str,
) -> None:
    file_data = to_conversation_file_data(conversation_data)
    binary_data = file_data["binaryData"]
    if binary_data:
        embeddings = binary_data.get("embeddings")
        if embeddings:
            with open(filename + EMBEDDING_FILE_SUFFIX, "wb") as f:
                f.write(embeddings)
    with open(filename + DATA_FILE_SUFFIX, "w") as f:
        # f.write(repr(file_data["jsonData"]))
        json.dump(file_data["jsonData"], f)


def serialize_embeddings(embeddings: NormalizedEmbeddings) -> NormalizedEmbeddings:
    return np.concatenate(embeddings)


def to_conversation_file_data[IMessageData](
    conversation_data: IConversationDataWithIndexes[IMessageData],
) -> ConversationFileData:
    file_header = create_file_header()
    embedding_file_header = EmbeddingFileHeader()

    buffer = bytearray()

    related_terms_index_data = conversation_data.get("relatedTermsIndexData")
    if related_terms_index_data is not None:
        text_embedding_data = related_terms_index_data.get("textEmbeddingData")
        if text_embedding_data is not None:
            embeddings = text_embedding_data.get("embeddings")
            if embeddings is not None:
                buffer.extend(embeddings)
                text_embedding_data["embeddings"] = None
                embedding_file_header["relatedCount"] = len(embeddings)

    message_index_data = conversation_data.get("messageIndexData")
    if message_index_data is not None:
        text_embedding_data = message_index_data.get("indexData")
        if text_embedding_data is not None:
            index_embeddings = text_embedding_data.get("embeddings")
            if index_embeddings is not None:
                embeddings = index_embeddings.get("embeddings")
                if embeddings is not None:
                    buffer.extend(embeddings)
                    index_embeddings["embeddings"] = None
                    embedding_file_header["messageCount"] = len(embeddings)

    binary_data = ConversationBinaryData(embeddings=buffer)
    json_data = ConversationJsonData(
        **conversation_data,
        fileHeader=file_header,
        embeddingFileHeader=embedding_file_header,
    )
    file_data = ConversationFileData(
        jsonData=json_data,
        binaryData=binary_data,
    )

    return file_data


@overload
def serialize_object(arg: None) -> None: ...
@overload
def serialize_object(arg: object) -> Any:
    # NOTE: Actually this only takes objects with a __dict__.
    ...


def serialize_object(arg: Any) -> Any | None:
    if arg is None:
        return None
    assert hasattr(arg, "__dict__"), f"Cannot serialize knowledge of type {type(arg)}"
    result = to_json(arg)
    assert isinstance(result, dict), f"Serialized knowledge is not a dict: {result}"
    return result


def to_json(obj: Any) -> Any:
    if obj is None:
        return None
    d = getattr(obj, "__dict__", None)
    if d is not None:
        obj = d
    tp = type(obj)
    if tp is dict:
        return {to_camel(key): to_json(value) for key, value in obj.items()}
    if tp is list:
        return [to_json(value) for value in obj]
    if tp in (str, int, float, bool, None):
        return obj
    raise TypeError(f"Cannot jsonify {tp}: {obj!r}")


@functools.cache
def to_camel(name: str):
    assert isinstance(name, str), f"Cannot convert {name!r} to camel case"
    # Name must be of the form lowercase_words_separated_by_underscores.
    # Response will be lowercaseWordsSeparatedByUnderscores.
    # Don't pass edge cases.
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
