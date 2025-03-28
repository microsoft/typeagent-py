# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from typing import Any, NotRequired, TypedDict, cast

from ..aitools.embeddings import NormalizedEmbeddings
from .interfaces import IConversationDataWithIndexes

import numpy as np
from numpy.typing import NDArray

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

    json_data = cast(ConversationJsonData, conversation_data.copy())
    json_data["fileHeader"] = file_header
    json_data["embeddingFileHeader"] = embedding_file_header

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

    file_data = ConversationFileData(
        jsonData=json_data,
        binaryData=binary_data,
    )

    return file_data
