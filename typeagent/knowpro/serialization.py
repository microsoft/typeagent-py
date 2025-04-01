# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import is_dataclass, MISSING
import functools
import json
import types
from typing import (
    Annotated,
    Any,
    Literal,
    get_origin,
    get_args,
    Union,
    NotRequired,
    overload,
    TypedDict,
)

import numpy as np

from ..aitools.embeddings import NormalizedEmbeddings
from .interfaces import (
    IConversationDataWithIndexes,
    Tag,
    Topic,
)
from . import kplib


# -------------------
# Shared definitions
# -------------------


DATA_FILE_SUFFIX = "_data.json"
EMBEDDING_FILE_SUFFIX = "_embeddings.bin"


class FileHeader(TypedDict):
    version: str


# Needed to create a TypedDict.
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


# --------------
# Serialization
# ---------------


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


# This converts any vanilla class instance to a dict, recursively,
# with field named converted to camelCase.
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
def to_camel(name: str) -> str:
    assert isinstance(name, str), f"Cannot convert {name!r} to camel case"
    # Name must be of the form foo_bar_baz.
    # Result will be fooBarBaz.
    # Don't pass edge cases.
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


# ----------------
# Deserialization
# -----------------


# No exceptions are caught; they just bubble out.
async def read_conversation_data_from_file(
    filename: str, embedding_size: int | None = None
) -> IConversationDataWithIndexes | None:
    with open(filename + DATA_FILE_SUFFIX) as f:
        json_data: ConversationJsonData = json.load(f)
    if json_data is None:
        # A serialized None -- file contained exactly "null".
        return None
    # TODO: validate json_data.
    embeddings: NormalizedEmbeddings | None = None
    if embedding_size:
        with open(filename + EMBEDDING_FILE_SUFFIX, "rb") as f:
            embeddings = np.fromfile(f, dtype=np.float32).reshape((-1, embedding_size))
    else:
        embeddings = None
    file_data = ConversationFileData(
        jsonData=json_data,
        binaryData=ConversationBinaryData(
            embeddings=None if embeddings is None else bytearray(embeddings.tobytes())
        ),
    )
    if json_data.get("fileHeader") is None:
        json_data["fileHeader"] = create_file_header()
    return from_conversation_file_data(file_data)


def from_conversation_file_data(
    file_data: ConversationFileData,
) -> IConversationDataWithIndexes | None:
    json_data = file_data["jsonData"]
    binary_data = file_data["binaryData"]
    if binary_data:
        embeddings = binary_data.get("embeddings")
        if embeddings:
            embeddings = np.frombuffer(embeddings, dtype=np.float32)
    else:
        embeddings = None
    # TODO: proper return value, and remove '| None' from return type.


TYPE_MAP = {
    "entity": kplib.ConcreteEntity,
    "action": kplib.Action,
    "topic": Topic,
    "tag": Tag,
}


# Looks like this only works for knowledge...
def deserialize_knowledge(knowledge_type: str, obj: Any) -> Any:
    typ = TYPE_MAP[knowledge_type]
    return deserialize_object(typ, obj)


# The rest of this file was written by o3-mini=high, with few errors.


def is_primitive(typ: type) -> bool:
    return typ in (int, float, bool, str, type(None))


def deserialize_object(typ: Any, obj: Any) -> Any:
    origin = get_origin(typ)

    # Handle Annotated by substituting its first argument for typ.
    if origin is Annotated:
        typ = get_args(typ)[0]
        origin = get_origin(typ)  # Get the first type argument.

    # Non-generic: primitives and dataclasses.
    if origin is None:
        if is_primitive(typ):
            if typ is not type(None) and not isinstance(obj, typ):
                raise ValueError(f"Expected {typ} but got {type(obj)}")
            return obj
        elif isinstance(typ, type) and is_dataclass(typ):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected dict for {typ}, got {type(obj)}")
            kwargs = {}
            for field, field_type in typ.__annotations__.items():
                json_key = to_camel(field)
                if json_key in obj:
                    kwargs[field] = deserialize_object(field_type, obj[json_key])
            return typ(**kwargs)
        else:
            breakpoint()  # What would this be?
            return obj

    # Handle Lieral.
    if origin is Literal:
        if type(obj) is str and obj in get_args(typ):
            return obj
        raise ValueError(
            f"Expected one of {get_args(typ)} for Literal, but got {obj!r} of type {type(obj)}"
        )

    # Handle list[T] / List[T].
    if origin is list:
        if not isinstance(obj, list):
            raise ValueError(f"Expected list for list, got {type(obj)}")
        (elem_type,) = get_args(typ)
        return [deserialize_object(elem_type, item) for item in obj]

    # Handle tuple[T1, T2, etc.] / Tuple[T1, T2, etc.].
    if origin is tuple:
        if not isinstance(obj, list):
            raise ValueError(f"Expected list for tuple, got {type(obj)}")
        args = get_args(typ)
        if len(args) != len(obj):
            raise ValueError(
                f"Tuple length mismatch: expected {len(args)}, got {len(obj)}"
            )
        return tuple(deserialize_object(t, item) for t, item in zip(args, obj))

    # Handle Union[X, Y], Optional[X], and X | Y.
    if origin in (Union, types.UnionType):
        candidates = get_args(typ)
        # Disambiguate among dataclasses if possible.
        dataclass_candidates = [
            c for c in candidates if isinstance(c, type) and is_dataclass(c)
        ]
        if dataclass_candidates and isinstance(obj, dict):
            matching = []
            for candidate in dataclass_candidates:
                mandatory = {
                    to_camel(name)
                    for name, field in candidate.__dataclass_fields__.items()
                    if field.default is MISSING and field.default_factory is MISSING
                }
                if mandatory.issubset(obj.keys()):
                    matching.append(candidate)
            if len(matching) == 1:
                return deserialize_object(matching[0], obj)
            elif len(matching) > 1:
                raise ValueError(
                    "Ambiguous union: multiple dataclass candidates match: "
                    + str([c.__name__ for c in matching])
                )
        # Try each candidate until one succeeds.
        last_exc = None
        for candidate in candidates:
            try:
                return deserialize_object(candidate, obj)
            except (
                Exception
            ) as e:  # TODO: Something better than catching all exceptions.
                last_exc = e
        raise ValueError(f"No union candidate succeeded. Last error: {last_exc}")

    breakpoint()
    raise TypeError(f"Unsupported type {typ}, object {obj!r} of type {type(obj)}")
