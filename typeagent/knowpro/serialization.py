# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import is_dataclass, MISSING
from datetime import datetime
import functools
import json
import types
from typing import (
    Annotated,
    Any,
    cast,
    get_args,
    get_origin,
    Literal,
    NotRequired,
    overload,
    TypeAliasType,
    TypedDict,
    Union,
)

import numpy as np
from pydantic.alias_generators import to_camel

from ..aitools.embeddings import NormalizedEmbeddings

from .interfaces import (
    ConversationDataWithIndexes,
    SearchTermGroupTypes,
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


class ConversationJsonData[TMessageData](ConversationDataWithIndexes[TMessageData]):
    fileHeader: NotRequired[FileHeader | None]
    embeddingFileHeader: NotRequired[EmbeddingFileHeader | None]


class ConversationBinaryData(TypedDict):
    embeddingsList: NotRequired[list[NormalizedEmbeddings] | None]


class ConversationFileData[TMessageData](TypedDict):
    # This data goes into a JSON text file
    jsonData: ConversationJsonData[TMessageData]
    # This goes into a single binary file
    binaryData: ConversationBinaryData


# --------------
# Serialization
# ---------------


def write_conversation_data_to_file[TMessageData](
    conversation_data: ConversationDataWithIndexes[TMessageData],
    filename: str,
) -> None:
    file_data = to_conversation_file_data(conversation_data)
    binary_data = file_data["binaryData"]
    if binary_data:
        embeddings_list = binary_data.get("embeddingsList")
        if embeddings_list:
            with open(filename + EMBEDDING_FILE_SUFFIX, "wb") as f:
                for embeddings in embeddings_list:
                    embeddings.tofile(f)
    with open(filename + DATA_FILE_SUFFIX, "w", encoding="utf-8") as f:
        # f.write(repr(file_data["jsonData"]))
        json.dump(file_data["jsonData"], f)


def serialize_embeddings(embeddings: NormalizedEmbeddings) -> NormalizedEmbeddings:
    return np.concatenate(embeddings)


def to_conversation_file_data[TMessageData](
    conversation_data: ConversationDataWithIndexes[TMessageData],
) -> ConversationFileData[TMessageData]:
    file_header = create_file_header()
    embedding_file_header = EmbeddingFileHeader()

    embeddings_list: list[NormalizedEmbeddings] = []

    related_terms_index_data = conversation_data.get("relatedTermsIndexData")
    if related_terms_index_data is not None:
        text_embedding_data = related_terms_index_data.get("textEmbeddingData")
        if text_embedding_data is not None:
            embeddings = text_embedding_data.get("embeddings")
            if embeddings is not None:
                embeddings_list.append(embeddings)
                text_embedding_data["embeddings"] = None
                embedding_file_header["relatedCount"] = len(embeddings)

    message_index_data = conversation_data.get("messageIndexData")
    if message_index_data is not None:
        text_embedding_data = message_index_data.get("indexData")
        if text_embedding_data is not None:
            embeddings = text_embedding_data.get("embeddings")
            if embeddings is not None:
                embeddings_list.append(embeddings)
                text_embedding_data["embeddings"] = None
                embedding_file_header["messageCount"] = len(embeddings)

    binary_data = ConversationBinaryData(embeddingsList=embeddings_list)
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


# This converts any Pydantic dataclass instance to a dict, recursively,
# with field names converted to camelCase.
@overload
def serialize_object(arg: None) -> None: ...
@overload
def serialize_object(arg: object) -> Any:
    # NOTE: This now works specifically with Pydantic dataclasses.
    ...


def serialize_object(arg: Any) -> Any | None:
    if arg is None:
        return None

    # Require Pydantic dataclass
    if not hasattr(arg, "__pydantic_serializer__"):
        raise TypeError(f"Object must be a Pydantic dataclass, got {type(arg)}")

    # Use Pydantic's serialization with aliases
    return arg.__pydantic_serializer__.to_python(arg, by_alias=True)  # type: ignore


# ----------------
# Deserialization
# -----------------


def from_conversation_file_data(
    file_data: ConversationFileData[Any],
) -> ConversationDataWithIndexes[Any]:
    json_data = file_data["jsonData"]
    file_header = json_data.get("fileHeader")
    if file_header is None:
        raise DeserializationError("Missing file header")
    if file_header["version"] != "0.1":
        raise DeserializationError(f"Unsupported file version {file_header['version']}")
    embedding_file_header = json_data.get("embeddingFileHeader")
    if embedding_file_header is None:
        raise DeserializationError("Missing embedding file header")

    binary_data = file_data["binaryData"]
    if binary_data:
        embeddings_list = binary_data.get("embeddingsList")
        if embeddings_list is None:
            raise DeserializationError("Missing embeddings list")
        if len(embeddings_list) != 1:
            raise ValueError(
                f"Expected embeddings list of lengt 1, got {len(embeddings_list)}"
            )
        embeddings = embeddings_list[0]
        pos = 0
        pos += get_embeddings_from_binary_data(
            embeddings,
            json_data,
            ("relatedTermsIndexData", "textEmbeddingData"),
            pos,
            embedding_file_header.get("relatedCount"),
        )
        pos += get_embeddings_from_binary_data(
            embeddings,
            json_data,
            ("messageIndexData", "indexData"),
            pos,
            embedding_file_header.get("messageCount"),
        )
    return json_data


def get_embeddings_from_binary_data(
    embeddings: NormalizedEmbeddings,
    json_data: ConversationJsonData[Any],
    keys: tuple[str, ...],
    offset: int,
    count: int | None,
) -> int:
    if count is None or count <= 0:
        return 0
    embeddings = embeddings[offset : offset + count]  # Simple np slice creates a view.
    if len(embeddings) != count:
        raise DeserializationError(
            f"Expected {count} embeddings, got {len(embeddings)}"
        )
    data: dict[str, object] = cast(
        dict[str, object], json_data
    )  # We know it's a dict, but pyright doesn't.
    # Traverse the keys to get to the embeddings.
    for key in keys:
        new_data = data.get(key)
        if new_data is None or type(new_data) is not dict:
            return 0
        data = new_data
    if "embeddings" in data:
        data["embeddings"] = embeddings
    return count


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


class DeserializationError(Exception):
    pass


@functools.cache
def is_primitive(typ: type) -> bool:
    return typ in (int, float, bool, str, type(None))


# TODO: Use type(obj) is X instead of isinstance(obj, X). It's faster.
# TODO: Design a consistent reporting format.
# TODO: Doesn't Pydantic have this functionality?
def deserialize_object(typ: Any, obj: Any) -> Any:
    if isinstance(typ, str):
        # A forward reference; special-case those that matter.
        match typ:
            case "SearchTermGroupTypes":
                typ = SearchTermGroupTypes
            case _:
                raise DeserializationError(f"Unknown forward type reference {typ!r}")
    elif typ is datetime:
        # Special case for datetime, which is serialized as a string.
        if isinstance(obj, str):
            try:
                return datetime.fromisoformat(obj)
            except ValueError as e:
                raise DeserializationError(f"Invalid datetime string {obj!r}") from e
        else:
            raise DeserializationError(
                f"Expected datetime string, got {type(obj)}: {obj!r}"
            )
    if typ.__class__ is TypeAliasType:
        typ = typ.__value__
    origin = get_origin(typ)

    # Handle Annotated by substituting its first argument for typ.
    if origin is Annotated:
        typ = get_args(typ)[0]
        if typ.__class__ is TypeAliasType:
            typ = typ.__value__
        origin = get_origin(typ)  # Get the first type argument.

    # Non-generic: primitives and dataclasses.
    if origin is None:
        if is_primitive(typ):
            if typ is int and type(obj) is float:
                return int(obj)
            if typ is float and type(obj) is int:
                return float(obj)
            if not isinstance(obj, typ):
                raise DeserializationError(f"Expected {typ} but got {type(obj)}")
            return obj
        elif isinstance(typ, type) and is_dataclass(typ):
            if not isinstance(obj, dict):
                raise DeserializationError(f"Expected dict for {typ}, got {type(obj)}")

            # Require Pydantic dataclass
            if not hasattr(typ, "__pydantic_validator__"):
                raise TypeError(f"Type must be a Pydantic dataclass, got {typ}")

            try:
                # Use Pydantic's validator with aliases
                return typ.__pydantic_validator__.validate_python(obj)  # type: ignore
            except Exception as e:
                raise DeserializationError(
                    f"Pydantic validation failed for {typ.__name__}: {e}"
                ) from e
        else:
            # Could be a class that's not a dataclass -- we don't know the signature.
            raise TypeError(f"Unsupported origin-less type {typ}")

    # Handle Literal.
    if origin is Literal:
        if type(obj) is str and obj in get_args(typ):
            return obj
        raise DeserializationError(
            f"Expected one of {get_args(typ)} for Literal, but got {obj!r} of type {type(obj)}"
        )

    # Handle list[T] / List[T].
    if origin is list:
        if not isinstance(obj, list):
            raise DeserializationError(f"Expected list for list, got {type(obj)}")
        (elem_type,) = get_args(typ)
        return [deserialize_object(elem_type, item) for item in obj]

    # Handle tuple[T1, T2, etc.] / Tuple[T1, T2, etc.].
    if origin is tuple:
        if not isinstance(obj, list):
            raise DeserializationError(f"Expected list for tuple, got {type(obj)}")
        args = get_args(typ)
        if len(args) != len(obj):
            raise DeserializationError(
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
                raise TypeError(
                    f"Ambiguous union {typ}: multiple dataclass candidates match: "
                    + str([c.__name__ for c in matching])
                )
        # Try each candidate until one succeeds.
        all_excs = []
        for candidate in candidates:
            try:
                return deserialize_object(candidate, obj)
            except DeserializationError as e:
                all_excs.append(e)
        raise DeserializationError(
            f"No candidate from union {typ} succeeded -- errors: {all_excs}"
        )

    raise TypeError(f"Unsupported type {typ}, object {obj!r} of type {type(obj)}")
