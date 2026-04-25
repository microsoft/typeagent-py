# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from typeagent.knowpro.interfaces import (
    ConversationDataWithIndexes,
    MessageTextIndexData,
    Tag,
    TermsToRelatedTermsIndexData,
    TextToTextLocationIndexData,
    Topic,
)
from typeagent.knowpro.knowledge_schema import ConcreteEntity, Quantity
from typeagent.knowpro.serialization import (
    ConversationBinaryData,
    ConversationFileData,
    ConversationJsonData,
    create_file_header,
    DeserializationError,
    deserialize_knowledge,
    deserialize_object,
    from_conversation_file_data,
    is_primitive,
    serialize_embeddings,
    serialize_object,
    to_conversation_file_data,
    write_conversation_data_to_file,
)
from typeagent.podcasts.podcast import Podcast

type SampleData = Any  # Anything more refined causes type errors


@pytest.fixture
def sample_conversation_data() -> SampleData:
    """Fixture to provide sample conversation data."""
    return {
        "relatedTermsIndexData": {
            "textEmbeddingData": {
                "embeddings": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
            }
        },
        "messageIndexData": {
            "indexData": {
                "embeddings": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
            }
        },
    }


def test_serialize_object():
    """Test the serialize_object function."""
    entity = ConcreteEntity(name="ExampleEntity", type=["ExampleType"])
    serialized = serialize_object(entity)
    assert serialized == {
        "name": "ExampleEntity",
        "type": ["ExampleType"],
        "facets": None,
    }


def test_create_file_header():
    """Test the create_file_header function."""
    header = create_file_header()
    assert header["version"] == "0.1"


def test_serialize_embeddings():
    """Test the serialize_embeddings function."""
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    serialized = serialize_embeddings(embeddings)
    assert np.array_equal(serialized, embeddings.flatten())


def test_to_conversation_file_data(sample_conversation_data: SampleData):
    """Test the to_conversation_file_data function."""
    file_data = to_conversation_file_data(sample_conversation_data)
    assert "jsonData" in file_data
    assert "binaryData" in file_data
    embeddings_list = file_data["binaryData"].get("embeddingsList")
    assert embeddings_list is not None
    assert len(embeddings_list) == 2


def test_from_conversation_file_data():
    """Test the from_conversation_file_data function."""
    sample_conversation_data = ConversationDataWithIndexes(
        nameTag="mock name",
        messages=[],
        tags=[],
        semanticRefs=[],
        messageIndexData=MessageTextIndexData(
            indexData=TextToTextLocationIndexData(
                textLocations=[],
                embeddings=np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            )
        ),
        relatedTermsIndexData=TermsToRelatedTermsIndexData(),
    )

    file_data = to_conversation_file_data(sample_conversation_data)
    conversation_data = from_conversation_file_data(file_data)
    assert conversation_data is not None
    assert conversation_data.get("relatedTermsIndexData") is not None


def test_write_and_read_conversation_data(
    tmp_path: Path, sample_conversation_data: SampleData
):
    """Test writing and reading conversation data to and from files."""
    filename = tmp_path / "conversation"
    write_conversation_data_to_file(
        cast(ConversationDataWithIndexes, sample_conversation_data), str(filename)
    )

    # Read back the data
    read_data = Podcast._read_conversation_data_from_file(
        str(filename),
    )
    assert read_data is not None
    assert read_data.get("relatedTermsIndexData") is not None
    assert read_data.get("messageIndexData") is not None


def test_deserialize_object():
    """Test the deserialize_object function."""
    obj = {"amount": 5.0, "units": "kg"}
    deserialized = deserialize_object(Quantity, obj)
    assert isinstance(deserialized, Quantity)
    assert deserialized.amount == 5.0
    assert deserialized.units == "kg"


def test_deserialization_error():
    """Test that DeserializationError is raised for invalid data."""
    with pytest.raises(DeserializationError, match="Pydantic validation failed"):
        deserialize_object(Quantity, {"invalid_key": "value"})


# ---------------------------------------------------------------------------
# Additional tests for broader coverage
# ---------------------------------------------------------------------------


def test_from_conversation_file_data_missing_header_raises():
    """from_conversation_file_data raises when fileHeader is absent."""
    json_data: ConversationJsonData[Any] = ConversationJsonData(
        nameTag="x", messages=[], tags=[], semanticRefs=None
    )
    file_data: ConversationFileData[Any] = ConversationFileData(
        jsonData=json_data,
        binaryData=ConversationBinaryData(embeddingsList=[]),
    )
    with pytest.raises(DeserializationError, match="Missing file header"):
        from_conversation_file_data(file_data)


def test_from_conversation_file_data_bad_version_raises():
    """from_conversation_file_data raises on unsupported version."""
    json_data: ConversationJsonData[Any] = ConversationJsonData(
        nameTag="x",
        messages=[],
        tags=[],
        semanticRefs=None,
        fileHeader={"version": "99.9"},
        embeddingFileHeader={},
    )
    file_data: ConversationFileData[Any] = ConversationFileData(
        jsonData=json_data,
        binaryData=ConversationBinaryData(embeddingsList=[]),
    )
    with pytest.raises(DeserializationError, match="Unsupported file version"):
        from_conversation_file_data(file_data)


def test_from_conversation_file_data_missing_embedding_header_raises():
    """from_conversation_file_data raises when embeddingFileHeader is absent."""
    json_data: ConversationJsonData[Any] = ConversationJsonData(
        nameTag="x",
        messages=[],
        tags=[],
        semanticRefs=None,
        fileHeader={"version": "0.1"},
    )
    file_data: ConversationFileData[Any] = ConversationFileData(
        jsonData=json_data,
        binaryData=ConversationBinaryData(embeddingsList=[]),
    )
    with pytest.raises(DeserializationError, match="Missing embedding file header"):
        from_conversation_file_data(file_data)


def test_from_conversation_file_data_missing_embeddings_list_raises():
    """from_conversation_file_data raises when embeddingsList is None."""
    json_data: ConversationJsonData[Any] = ConversationJsonData(
        nameTag="x",
        messages=[],
        tags=[],
        semanticRefs=None,
        fileHeader={"version": "0.1"},
        embeddingFileHeader={},
    )
    file_data: ConversationFileData[Any] = ConversationFileData(
        jsonData=json_data,
        binaryData=ConversationBinaryData(embeddingsList=None),
    )
    with pytest.raises(DeserializationError, match="Missing embeddings list"):
        from_conversation_file_data(file_data)


def test_from_conversation_file_data_success_empty():
    """from_conversation_file_data succeeds with minimal valid data."""
    emb = np.zeros((0, 4), dtype=np.float32)
    json_data: ConversationJsonData[Any] = ConversationJsonData(
        nameTag="test",
        messages=[],
        tags=[],
        semanticRefs=None,
        fileHeader={"version": "0.1"},
        embeddingFileHeader={},
    )
    file_data: ConversationFileData[Any] = ConversationFileData(
        jsonData=json_data,
        binaryData=ConversationBinaryData(embeddingsList=[emb]),
    )
    result = from_conversation_file_data(file_data)
    assert result["nameTag"] == "test"


def test_is_primitive():
    """Test is_primitive classification."""
    for t in (int, float, bool, str, type(None)):
        assert is_primitive(t), f"Expected {t} to be primitive"
    assert not is_primitive(list)
    assert not is_primitive(dict)


def test_deserialize_object_union_none():
    """deserialize_object handles optional (X | None) type with None input."""
    result = deserialize_object(int | None, None)
    assert result is None


def test_deserialize_object_list_of_int():
    """deserialize_object can deserialize a list of ints."""
    result = deserialize_object(list[int], [1, 2, 3])
    assert result == [1, 2, 3]


def test_deserialize_knowledge_entity():
    """deserialize_knowledge reconstructs a ConcreteEntity."""
    obj = {"name": "Bob", "type": ["person"]}
    result = deserialize_knowledge("entity", obj)
    assert isinstance(result, ConcreteEntity)
    assert result.name == "Bob"


def test_deserialize_knowledge_topic():
    """deserialize_knowledge reconstructs a Topic."""
    obj = {"text": "AI ethics"}
    result = deserialize_knowledge("topic", obj)
    assert isinstance(result, Topic)
    assert result.text == "AI ethics"


def test_deserialize_knowledge_tag():
    """deserialize_knowledge reconstructs a Tag."""
    obj = {"text": "important"}
    result = deserialize_knowledge("tag", obj)
    assert isinstance(result, Tag)
