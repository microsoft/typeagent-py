# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for knowpro/textlocindex.py (TextToTextLocationIndex)."""

import numpy as np
import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.interfaces import TextLocation, TextToTextLocationIndexData
from typeagent.knowpro.textlocindex import TextToTextLocationIndex


@pytest.fixture
def settings() -> TextEmbeddingIndexSettings:
    return TextEmbeddingIndexSettings(create_test_embedding_model())


@pytest.fixture
def index(settings: TextEmbeddingIndexSettings) -> TextToTextLocationIndex:
    return TextToTextLocationIndex(settings)


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_size(index: TextToTextLocationIndex) -> None:
    assert await index.size() == 0


@pytest.mark.asyncio
async def test_empty_is_empty(index: TextToTextLocationIndex) -> None:
    assert await index.is_empty()


def test_get_out_of_range_returns_default(index: TextToTextLocationIndex) -> None:
    assert index.get(0) is None
    assert index.get(-1) is None
    assert index.get(0, TextLocation(99)) == TextLocation(99)


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_resets(index: TextToTextLocationIndex) -> None:
    loc = TextLocation(message_ordinal=0)
    await index.add_text_location("hello world", loc)
    assert await index.size() == 1
    index.clear()
    assert await index.size() == 0
    assert await index.is_empty()


# ---------------------------------------------------------------------------
# serialize / deserialize round-trip (no real embeddings needed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_serialize_empty(index: TextToTextLocationIndex) -> None:
    data = index.serialize()
    assert data["textLocations"] == []
    # embeddings may be None or an empty ndarray
    emb = data["embeddings"]
    assert emb is None or (hasattr(emb, "shape") and emb.size == 0)


def test_deserialize_raises_on_no_embeddings(
    index: TextToTextLocationIndex,
) -> None:
    data: TextToTextLocationIndexData = {
        "textLocations": [{"messageOrdinal": 0, "chunkOrdinal": 0}],
        "embeddings": None,
    }
    with pytest.raises(ValueError, match="No embeddings found"):
        index.deserialize(data)


def test_deserialize_raises_on_length_mismatch(
    index: TextToTextLocationIndex, settings: TextEmbeddingIndexSettings
) -> None:
    # The test embedding model uses size 3 by default.
    emb_size = 3
    fake_emb = np.zeros((3, emb_size), dtype=np.float32)
    data: TextToTextLocationIndexData = {
        # 2 locations but 3 embeddings → mismatch
        "textLocations": [
            {"messageOrdinal": 0, "chunkOrdinal": 0},
            {"messageOrdinal": 1, "chunkOrdinal": 0},
        ],
        "embeddings": fake_emb,
    }
    with pytest.raises(ValueError):
        index.deserialize(data)


def test_deserialize_valid_data(
    index: TextToTextLocationIndex, settings: TextEmbeddingIndexSettings
) -> None:
    emb_size = 3  # default size for create_test_embedding_model()
    n = 2
    fake_emb = np.zeros((n, emb_size), dtype=np.float32)
    data: TextToTextLocationIndexData = {
        "textLocations": [
            {"messageOrdinal": 0, "chunkOrdinal": 0},
            {"messageOrdinal": 1, "chunkOrdinal": 0},
        ],
        "embeddings": fake_emb,
    }
    index.deserialize(data)
    assert index.get(0) == TextLocation(0)
    assert index.get(1) == TextLocation(1)
    assert index.get(2) is None


# ---------------------------------------------------------------------------
# get() helper
# ---------------------------------------------------------------------------


def test_get_returns_correct_location(
    index: TextToTextLocationIndex, settings: TextEmbeddingIndexSettings
) -> None:
    emb_size = 3  # default size for create_test_embedding_model()
    n = 3
    fake_emb = np.zeros((n, emb_size), dtype=np.float32)
    data: TextToTextLocationIndexData = {
        "textLocations": [
            {"messageOrdinal": 10, "chunkOrdinal": 0},
            {"messageOrdinal": 20, "chunkOrdinal": 1},
            {"messageOrdinal": 30, "chunkOrdinal": 0},
        ],
        "embeddings": fake_emb,
    }
    index.deserialize(data)
    assert index.get(0) == TextLocation(10, 0)
    assert index.get(1) == TextLocation(20, 1)
    assert index.get(2) == TextLocation(30, 0)
    assert index.get(3) is None
