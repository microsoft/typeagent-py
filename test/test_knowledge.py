# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typechat import Result, Failure, Success

from typeagent.knowpro.knowledge import (
    create_knowledge_extractor,
    extract_knowledge_from_text,
    extract_knowledge_from_text_batch,
    merge_topics,
)
from typeagent.knowpro import convknowledge, kplib

from fixtures import needs_auth  # type: ignore  # Used!


@pytest.fixture
def mock_knowledge_extractor():
    """Fixture to create a mock KnowledgeExtractor."""

    class MockKnowledgeExtractor:
        async def extract(self, text: str) -> Result[kplib.KnowledgeResponse]:
            if text == "error":
                return Failure("Extraction failed")
            return Success(
                kplib.KnowledgeResponse(
                    entities=[], actions=[], inverse_actions=[], topics=[text]
                )
            )

    return MockKnowledgeExtractor()


def test_create_knowledge_extractor(needs_auth):
    """Test creating a knowledge extractor."""
    extractor = create_knowledge_extractor()
    assert isinstance(extractor, convknowledge.KnowledgeExtractor)


@pytest.mark.asyncio
async def test_extract_knowledge_from_text(mock_knowledge_extractor):
    """Test extracting knowledge from a single text input."""
    result = await extract_knowledge_from_text(mock_knowledge_extractor, "test text", 3)
    assert isinstance(result, Success)
    assert result.value.topics[0] == "test text"

    failure_result = await extract_knowledge_from_text(
        mock_knowledge_extractor, "error", 3
    )
    assert isinstance(failure_result, Failure)
    assert failure_result.message == "Extraction failed"


@pytest.mark.asyncio
async def test_extract_knowledge_from_text_batch(mock_knowledge_extractor):
    """Test extracting knowledge from a batch of text inputs."""
    text_batch = ["text 1", "text 2", "error"]
    results = await extract_knowledge_from_text_batch(
        mock_knowledge_extractor, text_batch, 2, 3
    )

    assert len(results) == 3
    assert isinstance(results[0], Success)
    assert results[0].value.topics[0] == "text 1"
    assert isinstance(results[1], Success)
    assert results[1].value.topics[0] == "text 2"
    assert isinstance(results[2], Failure)
    assert results[2].message == "Extraction failed"


def test_merge_topics():
    """Test merging a list of topics into a unique list."""
    topics = ["topic1", "topic2", "topic1", "topic3"]
    merged_topics = merge_topics(topics)

    assert len(merged_topics) == 3
    assert "topic1" in merged_topics
    assert "topic2" in merged_topics
    assert "topic3" in merged_topics
