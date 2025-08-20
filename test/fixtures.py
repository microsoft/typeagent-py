# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import AsyncGenerator, Iterator
import os
import tempfile
from typing import assert_never

import pytest
import pytest_asyncio

from typeagent.aitools import utils
from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.interfaces import IStorageProvider
from typeagent.knowpro.messageindex import MessageTextIndexSettings
from typeagent.knowpro.reltermsindex import RelatedTermIndexSettings
from typeagent.storage.memorystore import MemoryStorageProvider
from typeagent.storage.sqlitestore import SqliteStorageProvider


@pytest.fixture(scope="session")
def needs_auth() -> None:
    utils.load_dotenv()


@pytest.fixture(scope="session")
def embedding_model() -> AsyncEmbeddingModel:
    """Fixture to create a test embedding model with small embedding size for faster tests."""
    return AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)


@pytest.fixture
def temp_dir() -> Iterator[str]:
    with tempfile.TemporaryDirectory() as dir:
        yield dir


@pytest.fixture
def temp_db_path() -> Iterator[str]:
    """Create a temporary SQLite database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest_asyncio.fixture
async def memory_storage(embedding_model) -> MemoryStorageProvider:
    """Create a MemoryStorageProvider for testing."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    return await MemoryStorageProvider.create(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )


@pytest_asyncio.fixture
async def sqlite_storage(
    temp_db_path, embedding_model
) -> AsyncGenerator[SqliteStorageProvider, None]:
    """Create a SqliteStorageProvider for testing."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = await SqliteStorageProvider.create(
        message_text_settings, related_terms_settings, temp_db_path
    )
    yield provider
    await provider.close()


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def storage_provider_type(
    request, embedding_model, temp_db_path
) -> AsyncGenerator[tuple[IStorageProvider, str], None]:
    """Parameterized fixture that provides both memory and sqlite storage providers."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    match request.param:
        case "memory":
            provider = await MemoryStorageProvider.create(
                message_text_settings=message_text_settings,
                related_terms_settings=related_terms_settings,
            )
            yield provider, request.param
        case "sqlite":
            provider = await SqliteStorageProvider.create(
                message_text_settings, related_terms_settings, temp_db_path
            )
            yield provider, request.param
            await provider.close()
        case _:
            assert_never(request.param)
