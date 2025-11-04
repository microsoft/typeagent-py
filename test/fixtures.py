# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import AsyncGenerator, Iterator
import os
import tempfile
from typing import Any

import pytest
import pytest_asyncio

from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding
import tiktoken

from typeagent.aitools import utils
from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.storage.memory.collections import (
    MemoryMessageCollection,
    MemorySemanticRefCollection,
)
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces import (
    DeletionInfo,
    IConversation,
    IConversationSecondaryIndexes,
    IMessage,
    IMessageCollection,
    ISemanticRefCollection,
    IStorageProvider,
    ITermToSemanticRefIndex,
    SemanticRef,
    ScoredSemanticRefOrdinal,
    TextLocation,
)
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.knowpro.convsettings import (
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.knowpro.secindex import ConversationSecondaryIndexes
from typeagent.storage.memory import MemoryStorageProvider
from typeagent.storage import SqliteStorageProvider


@pytest.fixture(scope="session")
def needs_auth() -> None:
    utils.load_dotenv()


@pytest.fixture(scope="session")
def really_needs_auth() -> None:
    utils.load_dotenv()
    # Check if any of the supported API keys is set
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
        pytest.skip("No API key found")


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
    temp_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    path = temp_file.name
    temp_file.close()
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def memory_storage(
    embedding_model: AsyncEmbeddingModel,
) -> MemoryStorageProvider:
    """Create a memory storage provider with settings."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model=embedding_model)
    message_text_settings = MessageTextIndexSettings(
        embedding_index_settings=embedding_settings
    )
    related_terms_settings = RelatedTermIndexSettings(
        embedding_index_settings=embedding_settings
    )
    return MemoryStorageProvider(
        message_text_settings=message_text_settings,
        related_terms_settings=related_terms_settings,
    )


# Unified fake message and conversation classes for testing


class FakeMessage(IMessage):
    """Unified message implementation for testing purposes."""

    def __init__(
        self, text_chunks: list[str] | str, message_ordinal: int | None = None
    ):
        if isinstance(text_chunks, str):
            self.text_chunks = [text_chunks]
        else:
            self.text_chunks = text_chunks

        # Handle timestamp - for compatibility with mock message pattern
        if message_ordinal is not None:
            self.ordinal = message_ordinal
            self.timestamp = f"2020-01-01T{message_ordinal:02d}:00:00"
        else:
            self.timestamp = None

        self.tags: list[str] = []
        self.deletion_info: DeletionInfo | None = None
        self.text_location = TextLocation(0, 0)

    def get_knowledge(self) -> KnowledgeResponse:
        return KnowledgeResponse(
            entities=[],
            actions=[],
            inverse_actions=[],
            topics=[],
        )

    def get_text(self) -> str:
        return " ".join(self.text_chunks)

    def get_text_location(self) -> TextLocation:
        return self.text_location


@pytest_asyncio.fixture
async def sqlite_storage(
    temp_db_path: str, embedding_model: AsyncEmbeddingModel
) -> AsyncGenerator[SqliteStorageProvider[FakeMessage], None]:
    """Create a SqliteStorageProvider for testing."""
    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)

    provider = SqliteStorageProvider(
        db_path=temp_db_path,
        message_type=FakeMessage,
        message_text_index_settings=message_text_settings,
        related_term_index_settings=related_terms_settings,
    )
    yield provider
    await provider.close()


class FakeMessageCollection(MemoryMessageCollection[FakeMessage]):
    """Message collection for testing."""

    pass


class FakeTermIndex(ITermToSemanticRefIndex):
    """Simple term index for testing."""

    def __init__(
        self, term_to_refs: dict[str, list[ScoredSemanticRefOrdinal]] | None = None
    ):
        self.term_to_refs = term_to_refs or {}

    async def size(self) -> int:
        return len(self.term_to_refs)

    async def get_terms(self) -> list[str]:
        return list(self.term_to_refs.keys())

    async def add_term(
        self,
        term: str,
        semantic_ref_ordinal: int | ScoredSemanticRefOrdinal,
    ) -> str:
        if term not in self.term_to_refs:
            self.term_to_refs[term] = []
        if isinstance(semantic_ref_ordinal, int):
            scored_ref = ScoredSemanticRefOrdinal(semantic_ref_ordinal, 1.0)
        else:
            scored_ref = semantic_ref_ordinal
        self.term_to_refs[term].append(scored_ref)
        return term

    async def remove_term(self, term: str, semantic_ref_ordinal: int) -> None:
        if term in self.term_to_refs:
            self.term_to_refs[term] = [
                ref
                for ref in self.term_to_refs[term]
                if ref.semantic_ref_ordinal != semantic_ref_ordinal
            ]
            if not self.term_to_refs[term]:
                del self.term_to_refs[term]

    async def clear(self) -> None:
        """Clear all terms from the index."""
        self.term_to_refs.clear()

    async def lookup_term(self, term: str) -> list[ScoredSemanticRefOrdinal] | None:
        return self.term_to_refs.get(term)

    async def serialize(self) -> Any:
        raise RuntimeError

    async def deserialize(self, data: Any) -> None:
        raise RuntimeError


class FakeConversation(IConversation[FakeMessage, FakeTermIndex]):
    """Unified conversation implementation for testing purposes."""

    def __init__(
        self,
        name_tag: str = "FakeConversation",
        messages: list[FakeMessage] | None = None,
        semantic_refs: list[SemanticRef] | None = None,
        storage_provider: IStorageProvider | None = None,
        has_secondary_indexes: bool = True,
    ):
        self.name_tag = name_tag
        self.tags: list[str] = []

        # Set up messages
        if messages is None:
            messages = [FakeMessage("Hello world")]
        self.messages: IMessageCollection[FakeMessage] = FakeMessageCollection(messages)

        # Set up semantic refs
        self.semantic_refs: ISemanticRefCollection = MemorySemanticRefCollection(
            semantic_refs or []
        )

        # Set up term index
        self.semantic_ref_index: FakeTermIndex | None = FakeTermIndex()

        # Store settings with storage provider for access via conversation.settings.storage_provider
        if storage_provider is None:
            # Default storage provider will be created lazily in async context
            self._needs_async_init = True
            self.secondary_indexes = None
            self._storage_provider = None
            self._has_secondary_indexes = has_secondary_indexes
        else:
            # Create test model for settings
            test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
            self.settings = ConversationSettings(test_model, storage_provider)
            self._needs_async_init = False
            self._storage_provider = storage_provider

            if has_secondary_indexes:
                # Set up secondary indexes
                embedding_settings = TextEmbeddingIndexSettings(test_model)
                related_terms_settings = RelatedTermIndexSettings(embedding_settings)
                self.secondary_indexes: (
                    IConversationSecondaryIndexes[FakeMessage] | None
                ) = ConversationSecondaryIndexes(
                    storage_provider, related_terms_settings
                )
            else:
                self.secondary_indexes = None

    async def ensure_initialized(self):
        """Ensure async initialization is complete."""
        if self._needs_async_init:
            test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
            self.settings = ConversationSettings(test_model)
            storage_provider = await self.settings.get_storage_provider()
            self._storage_provider = storage_provider
            if self.semantic_ref_index is None:
                self.semantic_ref_index = await storage_provider.get_semantic_ref_index()  # type: ignore

            if self._has_secondary_indexes:
                # Set up secondary indexes
                embedding_settings = TextEmbeddingIndexSettings(test_model)
                related_terms_settings = RelatedTermIndexSettings(embedding_settings)
                self.secondary_indexes = ConversationSecondaryIndexes(
                    storage_provider, related_terms_settings
                )
            else:
                self.secondary_indexes = None

            self._needs_async_init = False


@pytest.fixture
def fake_conversation() -> FakeConversation:
    """Fixture to create a FakeConversation instance."""
    return FakeConversation()


@pytest.fixture
async def fake_conversation_with_storage(
    memory_storage: MemoryStorageProvider,
) -> FakeConversation:
    """Fixture to create a FakeConversation instance with storage provider."""
    return FakeConversation(storage_provider=memory_storage)


class FakeEmbeddings:

    def __init__(
        self,
        max_batch_size: int = 2048,
        max_chunk_size: int = 4096,
        max_elements_per_batch: int = 300_000,
        use_tiktoken: bool = False,
    ):
        self.model_name = "text-embedding-ada-002"
        self.call_count = 0
        self.max_batch_size = max_batch_size
        self.max_chunk_size = max_chunk_size
        self.max_elements_per_batch = max_elements_per_batch
        self.use_tiktoken = use_tiktoken

    def reset_counter(self):
        self.call_count = 0

    async def create(self, **kwargs):
        self.call_count += 1
        input = kwargs["input"]
        len_input = len(input)
        if len_input > self.max_batch_size:
            raise ValueError("Embedding model received batch larger 2048")
        dimensions = 1536
        if "dimensions" in kwargs:
            dimensions = kwargs["dimensions"]

        embedding_result = []
        total_elements = 0
        for index in range(len_input):
            entity = input[index]
            if self.use_tiktoken:
                enc_name = tiktoken.encoding_name_for_model(self.model_name)
                enc = tiktoken.get_encoding(enc_name)
                entity = enc.encode(entity)
            total_elements += len(entity)
            if len(entity) > self.max_chunk_size:
                raise ValueError(
                    f"Chunk size {len(entity)} larger than max size {self.max_chunk_size}"
                )
            value = index % 2
            embedding_result.append(
                Embedding(
                    embedding=[value] * dimensions, index=index, object="embedding"
                )
            )

        if total_elements > self.max_elements_per_batch:
            raise ValueError(
                f"Batch size {total_elements} larger than max tokens/chars per batch {self.max_elements_per_batch}"
            )

        response = CreateEmbeddingResponse(
            data=embedding_result,
            model="test_model",
            object="list",
            usage=Usage(prompt_tokens=0, total_tokens=0),
        )

        return response


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    """Fixture to create a FaceEmbedding instance"""
    return FakeEmbeddings(max_batch_size=2048, max_chunk_size=4096 * 3)


@pytest.fixture
def fake_embeddings_tiktoken() -> FakeEmbeddings:
    """Fixture to create a FaceEmbedding instance"""
    return FakeEmbeddings(max_batch_size=2048, max_chunk_size=4096, use_tiktoken=True)
