# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os

import numpy as np
from numpy.typing import NDArray

from openai import AsyncAzureOpenAI, AsyncOpenAI, DEFAULT_MAX_RETRIES, OpenAIError
from openai.types import Embedding
import tiktoken
from tiktoken import model as tiktoken_model
from tiktoken.core import Encoding

from .auth import AzureTokenProvider, get_shared_token_provider
from .utils import timelog

type NormalizedEmbedding = NDArray[np.float32]  # A single embedding
type NormalizedEmbeddings = NDArray[np.float32]  # An array of embeddings


DEFAULT_MODEL_NAME = "text-embedding-ada-002"
DEFAULT_EMBEDDING_SIZE = 1536  # Default embedding size (required for ada-002)
DEFAULT_ENVVAR = "AZURE_OPENAI_ENDPOINT_EMBEDDING"  # We support OpenAI and Azure OpenAI
TEST_MODEL_NAME = "test"
MAX_BATCH_SIZE = 2048
MAX_TOKEN_SIZE = 4096
MAX_TOKENS_PER_BATCH = 300_000
MAX_CHAR_SIZE = MAX_TOKEN_SIZE * 3
MAX_CHARS_PER_BATCH = MAX_TOKENS_PER_BATCH * 3

model_to_embedding_size_and_envvar: dict[str, tuple[int | None, str]] = {
    DEFAULT_MODEL_NAME: (DEFAULT_EMBEDDING_SIZE, DEFAULT_ENVVAR),
    "text-embedding-3-small": (1536, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"),
    "text-embedding-3-large": (3072, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE"),
    # For testing only, not a real model (insert real embeddings above)
    TEST_MODEL_NAME: (3, "SIR_NOT_APPEARING_IN_THIS_FILM"),
}


class AsyncEmbeddingModel:
    model_name: str
    embedding_size: int
    endpoint_envvar: str
    use_azure: bool
    azure_token_provider: AzureTokenProvider | None
    async_client: AsyncOpenAI | None
    azure_endpoint: str
    azure_api_version: str
    encoding: Encoding | None
    max_chunk_size: int
    max_size_per_batch: int

    _embedding_cache: dict[str, NormalizedEmbedding]

    def __init__(
        self,
        embedding_size: int | None = None,
        model_name: str | None = None,
        endpoint_envvar: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
        self.model_name = model_name

        suggested_embedding_size, suggested_endpoint_envvar = (
            model_to_embedding_size_and_envvar.get(model_name, (None, None))
        )

        if embedding_size is None:
            if suggested_embedding_size is not None:
                embedding_size = suggested_embedding_size
            else:
                embedding_size = DEFAULT_EMBEDDING_SIZE
        self.embedding_size = embedding_size

        if (
            model_name == DEFAULT_MODEL_NAME
            and embedding_size != DEFAULT_EMBEDDING_SIZE
        ):
            raise ValueError(
                f"Cannot customize embedding_size for default model {DEFAULT_MODEL_NAME}"
            )

        # Read API keys once
        openai_api_key = os.getenv("OPENAI_API_KEY")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")

        # Prefer OpenAI if both are set, use Azure if only Azure is set
        self.use_azure = bool(azure_api_key) and not bool(openai_api_key)

        if endpoint_envvar is None:
            # Check if OpenAI credentials are available, prefer OpenAI over Azure
            if openai_api_key:
                endpoint_envvar = "OPENAI_BASE_URL"  # Use OpenAI
            elif suggested_endpoint_envvar is not None:
                endpoint_envvar = suggested_endpoint_envvar
            else:
                endpoint_envvar = DEFAULT_ENVVAR

        self.endpoint_envvar = endpoint_envvar
        self.azure_token_provider = None

        if self.model_name == TEST_MODEL_NAME:
            self.async_client = None
        elif self.use_azure:
            if not azure_api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not found in environment.")
            with timelog("Using Azure OpenAI"):
                self._setup_azure(azure_api_key)
        else:
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment.")
            endpoint = os.getenv(self.endpoint_envvar)
            with timelog("Using OpenAI"):
                self.async_client = AsyncOpenAI(
                    base_url=endpoint, api_key=openai_api_key, max_retries=max_retries
                )

        if self.model_name in tiktoken_model.MODEL_TO_ENCODING:
            encoding_name = tiktoken.encoding_name_for_model(self.model_name)
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.max_chunk_size = MAX_TOKEN_SIZE
            self.max_size_per_batch = MAX_TOKENS_PER_BATCH
        else:
            self.encoding = None
            self.max_chunk_size = MAX_CHAR_SIZE
            self.max_size_per_batch = MAX_CHARS_PER_BATCH

        self._embedding_cache = {}

    def _setup_azure(self, azure_api_key: str) -> None:
        from .utils import get_azure_api_key, parse_azure_endpoint

        azure_api_key = get_azure_api_key(azure_api_key)
        self.azure_endpoint, self.azure_api_version = parse_azure_endpoint(
            self.endpoint_envvar
        )

        if azure_api_key != os.getenv("AZURE_OPENAI_API_KEY"):
            # If we got a token from identity, store the provider for refresh
            self.azure_token_provider = get_shared_token_provider()

        self.async_client = AsyncAzureOpenAI(
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=azure_api_key,
        )

    async def refresh_auth(self):
        """Update client when using a token provider and it's nearly expired."""
        # refresh_token is synchronous and slow -- run it in a separate thread
        assert self.azure_token_provider
        refresh_token = self.azure_token_provider.refresh_token
        loop = asyncio.get_running_loop()
        azure_api_key = await loop.run_in_executor(None, refresh_token)
        assert self.azure_api_version
        assert self.azure_endpoint
        self.async_client = AsyncAzureOpenAI(
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=azure_api_key,
        )

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        existing = self._embedding_cache.get(key)
        if existing is not None:
            assert np.array_equal(existing, embedding)
        else:
            self._embedding_cache[key] = embedding

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        embeddings = await self.get_embeddings_nocache([input])
        return embeddings[0]

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        if not input:
            empty = np.array([], dtype=np.float32)
            empty.shape = (0, self.embedding_size)
            return empty
        if self.azure_token_provider and self.azure_token_provider.needs_refresh():
            await self.refresh_auth()
        extra_args = {}
        if self.model_name != DEFAULT_MODEL_NAME:
            extra_args["dimensions"] = self.embedding_size
        if self.async_client is None:
            # Compute a random embedding for testing purposes.

            def hashish(s: str) -> int:
                # Primitive deterministic hash function (hash() varies per run)
                h = 0
                for ch in s:
                    h = (h * 31 + ord(ch)) & 0xFFFFFFFF
                return h

            prime = 1961
            fake_data: list[NormalizedEmbedding] = []
            for item in input:
                if not item:
                    raise OpenAIError
                length = len(item)
                floats = []
                for i in range(self.embedding_size):
                    cut = i % length
                    scrambled = item[cut:] + item[:cut]
                    hashed = hashish(scrambled)
                    reduced = (hashed % prime) / prime
                    floats.append(reduced)
                array = np.array(floats, dtype=np.float64)
                normalized = array / np.sqrt(np.dot(array, array))
                dot = np.dot(normalized, normalized)
                assert (
                    abs(dot - 1.0) < 1e-15
                ), f"Embedding {normalized} is not normalized: {dot}"
                fake_data.append(normalized)
            assert len(fake_data) == len(input), (len(fake_data), "!=", len(input))
            result = np.array(fake_data, dtype=np.float32)
            return result
        else:
            batches: list[list[str]] = []
            batch: list[str] = []
            batch_sum: int = 0
            for sentence in input:
                truncated_input, truncated_input_size = await self.truncate_input(
                    sentence
                )
                if (
                    len(batch) >= MAX_BATCH_SIZE
                    or batch_sum + truncated_input_size > self.max_size_per_batch
                ):
                    batches.append(batch)
                    batch = []
                    batch_sum = 0
                batch.append(truncated_input)
                batch_sum += truncated_input_size
            if batch:
                batches.append(batch)

            data: list[Embedding] = []
            for batch in batches:
                embeddings_data = (
                    await self.async_client.embeddings.create(
                        input=batch,
                        model=self.model_name,
                        encoding_format="float",
                        **extra_args,
                    )
                ).data
                data.extend(embeddings_data)

            assert len(data) == len(input), (len(data), "!=", len(input))
            return np.array([d.embedding for d in data], dtype=np.float32)

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        """Retrieve an embedding, using the cache."""
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        embedding = await self.get_embedding_nocache(key)
        self._embedding_cache[key] = embedding
        return embedding

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        """Retrieve embeddings for multiple keys, using the cache."""
        embeddings: list[NormalizedEmbedding | None] = []
        missing_keys: list[str] = []

        # Collect cached embeddings and identify missing keys
        for key in keys:
            if key in self._embedding_cache:
                embeddings.append(self._embedding_cache[key])
            else:
                embeddings.append(None)  # Placeholder for missing keys
                missing_keys.append(key)

        # Retrieve embeddings for missing keys
        if missing_keys:
            new_embeddings = await self.get_embeddings_nocache(missing_keys)
            for key, embedding in zip(missing_keys, new_embeddings):
                self._embedding_cache[key] = embedding

            # Replace placeholders with retrieved embeddings
            for i, key in enumerate(keys):
                if embeddings[i] is None:
                    embeddings[i] = self._embedding_cache[key]
        return np.array(embeddings, dtype=np.float32).reshape(
            (len(keys), self.embedding_size)
        )

    async def truncate_input(self, input: str) -> tuple[str, int]:
        """Truncate input strings to fit within model limits.

        args:
            input: The input string to truncate.

        returns:
            A tuple of (truncated string, size after truncation).
        """
        if self.encoding is None:
            # Non-token-aware truncation
            if len(input) > self.max_chunk_size:
                return input[: self.max_chunk_size], self.max_chunk_size
            else:
                return input, len(input)
        else:
            # Token-aware truncation
            tokens = self.encoding.encode(input)
            if len(tokens) > self.max_chunk_size:
                truncated_tokens = tokens[: self.max_chunk_size]
                return self.encoding.decode(truncated_tokens), self.max_chunk_size
            else:
                return input, len(tokens)
