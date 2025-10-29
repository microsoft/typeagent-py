# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os
import re

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI, AsyncAzureOpenAI, OpenAIError
import tiktoken
from tiktoken import model

from .utils import chunk_input

from .auth import get_shared_token_provider, AzureTokenProvider
from .utils import timelog

type NormalizedEmbedding = NDArray[np.float32]  # A single embedding
type NormalizedEmbeddings = NDArray[np.float32]  # An array of embeddings


DEFAULT_MODEL_NAME = "text-embedding-ada-002"
DEFAULT_EMBEDDING_SIZE = 1536  # Default embedding size (required for ada-002)
DEFAULT_ENVVAR = "AZURE_OPENAI_ENDPOINT_EMBEDDING"
TEST_MODEL_NAME = "test"
MAX_BATCH_SIZE = 2048
MAX_TOKEN_SIZE = 4096
MAX_TOKENS_PER_BATCH = 300_000
MAX_STRING_SIZE = MAX_TOKEN_SIZE * 3
MAX_CHARS_PER_BATCH = MAX_TOKENS_PER_BATCH

model_to_embedding_size_and_envvar: dict[str, tuple[int | None, str]] = {
    DEFAULT_MODEL_NAME: (DEFAULT_EMBEDDING_SIZE, DEFAULT_ENVVAR),
    "text-embedding-small": (None, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"),
    "text-embedding-large": (None, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE"),
    # For testing only, not a real model (insert real embeddings above)
    TEST_MODEL_NAME: (3, "SIR_NOT_APPEARING_IN_THIS_FILM"),
}


class AsyncEmbeddingModel:
    model_name: str
    embedding_size: int
    endpoint_envvar: str
    azure_token_provider: AzureTokenProvider | None
    async_client: AsyncOpenAI | None
    azure_endpoint: str
    azure_api_version: str
    encoding_name: str | None

    _embedding_cache: dict[str, NormalizedEmbedding]

    def __init__(
        self,
        embedding_size: int | None = None,
        model_name: str | None = None,
        endpoint_envvar: str | None = None,
    ):
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
        self.model_name = model_name

        required_embedding_size, required_endpoint_envvar = (
            model_to_embedding_size_and_envvar.get(model_name, (None, None))
        )
        if required_embedding_size is not None:
            if embedding_size is not None and embedding_size != required_embedding_size:
                raise ValueError(
                    f"Embedding size {embedding_size} does not match "
                    f"required size {required_embedding_size} for model {model_name}."
                )
            embedding_size = required_embedding_size
        if embedding_size is None or embedding_size <= 0:
            embedding_size = DEFAULT_EMBEDDING_SIZE
        self.embedding_size = embedding_size

        if required_endpoint_envvar is not None:
            if (
                endpoint_envvar is not None
                and endpoint_envvar != required_endpoint_envvar
            ):
                raise ValueError(
                    f"Environment variable for embedding endpoint {endpoint_envvar} does not match "
                    f"required environment variable {required_endpoint_envvar} for embedding model {model_name}."
                )
            endpoint_envvar = required_endpoint_envvar
        if endpoint_envvar is None:
            endpoint_envvar = DEFAULT_ENVVAR
        self.endpoint_envvar = endpoint_envvar

        self.azure_token_provider = None

        if self.model_name == TEST_MODEL_NAME:
            self.async_client = None
        else:
            openai_key_name = "OPENAI_API_KEY"
            azure_key_name = "AZURE_OPENAI_API_KEY"
            if openai_key := os.getenv(openai_key_name):
                endpoint = os.getenv(self.endpoint_envvar)
                with timelog(f"Using OpenAI"):
                    self.async_client = AsyncOpenAI(
                        base_url=endpoint, api_key=openai_key
                    )
            elif azure_api_key := os.getenv(azure_key_name):
                with timelog("Using Azure OpenAI"):
                    self._setup_azure(azure_api_key)
            else:
                raise ValueError(
                    f"Neither {openai_key_name} nor {azure_key_name} found in environment."
                )

        if self.model_name in model.MODEL_TO_ENCODING:
            encoding_name = tiktoken.encoding_name_for_model(self.model_name)
            self.encoding = tiktoken.get_encoding(encoding_name)
            self.max_chunk_size = MAX_TOKEN_SIZE
            self.max_size_per_batch = MAX_TOKENS_PER_BATCH
        else:
            self.encoding = None
            self.max_chunk_size = MAX_STRING_SIZE
            self.max_size_per_batch = MAX_CHARS_PER_BATCH

        self._embedding_cache = {}

    def _setup_azure(self, azure_api_key: str) -> None:
        # TODO: support different endpoint names
        endpoint_envvar = self.endpoint_envvar
        azure_endpoint = os.environ.get(endpoint_envvar)
        if not azure_endpoint:
            raise ValueError(f"Environment variable {endpoint_envvar} not found.")
        m = re.search(r"[?,]api-version=([^,]+)$", azure_endpoint)
        if not m:
            raise ValueError(
                f"{endpoint_envvar}={azure_endpoint} "
                f"doesn't end in api-version=<version>"
            )
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = m.group(1)
        if azure_api_key.lower() == "identity":
            self.azure_token_provider = get_shared_token_provider()
            azure_api_key = self.azure_token_provider.get_token()
            # print("Using shared TokenProvider")
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

            chunked_groups = {}
            chunked_input = []
            chunked_sizes = []
            for input_idx, entity in enumerate(input):
                entity_to_embed = entity
                if self.encoding:
                    entity_to_embed = self.encoding.encode(entity)

                entity_size = len(entity_to_embed)
                if entity_size > self.max_chunk_size:
                    chunked_entity = list(
                        chunk_input(entity_to_embed, self.max_chunk_size)
                    )
                    start_idx = len(chunked_input)
                    no_of_chunks = len(chunked_entity)
                    chunked_groups[input_idx] = (start_idx, no_of_chunks)
                    for chunk in chunked_entity:
                        chunked_sizes.append(len(chunk))
                        chunked_input.append(chunk)
                else:
                    chunked_sizes.append(entity_size)
                    chunked_input.append(entity_to_embed)

            input_embeddings = []
            batch = []
            size_of_batch = 0
            assert len(chunked_sizes) == len(chunked_input)
            for chunk_size, chunk in zip(chunked_sizes, chunked_input):
                if (
                    len(batch) == MAX_BATCH_SIZE
                    or size_of_batch + chunk_size > self.max_size_per_batch
                ):
                    data = (
                        await self.async_client.embeddings.create(
                            input=batch,
                            model=self.model_name,
                            encoding_format="float",
                            **extra_args,
                        )
                    ).data
                    input_embeddings.extend(data)
                    batch = [chunk]
                    size_of_batch = chunk_size
                else:
                    batch.append(chunk)
                    size_of_batch += chunk_size
            if batch:
                data = (
                    await self.async_client.embeddings.create(
                        input=batch,
                        model=self.model_name,
                        encoding_format="float",
                        **extra_args,
                    )
                ).data
                input_embeddings.extend(data)

            result = np.empty((len(input), self.embedding_size), dtype=np.float32)
            embedding_idx = 0
            for input_idx in range(len(input)):
                if input_idx in chunked_groups:
                    start_idx, no_of_chunks = chunked_groups[input_idx]
                    chunks = input_embeddings[start_idx : start_idx + no_of_chunks]
                    chunk_embeddings = np.average(
                        [chunk.embedding for chunk in chunks], axis=0
                    )
                    result[input_idx] = chunk_embeddings
                    embedding_idx += no_of_chunks
                else:
                    result[input_idx] = input_embeddings[embedding_idx].embedding
                    embedding_idx += 1

            assert len(result) == len(input), (len(result), "!=", len(input))
            return result

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
