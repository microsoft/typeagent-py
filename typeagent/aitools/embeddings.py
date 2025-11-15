# Copyright (c) Microsoft.
# Licensed under the MIT License.

import asyncio
import os

import numpy as np
from numpy.typing import NDArray

import litellm
from litellm import aembedding

import tiktoken
from tiktoken import model as tiktoken_model

from .auth import get_shared_token_provider, AzureTokenProvider
from .utils import timelog


type NormalizedEmbedding = NDArray[np.float32]
type NormalizedEmbeddings = NDArray[np.float32]


DEFAULT_MODEL_NAME = "text-embedding-ada-002"
DEFAULT_EMBEDDING_SIZE = 1536
DEFAULT_ENVVAR = "AZURE_OPENAI_ENDPOINT_EMBEDDING"

TEST_MODEL_NAME = "test"

MAX_BATCH_SIZE = 2048
MAX_TOKEN_SIZE = 4096
MAX_TOKENS_PER_BATCH = 300_000
MAX_CHAR_SIZE = MAX_TOKEN_SIZE * 3
MAX_CHARS_PER_BATCH = MAX_TOKENS_PER_BATCH * 3

# Embedding size + Azure envvar mapping
model_to_embedding_size_and_envvar: dict[str, tuple[int | None, str]] = {
    DEFAULT_MODEL_NAME: (DEFAULT_EMBEDDING_SIZE, DEFAULT_ENVVAR),
    "text-embedding-3-small": (1536, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"),
    "text-embedding-3-large": (3072, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE"),
    TEST_MODEL_NAME: (3, "NO_ENV"),
}

class AsyncEmbeddingModel:
    model_name: str
    embedding_size: int
    endpoint_envvar: str

    azure_token_provider: AzureTokenProvider | None
    litellm_args: dict

    encoding: tiktoken.core.Encoding | None
    max_chunk_size: int
    max_size_per_batch: int

    _embedding_cache: dict[str, NormalizedEmbedding]

    def __init__(
        self,
        embedding_size: int | None = None,
        model_name: str | None = None,
        endpoint_envvar: str | None = None,
        max_retries: int = litellm.DEFAULT_MAX_RETRIES,
    ):
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
        self.model_name = model_name

        suggested_emb_size, suggested_env = model_to_embedding_size_and_envvar.get(
            model_name, (None, None)
        )

        if embedding_size is None:
            embedding_size = suggested_emb_size or DEFAULT_EMBEDDING_SIZE
        self.embedding_size = embedding_size

        if endpoint_envvar is None:
            endpoint_envvar = suggested_env or DEFAULT_ENVVAR
        self.endpoint_envvar = endpoint_envvar
        self.litellm_args = {"model": self.model_name}

        openai_key = os.getenv("OPENAI_API_KEY")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")

        if model_name == TEST_MODEL_NAME:
            self.litellm_args = None  # Use fake embeddings
            self.azure_token_provider = None

        elif openai_key:
            with timelog("LiteLLM -> OpenAI"):
                litellm.api_key = openai_key
                self._setup_openai()

        elif azure_key:
            with timelog("LiteLLM -> Azure OpenAI"):
                self._setup_azure(azure_key)

        else:
            raise ValueError("Missing OPENAI_API_KEY or AZURE_OPENAI_API_KEY")

        if self.model_name in tiktoken_model.MODEL_TO_ENCODING:
            enc_name = tiktoken.encoding_name_for_model(self.model_name)
            self.encoding = tiktoken.get_encoding(enc_name)
            self.max_chunk_size = MAX_TOKEN_SIZE
            self.max_size_per_batch = MAX_TOKENS_PER_BATCH
        else:
            self.encoding = None
            self.max_chunk_size = MAX_CHAR_SIZE
            self.max_size_per_batch = MAX_CHARS_PER_BATCH

        self._embedding_cache = {}


    def _setup_openai(self):
        endpoint = os.getenv(self.endpoint_envvar)
        if endpoint:
            self.litellm_args["api_base"] = endpoint

    def _setup_azure(self, azure_api_key: str):
        from .utils import get_azure_api_key, parse_azure_endpoint

        azure_api_key = get_azure_api_key(azure_api_key)
        endpoint, version = parse_azure_endpoint(self.endpoint_envvar)

        self.azure_endpoint = endpoint
        self.azure_api_version = version

        if azure_api_key != os.getenv("AZURE_OPENAI_API_KEY"):
            self.azure_token_provider = get_shared_token_provider()
        else:
            self.azure_token_provider = None

        litellm.api_key = azure_api_key
        self.litellm_args.update(
            {
                "api_type": "azure",
                "api_base": self.azure_endpoint,
                "api_version": self.azure_api_version,
            }
        )

    async def refresh_auth(self):
        assert self.azure_token_provider
        loop = asyncio.get_running_loop()
        new_key = await loop.run_in_executor(
            None, self.azure_token_provider.refresh_token
        )

        litellm.api_key = new_key
        self.litellm_args["api_key"] = new_key


    def add_embedding(self, key: str, embedding: NormalizedEmbedding):
        self._embedding_cache[key] = embedding

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        if key in self._embedding_cache:
            return self._embedding_cache[key]
        emb = await self.get_embedding_nocache(key)
        self._embedding_cache[key] = emb
        return emb

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        out, missing = [], []

        for key in keys:
            if key in self._embedding_cache:
                out.append(self._embedding_cache[key])
            else:
                missing.append(key)
                out.append(None)

        if missing:
            new_embs = await self.get_embeddings_nocache(missing)
            for k, emb in zip(missing, new_embs):
                self._embedding_cache[k] = emb

            # fill in None slots
            for i, key in enumerate(keys):
                if out[i] is None:
                    out[i] = self._embedding_cache[key]

        return np.array(out, dtype=np.float32).reshape(len(keys), self.embedding_size)


    async def get_embedding_nocache(self, text: str) -> NormalizedEmbedding:
        return (await self.get_embeddings_nocache([text]))[0]

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        if not input:
            return np.zeros((0, self.embedding_size), dtype=np.float32)

        # Fake model
        if self.litellm_args is None:
            return self._fake_embeddings(input)

        # Azure refresh
        if self.azure_token_provider and self.azure_token_provider.needs_refresh():
            await self.refresh_auth()

        batches = []
        batch, count = [], 0

        for text in input:
            truncated, size = await self.truncate_input(text)
            if len(batch) >= MAX_BATCH_SIZE or count + size > self.max_size_per_batch:
                batches.append(batch)
                batch, count = [], 0
            batch.append(truncated)
            count += size

        if batch:
            batches.append(batch)

        out = []
        extra = {}
        if self.model_name != DEFAULT_MODEL_NAME:
            extra["dimensions"] = self.embedding_size

        for b in batches:
            resp = await aembedding(
                input=b,
                **self.litellm_args,
                **extra,
            )
            for d in resp.data:
                out.append(d["embedding"])

        return np.array(out, dtype=np.float32)

    def _fake_embeddings(self, input: list[str]) -> NormalizedEmbeddings:
        prime = 1961

        def hsh(s: str):
            h = 0
            for ch in s:
                h = (h * 31 + ord(ch)) & 0xFFFFFFFF
            return h

        out = []
        for text in input:
            vec = []
            L = len(text)
            for i in range(self.embedding_size):
                cut = i % L if L > 0 else 0
                sub = text[cut:] + text[:cut]
                vec.append((hsh(sub) % prime) / prime)

            arr = np.array(vec, dtype=np.float64)
            arr = arr / np.linalg.norm(arr)
            out.append(arr)

        return np.array(out, dtype=np.float32)


    async def truncate_input(self, text: str) -> tuple[str, int]:
        if self.encoding is None:
            if len(text) > self.max_chunk_size:
                return text[: self.max_chunk_size], self.max_chunk_size
            return text, len(text)

        tokens = self.encoding.encode(text)
        if len(tokens) > self.max_chunk_size:
            tokens = tokens[: self.max_chunk_size]
            return self.encoding.decode(tokens), self.max_chunk_size
        return text, len(tokens)
