# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os
import re

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI, AsyncAzureOpenAI

from .auth import get_shared_token_provider, AzureTokenProvider

type NormalizedEmbedding = NDArray[np.float32]  # A single embedding
type NormalizedEmbeddings = NDArray[np.float32]  # An array of embeddings


class AsyncEmbeddingModel:
    def __init__(self, embedding_size: int | None = None):
        if embedding_size is None or embedding_size <= 0:
            embedding_size = 1536
        self.embedding_size = embedding_size
        self.azure_token_provider: AzureTokenProvider | None = None
        openai_key_name = "OPENAI_API_KEY"
        azure_key_name = "AZURE_OPENAI_API_KEY"
        if os.getenv(openai_key_name):
            print(f"\nUsing OpenAI")
            self.async_client = AsyncOpenAI()
        elif azure_api_key := os.getenv(azure_key_name):
            print("\nUsing Azure OpenAI")
            self._setup_azure(azure_api_key)
        else:
            raise ValueError(
                f"Neither {openai_key_name} nor {azure_key_name} found in environment."
            )
        self._embedding_cache: dict[str, NormalizedEmbedding] = {}

    def _setup_azure(self, azure_api_key: str) -> None:
        # TODO: support different endpoint names
        endpoint_name = "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"
        self.azure_endpoint = os.environ.get(endpoint_name)
        if not self.azure_endpoint:
            raise ValueError(f"Environment variable {endpoint_name} not found.")
        m = re.search(r"[?,]api-version=([^,]+)$", self.azure_endpoint)
        if not m:
            raise ValueError(
                f"{endpoint_name}={self.azure_endpoint} doesn't end in api-version=<version>"
            )
        self.azure_api_version = m.group(1)
        if azure_api_key.lower() == "identity":
            self.azure_token_provider = get_shared_token_provider()
            azure_api_key = self.azure_token_provider.get_token()
            print("Using shared TokenProvider")
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
            assert existing == embedding
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
        data = (
            await self.async_client.embeddings.create(
                input=input,
                model="text-embedding-3-small",
                encoding_format="float",
                dimensions=self.embedding_size,
            )
        ).data
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


async def main():
    from . import auth

    auth.load_dotenv()

    async_model = AsyncEmbeddingModel()
    e = await async_model.get_embeddings([])
    print(repr(e))
    inputs = ["Hello, world", "Foo bar baz"]
    embeddings = await async_model.get_embeddings(inputs)
    print(repr(embeddings))
    for input, embedding in zip(inputs, embeddings, strict=True):
        print(f"{input}: {len(embedding)} {embedding[:5]}...{embedding[-5:]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
