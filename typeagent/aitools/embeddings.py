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


DEFAULT_MODEL_NAME = "ada-002"
DEFAULT_EMBEDDING_SIZE = 1536  # Default embedding size (required for ada-002)
DEFAULT_ENVVAR = "AZURE_OPENAI_ENDPOINT_EMBEDDING"

model_to_embedding_size_and_envvar = {
    DEFAULT_MODEL_NAME: (DEFAULT_EMBEDDING_SIZE, DEFAULT_ENVVAR),
    "text-embedding-small": (None, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"),
    "text-embedding-large": (None, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_LARGE"),
    # For testing only, not a real model (insert real embeddings above)
    "test": (3, "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"),
}


class AsyncEmbeddingModel:
    def __init__(
        self, embedding_size: int | None = None, model_name: str | None = None
    ):
        if model_name is None:
            model_name = DEFAULT_MODEL_NAME
        self.model_name = model_name

        required_embedding_size, endpoint_envvar = (
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

        if not endpoint_envvar:
            raise ValueError(
                f"Model {model_name} is not supported. "
                f"Supported models are: {', '.join(model_to_embedding_size_and_envvar.keys())}"
            )
        self.endpoint_envvar = endpoint_envvar

        self.azure_token_provider: AzureTokenProvider | None = None
        openai_key_name = "OPENAI_API_KEY"
        azure_key_name = "AZURE_OPENAI_API_KEY"
        if os.getenv(openai_key_name):
            print(f"Using OpenAI")
            self.async_client = AsyncOpenAI()
        elif azure_api_key := os.getenv(azure_key_name):
            print("Using Azure OpenAI")
            self._setup_azure(azure_api_key)
        else:
            raise ValueError(
                f"Neither {openai_key_name} nor {azure_key_name} found in environment."
            )
        self._embedding_cache: dict[str, NormalizedEmbedding] = {}

    def _setup_azure(self, azure_api_key: str) -> None:
        # TODO: support different endpoint names
        endpoint_envvar = self.endpoint_envvar
        self.azure_endpoint = os.environ.get(endpoint_envvar)
        if not self.azure_endpoint:
            raise ValueError(f"Environment variable {endpoint_envvar} not found.")
        m = re.search(r"[?,]api-version=([^,]+)$", self.azure_endpoint)
        if not m:
            raise ValueError(
                f"{endpoint_envvar}={self.azure_endpoint} "
                f"doesn't end in api-version=<version>"
            )
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
        extra_args = {}
        if self.model_name != DEFAULT_MODEL_NAME:
            extra_args["dimensions"] = self.embedding_size
        data = (
            await self.async_client.embeddings.create(
                input=input,
                model=self.model_name,
                encoding_format="float",
                **extra_args,
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
    from . import utils

    utils.load_dotenv()

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
