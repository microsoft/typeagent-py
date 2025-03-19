# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os
import re
import time

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI, AsyncAzureOpenAI

from .auth import get_shared_token_provider, AzureTokenProvider

NormalizedEmbedding = NDArray[np.float32]  # A single embedding
NormalizedEmbeddings = NDArray[np.float32]  # An array of embeddings


class AsyncEmbeddingModel:
    def __init__(self):
        self.azure_token_provider: AzureTokenProvider | None = None
        openai_key_name = "OPENAI_API_KEY"
        azure_key_name = "AZURE_OPENAI_API_KEY"
        if os.getenv(openai_key_name):
            print(f"Using OpenAI")
            self.async_client = AsyncOpenAI()
        elif azure_api_key := os.getenv(azure_key_name):
            print("Using Azure OpenAI")
            # TODO: support different endpoint names
            endpoint_name = "AZURE_OPENAI_ENDPOINT_EMBEDDING_3_SMALL"
            self.azure_endpoint = os.environ.get(endpoint_name)
            if not self.azure_endpoint:
                raise ValueError(f"Environment variable {endpoint_name} not found.")
            m = re.search(r"[?,]api-version=([^,]+)$", self.azure_endpoint)
            if not m:
                raise ValueError(
                    "{endpoint_name}={endpoint} doesn't end in api-version=<version>"
                )
            self.azure_api_version = m.group(1)
            if azure_api_key.lower() == "identity":
                print("Using shared TokenProvider")
                self.azure_token_provider = get_shared_token_provider()
                azure_api_key = self.azure_token_provider.get_token()
            self.async_client = AsyncAzureOpenAI(
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint,
                api_key=azure_api_key,
            )
        else:
            raise ValueError(
                f"Neither {openai_key_name} nor {azure_key_name} found in environment."
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

    async def get_embedding(self, input: str) -> NormalizedEmbedding:
        embeddings = await self.get_embeddings([input])
        return embeddings[0]

    async def get_embeddings(self, input: list[str]) -> NormalizedEmbeddings:
        if not input:
            # Save round trip and avoid error from reshape((0, N)) for N != 0.
            return np.array([], dtype=np.float32).reshape((0, 0))
        if self.azure_token_provider and self.azure_token_provider.needs_refresh():
            await self.refresh_auth()
        data = (
            await self.async_client.embeddings.create(
                input=input,
                model="text-embedding-3-small",
            )
        ).data
        assert len(data) == len(input), (len(data), "!=", len(input))
        return np.array([d.embedding for d in data], dtype=np.float32)


async def main():
    import dotenv

    dotenv.load_dotenv(os.path.expanduser("~/TypeAgent/ts/.env"))

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
