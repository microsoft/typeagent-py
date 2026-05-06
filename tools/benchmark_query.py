#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark lookup_term_filtered as a standalone script.

Usage:
    uv run python tools/benchmark_query.py
"""

from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable
import hashlib
import os
import shutil
import statistics
import tempfile
import time

import numpy as np

from typeagent.aitools.embeddings import (
    CachingEmbeddingModel,
    NormalizedEmbedding,
    NormalizedEmbeddings,
)
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces_core import Term
from typeagent.knowpro.query import lookup_term_filtered
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)


class DeterministicBenchmarkEmbedder:
    def __init__(self, embedding_size: int) -> None:
        self._embedding_size = embedding_size

    @property
    def model_name(self) -> str:
        return "benchmark-local"

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        return _compute_embedding(input, self._embedding_size)

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        if not input:
            raise ValueError("Cannot embed an empty list")
        return np.stack(
            [_compute_embedding(value, self._embedding_size) for value in input]
        ).astype(np.float32)


def _compute_embedding(text: str, embedding_size: int) -> NormalizedEmbedding:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    repeats = (embedding_size + len(digest) - 1) // len(digest)
    data = (digest * repeats)[:embedding_size]
    embedding = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
    embedding = embedding - np.float32(127.5)
    norm = np.float32(np.linalg.norm(embedding))
    if norm > 0:
        embedding = embedding / norm
    return embedding.astype(np.float32)


def create_benchmark_embedding_model(embedding_size: int) -> CachingEmbeddingModel:
    return CachingEmbeddingModel(DeterministicBenchmarkEmbedder(embedding_size))


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark lookup_term_filtered with a synthetic transcript.",
    )
    parser.add_argument(
        "--messages",
        type=int,
        default=200,
        help="Number of synthetic messages to index before running the benchmark.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Number of timed rounds to run.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=20,
        help="Number of untimed warmup rounds to run first.",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=16,
        help="Embedding size for the local deterministic benchmark model.",
    )
    return parser


def make_settings(embedding_size: int) -> ConversationSettings:
    settings = ConversationSettings(
        model=create_benchmark_embedding_model(embedding_size)
    )
    settings.semantic_ref_index_settings.auto_extract_knowledge = False
    return settings


def synthetic_messages(count: int) -> list[TranscriptMessage]:
    return [
        TranscriptMessage(
            text_chunks=[f"Message {i} about topic {i % 10}"],
            metadata=TranscriptMessageMeta(speaker=f"Speaker{i % 3}"),
            tags=[f"tag{i % 5}"],
        )
        for i in range(count)
    ]


async def create_indexed_transcript(
    settings: ConversationSettings,
    storage: SqliteStorageProvider,
    message_count: int,
) -> Transcript:
    settings.storage_provider = storage
    transcript = await Transcript.create(settings, name="benchmark-query")
    await transcript.add_messages_with_indexing(synthetic_messages(message_count))
    return transcript


async def find_best_term(transcript: Transcript) -> tuple[str, int]:
    semref_index = transcript.semantic_ref_index
    assert semref_index is not None

    best_term: str | None = None
    best_count = 0

    for term in await semref_index.get_terms():
        refs = await semref_index.lookup_term(term)
        ref_count = len(refs) if refs is not None else 0
        if ref_count > best_count:
            best_count = ref_count
            best_term = term

    if best_term is None:
        raise ValueError("No terms found after indexing")

    return best_term, best_count


async def run_benchmark(
    target: Callable[[], Awaitable[None]],
    rounds: int,
    warmup_rounds: int,
) -> list[float]:
    for _ in range(warmup_rounds):
        await target()

    samples_us: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        await target()
        elapsed_us = (time.perf_counter_ns() - start) / 1_000
        samples_us.append(elapsed_us)
    return samples_us


def print_report(
    label: str, samples_us: list[float], rounds: int, warmup_rounds: int
) -> None:
    print(label)
    print(f"  rounds: {rounds} ({warmup_rounds} warmup)")
    print(f"  min:    {min(samples_us):9.3f} us")
    print(f"  mean:   {statistics.fmean(samples_us):9.3f} us")
    print(f"  median: {statistics.median(samples_us):9.3f} us")
    print(f"  max:    {max(samples_us):9.3f} us")


async def main() -> None:
    args = create_arg_parser().parse_args()
    temp_dir = tempfile.mkdtemp(prefix="benchmark-query-")
    db_path = os.path.join(temp_dir, "query_bench.db")

    settings = make_settings(args.embedding_size)
    storage = SqliteStorageProvider(
        db_path,
        message_type=TranscriptMessage,
        message_text_index_settings=settings.message_text_index_settings,
        related_term_index_settings=settings.related_term_index_settings,
    )

    try:
        transcript = await create_indexed_transcript(settings, storage, args.messages)
        best_term, best_count = await find_best_term(transcript)
        print(f"Benchmarking term {best_term!r} with {best_count} matches")

        term = Term(text=best_term)
        semref_index = transcript.semantic_ref_index
        semantic_refs = transcript.semantic_refs
        assert semref_index is not None
        assert semantic_refs is not None

        async def target() -> None:
            results = await lookup_term_filtered(
                semref_index,
                term,
                semantic_refs,
                lambda _metadata, _scored: True,
            )
            if results is None:
                raise ValueError(f"No results found for {best_term!r}")

        samples_us = await run_benchmark(target, args.rounds, args.warmup_rounds)
        print_report(
            "lookup_term_filtered (accept-all filter)",
            samples_us,
            args.rounds,
            args.warmup_rounds,
        )
    finally:
        await storage.close()
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
