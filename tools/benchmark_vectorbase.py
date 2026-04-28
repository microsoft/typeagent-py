#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark VectorBase lookup methods as a standalone script.

Usage:
    uv run python tools/benchmark_vectorbase.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import statistics
import time

import numpy as np

from typeagent.aitools.embeddings import NormalizedEmbedding, NormalizedEmbeddings
from typeagent.aitools.vectorbase import (
    ScoredInt,
    TextEmbeddingIndexSettings,
    VectorBase,
)


class NullEmbeddingModel:
    @property
    def model_name(self) -> str:
        return "benchmark-local"

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        return None

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        raise RuntimeError("VectorBase benchmark does not use embedding generation")

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        raise RuntimeError("VectorBase benchmark does not use embedding generation")

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        raise RuntimeError("VectorBase benchmark does not use embedding generation")

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        raise RuntimeError("VectorBase benchmark does not use embedding generation")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark VectorBase lookup methods with synthetic vectors.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=200,
        help="Number of timed rounds to run for each benchmark.",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=20,
        help="Number of untimed warmup rounds to run first.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=384,
        help="Embedding dimension to generate.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1_000,
        help="Subset size for fuzzy_lookup_embedding_in_subset.",
    )
    return parser


def make_vectorbase(
    vector_count: int, dim: int, seed: int
) -> tuple[VectorBase, NormalizedEmbedding]:
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((vector_count, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors /= norms

    settings = TextEmbeddingIndexSettings(embedding_model=NullEmbeddingModel())
    vectorbase = VectorBase(settings)
    vectorbase.add_embeddings(None, vectors)

    query = rng.standard_normal(dim).astype(np.float32)
    query /= np.linalg.norm(query)
    return vectorbase, query


def run_benchmark(
    target: Callable[[], list[ScoredInt]], rounds: int, warmup_rounds: int
) -> list[float]:
    for _ in range(warmup_rounds):
        target()

    samples_us: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter_ns()
        target()
        elapsed_us = (time.perf_counter_ns() - start) / 1_000
        samples_us.append(elapsed_us)
    return samples_us


def validate_result(result: list[ScoredInt]) -> None:
    if len(result) != 10:
        raise ValueError(f"Expected 10 hits, got {len(result)}")
    if not all(isinstance(item, ScoredInt) for item in result):
        raise TypeError("Expected every result item to be a ScoredInt")


def print_report(
    label: str, samples_us: list[float], rounds: int, warmup_rounds: int
) -> None:
    print(label)
    print(f"  rounds: {rounds} ({warmup_rounds} warmup)")
    print(f"  min:    {min(samples_us):9.3f} us")
    print(f"  mean:   {statistics.fmean(samples_us):9.3f} us")
    print(f"  median: {statistics.median(samples_us):9.3f} us")
    print(f"  max:    {max(samples_us):9.3f} us")


def main() -> None:
    args = create_arg_parser().parse_args()

    vb_1k, query_1k = make_vectorbase(1_000, args.dim, seed=42)
    vb_10k, query_10k = make_vectorbase(10_000, args.dim, seed=43)
    subset_rng = np.random.default_rng(99)
    subset = subset_rng.choice(10_000, size=args.subset_size, replace=False).tolist()

    benchmarks: list[tuple[str, Callable[[], list[ScoredInt]]]] = [
        (
            "fuzzy_lookup_embedding (1k vectors)",
            lambda: vb_1k.fuzzy_lookup_embedding(query_1k, max_hits=10, min_score=0.0),
        ),
        (
            "fuzzy_lookup_embedding (10k vectors)",
            lambda: vb_10k.fuzzy_lookup_embedding(
                query_10k, max_hits=10, min_score=0.0
            ),
        ),
        (
            f"fuzzy_lookup_embedding_in_subset ({args.subset_size} of 10k)",
            lambda: vb_10k.fuzzy_lookup_embedding_in_subset(
                query_10k,
                subset,
                max_hits=10,
                min_score=0.0,
            ),
        ),
    ]

    for label, target in benchmarks:
        validate_result(target())
        samples_us = run_benchmark(target, args.rounds, args.warmup_rounds)
        print_report(label, samples_us, args.rounds, args.warmup_rounds)


if __name__ == "__main__":
    main()
