# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark retrieval settings for known embedding models.

This script evaluates the Adrian Tchaikovsky Episode 53 search dataset in
`tests/testdata/` and reports retrieval quality for combinations of
`min_score` and `max_hits`.

The benchmark is intentionally narrow:
- It only measures retrieval against `messageMatches` ground truth.
- It is meant to help choose repository defaults for known models.
- In practice, `min_score` is the primary library default this informs.
- It does not prove universal "best" settings for every dataset.

Usage:
    uv run python tools/benchmark_embeddings.py
    uv run python tools/benchmark_embeddings.py --model openai:text-embedding-3-small
    uv run python tools/benchmark_embeddings.py --model openai:text-embedding-3-small --min-score-start 0.01 --min-score-stop 0.20 --min-score-step 0.01
"""

import argparse
import asyncio
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from typeagent.aitools.embeddings import (
    IEmbeddingModel,
    NormalizedEmbeddings,
)
from typeagent.aitools.model_adapters import create_embedding_model
from typeagent.aitools.vectorbase import (
    TextEmbeddingIndexSettings,
    VectorBase,
)

DEFAULT_MIN_SCORES = [score / 10 for score in range(1, 10)]
DEFAULT_MAX_HITS = [5, 10, 15, 20]
DATA_DIR = Path("tests") / "testdata"
INDEX_DATA_PATH = DATA_DIR / "Episode_53_AdrianTchaikovsky_index_data.json"
SEARCH_RESULTS_PATH = DATA_DIR / "Episode_53_Search_results.json"


@dataclass
class SearchQueryCase:
    """A benchmark query paired with the message ordinals it should retrieve."""

    query: str
    expected_matches: list[int]


@dataclass
class SearchMetrics:
    """Aggregate retrieval quality metrics for one benchmark row."""

    hit_rate: float
    mean_reciprocal_rank: float


@dataclass
class TopScoreStats:
    """Observed top-1 score statistics across all benchmark queries."""

    min_top_score: float
    mean_top_score: float
    max_top_score: float


@dataclass
class BenchmarkRow:
    """One `(min_score, max_hits)` configuration evaluated by the benchmark."""

    min_score: float
    max_hits: int
    metrics: SearchMetrics


def parse_float_list(raw: str | None) -> list[float]:
    """Parse explicit min-score values or fall back to the coarse default grid."""

    if raw is None:
        return DEFAULT_MIN_SCORES
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--min-scores must contain at least one value")
    return values


def build_float_range(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive decimal-safe float range for score sweeps."""

    if step <= 0:
        raise ValueError("--min-score-step must be positive")
    if start > stop:
        raise ValueError("--min-score-start must be <= --min-score-stop")

    start_decimal = Decimal(str(start))
    stop_decimal = Decimal(str(stop))
    step_decimal = Decimal(str(step))
    values: list[float] = []
    current = start_decimal
    while current <= stop_decimal:
        values.append(float(current))
        current += step_decimal
    return values


def resolve_min_scores(
    raw: str | None,
    start: float | None,
    stop: float | None,
    step: float | None,
) -> list[float]:
    """Resolve the benchmark min-score grid from explicit values or a generated range."""

    range_args = [start, stop, step]
    using_range = any(value is not None for value in range_args)
    if using_range:
        if raw is not None:
            raise ValueError(
                "Use either --min-scores or the --min-score-start/stop/step range"
            )
        if any(value is None for value in range_args):
            raise ValueError(
                "--min-score-start, --min-score-stop, and --min-score-step must all be set together"
            )
        assert start is not None
        assert stop is not None
        assert step is not None
        return build_float_range(start, stop, step)
    return parse_float_list(raw)


def parse_int_list(raw: str | None) -> list[int]:
    """Parse positive integer arguments such as `max_hits` grids."""

    if raw is None:
        return DEFAULT_MAX_HITS
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--max-hits must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError("--max-hits values must be positive integers")
    return values


def load_message_texts(repo_root: Path) -> list[str]:
    """Load the benchmark corpus as one text blob per message."""

    index_data = json.loads((repo_root / INDEX_DATA_PATH).read_text(encoding="utf-8"))
    messages = index_data["messages"]
    return [" ".join(message.get("textChunks", [])) for message in messages]


def load_search_queries(repo_root: Path) -> list[SearchQueryCase]:
    """Load benchmark queries that include message-level ground-truth matches."""

    search_data = json.loads(
        (repo_root / SEARCH_RESULTS_PATH).read_text(encoding="utf-8")
    )
    cases: list[SearchQueryCase] = []
    for item in search_data:
        search_text = item.get("searchText")
        results = item.get("results", [])
        if not search_text or not results:
            continue
        expected_matches = results[0].get("messageMatches", [])
        if not expected_matches:
            continue
        cases.append(SearchQueryCase(search_text, expected_matches))
    return cases


async def build_vector_base(
    model_spec: str | None,
    message_texts: list[str],
    batch_size: int,
) -> tuple[IEmbeddingModel, VectorBase]:
    """Build a message-level vector index for the benchmark corpus."""

    model = create_embedding_model(model_spec)
    settings = TextEmbeddingIndexSettings(
        embedding_model=model,
        min_score=0.0,
        max_matches=None,
        batch_size=batch_size,
    )
    vector_base = VectorBase(settings)
    for start in range(0, len(message_texts), batch_size):
        batch = message_texts[start : start + batch_size]
        await vector_base.add_keys(batch)

    return model, vector_base


def evaluate_search_queries(
    vector_base: VectorBase,
    query_cases: list[SearchQueryCase],
    query_embeddings: NormalizedEmbeddings,
    min_score: float,
    max_hits: int,
) -> SearchMetrics:
    """Evaluate one benchmark row over every labeled query."""

    hit_count = 0
    reciprocal_ranks: list[float] = []

    for case, query_embedding in zip(query_cases, query_embeddings):
        scored_results = vector_base.fuzzy_lookup_embedding(
            query_embedding,
            max_hits=max_hits,
            min_score=min_score,
        )
        rank = 0
        for result_index, scored_result in enumerate(scored_results, start=1):
            if scored_result.item in case.expected_matches:
                rank = result_index
                break
        if rank > 0:
            hit_count += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return SearchMetrics(
        hit_rate=(hit_count / len(query_cases)) * 100,
        mean_reciprocal_rank=mean(reciprocal_ranks),
    )


def measure_top_score_stats(
    vector_base: VectorBase,
    query_embeddings: NormalizedEmbeddings,
) -> TopScoreStats:
    """Measure the achievable top-1 score range for the current model and corpus."""

    top_scores: list[float] = []
    for query_embedding in query_embeddings:
        scored_results = vector_base.fuzzy_lookup_embedding(
            query_embedding,
            max_hits=1,
            min_score=0.0,
        )
        top_scores.append(scored_results[0].score if scored_results else 0.0)

    return TopScoreStats(
        min_top_score=min(top_scores),
        mean_top_score=mean(top_scores),
        max_top_score=max(top_scores),
    )


def filter_min_scores_by_ceiling(
    min_scores: list[float], max_top_score: float
) -> tuple[list[float], list[float]]:
    """Discard score thresholds that cannot return any results for this run."""

    effective_scores = [
        min_score for min_score in min_scores if min_score <= max_top_score + 1e-9
    ]
    skipped_scores = [
        min_score for min_score in min_scores if min_score > max_top_score + 1e-9
    ]
    return effective_scores, skipped_scores


def select_best_row(rows: list[BenchmarkRow]) -> BenchmarkRow:
    """Prefer the strongest MRR/Hit Rate row, then the stricter score cutoff."""

    return max(
        rows,
        key=lambda row: (
            row.metrics.mean_reciprocal_rank,
            row.metrics.hit_rate,
            row.min_score,
            -row.max_hits,
        ),
    )


def print_rows(rows: list[BenchmarkRow]) -> None:
    """Print the benchmark grid in a reviewer-friendly table."""

    print("=" * 72)
    print("SEARCH BENCHMARK (Episode 53 messageMatches ground truth)")
    print("=" * 72)
    print(f"{'Min Score':<12} | {'Max Hits':<10} | {'Hit Rate (%)':<15} | {'MRR':<10}")
    print("-" * 65)
    for row in rows:
        print(
            f"{row.min_score:<12.2f} | {row.max_hits:<10d} | "
            f"{row.metrics.hit_rate:<15.2f} | "
            f"{row.metrics.mean_reciprocal_rank:<10.4f}"
        )
    print("-" * 65)


async def run_benchmark(
    model_spec: str | None,
    min_scores: list[float],
    max_hits_values: list[int],
    batch_size: int,
) -> None:
    """Run a single benchmark sweep and print the evaluated grid."""

    load_dotenv()

    repo_root = Path(__file__).resolve().parent.parent
    message_texts = load_message_texts(repo_root)
    query_cases = load_search_queries(repo_root)
    if not query_cases:
        raise ValueError("No search queries with messageMatches found in the dataset")
    model, vector_base = await build_vector_base(
        model_spec,
        message_texts,
        batch_size,
    )
    query_embeddings = await model.get_embeddings([case.query for case in query_cases])
    top_score_stats = measure_top_score_stats(vector_base, query_embeddings)
    effective_min_scores, skipped_min_scores = filter_min_scores_by_ceiling(
        min_scores,
        top_score_stats.max_top_score,
    )
    if not effective_min_scores:
        raise ValueError(
            "No requested min_score values are below the observed top-score ceiling "
            f"of {top_score_stats.max_top_score:.4f}"
        )

    rows: list[BenchmarkRow] = []
    for min_score in effective_min_scores:
        for max_hits in max_hits_values:
            metrics = evaluate_search_queries(
                vector_base,
                query_cases,
                query_embeddings,
                min_score,
                max_hits,
            )
            rows.append(BenchmarkRow(min_score, max_hits, metrics))

    print(f"Model: {model.model_name}")
    print(f"Messages indexed: {len(message_texts)}")
    print(f"Queries evaluated: {len(query_cases)}")
    print(
        "Observed top-1 score range: "
        f"{top_score_stats.min_top_score:.4f}..{top_score_stats.max_top_score:.4f} "
        f"(mean {top_score_stats.mean_top_score:.4f})"
    )
    if skipped_min_scores:
        print(
            f"Skipped {len(skipped_min_scores)} min_score values above "
            f"{top_score_stats.max_top_score:.4f}; they cannot return any matches."
        )
    print()
    print_rows(rows)

    best_row = select_best_row(rows)
    print()
    print("Best-scoring benchmark row:")
    print(f"  min_score={best_row.min_score:.2f}")
    print(f"  max_hits={best_row.max_hits}")
    print(f"  hit_rate={best_row.metrics.hit_rate:.2f}%")
    print(f"  mrr={best_row.metrics.mean_reciprocal_rank:.4f}")


def main() -> None:
    """Parse CLI arguments and run the benchmark once."""

    parser = argparse.ArgumentParser(
        description="Benchmark retrieval settings for an embedding model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Provider and model name, e.g. 'openai:text-embedding-3-small'",
    )
    parser.add_argument(
        "--min-scores",
        type=str,
        default=None,
        help="Comma-separated min_score values to test.",
    )
    parser.add_argument(
        "--min-score-start",
        type=float,
        default=None,
        help="Inclusive start of a generated min_score range.",
    )
    parser.add_argument(
        "--min-score-stop",
        type=float,
        default=None,
        help="Inclusive end of a generated min_score range.",
    )
    parser.add_argument(
        "--min-score-step",
        type=float,
        default=None,
        help="Step size for a generated min_score range.",
    )
    parser.add_argument(
        "--max-hits",
        type=str,
        default=None,
        help="Comma-separated max_hits values to test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used when building the index.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    asyncio.run(
        run_benchmark(
            model_spec=args.model,
            min_scores=resolve_min_scores(
                args.min_scores,
                args.min_score_start,
                args.min_score_stop,
                args.min_score_step,
            ),
            max_hits_values=parse_int_list(args.max_hits),
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
