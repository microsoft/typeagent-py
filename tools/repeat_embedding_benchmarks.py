# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run embedding benchmarks repeatedly and save raw/summary JSON results.

This script runs `tools/benchmark_embeddings.py` logic multiple times for each
embedding model, stores every run as JSON, and writes aggregate summaries that
can be used to justify tuned defaults.

Usage:
    uv run python tools/repeat_embedding_benchmarks.py
    uv run python tools/repeat_embedding_benchmarks.py --runs 30
    uv run python tools/repeat_embedding_benchmarks.py --models openai:text-embedding-3-small,openai:text-embedding-3-large,openai:text-embedding-ada-002
"""

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

from benchmark_embeddings import (
    BenchmarkRow,
    build_vector_base,
    DEFAULT_MAX_HITS,
    DEFAULT_MIN_SCORES,
    evaluate_search_queries,
    load_message_texts,
    load_search_queries,
    parse_float_list,
    parse_int_list,
    select_best_row,
)

DEFAULT_MODELS = [
    "openai:text-embedding-3-small",
    "openai:text-embedding-3-large",
    "openai:text-embedding-ada-002",
]
DEFAULT_OUTPUT_DIR = Path("benchmark_results")


@dataclass
class RunRow:
    min_score: float
    max_hits: int
    hit_rate: float
    mean_reciprocal_rank: float


@dataclass
class RunResult:
    run_index: int
    model_spec: str
    resolved_model_name: str
    message_count: int
    query_count: int
    rows: list[RunRow]
    best_row: RunRow


def sanitize_model_name(model_spec: str) -> str:
    return model_spec.replace(":", "__").replace("/", "_").replace("\\", "_")


def benchmark_row_to_run_row(row: BenchmarkRow) -> RunRow:
    return RunRow(
        min_score=row.min_score,
        max_hits=row.max_hits,
        hit_rate=row.metrics.hit_rate,
        mean_reciprocal_rank=row.metrics.mean_reciprocal_rank,
    )


def summarize_runs(model_spec: str, runs: list[RunResult]) -> dict[str, object]:
    summary_rows: dict[tuple[float, int], list[RunRow]] = {}
    for run in runs:
        for row in run.rows:
            summary_rows.setdefault((row.min_score, row.max_hits), []).append(row)

    averaged_rows: list[dict[str, float | int]] = []
    for (min_score, max_hits), rows in sorted(summary_rows.items()):
        averaged_rows.append(
            {
                "min_score": min_score,
                "max_hits": max_hits,
                "mean_hit_rate": mean(row.hit_rate for row in rows),
                "mean_mrr": mean(row.mean_reciprocal_rank for row in rows),
            }
        )

    best_rows = [run.best_row for run in runs]
    best_min_score_counts: dict[str, int] = {}
    best_max_hits_counts: dict[str, int] = {}
    for row in best_rows:
        best_min_score_counts[f"{row.min_score:.2f}"] = (
            best_min_score_counts.get(f"{row.min_score:.2f}", 0) + 1
        )
        best_max_hits_counts[str(row.max_hits)] = (
            best_max_hits_counts.get(str(row.max_hits), 0) + 1
        )

    averaged_best_row = max(
        averaged_rows,
        key=lambda row: (
            float(row["mean_mrr"]),
            float(row["mean_hit_rate"]),
            -float(row["min_score"]),
            -int(row["max_hits"]),
        ),
    )

    return {
        "model_spec": model_spec,
        "resolved_model_name": runs[0].resolved_model_name,
        "run_count": len(runs),
        "message_count": runs[0].message_count,
        "query_count": runs[0].query_count,
        "candidate_rows": averaged_rows,
        "recommended_row": averaged_best_row,
        "best_min_score_counts": best_min_score_counts,
        "best_max_hits_counts": best_max_hits_counts,
    }


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_markdown_summary(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Repeated Embedding Benchmark Summary",
        "",
        "| Model | Runs | Recommended min_score | Recommended max_hits | Mean hit rate | Mean MRR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        recommended_row = summary["recommended_row"]
        assert isinstance(recommended_row, dict)
        lines.append(
            "| "
            f"{summary['resolved_model_name']} | "
            f"{summary['run_count']} | "
            f"{recommended_row['min_score']:.2f} | "
            f"{recommended_row['max_hits']} | "
            f"{recommended_row['mean_hit_rate']:.2f} | "
            f"{recommended_row['mean_mrr']:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def run_single_model_benchmark(
    model_spec: str,
    runs: int,
    min_scores: list[float],
    max_hits_values: list[int],
    batch_size: int,
    output_dir: Path,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parent.parent
    message_texts = load_message_texts(repo_root)
    query_cases = load_search_queries(repo_root)
    model_output_dir = output_dir / sanitize_model_name(model_spec)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[RunResult] = []
    for run_index in range(1, runs + 1):
        model, vector_base = await build_vector_base(
            model_spec, message_texts, batch_size
        )
        query_embeddings = await model.get_embeddings(
            [case.query for case in query_cases]
        )
        benchmark_rows: list[BenchmarkRow] = []
        for min_score in min_scores:
            for max_hits in max_hits_values:
                metrics = evaluate_search_queries(
                    vector_base,
                    query_cases,
                    query_embeddings,
                    min_score,
                    max_hits,
                )
                benchmark_rows.append(BenchmarkRow(min_score, max_hits, metrics))

        best_row = select_best_row(benchmark_rows)
        run_result = RunResult(
            run_index=run_index,
            model_spec=model_spec,
            resolved_model_name=model.model_name,
            message_count=len(message_texts),
            query_count=len(query_cases),
            rows=[benchmark_row_to_run_row(row) for row in benchmark_rows],
            best_row=benchmark_row_to_run_row(best_row),
        )
        run_results.append(run_result)
        write_json(model_output_dir / f"run_{run_index:02d}.json", asdict(run_result))

    summary = summarize_runs(model_spec, run_results)
    write_json(model_output_dir / "summary.json", summary)
    return summary


async def run_repeated_benchmarks(
    models: list[str],
    runs: int,
    min_scores: list[float],
    max_hits_values: list[int],
    batch_size: int,
    output_root: Path,
) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at_utc": timestamp,
        "runs_per_model": runs,
        "models": models,
        "min_scores": min_scores,
        "max_hits_values": max_hits_values,
        "batch_size": batch_size,
    }
    write_json(output_dir / "metadata.json", metadata)

    summaries: list[dict[str, object]] = []
    for model_spec in models:
        print(f"Running {runs} benchmark iterations for {model_spec}...")
        summary = await run_single_model_benchmark(
            model_spec=model_spec,
            runs=runs,
            min_scores=min_scores,
            max_hits_values=max_hits_values,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        summaries.append(summary)

    write_json(output_dir / "summary.json", summaries)
    write_markdown_summary(output_dir / "summary.md", summaries)
    return output_dir


def parse_models(raw: str | None) -> list[str]:
    if raw is None:
        return DEFAULT_MODELS
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must contain at least one model")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run embedding benchmarks repeatedly and save JSON results."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model specs to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of repeated runs per model.",
    )
    parser.add_argument(
        "--min-scores",
        type=str,
        default=",".join(f"{score:.2f}" for score in DEFAULT_MIN_SCORES),
        help="Comma-separated min_score values to test.",
    )
    parser.add_argument(
        "--max-hits",
        type=str,
        default=",".join(str(value) for value in DEFAULT_MAX_HITS),
        help="Comma-separated max_hits values to test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used when building the index.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where benchmark results will be written.",
    )
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be a positive integer")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer")

    load_dotenv()
    output_dir = asyncio.run(
        run_repeated_benchmarks(
            models=parse_models(args.models),
            runs=args.runs,
            min_scores=parse_float_list(args.min_scores),
            max_hits_values=parse_int_list(args.max_hits),
            batch_size=args.batch_size,
            output_root=Path(args.output_dir),
        )
    )
    print(f"Wrote benchmark results to {output_dir}")


if __name__ == "__main__":
    main()
