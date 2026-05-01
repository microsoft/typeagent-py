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
    uv run python tools/repeat_embedding_benchmarks.py --models openai:text-embedding-3-small --min-score-start 0.01 --min-score-stop 0.20 --min-score-step 0.01
"""

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
import json
from pathlib import Path
from statistics import mean

from benchmark_embeddings import (
    ANSWER_EMBEDDING_SOURCE,
    BenchmarkRow,
    build_pipeline_conversation,
    build_text_vector_base,
    build_vector_base,
    CORPUS_EMBEDDING_SOURCE,
    DEFAULT_MAX_HITS,
    evaluate_grid,
    INDEX_DATA_PATH,
    INDEX_EMBEDDINGS_PATH,
    load_answer_queries,
    load_corpus_metadata,
    load_message_texts,
    load_pipeline_queries,
    load_related_term_queries,
    load_related_term_texts,
    load_search_queries,
    measure_top_score_stats,
    parse_int_list,
    PIPELINE_SCORING_PATHS,
    QUERY_EMBEDDING_SOURCE,
    resolve_min_scores,
    SEARCH_RESULTS_PATH,
    select_best_row,
)
from dotenv import load_dotenv

DEFAULT_MODELS = [
    "openai:text-embedding-3-small",
    "openai:text-embedding-3-large",
    "openai:text-embedding-ada-002",
]
DEFAULT_OUTPUT_DIR = Path("benchmark_results")


@dataclass
class RunRow:
    """Serialized benchmark row for one repeated run."""

    min_score: float
    max_hits: int
    hit_rate: float
    mean_reciprocal_rank: float
    pipeline_hit_rate: float | None
    pipeline_mean_reciprocal_rank: float | None
    pipeline_mean_result_count: float | None
    related_hit_rate: float | None
    related_mean_reciprocal_rank: float | None
    related_mean_result_count: float | None
    answerable_support: float | None
    no_answer_rejection_rate: float | None
    semantic_score: float | None


@dataclass
class RunResult:
    """All measurements captured for one benchmark repetition."""

    run_index: int
    model_spec: str
    resolved_model_name: str
    message_count: int
    query_count: int
    min_top_score: float
    mean_top_score: float
    max_top_score: float
    rows: list[RunRow]
    best_row: RunRow


def sanitize_model_name(model_spec: str) -> str:
    """Convert a model spec into a filesystem-safe directory name."""

    return model_spec.replace(":", "__").replace("/", "_").replace("\\", "_")


def benchmark_row_to_run_row(row: BenchmarkRow) -> RunRow:
    """Flatten a benchmark row into the JSON-friendly repeated-run shape."""

    return RunRow(
        min_score=row.min_score,
        max_hits=row.max_hits,
        hit_rate=row.metrics.hit_rate,
        mean_reciprocal_rank=row.metrics.mean_reciprocal_rank,
        pipeline_hit_rate=(
            row.pipeline_metrics.hit_rate if row.pipeline_metrics else None
        ),
        pipeline_mean_reciprocal_rank=(
            row.pipeline_metrics.mean_reciprocal_rank if row.pipeline_metrics else None
        ),
        pipeline_mean_result_count=(
            row.pipeline_metrics.mean_result_count if row.pipeline_metrics else None
        ),
        related_hit_rate=(
            row.related_metrics.hit_rate if row.related_metrics else None
        ),
        related_mean_reciprocal_rank=(
            row.related_metrics.mean_reciprocal_rank if row.related_metrics else None
        ),
        related_mean_result_count=(
            row.related_metrics.mean_result_count if row.related_metrics else None
        ),
        answerable_support=(
            row.answer_metrics.answerable_support if row.answer_metrics else None
        ),
        no_answer_rejection_rate=(
            row.answer_metrics.no_answer_rejection_rate if row.answer_metrics else None
        ),
        semantic_score=(
            row.answer_metrics.semantic_score if row.answer_metrics else None
        ),
    )


def summarize_runs(model_spec: str, runs: list[RunResult]) -> dict[str, object]:
    """Average repeated benchmark runs into a per-model summary payload."""

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
                "mean_pipeline_hit_rate": mean(
                    row.pipeline_hit_rate or 0.0 for row in rows
                ),
                "mean_pipeline_mrr": mean(
                    row.pipeline_mean_reciprocal_rank or 0.0 for row in rows
                ),
                "mean_pipeline_result_count": mean(
                    row.pipeline_mean_result_count or 0.0 for row in rows
                ),
                "mean_related_hit_rate": mean(
                    row.related_hit_rate or 0.0 for row in rows
                ),
                "mean_related_mrr": mean(
                    row.related_mean_reciprocal_rank or 0.0 for row in rows
                ),
                "mean_related_result_count": mean(
                    row.related_mean_result_count or 0.0 for row in rows
                ),
                "mean_answerable_support": mean(
                    row.answerable_support or 0.0 for row in rows
                ),
                "mean_no_answer_rejection_rate": mean(
                    row.no_answer_rejection_rate or 0.0 for row in rows
                ),
                "mean_semantic_score": mean(row.semantic_score or 0.0 for row in rows),
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
            float(row["mean_pipeline_mrr"]),
            float(row["mean_pipeline_hit_rate"]),
            float(row["mean_related_mrr"]),
            float(row["mean_related_hit_rate"]),
            float(row["mean_mrr"]),
            float(row["mean_hit_rate"]),
            float(row["min_score"]),
            float(row["mean_semantic_score"]),
            -int(row["max_hits"]),
        ),
    )

    return {
        "model_spec": model_spec,
        "resolved_model_name": runs[0].resolved_model_name,
        "run_count": len(runs),
        "message_count": runs[0].message_count,
        "query_count": runs[0].query_count,
        "min_top_score": mean(run.min_top_score for run in runs),
        "mean_top_score": mean(run.mean_top_score for run in runs),
        "max_top_score": mean(run.max_top_score for run in runs),
        "candidate_rows": averaged_rows,
        "recommended_row": averaged_best_row,
        "best_min_score_counts": best_min_score_counts,
        "best_max_hits_counts": best_max_hits_counts,
    }


def write_json(path: Path, data: object) -> None:
    """Write a JSON artifact with stable indentation for review and reuse."""

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_markdown_summary(path: Path, summaries: list[dict[str, object]]) -> None:
    """Write the reviewer-facing markdown summary for all benchmarked models."""

    lines = [
        "# Repeated Embedding Benchmark Summary",
        "",
        "Corpus and query embeddings are recomputed for each evaluated model.",
        (
            "The serialized benchmark sidecar "
            f"`{INDEX_EMBEDDINGS_PATH.name}` is ignored."
        ),
        "",
        "| Model | Runs | Recommended min_score | Recommended max_hits | Pipeline hit rate | Pipeline MRR | Related hit rate | Related MRR | Mean hit rate | Mean MRR | Answer support | No-answer rejection | Semantic score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            f"{recommended_row['mean_pipeline_hit_rate']:.2f} | "
            f"{recommended_row['mean_pipeline_mrr']:.4f} | "
            f"{recommended_row['mean_related_hit_rate']:.2f} | "
            f"{recommended_row['mean_related_mrr']:.4f} | "
            f"{recommended_row['mean_hit_rate']:.2f} | "
            f"{recommended_row['mean_mrr']:.4f} | "
            f"{recommended_row['mean_answerable_support']:.4f} | "
            f"{recommended_row['mean_no_answer_rejection_rate']:.2f} | "
            f"{recommended_row['mean_semantic_score']:.2f} |"
        )
    lines.append("")
    for summary in summaries:
        lines.append(
            f"- {summary['resolved_model_name']}: observed top-1 score range "
            f"{summary['min_top_score']:.4f}..{summary['max_top_score']:.4f} "
            f"(mean {summary['mean_top_score']:.4f})."
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_run_suite_metadata(
    repo_root: Path,
    timestamp: str,
    models: list[str],
    runs: int,
    min_scores: list[float],
    max_hits_values: list[int],
    batch_size: int,
) -> dict[str, object]:
    """Build the shared metadata payload for one repeated benchmark suite."""

    corpus_metadata = load_corpus_metadata(repo_root)
    return {
        "created_at_utc": timestamp,
        "runs_per_model": runs,
        "models": models,
        "message_source_json": str(INDEX_DATA_PATH),
        "pipeline_source_json": str(SEARCH_RESULTS_PATH),
        "related_term_source_json": str(SEARCH_RESULTS_PATH),
        "pipeline_scoring_paths": PIPELINE_SCORING_PATHS,
        "ignored_serialized_embedding_sidecar": str(INDEX_EMBEDDINGS_PATH),
        "ignored_serialized_embedding_size": (
            corpus_metadata.serialized_embedding_size
        ),
        "ignored_serialized_message_embedding_count": (
            corpus_metadata.serialized_message_count
        ),
        "ignored_serialized_related_embedding_count": (
            corpus_metadata.serialized_related_count
        ),
        "ignored_serialized_total_embedding_count": (
            corpus_metadata.serialized_total_embedding_count
        ),
        "corpus_embedding_source": CORPUS_EMBEDDING_SOURCE,
        "query_embedding_source": QUERY_EMBEDDING_SOURCE,
        "answer_embedding_source": ANSWER_EMBEDDING_SOURCE,
        "min_scores": min_scores,
        "max_hits_values": max_hits_values,
        "min_score_count": len(min_scores),
        "max_hits_count": len(max_hits_values),
        "grid_row_count": len(min_scores) * len(max_hits_values),
        "batch_size": batch_size,
    }


async def run_single_model_benchmark(
    model_spec: str,
    runs: int,
    min_scores: list[float],
    max_hits_values: list[int],
    batch_size: int,
    output_dir: Path,
) -> dict[str, object]:
    """Run the benchmark repeatedly for one model and persist raw artifacts."""

    repo_root = Path(__file__).resolve().parent.parent
    message_texts = load_message_texts(repo_root)
    related_terms = load_related_term_texts(repo_root)
    query_cases = load_search_queries(repo_root)
    pipeline_query_cases = load_pipeline_queries(repo_root)
    related_query_cases = load_related_term_queries(repo_root)
    answer_cases = load_answer_queries(repo_root)
    model_output_dir = output_dir / sanitize_model_name(model_spec)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    grid_row_count = len(min_scores) * len(max_hits_values)
    print(
        "  Loaded fixtures: "
        f"{len(message_texts)} messages, "
        f"{len(query_cases)} direct queries, "
        f"{len(pipeline_query_cases)} pipeline queries, "
        f"{len(related_query_cases)} related-term queries, "
        f"{len(answer_cases)} answer queries.",
        flush=True,
    )
    print(
        f"  Preparing {model_spec} indexes and embeddings once "
        f"for {grid_row_count} grid rows "
        f"({len(min_scores)} min_score values x "
        f"{len(max_hits_values)} max_hits values)...",
        flush=True,
    )
    model, vector_base = await build_vector_base(
        model_spec,
        message_texts,
        batch_size,
    )
    print("  Message vector index ready.", flush=True)
    related_vector_base = await build_text_vector_base(
        model,
        related_terms,
        batch_size,
    )
    print("  Related-term vector index ready.", flush=True)
    pipeline_conversation = await build_pipeline_conversation(
        repo_root,
        model,
    )
    print("  True pipeline conversation indexes ready.", flush=True)
    query_embeddings = await model.get_embeddings_nocache(
        [case.query for case in query_cases]
    )
    related_query_embeddings = await model.get_embeddings_nocache(
        [case.term for case in related_query_cases]
    )
    answer_question_embeddings = await model.get_embeddings_nocache(
        [case.question for case in answer_cases]
    )
    answer_embeddings = await model.get_embeddings_nocache(
        [case.answer for case in answer_cases]
    )
    top_score_stats = measure_top_score_stats(
        vector_base,
        query_embeddings,
    )
    print(
        "  Direct query-to-message diagnostic ready "
        f"(best-match score range {top_score_stats.min_top_score:.4f}.."
        f"{top_score_stats.max_top_score:.4f}; not used to cap the sweep).",
        flush=True,
    )

    run_results: list[RunResult] = []
    for run_index in range(1, runs + 1):
        print(
            f"  Run {run_index}/{runs}: evaluating {grid_row_count} grid rows...",
            flush=True,
        )
        benchmark_rows = await evaluate_grid(
            vector_base,
            query_cases,
            query_embeddings,
            min_scores,
            max_hits_values,
            pipeline_conversation,
            pipeline_query_cases,
            related_vector_base,
            related_terms,
            related_query_cases,
            related_query_embeddings,
            answer_cases,
            answer_question_embeddings,
            answer_embeddings,
            progress_label=f"  Run {run_index}/{runs}",
        )

        best_row = select_best_row(benchmark_rows)
        print(
            "  Run "
            f"{run_index}/{runs}: best min_score={best_row.min_score:.2f}, "
            f"max_hits={best_row.max_hits}.",
            flush=True,
        )
        run_result = RunResult(
            run_index=run_index,
            model_spec=model_spec,
            resolved_model_name=model.model_name,
            message_count=len(message_texts),
            query_count=len(query_cases),
            min_top_score=top_score_stats.min_top_score,
            mean_top_score=top_score_stats.mean_top_score,
            max_top_score=top_score_stats.max_top_score,
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
    """Run the benchmark suite for each requested model and save the artifacts."""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    metadata = build_run_suite_metadata(
        repo_root=repo_root,
        timestamp=timestamp,
        models=models,
        runs=runs,
        min_scores=min_scores,
        max_hits_values=max_hits_values,
        batch_size=batch_size,
    )
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
    """Parse the model list or fall back to the built-in OpenAI benchmark set."""

    if raw is None:
        return DEFAULT_MODELS
    models = [item.strip() for item in raw.split(",") if item.strip()]
    if not models:
        raise ValueError("--models must contain at least one model")
    return models


def main() -> None:
    """Parse CLI arguments and run repeated embedding benchmarks."""

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
            min_scores=resolve_min_scores(
                args.min_scores,
                args.min_score_start,
                args.min_score_stop,
                args.min_score_step,
            ),
            max_hits_values=parse_int_list(args.max_hits),
            batch_size=args.batch_size,
            output_root=Path(args.output_dir),
        )
    )
    print(f"Wrote benchmark results to {output_dir}")


if __name__ == "__main__":
    main()
