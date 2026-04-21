# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "benchmark_embeddings.py"
)
SPEC = spec_from_file_location("benchmark_embeddings_for_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
BENCHMARK_EMBEDDINGS = module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK_EMBEDDINGS)

BenchmarkRow = BENCHMARK_EMBEDDINGS.BenchmarkRow
SearchMetrics = BENCHMARK_EMBEDDINGS.SearchMetrics
build_float_range = BENCHMARK_EMBEDDINGS.build_float_range
filter_min_scores_by_ceiling = BENCHMARK_EMBEDDINGS.filter_min_scores_by_ceiling
load_message_texts = BENCHMARK_EMBEDDINGS.load_message_texts
parse_float_list = BENCHMARK_EMBEDDINGS.parse_float_list
resolve_min_scores = BENCHMARK_EMBEDDINGS.resolve_min_scores
select_best_row = BENCHMARK_EMBEDDINGS.select_best_row


def make_row(
    min_score: float,
    max_hits: int,
    hit_rate: float,
    mean_reciprocal_rank: float,
) -> BenchmarkRow:
    """Build a benchmark row without repeating nested metrics boilerplate."""

    return BenchmarkRow(
        min_score=min_score,
        max_hits=max_hits,
        metrics=SearchMetrics(
            hit_rate=hit_rate,
            mean_reciprocal_rank=mean_reciprocal_rank,
        ),
    )


def test_select_best_row_prefers_higher_min_score_on_metric_tie() -> None:
    rows = [
        make_row(0.25, 15, 98.5, 0.7514),
        make_row(0.70, 15, 98.5, 0.7514),
    ]

    best_row = select_best_row(rows)

    assert best_row.min_score == 0.70
    assert best_row.max_hits == 15


def test_select_best_row_prefers_lower_max_hits_on_full_tie() -> None:
    rows = [
        make_row(0.70, 20, 98.5, 0.7514),
        make_row(0.70, 15, 98.5, 0.7514),
    ]

    best_row = select_best_row(rows)

    assert best_row.min_score == 0.70
    assert best_row.max_hits == 15


def test_parse_float_list_defaults_to_tenth_point_grid() -> None:
    assert parse_float_list(None) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_build_float_range_supports_hundredth_point_sweeps() -> None:
    assert build_float_range(0.01, 0.05, 0.01) == [0.01, 0.02, 0.03, 0.04, 0.05]


def test_resolve_min_scores_uses_generated_range() -> None:
    assert resolve_min_scores(None, 0.01, 0.03, 0.01) == [0.01, 0.02, 0.03]


def test_resolve_min_scores_rejects_mixed_inputs() -> None:
    with pytest.raises(ValueError, match="Use either --min-scores"):
        resolve_min_scores("0.1,0.2", 0.01, 0.03, 0.01)


def test_filter_min_scores_by_ceiling_skips_guaranteed_zero_rows() -> None:
    effective_scores, skipped_scores = filter_min_scores_by_ceiling(
        [0.01, 0.16, 0.17, 0.5],
        0.16,
    )

    assert effective_scores == [0.01, 0.16]
    assert skipped_scores == [0.17, 0.5]


def test_load_message_texts_returns_one_text_blob_per_message() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    message_texts = load_message_texts(repo_root)

    assert message_texts
    assert all(isinstance(text, str) for text in message_texts)
