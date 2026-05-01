# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pytest

from typeagent.aitools.embeddings import NormalizedEmbedding, NormalizedEmbeddings

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "benchmark_embeddings.py"
)
SPEC = spec_from_file_location("benchmark_embeddings_for_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
BENCHMARK_EMBEDDINGS = module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK_EMBEDDINGS)

BenchmarkRow = BENCHMARK_EMBEDDINGS.BenchmarkRow
AnswerMetrics = BENCHMARK_EMBEDDINGS.AnswerMetrics
PipelineMetrics = BENCHMARK_EMBEDDINGS.PipelineMetrics
RelatedTermMetrics = BENCHMARK_EMBEDDINGS.RelatedTermMetrics
RelatedTermQueryCase = BENCHMARK_EMBEDDINGS.RelatedTermQueryCase
SearchMetrics = BENCHMARK_EMBEDDINGS.SearchMetrics
build_float_range = BENCHMARK_EMBEDDINGS.build_float_range
create_benchmark_conversation_settings = (
    BENCHMARK_EMBEDDINGS.create_benchmark_conversation_settings
)
evaluate_answer_queries = BENCHMARK_EMBEDDINGS.evaluate_answer_queries
evaluate_related_term_queries = BENCHMARK_EMBEDDINGS.evaluate_related_term_queries
load_corpus_metadata = BENCHMARK_EMBEDDINGS.load_corpus_metadata
load_pipeline_queries = BENCHMARK_EMBEDDINGS.load_pipeline_queries
load_message_texts = BENCHMARK_EMBEDDINGS.load_message_texts
load_related_term_queries = BENCHMARK_EMBEDDINGS.load_related_term_queries
load_related_term_texts = BENCHMARK_EMBEDDINGS.load_related_term_texts
parse_float_list = BENCHMARK_EMBEDDINGS.parse_float_list
resolve_min_scores = BENCHMARK_EMBEDDINGS.resolve_min_scores
select_best_row = BENCHMARK_EMBEDDINGS.select_best_row


class FakeEmbeddingModel:
    """Minimal embedding model stub for settings tests."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        del key, embedding

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        del input
        return np.array([1.0], dtype=np.float32)

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        del input
        return np.array([[1.0]], dtype=np.float32)

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        del key
        return np.array([1.0], dtype=np.float32)

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        del keys
        return np.array([[1.0]], dtype=np.float32)


def make_row(
    min_score: float,
    max_hits: int,
    hit_rate: float,
    mean_reciprocal_rank: float,
    semantic_score: float | None = None,
    pipeline_hit_rate: float | None = None,
    pipeline_mean_reciprocal_rank: float | None = None,
    related_hit_rate: float | None = None,
    related_mean_reciprocal_rank: float | None = None,
) -> BenchmarkRow:
    """Build a benchmark row without repeating nested metrics boilerplate."""

    answer_metrics = (
        AnswerMetrics(
            answerable_support=semantic_score / 100,
            no_answer_rejection_rate=0.0,
            semantic_score=semantic_score,
        )
        if semantic_score is not None
        else None
    )
    related_metrics = (
        RelatedTermMetrics(
            hit_rate=related_hit_rate,
            mean_reciprocal_rank=related_mean_reciprocal_rank,
            mean_result_count=10.0,
        )
        if related_hit_rate is not None and related_mean_reciprocal_rank is not None
        else None
    )
    pipeline_metrics = (
        PipelineMetrics(
            hit_rate=pipeline_hit_rate,
            mean_reciprocal_rank=pipeline_mean_reciprocal_rank,
            mean_result_count=10.0,
        )
        if pipeline_hit_rate is not None and pipeline_mean_reciprocal_rank is not None
        else None
    )
    return BenchmarkRow(
        min_score=min_score,
        max_hits=max_hits,
        metrics=SearchMetrics(
            hit_rate=hit_rate,
            mean_reciprocal_rank=mean_reciprocal_rank,
        ),
        pipeline_metrics=pipeline_metrics,
        related_metrics=related_metrics,
        answer_metrics=answer_metrics,
    )


@pytest.mark.parametrize(
    ("model_name", "expected_min_score"),
    [
        ("text-embedding-3-large", 0.74),
        ("text-embedding-3-small", 0.73),
        ("text-embedding-ada-002", 0.93),
    ],
)
def test_benchmark_conversation_settings_use_model_default(
    model_name: str,
    expected_min_score: float,
) -> None:
    settings = create_benchmark_conversation_settings(FakeEmbeddingModel(model_name))

    assert (
        settings.related_term_index_settings.embedding_index_settings.min_score
        == expected_min_score
    )
    assert settings.thread_settings.min_score == expected_min_score
    assert (
        settings.message_text_index_settings.embedding_index_settings.min_score
        == expected_min_score
    )
    assert (
        settings.related_term_index_settings.embedding_index_settings.max_matches == 50
    )


def test_select_best_row_prefers_higher_min_score_on_metric_tie() -> None:
    rows = [
        make_row(0.25, 15, 98.5, 0.7514, semantic_score=90.0),
        make_row(0.70, 15, 98.5, 0.7514, semantic_score=80.0),
    ]

    best_row = select_best_row(rows)

    assert best_row.min_score == 0.70
    assert best_row.max_hits == 15


def test_select_best_row_only_uses_answer_context_after_min_score_tie() -> None:
    rows = [
        make_row(0.70, 15, 98.5, 0.7514, semantic_score=80.0),
        make_row(0.70, 15, 98.5, 0.7514, semantic_score=90.0),
    ]

    best_row = select_best_row(rows)

    assert best_row.answer_metrics is not None
    assert best_row.answer_metrics.semantic_score == 90.0


def test_select_best_row_prefers_related_term_quality_before_message_quality() -> None:
    rows = [
        make_row(
            0.80,
            15,
            98.5,
            0.90,
            related_hit_rate=90.0,
            related_mean_reciprocal_rank=0.70,
        ),
        make_row(
            0.70,
            15,
            98.5,
            0.80,
            related_hit_rate=95.0,
            related_mean_reciprocal_rank=0.75,
        ),
    ]

    best_row = select_best_row(rows)

    assert best_row.min_score == 0.70


def test_select_best_row_prefers_pipeline_quality_before_related_term_quality() -> None:
    rows = [
        make_row(
            0.80,
            15,
            98.5,
            0.90,
            pipeline_hit_rate=90.0,
            pipeline_mean_reciprocal_rank=0.70,
            related_hit_rate=99.0,
            related_mean_reciprocal_rank=0.99,
        ),
        make_row(
            0.70,
            15,
            98.5,
            0.80,
            pipeline_hit_rate=95.0,
            pipeline_mean_reciprocal_rank=0.75,
            related_hit_rate=90.0,
            related_mean_reciprocal_rank=0.70,
        ),
    ]

    best_row = select_best_row(rows)

    assert best_row.min_score == 0.70


def test_evaluate_related_term_queries_scores_expected_terms() -> None:
    vector_base = BENCHMARK_EMBEDDINGS.VectorBase(
        BENCHMARK_EMBEDDINGS.TextEmbeddingIndexSettings(
            BENCHMARK_EMBEDDINGS.create_embedding_model("test")
        )
    )
    vector_base.add_embedding(None, np.array([1.0, 0.0], dtype=np.float32))
    vector_base.add_embedding(None, np.array([0.0, 1.0], dtype=np.float32))

    metrics = evaluate_related_term_queries(
        vector_base,
        ["alpha", "beta"],
        [RelatedTermQueryCase("query", ["beta"])],
        np.array([[0.0, 1.0]], dtype=np.float32),
        min_score=0.0,
        max_hits=2,
    )

    assert metrics.hit_rate == 100.0
    assert metrics.mean_reciprocal_rank == 1.0
    assert metrics.mean_result_count == 2.0


def test_evaluate_answer_queries_reports_normalized_support_score() -> None:
    vector_base = BENCHMARK_EMBEDDINGS.VectorBase(
        BENCHMARK_EMBEDDINGS.TextEmbeddingIndexSettings(
            BENCHMARK_EMBEDDINGS.create_embedding_model("test")
        )
    )
    vector_base.add_embedding(None, np.array([1.0, 0.0], dtype=np.float32))
    answer_cases = [
        BENCHMARK_EMBEDDINGS.AnswerQueryCase(
            question="question",
            answer="answer",
            has_no_answer=False,
        )
    ]

    metrics = evaluate_answer_queries(
        vector_base,
        answer_cases,
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0.0, 1.0]], dtype=np.float32),
        min_score=0.0,
        max_hits=1,
    )

    assert metrics.answerable_support == 0.5
    assert metrics.semantic_score == 75.0


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


def test_load_message_texts_returns_one_text_blob_per_message() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    message_texts = load_message_texts(repo_root)

    assert message_texts
    assert all(isinstance(text, str) for text in message_texts)


def test_load_related_term_texts_returns_fixture_terms() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    terms = load_related_term_texts(repo_root)

    assert len(terms) == 1188
    assert "adrian tchaikovsky" in terms


def test_load_related_term_queries_returns_compiled_related_terms() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    cases = load_related_term_queries(repo_root)

    assert cases
    assert all(case.expected_related_terms for case in cases)


def test_load_pipeline_queries_strips_cached_related_terms() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    cases = load_pipeline_queries(repo_root)

    assert cases
    for case in cases:
        for obj in BENCHMARK_EMBEDDINGS.iter_dicts(
            case.query_exprs[0].__pydantic_serializer__.to_python(
                case.query_exprs[0],
                by_alias=True,
            )
        ):
            assert obj.get("relatedTerms") in (None, [])


def test_load_message_texts_ignores_serialized_embedding_sidecar(
    tmp_path: Path,
) -> None:
    testdata_dir = tmp_path / "tests" / "testdata"
    testdata_dir.mkdir(parents=True)
    (testdata_dir / "Episode_53_AdrianTchaikovsky_index_data.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"textChunks": ["hello", "world"]},
                    {"textChunks": ["goodbye"]},
                ],
                "embeddingFileHeader": {
                    "messageCount": 2,
                    "relatedCount": 0,
                    "modelMetadata": {"embeddingSize": 1536},
                },
            }
        ),
        encoding="utf-8",
    )
    (testdata_dir / "Episode_53_AdrianTchaikovsky_index_embeddings.bin").write_bytes(
        b"not real embeddings"
    )

    message_texts = load_message_texts(tmp_path)

    assert message_texts == ["hello world", "goodbye"]


def test_load_corpus_metadata_reports_serialized_sidecar_details() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    metadata = load_corpus_metadata(repo_root)

    assert metadata.message_count > 0
    assert metadata.serialized_embedding_size == 1536
    assert metadata.serialized_message_count == 106
    assert metadata.serialized_related_count == 1188
    assert metadata.serialized_total_embedding_count == 1294


def test_load_corpus_metadata_rejects_inconsistent_sidecar_size(
    tmp_path: Path,
) -> None:
    testdata_dir = tmp_path / "tests" / "testdata"
    testdata_dir.mkdir(parents=True)
    (testdata_dir / "Episode_53_AdrianTchaikovsky_index_data.json").write_text(
        json.dumps(
            {
                "messages": [{"textChunks": ["hello"]}],
                "embeddingFileHeader": {
                    "messageCount": 1,
                    "relatedCount": 1,
                    "modelMetadata": {"embeddingSize": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    (testdata_dir / "Episode_53_AdrianTchaikovsky_index_embeddings.bin").write_bytes(
        b"bad-sidecar"
    )

    with pytest.raises(ValueError, match="Serialized benchmark sidecar size"):
        load_corpus_metadata(tmp_path)
