# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "tools" / "repeat_embedding_benchmarks.py"
)
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = spec_from_file_location("repeat_embedding_benchmarks_for_test", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
REPEAT_BENCHMARKS = module_from_spec(SPEC)
SPEC.loader.exec_module(REPEAT_BENCHMARKS)

build_run_suite_metadata = REPEAT_BENCHMARKS.build_run_suite_metadata


def test_build_run_suite_metadata_records_ignored_sidecar() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    metadata = build_run_suite_metadata(
        repo_root=repo_root,
        timestamp="20260424T000000Z",
        models=["openai:text-embedding-3-small"],
        runs=3,
        min_scores=[0.01, 0.02],
        max_hits_values=[5, 10],
        batch_size=16,
    )

    assert metadata["ignored_serialized_embedding_size"] == 1536
    assert metadata["ignored_serialized_message_embedding_count"] == 106
    assert metadata["ignored_serialized_related_embedding_count"] == 1188
    assert metadata["ignored_serialized_total_embedding_count"] == 1294
    assert metadata["pipeline_source_json"] == (
        "tests\\testdata\\Episode_53_Search_results.json"
    )
    assert metadata["related_term_source_json"] == (
        "tests\\testdata\\Episode_53_Search_results.json"
    )
    assert metadata["pipeline_scoring_paths"] == [
        "src/typeagent/knowpro/search.py::run_search_query",
        "src/typeagent/knowpro/query.py::MatchSearchTermExpr.accumulate_matches_for_term",
        "src/typeagent/knowpro/collections.py::SemanticRefAccumulator.add_term_matches",
        "src/typeagent/knowpro/collections.py::add_smooth_related_score_to_match_score",
        "src/typeagent/knowpro/query.py::message_matches_from_knowledge_matches",
        "src/typeagent/knowpro/collections.py::MessageAccumulator.smooth_scores",
    ]
    assert metadata["corpus_embedding_source"] == (
        "recomputed_per_model_from_message_text"
    )
    assert metadata["query_embedding_source"] == (
        "recomputed_per_model_from_search_text"
    )
    assert metadata["min_score_count"] == 2
    assert metadata["max_hits_count"] == 2
    assert metadata["grid_row_count"] == 4
