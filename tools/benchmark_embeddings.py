# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark retrieval settings for known embedding models.

This script evaluates the Adrian Tchaikovsky Episode 53 search dataset in
`tests/testdata/` and reports retrieval quality for combinations of
`min_score` and `max_hits`.

Methodology:
- Load only message text from the benchmark `_data.json` payload.
- Treat the serialized `_embeddings.bin` sidecar as metadata only.
- Recompute corpus and query embeddings with the requested model.

The benchmark is intentionally narrow:
- It only measures retrieval against `messageMatches` ground truth.
- It is meant to help choose repository defaults for known models.
- In practice, `min_score` is the primary library default this informs.
- It does not prove universal "best" settings for every dataset.
It also includes a semantic answer-context signal from the answer fixture:
- Answerable questions should retrieve messages close to the expected answer.
- No-answer questions should avoid high-confidence retrieved context.

Usage:
    uv run python tools/benchmark_embeddings.py
    uv run python tools/benchmark_embeddings.py --model openai:text-embedding-3-small
    uv run python tools/benchmark_embeddings.py --model openai:text-embedding-3-small --min-score-start 0.01 --min-score-stop 0.20 --min-score-step 0.01
"""

import argparse
import asyncio
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
import numpy as np

from typeagent.aitools.embeddings import (
    IEmbeddingModel,
    NormalizedEmbeddings,
)
from typeagent.aitools.model_adapters import create_embedding_model
from typeagent.aitools.vectorbase import (
    TextEmbeddingIndexSettings,
    VectorBase,
)
from typeagent.knowpro import search, secindex, serialization
from typeagent.knowpro.convsettings import (
    ConversationSettings,
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.podcasts import podcast

DEFAULT_MIN_SCORES = [score / 10 for score in range(1, 10)]
DEFAULT_MAX_HITS = [5, 10, 15, 20]
DATA_DIR = Path("tests") / "testdata"
INDEX_PREFIX_PATH = DATA_DIR / "Episode_53_AdrianTchaikovsky_index"
INDEX_DATA_PATH = DATA_DIR / "Episode_53_AdrianTchaikovsky_index_data.json"
INDEX_EMBEDDINGS_PATH = DATA_DIR / "Episode_53_AdrianTchaikovsky_index_embeddings.bin"
SEARCH_RESULTS_PATH = DATA_DIR / "Episode_53_Search_results.json"
ANSWER_RESULTS_PATH = DATA_DIR / "Episode_53_Answer_results.json"
CORPUS_EMBEDDING_SOURCE = "recomputed_per_model_from_message_text"
QUERY_EMBEDDING_SOURCE = "recomputed_per_model_from_search_text"
ANSWER_EMBEDDING_SOURCE = "recomputed_per_model_from_expected_answer_text"
PIPELINE_SCORING_PATHS = [
    "src/typeagent/knowpro/search.py::run_search_query",
    "src/typeagent/knowpro/query.py::MatchSearchTermExpr.accumulate_matches_for_term",
    "src/typeagent/knowpro/collections.py::SemanticRefAccumulator.add_term_matches",
    "src/typeagent/knowpro/collections.py::add_smooth_related_score_to_match_score",
    "src/typeagent/knowpro/query.py::message_matches_from_knowledge_matches",
    "src/typeagent/knowpro/collections.py::MessageAccumulator.smooth_scores",
]


def score_from_cosine(cosine_similarity: np.ndarray) -> np.ndarray:
    """Map cosine similarity from -1..1 to the public 0..1 score scale."""

    return np.clip((cosine_similarity + 1.0) / 2.0, 0.0, 1.0)


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
class PipelineQueryCase:
    """A compiled query fixture with message-level ground truth."""

    query: str
    query_exprs: list[search.SearchQueryExpr]
    expected_matches: list[int]


@dataclass
class PipelineMetrics:
    """Aggregate metrics from the real query scoring pipeline."""

    hit_rate: float
    mean_reciprocal_rank: float
    mean_result_count: float


@dataclass
class RelatedTermQueryCase:
    """A search term paired with related terms from the compiled query fixture."""

    term: str
    expected_related_terms: list[str]


@dataclass
class RelatedTermMetrics:
    """Aggregate fuzzy related-term retrieval metrics for one benchmark row."""

    hit_rate: float
    mean_reciprocal_rank: float
    mean_result_count: float


@dataclass
class AnswerQueryCase:
    """A benchmark answer case paired with its expected answerability."""

    question: str
    answer: str
    has_no_answer: bool


@dataclass
class AnswerMetrics:
    """Aggregate semantic answer-context metrics for one benchmark row."""

    answerable_support: float
    no_answer_rejection_rate: float
    semantic_score: float


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
    pipeline_metrics: PipelineMetrics | None = None
    related_metrics: RelatedTermMetrics | None = None
    answer_metrics: AnswerMetrics | None = None


@dataclass
class CorpusMetadata:
    """Metadata about the serialized benchmark corpus fixture."""

    message_count: int
    serialized_embedding_size: int | None
    serialized_message_count: int | None
    serialized_related_count: int | None
    serialized_total_embedding_count: int | None


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
    """Load the benchmark corpus as one text blob per message.

    The JSON fixture also points at a serialized embedding sidecar, but that
    sidecar is deliberately ignored here. Cross-model comparisons are only
    meaningful when every evaluated model embeds the same raw message text.
    """

    index_data = json.loads((repo_root / INDEX_DATA_PATH).read_text(encoding="utf-8"))
    messages = index_data["messages"]
    return [" ".join(message.get("textChunks", [])) for message in messages]


def load_related_term_texts(repo_root: Path) -> list[str]:
    """Load the term corpus used by fuzzy related-term lookup."""

    index_data = json.loads((repo_root / INDEX_DATA_PATH).read_text(encoding="utf-8"))
    related_terms_index_data = index_data.get("relatedTermsIndexData") or {}
    text_embedding_data = related_terms_index_data.get("textEmbeddingData") or {}
    text_items = text_embedding_data.get("textItems")
    if isinstance(text_items, list) and text_items:
        return [text for text in text_items if isinstance(text, str)]

    semantic_index_data = index_data.get("semanticIndexData") or {}
    items = semantic_index_data.get("items") or []
    return [item["term"] for item in items if isinstance(item.get("term"), str)]


def load_corpus_metadata(repo_root: Path) -> CorpusMetadata:
    """Load sidecar metadata without loading the sidecar embeddings."""

    index_data = json.loads((repo_root / INDEX_DATA_PATH).read_text(encoding="utf-8"))
    embedding_file_header = index_data.get("embeddingFileHeader") or {}
    model_metadata = embedding_file_header.get("modelMetadata") or {}
    serialized_embedding_size = model_metadata.get("embeddingSize")
    serialized_message_count = embedding_file_header.get("messageCount")
    serialized_related_count = embedding_file_header.get("relatedCount")
    serialized_total_embedding_count: int | None = None

    sidecar_path = repo_root / INDEX_EMBEDDINGS_PATH
    if serialized_embedding_size is not None and sidecar_path.exists():
        bytes_per_embedding = serialized_embedding_size * np.dtype(np.float32).itemsize
        if bytes_per_embedding <= 0:
            raise ValueError(
                "Serialized benchmark corpus has a non-positive embedding size"
            )
        sidecar_size_bytes = sidecar_path.stat().st_size
        if sidecar_size_bytes % bytes_per_embedding != 0:
            raise ValueError(
                "Serialized benchmark sidecar size is not divisible by the declared "
                f"embedding width of {serialized_embedding_size}"
            )
        serialized_total_embedding_count = sidecar_size_bytes // bytes_per_embedding
        declared_total_count = (serialized_message_count or 0) + (
            serialized_related_count or 0
        )
        if (
            declared_total_count
            and declared_total_count != serialized_total_embedding_count
        ):
            raise ValueError(
                "Serialized benchmark sidecar row count does not match the counts "
                "declared in the JSON metadata"
            )

    return CorpusMetadata(
        message_count=len(index_data.get("messages", [])),
        serialized_embedding_size=serialized_embedding_size,
        serialized_message_count=serialized_message_count,
        serialized_related_count=serialized_related_count,
        serialized_total_embedding_count=serialized_total_embedding_count,
    )


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


def strip_related_terms(value: object) -> None:
    """Remove cached related-term expansions so each model resolves its own."""

    for obj in iter_dicts(value):
        related_terms = obj.get("relatedTerms")
        if isinstance(related_terms, list) and related_terms:
            obj["relatedTerms"] = None


def load_pipeline_queries(repo_root: Path) -> list[PipelineQueryCase]:
    """Load compiled query fixtures for the real semantic scoring pipeline."""

    search_data = json.loads(
        (repo_root / SEARCH_RESULTS_PATH).read_text(encoding="utf-8")
    )
    cases: list[PipelineQueryCase] = []
    for item in search_data:
        search_text = item.get("searchText")
        compiled_query_expr = item.get("compiledQueryExpr")
        results = item.get("results", [])
        if not (
            isinstance(search_text, str)
            and isinstance(compiled_query_expr, list)
            and results
        ):
            continue
        expected_matches = results[0].get("messageMatches", [])
        if not expected_matches:
            continue
        strip_related_terms(compiled_query_expr)
        query_exprs = serialization.deserialize_object(
            list[search.SearchQueryExpr],
            compiled_query_expr,
        )
        cases.append(
            PipelineQueryCase(
                query=search_text,
                query_exprs=query_exprs,
                expected_matches=expected_matches,
            )
        )
    return cases


def iter_dicts(value: object):
    """Yield dictionaries recursively from a decoded JSON value."""

    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def load_related_term_queries(repo_root: Path) -> list[RelatedTermQueryCase]:
    """Load expected fuzzy related-term outputs from compiled query fixtures.

    These compiled fixtures are closer to the real query pipeline than raw
    query-to-message similarity: `min_score` normally gates fuzzy related-term
    expansion before semantic-ref and message scores are accumulated.
    """

    search_data = json.loads(
        (repo_root / SEARCH_RESULTS_PATH).read_text(encoding="utf-8")
    )
    cases: list[RelatedTermQueryCase] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for item in search_data:
        for obj in iter_dicts(item.get("compiledQueryExpr", [])):
            term = obj.get("term")
            related_terms = obj.get("relatedTerms")
            if not (
                isinstance(term, dict)
                and isinstance(term.get("text"), str)
                and isinstance(related_terms, list)
                and related_terms
            ):
                continue
            expected: list[str] = []
            for related in related_terms:
                if isinstance(related, dict):
                    related_text = related.get("text")
                    if isinstance(related_text, str):
                        expected.append(related_text)
            if not expected:
                continue
            key = (term["text"], tuple(expected))
            if key not in seen:
                seen.add(key)
                cases.append(RelatedTermQueryCase(term["text"], expected))
    return cases


def load_answer_queries(repo_root: Path) -> list[AnswerQueryCase]:
    """Load expected answers for semantic answer-context benchmarking."""

    answer_data = json.loads(
        (repo_root / ANSWER_RESULTS_PATH).read_text(encoding="utf-8")
    )
    cases: list[AnswerQueryCase] = []
    for item in answer_data:
        question = item.get("question")
        answer = item.get("answer")
        has_no_answer = item.get("hasNoAnswer")
        if (
            isinstance(question, str)
            and isinstance(answer, str)
            and isinstance(has_no_answer, bool)
        ):
            cases.append(AnswerQueryCase(question, answer, has_no_answer))
    return cases


async def build_vector_base(
    model_spec: str | None,
    message_texts: list[str],
    batch_size: int,
) -> tuple[IEmbeddingModel, VectorBase]:
    """Build a message-level vector index for the benchmark corpus.

    This computes fresh embeddings for `message_texts` with the requested
    model. It does not deserialize or consult the fixture's `_embeddings.bin`
    sidecar, which may have been generated by a different embedding model.
    """

    model = create_embedding_model(model_spec)
    vector_base = await build_text_vector_base(model, message_texts, batch_size)
    return model, vector_base


async def build_pipeline_conversation(
    repo_root: Path,
    model: IEmbeddingModel,
) -> podcast.Podcast:
    """Build the benchmark conversation with per-model secondary indexes.

    The fixture's serialized embedding sidecar is deliberately not used here.
    We keep the semantic refs and exact semantic index, then rebuild related-term
    and message-text indexes with the requested model so `min_score` gates the
    same fuzzy expansion path the runtime uses.
    """

    settings = create_benchmark_conversation_settings(model)
    data = podcast.Podcast._read_conversation_data_from_file(
        str(repo_root / INDEX_PREFIX_PATH)
    )
    data.pop("relatedTermsIndexData", None)
    data.pop("messageIndexData", None)
    conversation = await podcast.Podcast.create(settings)
    await conversation.deserialize(data)
    await secindex.build_secondary_indexes(conversation, settings)
    return conversation


def create_benchmark_conversation_settings(
    model: IEmbeddingModel,
) -> ConversationSettings:
    """Use benchmarked model defaults without changing normal app settings."""

    settings = ConversationSettings(model=model)
    benchmark_min_score = TextEmbeddingIndexSettings(model).min_score
    settings.related_term_index_settings = RelatedTermIndexSettings(
        TextEmbeddingIndexSettings(
            model,
            min_score=benchmark_min_score,
            max_matches=50,
        )
    )
    settings.thread_settings = TextEmbeddingIndexSettings(
        model,
        min_score=benchmark_min_score,
    )
    settings.message_text_index_settings = MessageTextIndexSettings(
        TextEmbeddingIndexSettings(
            model,
            min_score=benchmark_min_score,
        )
    )
    return settings


async def build_text_vector_base(
    model: IEmbeddingModel,
    texts: list[str],
    batch_size: int,
) -> VectorBase:
    """Build a vector index for already selected benchmark text items."""

    settings = TextEmbeddingIndexSettings(
        embedding_model=model,
        min_score=0.0,
        max_matches=None,
        batch_size=batch_size,
    )
    vector_base = VectorBase(settings)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings = await model.get_embeddings_nocache(batch)
        vector_base.add_embeddings(None, embeddings)
    return vector_base


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


async def evaluate_pipeline_queries(
    conversation: podcast.Podcast,
    query_cases: list[PipelineQueryCase],
    min_score: float,
    max_hits: int,
) -> PipelineMetrics:
    """Evaluate compiled queries through the runtime semantic scoring path."""

    related_settings = (
        conversation.settings.related_term_index_settings.embedding_index_settings
    )
    related_settings.min_score = min_score
    hit_count = 0
    reciprocal_ranks: list[float] = []
    result_counts: list[int] = []
    options = search.SearchOptions(max_message_matches=max_hits)

    for case in query_cases:
        query_exprs = deepcopy(case.query_exprs)
        scored_results = []
        for query_expr in query_exprs:
            search_results = await search.run_search_query(
                conversation,
                query_expr,
                options,
            )
            for result in search_results:
                scored_results.extend(result.message_matches)
        scored_results.sort(key=lambda result: result.score, reverse=True)
        result_counts.append(len(scored_results))
        expected_matches = set(case.expected_matches)
        rank = 0
        for result_index, scored_result in enumerate(scored_results, start=1):
            if scored_result.message_ordinal in expected_matches:
                rank = result_index
                break
        if rank > 0:
            hit_count += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return PipelineMetrics(
        hit_rate=(hit_count / len(query_cases)) * 100,
        mean_reciprocal_rank=mean(reciprocal_ranks),
        mean_result_count=mean(result_counts),
    )


def evaluate_related_term_queries(
    vector_base: VectorBase,
    related_terms: list[str],
    query_cases: list[RelatedTermQueryCase],
    query_embeddings: NormalizedEmbeddings,
    min_score: float,
    max_hits: int,
) -> RelatedTermMetrics:
    """Evaluate fuzzy related-term retrieval against compiled query fixtures."""

    hit_count = 0
    reciprocal_ranks: list[float] = []
    result_counts: list[int] = []

    for case, query_embedding in zip(query_cases, query_embeddings):
        expected_terms = set(case.expected_related_terms)
        scored_results = vector_base.fuzzy_lookup_embedding(
            query_embedding,
            max_hits=max_hits,
            min_score=min_score,
        )
        result_counts.append(len(scored_results))
        rank = 0
        for result_index, scored_result in enumerate(scored_results, start=1):
            if related_terms[scored_result.item] in expected_terms:
                rank = result_index
                break
        if rank > 0:
            hit_count += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return RelatedTermMetrics(
        hit_rate=(hit_count / len(query_cases)) * 100,
        mean_reciprocal_rank=mean(reciprocal_ranks),
        mean_result_count=mean(result_counts),
    )


def evaluate_answer_queries(
    vector_base: VectorBase,
    answer_cases: list[AnswerQueryCase],
    question_embeddings: NormalizedEmbeddings,
    answer_embeddings: NormalizedEmbeddings,
    min_score: float,
    max_hits: int,
) -> AnswerMetrics:
    """Evaluate whether retrieved message context semantically supports answers."""

    answerable_support_scores: list[float] = []
    no_answer_rejections = 0
    no_answer_count = 0
    corpus_embeddings = vector_base.serialize()

    for case, question_embedding, answer_embedding in zip(
        answer_cases,
        question_embeddings,
        answer_embeddings,
        strict=True,
    ):
        scored_results = vector_base.fuzzy_lookup_embedding(
            question_embedding,
            max_hits=max_hits,
            min_score=min_score,
        )
        if case.has_no_answer:
            no_answer_count += 1
            if not scored_results:
                no_answer_rejections += 1
            continue

        if not scored_results:
            answerable_support_scores.append(0.0)
            continue

        retrieved_embeddings = corpus_embeddings[
            [scored_result.item for scored_result in scored_results]
        ]
        scores = score_from_cosine(np.dot(retrieved_embeddings, answer_embedding))
        answerable_support_scores.append(float(np.max(scores)))

    answerable_support = (
        mean(answerable_support_scores) if answerable_support_scores else 0.0
    )
    no_answer_rejection_rate = (
        (no_answer_rejections / no_answer_count) * 100 if no_answer_count else 100.0
    )
    semantic_score = (answerable_support * 100 + no_answer_rejection_rate) / 2
    return AnswerMetrics(
        answerable_support=answerable_support,
        no_answer_rejection_rate=no_answer_rejection_rate,
        semantic_score=semantic_score,
    )


async def evaluate_grid(
    vector_base: VectorBase,
    query_cases: list[SearchQueryCase],
    query_embeddings: NormalizedEmbeddings,
    min_scores: list[float],
    max_hits_values: list[int],
    pipeline_conversation: podcast.Podcast | None = None,
    pipeline_query_cases: list[PipelineQueryCase] | None = None,
    related_vector_base: VectorBase | None = None,
    related_terms: list[str] | None = None,
    related_query_cases: list[RelatedTermQueryCase] | None = None,
    related_query_embeddings: NormalizedEmbeddings | None = None,
    answer_cases: list[AnswerQueryCase] | None = None,
    answer_question_embeddings: NormalizedEmbeddings | None = None,
    answer_embeddings: NormalizedEmbeddings | None = None,
    progress_label: str | None = None,
) -> list[BenchmarkRow]:
    """Evaluate every `(min_score, max_hits)` row in the requested grid."""

    rows: list[BenchmarkRow] = []
    for min_score_index, min_score in enumerate(min_scores, start=1):
        if progress_label and (
            min_score_index == 1
            or min_score_index == len(min_scores)
            or min_score_index % 10 == 0
        ):
            print(
                f"{progress_label}: min_score {min_score:.2f} "
                f"({min_score_index}/{len(min_scores)})...",
                flush=True,
            )
        for max_hits in max_hits_values:
            metrics = evaluate_search_queries(
                vector_base,
                query_cases,
                query_embeddings,
                min_score,
                max_hits,
            )
            pipeline_metrics = None
            if pipeline_conversation is not None and pipeline_query_cases is not None:
                pipeline_metrics = await evaluate_pipeline_queries(
                    pipeline_conversation,
                    pipeline_query_cases,
                    min_score,
                    max_hits,
                )
            related_metrics = None
            if (
                related_vector_base is not None
                and related_terms is not None
                and related_query_cases is not None
                and related_query_embeddings is not None
            ):
                related_metrics = evaluate_related_term_queries(
                    related_vector_base,
                    related_terms,
                    related_query_cases,
                    related_query_embeddings,
                    min_score,
                    max_hits,
                )
            answer_metrics = None
            if (
                answer_cases is not None
                and answer_question_embeddings is not None
                and answer_embeddings is not None
            ):
                answer_metrics = evaluate_answer_queries(
                    vector_base,
                    answer_cases,
                    answer_question_embeddings,
                    answer_embeddings,
                    min_score,
                    max_hits,
                )
            rows.append(
                BenchmarkRow(
                    min_score,
                    max_hits,
                    metrics,
                    pipeline_metrics,
                    related_metrics,
                    answer_metrics,
                )
            )
    return rows


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
    """Keep the requested min-score grid intact."""

    _ = max_top_score
    return list(min_scores), []


def select_best_row(rows: list[BenchmarkRow]) -> BenchmarkRow:
    """Prefer true pipeline quality, then related-term quality and strictness."""

    return max(
        rows,
        key=lambda row: (
            row.pipeline_metrics.mean_reciprocal_rank if row.pipeline_metrics else 0.0,
            row.pipeline_metrics.hit_rate if row.pipeline_metrics else 0.0,
            row.related_metrics.mean_reciprocal_rank if row.related_metrics else 0.0,
            row.related_metrics.hit_rate if row.related_metrics else 0.0,
            row.metrics.mean_reciprocal_rank,
            row.metrics.hit_rate,
            row.min_score,
            row.answer_metrics.semantic_score if row.answer_metrics else 0.0,
            -row.max_hits,
        ),
    )


def print_rows(rows: list[BenchmarkRow]) -> None:
    """Print the benchmark grid in a reviewer-friendly table."""

    print("=" * 72)
    print("PIPELINE + SEARCH + ANSWER-CONTEXT BENCHMARK (Episode 53 fixtures)")
    print("=" * 72)
    print(
        f"{'Min Score':<12} | {'Max Hits':<10} | {'Hit Rate (%)':<15} | "
        f"{'MRR':<10} | {'Pipe Hit':<10} | {'Pipe MRR':<10} | "
        f"{'Pipe Cnt':<10} | {'Rel Hit':<10} | {'Rel MRR':<10} | "
        f"{'Rel Cnt':<10} | {'Ans Sup':<10} | {'NoAns (%)':<10} | {'Sem':<10}"
    )
    print("-" * 174)
    for row in rows:
        pipeline_hit_rate = (
            f"{row.pipeline_metrics.hit_rate:<10.2f}"
            if row.pipeline_metrics
            else f"{'n/a':<10}"
        )
        pipeline_mrr = (
            f"{row.pipeline_metrics.mean_reciprocal_rank:<10.4f}"
            if row.pipeline_metrics
            else f"{'n/a':<10}"
        )
        pipeline_count = (
            f"{row.pipeline_metrics.mean_result_count:<10.2f}"
            if row.pipeline_metrics
            else f"{'n/a':<10}"
        )
        related_hit_rate = (
            f"{row.related_metrics.hit_rate:<10.2f}"
            if row.related_metrics
            else f"{'n/a':<10}"
        )
        related_mrr = (
            f"{row.related_metrics.mean_reciprocal_rank:<10.4f}"
            if row.related_metrics
            else f"{'n/a':<10}"
        )
        related_count = (
            f"{row.related_metrics.mean_result_count:<10.2f}"
            if row.related_metrics
            else f"{'n/a':<10}"
        )
        answer_support = (
            f"{row.answer_metrics.answerable_support:<10.4f}"
            if row.answer_metrics
            else f"{'n/a':<10}"
        )
        no_answer = (
            f"{row.answer_metrics.no_answer_rejection_rate:<10.2f}"
            if row.answer_metrics
            else f"{'n/a':<10}"
        )
        semantic_score = (
            f"{row.answer_metrics.semantic_score:<10.2f}"
            if row.answer_metrics
            else f"{'n/a':<10}"
        )
        print(
            f"{row.min_score:<12.2f} | {row.max_hits:<10d} | "
            f"{row.metrics.hit_rate:<15.2f} | "
            f"{row.metrics.mean_reciprocal_rank:<10.4f} | "
            f"{pipeline_hit_rate} | {pipeline_mrr} | {pipeline_count} | "
            f"{related_hit_rate} | {related_mrr} | {related_count} | "
            f"{answer_support} | {no_answer} | {semantic_score}"
        )
    print("-" * 174)


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
    related_terms = load_related_term_texts(repo_root)
    corpus_metadata = load_corpus_metadata(repo_root)
    query_cases = load_search_queries(repo_root)
    pipeline_query_cases = load_pipeline_queries(repo_root)
    related_query_cases = load_related_term_queries(repo_root)
    answer_cases = load_answer_queries(repo_root)
    if not query_cases:
        raise ValueError("No search queries with messageMatches found in the dataset")
    model, vector_base = await build_vector_base(
        model_spec,
        message_texts,
        batch_size,
    )
    related_vector_base = await build_text_vector_base(
        model,
        related_terms,
        batch_size,
    )
    pipeline_conversation = await build_pipeline_conversation(repo_root, model)
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
    top_score_stats = measure_top_score_stats(vector_base, query_embeddings)
    rows = await evaluate_grid(
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
    )

    print(f"Model: {model.model_name}")
    print(f"Messages indexed: {len(message_texts)}")
    print(f"Related terms indexed: {len(related_terms)}")
    print(f"Queries evaluated: {len(query_cases)}")
    print(f"Pipeline query cases evaluated: {len(pipeline_query_cases)}")
    print(f"Related-term cases evaluated: {len(related_query_cases)}")
    print(f"Answer cases evaluated: {len(answer_cases)}")
    print("Pipeline scoring paths:")
    for path in PIPELINE_SCORING_PATHS:
        print(f"  {path}")
    if corpus_metadata.serialized_total_embedding_count is not None:
        print(
            "Serialized sidecar rows ignored: "
            f"{corpus_metadata.serialized_total_embedding_count} "
            f"({INDEX_EMBEDDINGS_PATH.name})"
        )
    elif corpus_metadata.serialized_embedding_size is not None:
        print(
            "Serialized sidecar metadata found and ignored: "
            f"embedding_size={corpus_metadata.serialized_embedding_size}"
        )
    print(f"Corpus embeddings: {CORPUS_EMBEDDING_SOURCE}")
    print(f"Query embeddings: {QUERY_EMBEDDING_SOURCE}")
    print(f"Answer embeddings: {ANSWER_EMBEDDING_SOURCE}")
    print(
        "Observed top-1 score range: "
        f"{top_score_stats.min_top_score:.4f}..{top_score_stats.max_top_score:.4f} "
        f"(mean {top_score_stats.mean_top_score:.4f})"
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
    if best_row.pipeline_metrics:
        print(f"  pipeline_hit_rate={best_row.pipeline_metrics.hit_rate:.2f}%")
        print(f"  pipeline_mrr={best_row.pipeline_metrics.mean_reciprocal_rank:.4f}")
        print(
            "  pipeline_mean_result_count="
            f"{best_row.pipeline_metrics.mean_result_count:.2f}"
        )
    if best_row.related_metrics:
        print(f"  related_hit_rate={best_row.related_metrics.hit_rate:.2f}%")
        print(f"  related_mrr={best_row.related_metrics.mean_reciprocal_rank:.4f}")
        print(
            "  related_mean_result_count="
            f"{best_row.related_metrics.mean_result_count:.2f}"
        )
    if best_row.answer_metrics:
        print(f"  answerable_support={best_row.answer_metrics.answerable_support:.4f}")
        print(
            "  no_answer_rejection_rate="
            f"{best_row.answer_metrics.no_answer_rejection_rate:.2f}%"
        )
        print(f"  semantic_score={best_row.answer_metrics.semantic_score:.2f}")


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
