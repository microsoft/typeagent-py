#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Utility script to benchmark different TextEmbeddingIndexSettings parameters.

Uses the Adrian Tchaikovsky podcast dataset (Episode 53) which contains:
- Index data: ~96 messages from the podcast conversation
- Search results: Queries with expected messageMatches (ground truth for retrieval)
- Answer results: Curated Q&A pairs with expected answers (ground truth for Q&A quality)

The benchmark evaluates embedding model retrieval quality using:
1. Search-based evaluation: Compares fuzzy_lookup results against expected messageMatches
2. Answer-based evaluation: Tests if queries from the Answer dataset retrieve messages
   that contain the expected answer content (substring matching)

Metrics:
- Hit Rate: Percentage of queries where at least one expected result was retrieved
- MRR (Mean Reciprocal Rank): Average of 1/rank of the first relevant result

Usage:
    uv run python tools/benchmark_embeddings.py [--model provider:model]
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from statistics import mean
import sys
from typing import Any

from typeagent.aitools.model_adapters import create_embedding_model
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings, VectorBase


async def run_benchmark(model_spec: str | None) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    index_data_path = repo_root / "tests" / "testdata" / "Episode_53_AdrianTchaikovsky_index_data.json"
    search_data_path = repo_root / "tests" / "testdata" / "Episode_53_Search_results.json"
    answer_data_path = repo_root / "tests" / "testdata" / "Episode_53_Answer_results.json"

    # ── Load index data (messages to embed) ──
    logger.info(f"Loading index data from {index_data_path}")
    try:
        with open(index_data_path, "r", encoding="utf-8") as f:
            index_json = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load index data: {e}")
        return

    messages = index_json.get("messages", [])
    message_texts = [" ".join(m.get("textChunks", [])) for m in messages]

    # ── Load search queries (ground truth: messageMatches) ──
    logger.info(f"Loading search queries from {search_data_path}")
    try:
        with open(search_data_path, "r", encoding="utf-8") as f:
            search_json = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load search queries: {e}")
        return

    # Filter out ones without results or expected matches
    search_queries: list[tuple[str, list[int]]] = []
    for item in search_json:
        search_text = item.get("searchText")
        results = item.get("results", [])
        if not results:
            continue
        expected = results[0].get("messageMatches", [])
        if not expected:
            continue
        search_queries.append((search_text, expected))

    # ── Load answer results (Q&A ground truth from Adrian Tchaikovsky dataset) ──
    answer_queries: list[tuple[str, str, bool]] = []  # (question, answer, hasNoAnswer)
    logger.info(f"Loading answer results from {answer_data_path}")
    try:
        with open(answer_data_path, "r", encoding="utf-8") as f:
            answer_json = json.load(f)
        for item in answer_json:
            question = item.get("question", "")
            answer = item.get("answer", "")
            has_no_answer = item.get("hasNoAnswer", False)
            if question and answer:
                answer_queries.append((question, answer, has_no_answer))
        logger.info(f"Found {len(answer_queries)} answer Q&A pairs "
                     f"({sum(1 for _, _, h in answer_queries if not h)} with answers, "
                     f"{sum(1 for _, _, h in answer_queries if h)} with no-answer).")
    except Exception as e:
        logger.warning(f"Failed to load answer results (continuing without): {e}")

    logger.info(f"Found {len(message_texts)} messages to embed.")
    logger.info(f"Found {len(search_queries)} search queries with expected matches.")

    # ── Create embedding model and index ──
    try:
        if model_spec == "test:fake":
            from typeagent.aitools.model_adapters import create_test_embedding_model
            model = create_test_embedding_model(embedding_size=384)
        else:
            model = create_embedding_model(model_spec)
    except Exception as e:
        logger.error(f"Failed to create embedding model: {e}")
        logger.info("Are your environment variables (e.g. OPENAI_API_KEY) set?")
        return
    settings = TextEmbeddingIndexSettings(model)
    vbase = VectorBase(settings)

    logger.info("Computing embeddings for messages (this may take some time...)")
    # Batch the embeddings
    batch_size = 50
    for i in range(0, len(message_texts), batch_size):
        batch = message_texts[i : i + batch_size]
        await vbase.add_keys(batch)
        print(f"  ... embedded {min(i + batch_size, len(message_texts))}/{len(message_texts)}")

    # ── Compute query embeddings ──
    logger.info("Computing embeddings for search queries...")
    search_query_texts = [q[0] for q in search_queries]
    search_query_embeddings = await model.get_embeddings(search_query_texts)

    answer_query_embeddings = None
    if answer_queries:
        logger.info("Computing embeddings for answer queries...")
        answer_query_texts = [q[0] for q in answer_queries]
        answer_query_embeddings = await model.get_embeddings(answer_query_texts)

    # ──────────────────────────────────────────────────────────────────────
    # Section 1: Grid Search using Search Results (messageMatches)
    # ──────────────────────────────────────────────────────────────────────

    # Grid search config
    min_scores_to_test = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    max_hits_to_test = [5, 10, 15, 20]

    logger.info(f"Starting grid search over model: {model.model_name}")
    print()
    print("=" * 72)
    print("  SEARCH RESULTS BENCHMARK (messageMatches ground truth)")
    print("=" * 72)
    print(f"{'Min Score':<12} | {'Max Hits':<10} | {'Hit Rate (%)':<15} | {'MRR':<10}")
    print("-" * 65)

    best_mrr = -1.0
    best_config = None

    for ms in min_scores_to_test:
        for mh in max_hits_to_test:
            hits = 0
            reciprocal_ranks = []
            
            for (query_text, expected_indices), q_emb in zip(search_queries, search_query_embeddings):
                scored_results = vbase.fuzzy_lookup_embedding(q_emb, max_hits=mh, min_score=ms)
                retrieved_indices = [sr.item for sr in scored_results]

                # Check if any of the expected items are in the retrieved answers
                rank = -1
                for r_idx, retrieved in enumerate(retrieved_indices):
                    if retrieved in expected_indices:
                        rank = r_idx + 1
                        break

                if rank > 0:
                    hits += 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

            hit_rate = (hits / len(search_queries)) * 100
            mrr = mean(reciprocal_ranks)
            
            print(f"{ms:<12.2f} | {mh:<10d} | {hit_rate:<15.2f} | {mrr:<10.4f}")

            if mrr > best_mrr:
                best_mrr = mrr
                best_config = (ms, mh)

    print("-" * 65)
    if best_config:
        logger.info(f"Search benchmark optimal: min_score={best_config[0]}, "
                     f"max_hits={best_config[1]} (MRR={best_mrr:.4f})")
    else:
        logger.info("Could not determine optimal parameters (no hits).")

    # ──────────────────────────────────────────────────────────────────────
    # Section 2: Answer Results Benchmark (Adrian Tchaikovsky Q&A pairs)
    # ──────────────────────────────────────────────────────────────────────

    if answer_queries and answer_query_embeddings is not None:
        print()
        print("=" * 72)
        print("  ANSWER RESULTS BENCHMARK (Adrian Tchaikovsky Q&A ground truth)")
        print("=" * 72)
        print()

        # For each answer query, check if retrieved messages contain key terms
        # from the expected answer. This is a content-based relevance check.
        #
        # We split answers with hasNoAnswer=True vs False to evaluate separately.

        answerable = [(q, a, emb) for (q, a, h), emb
                       in zip(answer_queries, answer_query_embeddings) if not h]
        unanswerable = [(q, a, emb) for (q, a, h), emb
                         in zip(answer_queries, answer_query_embeddings) if h]

        print(f"Answerable queries: {len(answerable)}")
        print(f"Unanswerable queries (hasNoAnswer=True): {len(unanswerable)}")
        print()

        # Extract key terms from expected answers for content matching
        def extract_answer_keywords(answer_text: str) -> list[str]:
            """Extract distinctive keywords/phrases from an answer for matching."""
            # Look for quoted items, proper nouns, and distinctive phrases
            keywords = []
            # Extract quoted phrases
            import re
            quoted = re.findall(r"'([^']+)'", answer_text)
            keywords.extend(quoted)
            quoted2 = re.findall(r'"([^"]+)"', answer_text)
            keywords.extend(quoted2)

            # Extract proper-noun-like terms (capitalized words that aren't sentence starters)
            # and key named entities from the Adrian Tchaikovsky dataset
            known_entities = [
                "Adrian Tchaikovsky", "Tchaikovsky", "Kevin Scott", "Christina Warren",
                "Children of Time", "Children of Ruin", "Children of Memory",
                "Shadows of the Apt", "Empire in Black and Gold",
                "Final Architecture", "Lords of Uncreation",
                "Dragonlance Chronicles", "Skynet", "Portids", "Corvids",
                "University of Reading", "Magnus Carlsen", "Warhammer",
                "Asimov", "Peter Watts", "William Gibson", "Iain Banks",
                "Peter Hamilton", "Arthur C. Clarke", "Profiles of the Future",
                "Dune", "Brave New World", "Iron Sunrise", "Wall-E",
                "George RR Martin", "Alastair Reynolds", "Ovid",
                "zoology", "psychology", "spiders", "arachnids", "insects",
            ]
            for entity in known_entities:
                if entity.lower() in answer_text.lower():
                    keywords.append(entity)

            return keywords

        # Run answer benchmark with the best config from search benchmark
        if best_config:
            eval_min_score, eval_max_hits = best_config
        else:
            eval_min_score, eval_max_hits = 0.80, 10

        print(f"Using parameters: min_score={eval_min_score}, max_hits={eval_max_hits}")
        print("-" * 72)
        print(f"{'#':<4} | {'Question':<45} | {'Keywords Found':<14} | {'Msgs':<5}")
        print("-" * 72)

        answer_hits = 0
        answer_keyword_scores: list[float] = []

        for idx, (question, answer, q_emb) in enumerate(answerable, 1):
            scored_results = vbase.fuzzy_lookup_embedding(
                q_emb, max_hits=eval_max_hits, min_score=eval_min_score
            )
            retrieved_indices = [sr.item for sr in scored_results]

            # Concatenate the text of all retrieved messages
            retrieved_text = " ".join(
                message_texts[i] for i in retrieved_indices if i < len(message_texts)
            )

            # Check how many answer keywords appear in retrieved text
            keywords = extract_answer_keywords(answer)
            if keywords:
                found = sum(
                    1 for kw in keywords
                    if kw.lower() in retrieved_text.lower()
                )
                keyword_score = found / len(keywords)
            else:
                # No keywords extracted — just check if we retrieved anything
                keyword_score = 1.0 if retrieved_indices else 0.0

            if keyword_score > 0:
                answer_hits += 1
            answer_keyword_scores.append(keyword_score)

            q_display = question[:42] + "..." if len(question) > 45 else question
            kw_display = f"{int(keyword_score * 100):>3}%"
            if keywords:
                kw_display += f" ({sum(1 for kw in keywords if kw.lower() in retrieved_text.lower())}/{len(keywords)})"
            print(f"{idx:<4} | {q_display:<45} | {kw_display:<14} | {len(retrieved_indices):<5}")

        print("-" * 72)

        if answerable:
            answer_hit_rate = (answer_hits / len(answerable)) * 100
            avg_keyword_score = mean(answer_keyword_scores) * 100
            print(f"Answer Hit Rate:        {answer_hit_rate:.1f}% "
                  f"({answer_hits}/{len(answerable)} queries found relevant content)")
            print(f"Avg Keyword Coverage:   {avg_keyword_score:.1f}%")

        # Evaluate unanswerable queries — ideally these should retrieve fewer/no results
        if unanswerable:
            print()
            print("-" * 72)
            print("Unanswerable queries (should ideally retrieve less relevant content):")
            print("-" * 72)
            false_positive_count = 0
            for question, answer, q_emb in unanswerable:
                scored_results = vbase.fuzzy_lookup_embedding(
                    q_emb, max_hits=eval_max_hits, min_score=eval_min_score
                )
                n_results = len(scored_results)
                avg_score = mean(sr.score for sr in scored_results) if scored_results else 0.0
                q_display = question[:55] + "..." if len(question) > 58 else question
                flag = "[!]" if n_results > 3 else "[ok]"
                if n_results > 3:
                    false_positive_count += 1
                print(f"  {flag} {q_display:<58} | {n_results:>3} results (avg={avg_score:.3f})")
            print(f"\nFalse positives (>3 results): {false_positive_count}/{len(unanswerable)}")

    # ── Summary ──
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"Model:                  {model.model_name}")
    print(f"Messages indexed:       {len(message_texts)}")
    print(f"Search queries tested:  {len(search_queries)}")
    if best_config:
        print(f"Best search params:     min_score={best_config[0]}, max_hits={best_config[1]}")
        print(f"Best search MRR:        {best_mrr:.4f}")
    if answer_queries:
        print(f"Answer queries tested:  {len(answerable)} answerable, {len(unanswerable)} unanswerable")
        if answerable:
            print(f"Answer hit rate:        {answer_hit_rate:.1f}%")
            print(f"Keyword coverage:       {avg_keyword_score:.1f}%")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding model parameters.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Provider and model name, e.g. 'openai:text-embedding-3-small'",
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.model))


if __name__ == "__main__":
    main()
