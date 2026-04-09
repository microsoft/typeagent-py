#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Utility script to benchmark different TextEmbeddingIndexSettings parameters.

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

    logger.info(f"Loading index data from {index_data_path}")
    try:
        with open(index_data_path, "r", encoding="utf-8") as f:
            index_json = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load index data: {e}")
        return

    messages = index_json.get("messages", [])
    message_texts = [" ".join(m.get("textChunks", [])) for m in messages]

    logger.info(f"Loading search queries from {search_data_path}")
    try:
        with open(search_data_path, "r", encoding="utf-8") as f:
            search_json = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load search queries: {e}")
        return

    # Filter out ones without results or expected matches
    queries = []
    for item in search_json:
        search_text = item.get("searchText")
        results = item.get("results", [])
        if not results:
            continue
        expected = results[0].get("messageMatches", [])
        if not expected:
            continue
        queries.append((search_text, expected))

    logger.info(f"Found {len(message_texts)} messages to embed.")
    logger.info(f"Found {len(queries)} queries with expected matches to test.")

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

    logger.info("Computing embeddings for queries...")
    query_texts = [q[0] for q in queries]
    query_embeddings = await model.get_embeddings(query_texts)

    # Grid search config
    min_scores_to_test = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    max_hits_to_test = [5, 10, 15, 20]

    logger.info(f"Starting grid search over model: {model.model_name}")
    print("-" * 65)
    print(f"{'Min Score':<12} | {'Max Hits':<10} | {'Hit Rate (%)':<15} | {'MRR':<10}")
    print("-" * 65)

    best_mrr = -1.0
    best_config = None

    for ms in min_scores_to_test:
        for mh in max_hits_to_test:
            hits = 0
            reciprocal_ranks = []
            
            for (query_text, expected_indices), q_emb in zip(queries, query_embeddings):
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

            hit_rate = (hits / len(queries)) * 100
            mrr = mean(reciprocal_ranks)
            
            print(f"{ms:<12.2f} | {mh:<10d} | {hit_rate:<15.2f} | {mrr:<10.4f}")

            if mrr > best_mrr:
                best_mrr = mrr
                best_config = (ms, mh)

    print("-" * 65)
    if best_config:
        logger.info(f"Optimal parameters found: min_score={best_config[0]}, max_hits={best_config[1]} (MRR={best_mrr:.4f})")
    else:
        logger.info("Could not determine optimal parameters (no hits).")


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
