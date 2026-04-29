#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark semref index write strategies: per-item vs batched.

No API keys or network access required — uses synthetic knowledge data
and the deterministic test embedding model.

The "individual" path inlines the pre-optimization logic (one append +
add_term per entity/action/topic) so results are comparable on any
branch without switching.

Usage:
    uv run python tools/benchmark_semref_writes.py
    uv run python tools/benchmark_semref_writes.py --chunks 100 --rounds 20
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import statistics
import tempfile
import time

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro import knowledge_schema as kplib
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces_core import SemanticRef, Topic
from typeagent.storage.memory.semrefindex import (
    add_knowledge_batch_to_semantic_ref_index,
    text_range_from_message_chunk,
    validate_entity,
    verify_has_semantic_ref_index,
)
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)

# ---------------------------------------------------------------------------
# Inlined pre-optimization write path (one append + add_term per item)
# ---------------------------------------------------------------------------


async def _individual_add_knowledge(
    conversation,
    message_ordinal,
    chunk_ordinal,
    knowledge,
):
    """Reproduces the pre-optimization per-item write logic."""
    verify_has_semantic_ref_index(conversation)
    semantic_refs = conversation.semantic_refs
    assert semantic_refs is not None
    semantic_ref_index = conversation.semantic_ref_index
    assert semantic_ref_index is not None

    for entity in knowledge.entities:
        if not validate_entity(entity):
            continue
        ordinal = await semantic_refs.size()
        await semantic_refs.append(
            SemanticRef(
                semantic_ref_ordinal=ordinal,
                range=text_range_from_message_chunk(message_ordinal, chunk_ordinal),
                knowledge=entity,
            )
        )
        await semantic_ref_index.add_term(entity.name, ordinal)
        for type_name in entity.type:
            await semantic_ref_index.add_term(type_name, ordinal)
        if entity.facets:
            for facet in entity.facets:
                if facet is not None:
                    await semantic_ref_index.add_term(facet.name, ordinal)
                    if facet.value is not None:
                        await semantic_ref_index.add_term(str(facet.value), ordinal)

    for action in list(knowledge.actions) + list(knowledge.inverse_actions):
        ordinal = await semantic_refs.size()
        await semantic_refs.append(
            SemanticRef(
                semantic_ref_ordinal=ordinal,
                range=text_range_from_message_chunk(message_ordinal, chunk_ordinal),
                knowledge=action,
            )
        )
        await semantic_ref_index.add_term(" ".join(action.verbs), ordinal)
        if action.subject_entity_name != "none":
            await semantic_ref_index.add_term(action.subject_entity_name, ordinal)
        if action.object_entity_name != "none":
            await semantic_ref_index.add_term(action.object_entity_name, ordinal)
        if action.indirect_object_entity_name != "none":
            await semantic_ref_index.add_term(
                action.indirect_object_entity_name, ordinal
            )
        if action.params:
            for param in action.params:
                if isinstance(param, str):
                    await semantic_ref_index.add_term(param, ordinal)
                else:
                    await semantic_ref_index.add_term(param.name, ordinal)
                    if isinstance(param.value, str):
                        await semantic_ref_index.add_term(param.value, ordinal)
        if action.subject_entity_facet is not None:
            await semantic_ref_index.add_term(action.subject_entity_facet.name, ordinal)
            if action.subject_entity_facet.value is not None:
                await semantic_ref_index.add_term(
                    str(action.subject_entity_facet.value), ordinal
                )

    for topic_text in knowledge.topics:
        ordinal = await semantic_refs.size()
        await semantic_refs.append(
            SemanticRef(
                semantic_ref_ordinal=ordinal,
                range=text_range_from_message_chunk(message_ordinal, chunk_ordinal),
                knowledge=Topic(text=topic_text),
            )
        )
        await semantic_ref_index.add_term(topic_text, ordinal)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def synthetic_knowledge(chunk_index: int) -> kplib.KnowledgeResponse:
    return kplib.KnowledgeResponse(
        entities=[
            kplib.ConcreteEntity(
                name=f"entity_{chunk_index}_{j}",
                type=[f"type_{j}", f"category_{chunk_index % 5}"],
                facets=[
                    kplib.Facet(name=f"facet_{j}", value=f"value_{j}") for j in range(2)
                ],
            )
            for j in range(3)
        ],
        actions=[
            kplib.Action(
                verbs=[f"verb_{chunk_index}"],
                verb_tense="past",
                subject_entity_name=f"entity_{chunk_index}_0",
                object_entity_name=f"entity_{chunk_index}_1",
                indirect_object_entity_name="none",
                params=[f"param_{chunk_index}"],
            )
        ],
        inverse_actions=[],
        topics=[f"topic_{chunk_index}", f"theme_{chunk_index % 3}"],
    )


def synthetic_messages(count: int) -> list[TranscriptMessage]:
    return [
        TranscriptMessage(
            text_chunks=[f"Message {i} about topic {i % 10}"],
            metadata=TranscriptMessageMeta(speaker=f"Speaker{i % 3}"),
            tags=[f"tag{i % 5}"],
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------


async def create_transcript(db_path: str) -> Transcript:
    model = create_test_embedding_model()
    settings = ConversationSettings(model=model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = False
    storage = SqliteStorageProvider(
        db_path,
        message_type=TranscriptMessage,
        message_text_index_settings=settings.message_text_index_settings,
        related_term_index_settings=settings.related_term_index_settings,
    )
    settings.storage_provider = storage
    return await Transcript.create(settings, name="bench-semref")


async def bench_individual(transcript: Transcript, chunks: int) -> None:
    for i in range(chunks):
        await _individual_add_knowledge(transcript, i, 0, synthetic_knowledge(i))


async def bench_batched(transcript: Transcript, chunks: int) -> None:
    items = [(i, 0, synthetic_knowledge(i)) for i in range(chunks)]
    await add_knowledge_batch_to_semantic_ref_index(transcript, items)


async def run_benchmark(
    label: str,
    factory,
    chunks: int,
    rounds: int,
    warmup: int,
) -> list[float]:
    samples_us: list[float] = []
    for r in range(warmup + rounds):
        temp_dir = tempfile.mkdtemp(prefix="bench-semref-")
        db_path = os.path.join(temp_dir, "bench.db")
        try:
            transcript = await create_transcript(db_path)
            msgs = synthetic_messages(chunks)
            await transcript.add_messages_with_indexing(msgs)

            start = time.perf_counter_ns()
            await factory(transcript, chunks)
            elapsed_us = (time.perf_counter_ns() - start) / 1_000

            if r >= warmup:
                samples_us.append(elapsed_us)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    return samples_us


def print_report(label: str, samples_us: list[float], rounds: int, warmup: int) -> None:
    print(f"\n{label}")
    print(f"  rounds: {rounds} ({warmup} warmup)")
    print(f"  min:    {min(samples_us):12.1f} us")
    print(f"  mean:   {statistics.fmean(samples_us):12.1f} us")
    print(f"  median: {statistics.median(samples_us):12.1f} us")
    print(f"  max:    {max(samples_us):12.1f} us")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark semref index write strategies.",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=50,
        help="Number of knowledge chunks to write per run (default: 50).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of timed rounds (default: 10).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of untimed warmup rounds (default: 2).",
    )
    args = parser.parse_args()

    knowledge_sample = synthetic_knowledge(0)
    refs_per_chunk = (
        len([e for e in knowledge_sample.entities if e.name])
        + len(knowledge_sample.actions)
        + len(knowledge_sample.inverse_actions)
        + len(knowledge_sample.topics)
    )
    print(f"Chunks per run: {args.chunks}")
    print(f"Semrefs per chunk: ~{refs_per_chunk}")
    print(f"Total semrefs per run: ~{refs_per_chunk * args.chunks}")

    individual = await run_benchmark(
        "Individual writes",
        bench_individual,
        args.chunks,
        args.rounds,
        args.warmup,
    )
    print_report(
        "Individual writes (per-entity append + add_term)",
        individual,
        args.rounds,
        args.warmup,
    )

    batched = await run_benchmark(
        "Batched writes",
        bench_batched,
        args.chunks,
        args.rounds,
        args.warmup,
    )
    print_report(
        "Batched writes (bulk extend + add_terms_batch)",
        batched,
        args.rounds,
        args.warmup,
    )

    speedup = statistics.fmean(individual) / statistics.fmean(batched)
    print(f"\nSpeedup: {speedup:.2f}x")


if __name__ == "__main__":
    asyncio.run(main())
