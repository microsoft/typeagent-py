# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmark for lookup_term_filtered — measures the N+1 query pattern.

After indexing 200 synthetic messages, looks up a high-frequency term
and filters results via lookup_term_filtered. Each call triggers
one get_item() SELECT per matching semantic ref (N+1 pattern).

Run:
    uv run python -m pytest tests/benchmarks/test_benchmark_query.py -v -s
"""

import os
import tempfile

import pytest

from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces_core import Term
from typeagent.knowpro.query import lookup_term_filtered
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)


def make_settings() -> ConversationSettings:
    model = create_test_embedding_model()
    settings = ConversationSettings(model=model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = False
    return settings


def synthetic_messages(n: int) -> list[TranscriptMessage]:
    return [
        TranscriptMessage(
            text_chunks=[f"Message {i} about topic {i % 10}"],
            metadata=TranscriptMessageMeta(speaker=f"Speaker{i % 3}"),
            tags=[f"tag{i % 5}"],
        )
        for i in range(n)
    ]


async def create_indexed_transcript(
    db_path: str, settings: ConversationSettings, n_messages: int
) -> Transcript:
    """Create and index a transcript, returning it ready for queries."""
    storage = SqliteStorageProvider(
        db_path,
        message_type=TranscriptMessage,
        message_text_index_settings=settings.message_text_index_settings,
        related_term_index_settings=settings.related_term_index_settings,
    )
    settings.storage_provider = storage
    transcript = await Transcript.create(settings, name="bench")
    messages = synthetic_messages(n_messages)
    await transcript.add_messages_with_indexing(messages)
    return transcript


@pytest.mark.asyncio
async def test_benchmark_lookup_term_filtered(async_benchmark):
    """Benchmark lookup_term_filtered with N+1 get_item pattern."""
    settings = make_settings()
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "query_bench.db")

    transcript = await create_indexed_transcript(db_path, settings, 200)

    # Find a high-frequency term to look up.
    semref_index = transcript.semantic_ref_index
    terms = await semref_index.get_terms()
    # Pick the term with the most matches.
    best_term = None
    best_count = 0
    for t in terms:
        refs = await semref_index.lookup_term(t)
        if refs and len(refs) > best_count:
            best_count = len(refs)
            best_term = t

    assert best_term is not None, "No terms found after indexing"
    print(f"\nBenchmarking term '{best_term}' with {best_count} matches")

    term = Term(text=best_term)
    semantic_refs = transcript.semantic_refs
    # Filter that accepts all — isolates the get_item overhead.
    accept_all = lambda sr, scored: True

    async def target():
        await lookup_term_filtered(semref_index, term, semantic_refs, accept_all)

    try:
        await async_benchmark.pedantic(target, rounds=200, warmup_rounds=20)
    finally:
        await settings.storage_provider.close()
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)
