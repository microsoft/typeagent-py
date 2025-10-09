#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fast test for related terms index functionality (replaces slow Episode 53 test)."""

import tempfile
import os
import pytest

from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.podcasts.podcast import Podcast, PodcastMessage, PodcastMessageMeta
from typeagent.storage import SqliteStorageProvider
from typeagent.knowpro.interfaces import SemanticRef, TextRange, TextLocation
from typeagent.knowpro.kplib import ConcreteEntity


@pytest.mark.asyncio
async def test_related_terms_index_minimal():
    """Fast test with minimal data to verify related terms functionality."""
    temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    temp_db_path = temp_db_file.name
    temp_db_file.close()

    try:
        # Create minimal test data with test embedding model (no API keys needed)
        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings = ConversationSettings(model=test_model)

        # Use a simple storage provider without AI embeddings
        storage_provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=PodcastMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage_provider

        pod = await Podcast.create(settings)

        # Add minimal test messages
        test_messages = [
            PodcastMessage(
                text_chunks=["Hello world, this is about Python programming"],
                tags=["test"],
                metadata=PodcastMessageMeta(speaker="Alice"),
            ),
            PodcastMessage(
                text_chunks=["Python is a great language for data science"],
                tags=["test"],
                metadata=PodcastMessageMeta(speaker="Bob"),
            ),
            PodcastMessage(
                text_chunks=["Data science and machine learning go together"],
                tags=["test"],
                metadata=PodcastMessageMeta(speaker="Alice"),
            ),
        ]

        # Add messages
        for msg in test_messages:
            await pod.messages.append(msg)

        # Add a few semantic refs manually (without AI)
        test_semantic_refs = [
            SemanticRef(
                semantic_ref_ordinal=0,
                range=TextRange(TextLocation(0, 0), TextLocation(0, 1)),
                knowledge=ConcreteEntity(
                    name="Python", type=["programming language"], facets=[]
                ),
            ),
            SemanticRef(
                semantic_ref_ordinal=1,
                range=TextRange(TextLocation(1, 0), TextLocation(1, 1)),
                knowledge=ConcreteEntity(
                    name="data science", type=["field"], facets=[]
                ),
            ),
        ]

        for sem_ref in test_semantic_refs:
            await pod.semantic_refs.append(sem_ref)

        # Build basic indexes without heavy computation
        if pod.secondary_indexes and pod.secondary_indexes.term_to_related_terms_index:
            # Add some basic terms manually instead of computing embeddings
            aliases = pod.secondary_indexes.term_to_related_terms_index.aliases
            from typeagent.knowpro.interfaces import Term

            await aliases.add_related_term(
                "python", [Term("programming", 1.0), Term("coding", 0.8)]
            )
            await aliases.add_related_term(
                "data science", [Term("analytics", 0.9), Term("statistics", 0.7)]
            )

        # Verify basic functionality
        msg_count = await pod.messages.size()
        sem_ref_count = await pod.semantic_refs.size()

        assert msg_count == 3
        assert sem_ref_count == 2

        # Test that related terms work
        if pod.secondary_indexes and pod.secondary_indexes.term_to_related_terms_index:
            aliases = pod.secondary_indexes.term_to_related_terms_index.aliases
            python_related = await aliases.lookup_term("python")
            assert python_related is not None
            assert len(python_related) == 2

            alias_count = await aliases.size()
            assert alias_count == 2

        await storage_provider.close()
        print("✅ Fast related terms test passed!")

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
