#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test to verify related terms index population in storage providers."""

import tempfile
import os
import pytest

from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.aitools.utils import load_dotenv
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.interfaces import SemanticRef, TextRange, TextLocation
from typeagent.knowpro import kplib
from typeagent.knowpro.convsettings import (
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.podcasts.podcast import PodcastMessage, PodcastMessageMeta
from typeagent.storage import SqliteStorageProvider

from fixtures import really_needs_auth


@pytest.mark.asyncio
async def test_related_terms_index_population_from_database(really_needs_auth):
    """Test that related terms index is correctly populated when reopening a database."""
    load_dotenv()
    temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    temp_db_path = temp_db_file.name
    temp_db_file.close()

    try:
        # Use the test model that's already configured in the system
        embedding_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        # Create and populate database
        storage1 = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=PodcastMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        # Add test messages
        test_messages = [
            PodcastMessage(
                text_chunks=["Hello, I'm discussing artificial intelligence today."],
                metadata=PodcastMessageMeta(speaker="Alice"),
            ),
            PodcastMessage(
                text_chunks=["Let me explain machine learning concepts."],
                metadata=PodcastMessageMeta(speaker="Bob"),
            ),
            PodcastMessage(
                text_chunks=["Python is a great programming language."],
                metadata=PodcastMessageMeta(speaker="Charlie"),
            ),
        ]

        msg_collection = await storage1.get_message_collection()
        for message in test_messages:
            await msg_collection.append(message)

        # Add some semantic refs to create terms for the related terms index
        sem_ref_collection = await storage1.get_semantic_ref_collection()

        # Add some entities
        entity_refs = [
            SemanticRef(
                semantic_ref_ordinal=0,
                range=TextRange(start=TextLocation(message_ordinal=0, chunk_ordinal=0)),
                knowledge=kplib.ConcreteEntity(
                    name="artificial intelligence", type=["technology", "concept"]
                ),
            ),
            SemanticRef(
                semantic_ref_ordinal=1,
                range=TextRange(start=TextLocation(message_ordinal=1, chunk_ordinal=0)),
                knowledge=kplib.ConcreteEntity(
                    name="machine learning", type=["technology", "subset of AI"]
                ),
            ),
            SemanticRef(
                semantic_ref_ordinal=2,
                range=TextRange(start=TextLocation(message_ordinal=2, chunk_ordinal=0)),
                knowledge=kplib.ConcreteEntity(
                    name="Python", type=["programming language"]
                ),
            ),
        ]

        for sem_ref in entity_refs:
            await sem_ref_collection.append(sem_ref)

        # Manually populate the semantic ref index since the user guarantees it's complete externally
        semantic_ref_index = await storage1.get_semantic_ref_index()

        for sem_ref in entity_refs:
            knowledge = sem_ref.knowledge
            ref_ordinal = sem_ref.semantic_ref_ordinal

            if isinstance(knowledge, kplib.ConcreteEntity):
                await semantic_ref_index.add_term(knowledge.name, ref_ordinal)
                for type_name in knowledge.type:
                    await semantic_ref_index.add_term(type_name, ref_ordinal)

        await storage1.close()

        # Reopen database and verify related terms index
        storage2 = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=PodcastMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        # Check message collection size
        msg_collection2 = await storage2.get_message_collection()
        msg_count = await msg_collection2.size()
        print(f"Message collection size: {msg_count}")
        assert msg_count == len(
            test_messages
        ), f"Expected {len(test_messages)} messages, got {msg_count}"

        # Check semantic ref collection size
        sem_ref_collection2 = await storage2.get_semantic_ref_collection()
        sem_ref_count = await sem_ref_collection2.size()
        print(f"Semantic ref collection size: {sem_ref_count}")
        assert sem_ref_count == len(
            entity_refs
        ), f"Expected {len(entity_refs)} semantic refs, got {sem_ref_count}"

        # Create a test conversation and build related terms index
        from typeagent.podcasts.podcast import Podcast
        from typeagent.knowpro.convsettings import ConversationSettings
        from typeagent.storage.memory.reltermsindex import build_related_terms_index
        from typeagent.storage.sqlite.reltermsindex import SqliteRelatedTermsIndex

        settings2 = ConversationSettings()
        settings2.storage_provider = storage2
        conversation = await Podcast.create(settings2)

        # Build related terms index from the semantic refs
        await build_related_terms_index(conversation, related_terms_settings)

        # Check related terms index
        related_terms_index = await storage2.get_related_terms_index()
        assert isinstance(related_terms_index, SqliteRelatedTermsIndex)

        # Check if fuzzy index has entries
        fuzzy_index = related_terms_index.fuzzy_index
        assert fuzzy_index is not None

        fuzzy_index_size = await fuzzy_index.size()
        print(f"Related terms fuzzy index size: {fuzzy_index_size}")

        # The fuzzy index should have entries for all the terms that were added to the semantic ref index
        # This includes entity names and their types
        assert (
            fuzzy_index_size > 0
        ), f"Related terms fuzzy index should have entries, got {fuzzy_index_size}"

        # We expect terms like: "artificial intelligence", "technology", "concept", "machine learning",
        # "subset of AI", "Python", "programming language"
        # So at least 7 unique terms
        expected_min_terms = 7
        assert (
            fuzzy_index_size >= expected_min_terms
        ), f"Expected at least {expected_min_terms} terms in fuzzy index, got {fuzzy_index_size}"

        print(
            f"✓ Related terms index successfully populated with {fuzzy_index_size} terms"
        )

        await storage2.close()

        print("✅ Related terms index population test passed!")

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
