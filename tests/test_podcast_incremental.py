# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test add_messages_with_indexing functionality for podcasts."""

import os
import tempfile

import pytest

from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.podcasts.podcast import Podcast, PodcastMessage, PodcastMessageMeta
from typeagent.storage.sqlite.provider import SqliteStorageProvider


@pytest.mark.asyncio
async def test_podcast_add_messages_with_indexing():
    """Test basic add_messages_with_indexing functionality for podcasts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings = ConversationSettings(model=test_model)
        settings.semantic_ref_index_settings.auto_extract_knowledge = False

        storage = SqliteStorageProvider(
            db_path,
            message_type=PodcastMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        podcast = await Podcast.create(settings, name="test")

        metadata1 = PodcastMessageMeta(speaker="Host", recipients=["Guest"])
        metadata2 = PodcastMessageMeta(speaker="Guest", recipients=["Host"])

        messages = [
            PodcastMessage(text_chunks=["Welcome to the podcast!"], metadata=metadata1),
            PodcastMessage(text_chunks=["Thanks for having me!"], metadata=metadata2),
        ]

        result = await podcast.add_messages_with_indexing(messages)

        assert result.messages_added == 2
        assert result.semrefs_added >= 4  # At least Host, Guest entities and actions
        assert await podcast.messages.size() == 2
        assert await podcast.semantic_refs.size() >= 4

        await storage.close()


@pytest.mark.asyncio
async def test_podcast_add_messages_batched():
    """Test batched message addition for podcasts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        test_model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
        settings = ConversationSettings(model=test_model)
        settings.semantic_ref_index_settings.auto_extract_knowledge = False

        storage = SqliteStorageProvider(
            db_path,
            message_type=PodcastMessage,
            message_text_index_settings=settings.message_text_index_settings,
            related_term_index_settings=settings.related_term_index_settings,
        )
        settings.storage_provider = storage
        podcast = await Podcast.create(settings, name="test")

        # Add first batch
        metadata1 = PodcastMessageMeta(speaker="Host")
        batch1 = [
            PodcastMessage(text_chunks=["Episode 1"], metadata=metadata1),
            PodcastMessage(text_chunks=["Episode 2"], metadata=metadata1),
        ]
        result1 = await podcast.add_messages_with_indexing(batch1)
        assert result1.messages_added == 2

        # Add second batch
        metadata2 = PodcastMessageMeta(speaker="Guest")
        batch2 = [
            PodcastMessage(text_chunks=["Episode 3"], metadata=metadata2),
            PodcastMessage(text_chunks=["Episode 4"], metadata=metadata2),
        ]
        result2 = await podcast.add_messages_with_indexing(batch2)
        assert result2.messages_added == 2

        assert await podcast.messages.size() == 4
        assert await podcast.semantic_refs.size() >= 4

        await storage.close()
