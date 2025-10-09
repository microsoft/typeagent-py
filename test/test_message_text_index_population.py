#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test to verify message text index population in storage providers."""

import asyncio
import tempfile
import os
import pytest
from typeagent.storage import SqliteStorageProvider
from typeagent.storage.memory.messageindex import MessageTextIndex
from typeagent.knowpro.convsettings import MessageTextIndexSettings
from typeagent.knowpro.convsettings import RelatedTermIndexSettings
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
from typeagent.podcasts.podcast import PodcastMessage, PodcastMessageMeta
from typeagent.aitools.utils import load_dotenv
import numpy as np


@pytest.mark.asyncio
async def test_message_text_index_population_from_database():
    """Test that message text index is correctly populated when reopening a database."""
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
        await msg_collection.extend(test_messages)
        assert await msg_collection.size() == len(test_messages)

        await storage1.close()

        # Reopen datawase and verify message text index
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

        # Check message text index
        msg_text_index = await storage2.get_message_text_index()
        # Check that it implements the interface correctly
        from typeagent.knowpro.interfaces import IMessageTextIndex

        assert isinstance(msg_text_index, IMessageTextIndex)

        # Check if index has entries (debug info)
        index_size = await msg_text_index.size()
        print(f"Message text index size: {index_size}")
        print(f"Message text index is empty: {await msg_text_index.is_empty()}")

        # Let's also try to manually check the first few messages
        for i in range(min(3, msg_count)):
            message = await msg_collection2.get_item(i)
            print(f"Message {i}: {message.text_chunks}")

        # Test message text index functionality by verifying it has entries
        # The main goal is to verify that the index was populated (non-zero size)
        # not that search works perfectly (which depends on embedding quality)

        # At minimum, the index should have some entries after population
        assert (
            index_size > 0
        ), f"Message text index should have entries, got {index_size}"

        # Each message has one chunk, so we should have 3 entries
        assert (
            index_size == 3
        ), f"Expected 3 index entries (one per message chunk), got {index_size}"

        print(f"✓ Message text index successfully populated with {index_size} entries")

        await storage2.close()

        print("✅ Message text index population test passed!")

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
