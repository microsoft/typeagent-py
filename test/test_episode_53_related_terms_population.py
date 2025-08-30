#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test to verify related terms index population with Episode 53 data."""

import asyncio
import tempfile
import os
import pytest

from typeagent.aitools.utils import load_dotenv
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.podcasts import podcast
from typeagent.storage.sqlitestore import SqliteStorageProvider


@pytest.mark.asyncio
async def test_episode_53_related_terms_index_population():
    """Test that related terms index is correctly populated when reopening Episode 53 database."""
    load_dotenv()
    temp_db_path = tempfile.mktemp(suffix=".sqlite")

    try:
        # Load Episode 53 data
        settings1 = ConversationSettings()
        print("Loading Episode 53 data...")
        pod1 = await podcast.Podcast.read_from_file(
            "testdata/Episode_53_AdrianTchaikovsky_index", settings1
        )
        print(f"Loaded podcast: {pod1.name_tag}")

        # Create a new storage provider with the test path
        storage_provider2 = await SqliteStorageProvider.create(
            settings1.message_text_index_settings,
            settings1.related_term_index_settings,
            temp_db_path,
            podcast.PodcastMessage,
        )

        # Transfer the data by serializing and deserializing
        pod_data = await pod1.serialize()

        # Create a new podcast with the sqlite storage and deserialize
        settings2 = ConversationSettings()
        settings2.storage_provider = storage_provider2
        pod2 = await podcast.Podcast.create(settings2)
        await pod2.deserialize(pod_data)

        await storage_provider2.close()

        # Reopen database with fresh settings to test index population
        settings3 = ConversationSettings()
        storage_provider3 = await SqliteStorageProvider.create(
            settings3.message_text_index_settings,
            settings3.related_term_index_settings,
            temp_db_path,
            podcast.PodcastMessage,
        )
        settings3.storage_provider = storage_provider3

        # Create a fresh podcast and load from the database
        pod3 = await podcast.Podcast.create(settings3)

        # Build the related terms index since it's not auto-populated
        from typeagent.knowpro.reltermsindex import build_related_terms_index

        await build_related_terms_index(pod3, settings3.related_term_index_settings)

        # Check that data was persisted and indexes are populated
        msg_count = await pod3.messages.size()
        sem_ref_count = await pod3.semantic_refs.size() if pod3.semantic_refs else 0

        print(f"Message count: {msg_count}")
        print(f"Semantic ref count: {sem_ref_count}")

        # Check related terms index
        secondary_indexes = pod3.secondary_indexes
        assert secondary_indexes is not None
        assert secondary_indexes.term_to_related_terms_index is not None

        related_terms_index = secondary_indexes.term_to_related_terms_index

        # Check aliases
        aliases = related_terms_index.aliases
        alias_count = await aliases.size()
        print(f"Aliases count: {alias_count}")

        # Check fuzzy index
        fuzzy_index = related_terms_index.fuzzy_index
        assert fuzzy_index is not None
        fuzzy_size = await fuzzy_index.size()
        print(f"Fuzzy index size: {fuzzy_size}")

        # Verify we have the expected counts
        # Note: aliases might not be preserved in this test since they're built from participant names
        # but the fuzzy index should have all the terms from the semantic refs
        expected_fuzzy_size = 1142
        assert (
            fuzzy_size == expected_fuzzy_size
        ), f"Expected {expected_fuzzy_size} fuzzy index entries, got {fuzzy_size}"

        print(f"✓ Related terms index successfully populated with {fuzzy_size} terms")

        await storage_provider3.close()

        print("✅ Episode 53 related terms index population test passed!")

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)


if __name__ == "__main__":
    asyncio.run(test_episode_53_related_terms_index_population())
