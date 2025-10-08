# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os
import pytest
from datetime import timezone

from fixtures import needs_auth, temp_dir, embedding_model  # type: ignore  # Yes they are used!

from typeagent.podcasts.podcast import Podcast
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces import Datetime
from typeagent.podcasts import podcast_ingest
from typeagent.knowpro.serialization import DATA_FILE_SUFFIX, EMBEDDING_FILE_SUFFIX
from typeagent.aitools.embeddings import AsyncEmbeddingModel


@pytest.mark.asyncio
async def test_ingest_podcast(
    needs_auth: None, temp_dir: str, embedding_model: AsyncEmbeddingModel
):
    # Import the podcast
    settings = ConversationSettings(embedding_model)
    pod = await podcast_ingest.ingest_podcast(
        "testdata/FakePodcast.txt",
        settings,
        None,
        Datetime.now(timezone.utc),  # Use timezone-aware datetime
        3.0,
    )

    # Basic assertions about the imported podcast
    assert pod.name_tag is not None
    assert len(pod.tags) > 0
    assert await pod.messages.size() > 0

    # Verify the semantic refs exist
    assert pod.semantic_refs is not None

    # Write the podcast to files
    filename_prefix = os.path.join(temp_dir, "podcast")
    await pod.write_to_file(filename_prefix)

    # Verify the files were created
    assert os.path.exists(filename_prefix + DATA_FILE_SUFFIX)
    assert os.path.exists(filename_prefix + EMBEDDING_FILE_SUFFIX)

    # Load and verify the podcast with a fresh settings object
    settings2 = ConversationSettings(embedding_model)
    pod2 = await Podcast.read_from_file(filename_prefix, settings2)
    assert pod2 is not None

    # Assertions for the loaded podcast
    assert pod2.name_tag == pod.name_tag, "Name tags do not match"
    assert pod2.tags == pod.tags, "Tags do not match"
    assert (
        await pod2.messages.size() == await pod.messages.size()
    ), "Number of messages do not match"

    # Compare messages (simplified check since we can't iterate over async collections directly)
    pod_msgs_size = await pod.messages.size()
    pod2_msgs_size = await pod2.messages.size()
    assert pod_msgs_size == pod2_msgs_size, "Message counts don't match"

    # Check first few messages match
    for i in range(min(3, pod_msgs_size)):  # Check first 3 messages
        m1 = await pod.messages.get_item(i)
        m2 = await pod2.messages.get_item(i)
        assert m1.serialize() == m2.serialize(), f"Message {i} doesn't match"

    # Write to another pair of files and check they match
    filename2 = os.path.join(temp_dir, "podcast2")
    await pod2.write_to_file(filename2)
    assert os.path.exists(filename2 + DATA_FILE_SUFFIX)
    assert os.path.exists(filename2 + EMBEDDING_FILE_SUFFIX)

    # Check that the files at filename2 are identical to those at filename
    with (
        open(filename_prefix + DATA_FILE_SUFFIX, "r") as f1,
        open(filename2 + DATA_FILE_SUFFIX, "r") as f2,
    ):
        assert f1.read() == f2.read(), "Data (json) files do not match"
    with (
        open(filename_prefix + EMBEDDING_FILE_SUFFIX, "rb") as f1,
        open(filename2 + EMBEDDING_FILE_SUFFIX, "rb") as f2,
    ):
        assert f1.read() == f2.read(), "Embedding (binary) files do not match"
