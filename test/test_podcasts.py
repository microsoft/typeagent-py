# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os

from fixtures import needs_auth, temp_dir  # type: ignore  # Yes they are used!

from typeagent.podcasts.podcast import Podcast
from typeagent.knowpro import importing
from typeagent.knowpro.interfaces import Datetime
from typeagent.podcasts import podcast_import
from typeagent.knowpro.serialization import DATA_FILE_SUFFIX, EMBEDDING_FILE_SUFFIX


def test_import_podcast(needs_auth, temp_dir):
    # Import the podcast
    settings = importing.ConversationSettings()
    pod = podcast_import.import_podcast(
        "testdata/FakePodcast.txt",
        None,
        Datetime.now(),
        3.0,
        settings=settings,
    )

    # Basic assertions about the imported podcast
    assert pod.name_tag is not None
    assert len(pod.tags) > 0
    assert len(pod.messages) > 0

    # Build the index
    indexing_result = asyncio.run(pod.build_index())
    assert indexing_result.semantic_refs is not None
    assert indexing_result.semantic_refs.error is None

    # Write the podcast to files
    filename = os.path.join(temp_dir, "podcast")
    pod.write_to_file(filename)

    # Verify the files were created
    assert os.path.exists(filename + DATA_FILE_SUFFIX)
    assert os.path.exists(filename + EMBEDDING_FILE_SUFFIX)

    # Load and verify the podcast
    pod2 = Podcast.read_from_file(filename)
    assert pod2 is not None

    # Assertions for the loaded podcast
    assert pod2.name_tag == pod.name_tag, "Name tags do not match"
    assert pod2.tags == pod.tags, "Tags do not match"
    assert len(pod2.messages) == len(pod.messages), "Number of messages do not match"
    assert all(
        m1.serialize() == m2.serialize() for m1, m2 in zip(pod.messages, pod2.messages)
    ), "Messages do not match"

    # Write to another pair of files and check they match
    filename2 = os.path.join(temp_dir, "podcast2")
    pod2.write_to_file(filename2)
    assert os.path.exists(filename2 + DATA_FILE_SUFFIX)
    assert os.path.exists(filename2 + EMBEDDING_FILE_SUFFIX)

    # Check that the files at filename2 are identical to those at filename
    with open(filename + DATA_FILE_SUFFIX, "r") as f1, open(
        filename2 + DATA_FILE_SUFFIX, "r"
    ) as f2:
        assert f1.read() == f2.read(), "Data (json) files do not match"
    with open(filename + EMBEDDING_FILE_SUFFIX, "rb") as f1, open(
        filename2 + EMBEDDING_FILE_SUFFIX, "rb"
    ) as f2:
        assert f1.read() == f2.read(), "Embedding (binary) files do not match"
