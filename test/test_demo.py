# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
import os
import textwrap
import time

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces import ScoredSemanticRefOrdinal
from typeagent.podcasts import podcast

from fixtures import really_needs_auth

tests_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(tests_dir)
DEFAULT_FILE = os.path.join(root_dir, "testdata", "Episode_53_AdrianTchaikovsky_index")

parser = argparse.ArgumentParser()
parser.add_argument(
    "filename",
    nargs="?",
    type=str,
    default=DEFAULT_FILE,
)


def test_main(really_needs_auth: None):
    # auth is needed because we use embeddings.
    # TODO: Only use the embeddings loaded from the file and cached.
    asyncio.run(main(DEFAULT_FILE))


async def main(filename_prefix: str):
    print("Create conversation settings ...")
    settings = ConversationSettings()
    model = settings.embedding_model
    assert model is not None
    assert isinstance(model, AsyncEmbeddingModel), f"model is {model!r}"
    assert settings.thread_settings.embedding_model is model
    assert (
        settings.message_text_index_settings.embedding_index_settings.embedding_model
        is model
    )
    assert (
        settings.related_term_index_settings.embedding_index_settings.embedding_model
        is model
    )

    print(f"Loading {filename_prefix} ...")
    t0 = time.time()
    pod = await podcast.Podcast.read_from_file(filename_prefix, settings)
    t1 = time.time()
    print(f"Loading took {t1-t0:.3f} seconds")
    assert pod is not None, "Failed to load podcast"
    assert isinstance(pod, podcast.Podcast), f"pod is {pod!r}"

    term = "book"
    print(f"\nSearching {pod.name_tag!r} for term {term!r} ...")
    assert pod.semantic_ref_index is not None, "semantic_ref_index should not be None"
    results = await pod.semantic_ref_index.lookup_term(term)
    assert results is not None
    assert isinstance(results, list), f"results is {results!r}"
    assert len(results) > 0, f"results is {results!r}"
    assert isinstance(
        results[0], ScoredSemanticRefOrdinal
    ), f"results[0] is {results[0]!r}"

    for scored_ord in results:
        ord = scored_ord.semantic_ref_ordinal
        assert pod.semantic_refs is not None
        assert 0 <= ord < await pod.semantic_refs.size()
        sref = await pod.semantic_refs.get_item(ord)
        assert sref.semantic_ref_ordinal == ord
        print(f"\n{ord}: Term {term!r} has knowledge", end=" ")
        print(f"of type {sref.knowledge.knowledge_type!r} at {sref.range}:")
        print(" ", sref.knowledge)
        # Now dig up the messages
        start_msg_ord = sref.range.start.message_ordinal
        end_msg_ord = (
            sref.range.end.message_ordinal if sref.range.end else start_msg_ord + 1
        )
        messages = await pod.messages.get_slice(start_msg_ord, end_msg_ord)
        assert len(messages) > 0, f"messages is {messages!r}"
        for i, message in enumerate(messages, start_msg_ord):
            print(f" MESSAGE {i}:")
            text = " ".join(message.text_chunks).strip()
            wrapped = textwrap.wrap(text)
            if len(wrapped) > 6:
                print(f"  {1}: {wrapped[0]}")
                print(f"  {2}: {wrapped[1]}")
                print(f"  ... {len(wrapped) - 4} more lines ...")
                print(f"  {len(wrapped) - 1}: {wrapped[-2]}")
                print(f"  {len(wrapped)}: {wrapped[-1]}")
            else:
                for i, line in enumerate(wrapped):
                    print(f"  {i}: {line}")

    print(f"\nChecking that serialize -> deserialize -> serialize roundtrips ...")
    ser1 = await pod.serialize()
    assert ser1 is not None, "Failed to serialize podcast"
    assert isinstance(ser1, dict), f"ser1 is not dict but {type(ser1)!r}"
    assert len(ser1) > 0, f"ser1 is empty {ser1!r}"
    assert "semanticRefs" in ser1, f"'semantic_refs' is not a key in {ser1.keys()!r}"

    # Create a fresh settings object with a new storage provider for the second podcast
    # to avoid conflicts with the first podcast's data.
    # TODO: Solve this differently, by passing storage provider separately everywhere.
    settings2 = ConversationSettings()
    pod2 = await podcast.Podcast.create(settings2)
    assert pod2 is not None, "Failed to create podcast"
    assert isinstance(pod2, podcast.Podcast), f"pod2 is not Podcast but {type(pod2)!r}"

    await pod2.deserialize(ser1)
    assert (
        pod2.name_tag == pod.name_tag
    ), f"pod2.name_tag is {pod2.name_tag!r} but expected {pod.name_tag!r}"

    ser2 = await pod2.serialize()
    assert ser2 is not None, "Failed to serialize podcast"
    assert isinstance(ser2, dict), f"ser2 is not dict but {type(ser2)!r}"
    assert len(ser2) > 0, f"ser2 is empty {ser2!r}"
    assert "semanticRefs" in ser2, f"'semantic_refs' is not a key in {ser2.keys()!r}"
    assert str(ser1) == str(ser2), f"ser1 != ser2"
