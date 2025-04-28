#!/usr/bin/env python3.13
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Check Python version before importing anything else.
import sys

minver = (3, 12)
if sys.version_info < minver:
    sys.exit(
        f"Error: Python {minver[0]}.{minver[1]}+ required, "
        + f"got {sys.version_info[0]}.{sys.version_info[1]}"
    )

import argparse
import os

# Use absolute imports so you can run this as a script file.
from typeagent.aitools import auth
from typeagent.knowpro import importing, serialization
from typeagent.knowpro.interfaces import (
    Datetime,
    IndexingEventHandlers,
    MessageOrdinal,
    TextLocation,
)
from typeagent.podcasts.podcast import Podcast
from typeagent.podcasts.podcast_import import import_podcast


async def main():
    auth.load_dotenv()
    parser = argparse.ArgumentParser(description="Import a podcast")
    parser.add_argument("filename", nargs="?", help="The filename to import")
    # TODO: Add more arguments for the import_podcast function.
    args = parser.parse_args()
    if not args.filename:
        args.filename = os.path.expanduser(
            "~/TypeAgent/python/ta/testdata/FakePodcast.txt"
        )
    settings = importing.ConversationSettings()
    pod = import_podcast(args.filename, None, Datetime.now(), 3.0, settings=settings)
    print()
    print("Name-Tag:", pod.name_tag)
    print("Tags:", ", ".join(pod.tags))
    for msg in pod.messages:
        print()
        print(msg)

    def on_knowledge_extracted(chunk, knowledge_result) -> bool:
        print("Knowledge extracted:", chunk, "\n    ", knowledge_result)
        return True

    def on_embeddings_created(source_texts, batch, batch_start_at) -> bool:
        print("Embeddings extracted:", source_texts)
        return True

    def on_text_indexed(
        text_and_locations: list[tuple[str, TextLocation]],
        batch: list[tuple[str, TextLocation]],
        batch_start_at: int,
    ) -> bool:
        print("Text indexed:", text_and_locations)
        return True

    def on_message_started(message_order: MessageOrdinal) -> bool:
        print("\nMESSAGE STARTED:", message_order)
        return True

    handler = IndexingEventHandlers(
        on_knowledge_extracted,
        on_embeddings_created,
        on_text_indexed,
        on_message_started,
    )

    indexing_result = await pod.build_index(handler)

    print()
    print(indexing_result)
    if indexing_result.semantic_refs is not None:
        if error := indexing_result.semantic_refs.error:
            raise RuntimeError(error)

    print()
    filename = "podcast"
    print(
        f"Dumping to {filename}{serialization.DATA_FILE_SUFFIX}, {filename}{serialization.EMBEDDING_FILE_SUFFIX}..."
    )
    pod.write_to_file(filename)
    print(f"Dump complete.")

    ser1 = pod.serialize()
    pod2 = Podcast(settings=pod.settings)
    pod2.deserialize(ser1)
    ser2 = pod2.serialize()
    if ser1 == ser2:
        print("Serialized data matches original")
    else:
        print("Serialized data does not match original")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
