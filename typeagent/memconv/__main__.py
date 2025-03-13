#!/usr/bin/env python3.13
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from datetime import datetime as Datetime
import json
import sys
from typing import cast

from ..knowpro.convindex import ConversationIndex

minver = (3, 12)
assert sys.version_info >= minver, f"Needs Python {minver[0]}.{minver[1]}+"
del minver

# TODO: Get rid of this dummy import.
import typechat

from ..knowpro.interfaces import IndexingEventHandlers, TextLocation
from ..memconv.import_podcasts import import_podcast


async def main():
    parser = argparse.ArgumentParser(description="Import a podcast")
    parser.add_argument("filename", help="The filename to import")
    # TODO: Add more arguments for the import_podcast function.
    args = parser.parse_args()
    pod = import_podcast(args.filename, None, Datetime.now(), 3.0)
    print("Name-Tag:", pod.name_tag)
    print("Tags:", ", ".join(pod.tags))
    for msg in pod.messages:
        print()
        print(msg)

    def on_knowledge_extracted(chunk, knowledge_result) -> bool:
        print("Knowledge extracted:", chunk, knowledge_result)
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

    handler = IndexingEventHandlers(
        on_knowledge_extracted,
        on_embeddings_created,
        on_text_indexed,
    )
    print()
    await pod.build_index(handler)
    if pod.semantic_ref_index is not None:

        data = pod.semantic_ref_index.serialize()
        new = ConversationIndex(data)
        assert new.serialize() == data
        # print(json.dumps(data, indent=2))

    # print(await pod.serialize())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
