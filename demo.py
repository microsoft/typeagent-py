# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
import os
import time

import dotenv

from typeagent.knowpro.importing import ConversationSettings
from typeagent.podcasts import podcast

parser = argparse.ArgumentParser()
parser.add_argument(
    "filename",
    nargs="?",
    type=str,
    default=os.path.expanduser(
        "~/TypeAgent/python/ta/testdata/Episode_53_AdrianTchaikovsky_index"
    ),
)


async def main():
    dotenv.load_dotenv(
        os.path.expanduser("~/TypeAgent/ts/.env")
    )  # TODO: Only works in dev tree
    # for k, v in os.environ.items():
    #     if "KEY" in k:
    #         print(f"{k}={v!r}")
    args = parser.parse_args()
    print("Create conversation settings...")
    settings = ConversationSettings()
    print(f"Loading {args.filename}...")
    t0 = time.time()
    pod = await podcast.Podcast.read_from_file(args.filename, settings)
    t1 = time.time()
    print(f"Loading took {t1-t0:.3f} seconds")
    if pod is None:
        print("Failed to read podcast")
        return

    book = pod.semantic_ref_index.lookup_term("book")
    print(book)

    ser1 = pod.serialize()
    pod2 = podcast.Podcast(settings=settings)
    pod2.deserialize(ser1)
    ser2 = pod.serialize()
    if ser2 != ser1:
        print("Serialized data does not match original")
    else:
        print("Serialized data matches original")


asyncio.run(main())
