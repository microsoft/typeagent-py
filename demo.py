# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os

import dotenv

from typeagent.podcasts import podcast


async def main():
    dotenv.load_dotenv(os.path.expanduser("~/TypeAgent/ts/.env"))  # TODO: Only works in dev tree
    # for k, v in os.environ.items():
    #     if "KEY" in k:
    #         print(f"{k}={v!r}")
    pod = await podcast.Podcast.read_from_file("podcast")
    if pod is None:
        print("Failed to read podcast")
        return
    ser1 = pod.serialize()
    pod2 = podcast.Podcast()
    pod2.deserialize(ser1)
    ser2 = pod.serialize()
    if ser2 != ser1:
        print("Serialized data does not match original")
    else:
        print("Serialized data matches original")


asyncio.run(main())
