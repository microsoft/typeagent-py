# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os

import dotenv

from typeagent.knowpro import convknowledge


async def main():
    dotenv.load_dotenv(os.path.expanduser("~/TypeAgent/ts/.env"))  # TODO: Only works in dev tree
    # for k, v in os.environ.items():
    #     if "KEY" in k:
    #         print(f"{k}={v!r}")
    ke = convknowledge.KnowledgeExtractor()
    print(await ke.extract("There is a book about hobbits called the Lord of the Rings."))


asyncio.run(main())
