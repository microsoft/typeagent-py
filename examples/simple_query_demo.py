#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Simple demo of the conversation.query() method.

This demonstrates the end-to-end query pattern:
    question = input("typeagent> ")
    answer = await conv.query(question)
    print(answer)
"""

import asyncio

from dotenv import load_dotenv

from typeagent import create_conversation
from typeagent.transcripts.transcript import TranscriptMessage, TranscriptMessageMeta


async def main():
    """Demo the simple query API."""
    # Load API keys
    load_dotenv()

    # Create a conversation with some sample content
    print("Creating conversation...")
    conv = await create_conversation(
        None,
        TranscriptMessage,
        name="Demo Conversation",
    )

    # Add some sample messages
    messages = [
        TranscriptMessage(
            text_chunks=["Welcome to the Python programming tutorial."],
            metadata=TranscriptMessageMeta(speaker="Instructor"),
        ),
        TranscriptMessage(
            text_chunks=["Today we'll learn about async/await in Python."],
            metadata=TranscriptMessageMeta(speaker="Instructor"),
        ),
        TranscriptMessage(
            text_chunks=["Python is a great language for beginners and experts alike."],
            metadata=TranscriptMessageMeta(speaker="Instructor"),
        ),
        TranscriptMessage(
            text_chunks=["The async keyword is used to define asynchronous functions."],
            metadata=TranscriptMessageMeta(speaker="Instructor"),
        ),
        TranscriptMessage(
            text_chunks=[
                "You use await to wait for asynchronous operations to complete."
            ],
            metadata=TranscriptMessageMeta(speaker="Instructor"),
        ),
    ]

    print("Adding messages and building indexes...")
    result = await conv.add_messages_with_indexing(messages)
    print(f"Conversation ready with {await conv.messages.size()} messages.")
    print(
        f"Added {result.messages_added} messages, {result.semrefs_added} semantic refs"
    )

    # Check indexes
    if conv.secondary_indexes:
        if conv.secondary_indexes.message_index:
            msg_index_size = await conv.secondary_indexes.message_index.size()
            print(f"Message index has {msg_index_size} entries")
    print()

    # Interactive query loop
    print("You can now ask questions about the conversation.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question: str = input("typeagent> ")
            if not question.strip():
                continue
            if question.strip().lower() in ("quit", "exit", "q"):
                break

            # This is the simple API pattern
            answer: str = await conv.query(question)
            print(answer)
            print()

        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())
