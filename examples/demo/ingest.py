from dotenv import load_dotenv

from typeagent import create_conversation
from typeagent.transcripts.transcript import TranscriptMessage, TranscriptMessageMeta

load_dotenv()


def read_messages(filename) -> list[TranscriptMessage]:
    messages: list[TranscriptMessage] = []
    with open(filename, "r") as f:
        for line in f:
            # Parse each line into a TranscriptMessage
            speaker, text_chunk = line.split(None, 1)
            message = TranscriptMessage(
                text_chunks=[text_chunk],
                metadata=TranscriptMessageMeta(speaker=speaker),
            )
            messages.append(message)
    return messages


async def main():
    conversation = await create_conversation("demo.db", TranscriptMessage)
    messages = read_messages("testdata.txt")
    print(f"Indexing {len(messages)} messages...")
    results = await conversation.add_messages_with_indexing(messages)
    print(f"Indexed {results.messages_added} messages.")
    print(f"Got {results.semrefs_added} semantic refs.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
