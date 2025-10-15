# Getting Started

## Installation

```sh
$ pip install typeagent
```

You might also want to use a
[virtual environment](https://docs.python.org/3/library/venv.html)
or another tool like [poetry](https://python-poetry.org/)
or [uv](https://docs.astral.sh/uv/), as long as your tool can
install wheels from [PyPI](https://pypi.org).

## "Hello world" ingestion program

### 1. Create a file named `transcript.txt` containing messages to index, e.g.:

```txt
STEVE We should really make a Python library for Structured RAG.
UMESH Who would be a good person to do the Python library?
GUIDO I volunteer to do the Python library. Give me a few months.
```

### 2. Write a small program like this:

```py
from typeagent import create_conversation
from typeagent.transcripts.transcript import (
    TranscriptMessage,
    TranscriptMessageMeta,
)


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
    messages = read_messages("transcript.txt")
    print(f"Indexing {len(messages)} messages...")
    results = await conversation.add_messages_with_indexing(messages)
    print(f"Indexed {results.messages_added} messages.")
    print(f"Got {results.semrefs_added} semantic refs.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

### 3. Set up your environment for using OpenAI

The minimal set of environment variables seems to be:

```sh
export OPENAI_API_KEY=your-very-secret-openai-api-key
export OPENAI_MODEL=gpt-4o
```

(See [Environment Variables](env-vars.md) for more information.)

### 4. Run your program

Expected output is something like:

```txt
0.027s -- Using OpenAI
Indexing 3 messages...
Indexed 3 messages.
Got 26 semantic refs.
```

## "Hello world" query program

TBD.
