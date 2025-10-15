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

### 1. Create a text file named `transcript.txt`

```txt
STEVE We should really make a Python library for Structured RAG.
UMESH Who would be a good person to do the Python library?
GUIDO I volunteer to do the Python library. Give me a few months.
```

### 2. Create a Python file named `demo.py`

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

The minimal set of environment variables is:

```sh
export OPENAI_API_KEY=your-very-secret-openai-api-key
export OPENAI_MODEL=gpt-4o
```

Some OpenAI setups will require some additional environment variables.
See [Environment Variables](env-vars.md) for more information.
You will also find information there on how to use
Azure-hosted OpenAI models.

### 4. Run your program

```sh
$ python demo.py
```

Expected output looks like:

```txt
0.027s -- Using OpenAI
Indexing 3 messages...
Indexed 3 messages.
Got 26 semantic refs.
```

## "Hello world" query program

### 1. Write this small program

```py
from typeagent import create_conversation
from typeagent.transcripts.transcript import TranscriptMessage


async def main():
    conversation = await create_conversation("demo.db", TranscriptMessage)
    question = "Who volunteered to do the python library?"
    print("Q:", question)
    answer = await conversation.query(question)
    print("A:", answer)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 2. Set up your environment like above

### 3. Run your program

```sh
$ python query.py
```

Expected output looks like:

```txt
0.019s -- Using OpenAI
Q: Who volunteered to do the python library?
A: Guido volunteered to do the Python library.
```

## Next steps

You can study the full documentation for `create_conversation()`
and `conversation.query()` in [High-level API](high-level-api.md).

You can also study the source code at the
[typeagent-py repo](https://github.com/microsoft/typeagent-py).