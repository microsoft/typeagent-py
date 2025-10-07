# Things to do before Oct 18

Talk at PyBay is on Sat, Oct 18 in SF

## Software

- Rename utool.py to query.py
- Unify Podcast and VTT ingestion (use shared message and metadata classes)?
- Code structure (do podcasts and transcripts need to be under typeagent?)?
- Distinguish between release deps and build/dev deps?

### Specifically for VTT import:

#### MAJOR (must do before talk)

- None

### Minor (can do without)

- Reduce duplication between ingest_vtt.py and typeagent/transcripts/
- `get_transcript_speakers` and `get_transcript_duration` should not re-parse the transcript -- they should just take the parsed vtt object.

### Not doing:

- Fix MCP service (should use host's LLM, not its own)
- Handle embeddings in MCP, even though MCP doesn't support it yet
  - GPT5 suggests to run a separate MCP service for this
  - Batch 128-256 items at a time
  - Explicitly handle truncation by counting tokens
- Handle caching using sha256() of text?

## Documentation

- Getting Started
- Document the high-level API
- Document what should go in `.env` and where it should live
  - And alternatively (first?) what to put in shell env directly
- Document test/build/release process
- Document how to run evaluations (but don't reveal all the data)

## Demos

- Adrian Tchaikovsky Podcast: ready
- Monty Python Episode: ready but need to pick a list of sketches to index
- Email demo: Umesh has a working prototype

## Talk

- Practice in private, timing, updating slides as needed
- Practice run for the team?
- Anticipate questions about (Lazy) GraphRAG?
  The answer is we give equiv/better results without waiting minutes for
  reindexing [in theory]


# Appendix

## Official abstract: "Structured RAG is better than RAG!"

At Microsoft I've been contributing to an open source project
demonstrating what we call Structured RAG.
This is an improvement over the popular AI tactic named RAG (look it up)
that can answer questions over large collections of text or images
better and faster than RAG. We use this as the basis for long-term AI
memory.

I will explain the Structured RAG algorithm and show some demos with
real-world data. I will also discuss the Python library we are releasing
this summer and its API.

## Scratch space for talk drafting

1. Explain Structured RAG (SRAG)

   1. Explain RAG
   2. Explain how SRAG works instead
   3. Show how SRAG is better (how?)

1a. My process
  - Over time using more and more AI (mostly Claude)
  - Latest changes almost entirely written by AI (with my strict supervision :-)

2. Demos

   1. Podcast demo queries (clean up utool.py for this?)
   2. Document demo, show ingest and query (very briefly)
   3. MP movie? Email?

3. Basics for using the library
   1. Install:
      ```sh
      pip install typeagent-py  # Installs typeagent and dependencies
      ```
   2. Create conversation (TENTATIVE):
      ```py
      import typeagent

      conv = typeagent.get_conversation(dbfile="mymemory.sqlite")
      # Could be empty (new) or could contain previously ingested data
      # You can always ingest additional messages
      ```
   3. Ingest messages (TENTATIVE):
      ```py
      for message in ...:  # Source of message strings
          metadata = ...  # Set date/time, speaker(s), listener(s)
          conv.ingest_message(message, metadata)
      ```
   4. Query (TENTATIVE):
      ```py
      request = input("> ")
      answer = conv.query(request)
      print(request)
      ```
   5. Demo using podcast example data

4. Links

- To PyPI project
- To GitHub (microsoft/typeagent-py)
- To docs
