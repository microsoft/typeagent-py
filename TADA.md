# Things to do before Oct 18

Talk at PyBay is on Sat, Oct 18 in SF

## Software

- Test the ingestion pipeline and fix issues
- Don't work on MCP, do that later
  - Fix MCP service (should use host's LLM, not its own)
  - Handle embeddings in MCP, even though MCP doesn't support it yet
    - GPT5 suggests to run a separate MCP service for this
    - Batch 128-256 items at a time
    - Explicitly handle truncation by counting tokens
  - Handle caching using sha256() of text?
- Design and implement high-level API to support ingestion and querying
- Add transactions to ingestion APIs?
- Code structure (does podcasts need to be under typeagent?)
- Move to typeagent-py repo?
- Rename PyPI package name to typeagent?

## Documentation

- Getting Started
- Document the high-level API
- Document the MCP API
- Document what should go in `.env` and where it should live
  - And alternatively what to put in shell env directly
- Document build/release process
- Document how to run evals (but don't reveal all the data)

## Demos

- Podcast demo (done)
- Different podcast?
- VTT (Python Documentary?)
- Documents demo (doesn't look so easy)
- Rob: Monty Python movie script (Rob will track down scripts)
- Email demo?! Maybe Umesh can help?? (Umesh thinks may be too complex)

## Talk

- Re-read abstract to discover essential points (done)
- Write slides
- Make a pretty design for slides?
- Practice in private, timing, updating slides as needed
- Practice run for the team?
- Anticipate questions about (Lazy) GraphRAG


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

2. Demos

   1. Podcast demo queries (clean up utool.py for this?)
   2. Document demo, show ingest and query (very briefly)
   3. MP movie? Email?

3. Basics for using the library
   1. Install:
      ```sh
      pip install typeagent-py  # Installs typeagent and dependencies
      ```
   2. Create conversation:
      ```py
      import typeagent

      conv = typeagent.get_conversation(dbfile="mymemory.sqlite")
      # Could be empty (new) or could contain previously ingested data
      # You can always ingest additional messages
      ```
   3. Ingest messages:
      ```py
      for message in ...:  # Source of message strings
          metadata = ...  # Set date/time, speaker(s), listener(s)
          conv.ingest_message(message, metadata)
      ```
   4. Query:
      ```py
      request = input("> ")
      answer = conv.query(request)
      print(request)
      ```
   5. Demo using podcast example data

4. Links

- To PyPI project
- To GitHub (typeagent-py or TypeAgent/python/ta?)
- To docs

