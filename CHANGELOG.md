# Typeagent Change Log

## 2026

### 0.5.0 (May 26)

This release focuses on ingestion throughput, robustness, and developer tooling,
with a new streaming ingestion pipeline and substantial performance work.
Thanks go out to:
  - Bernhard Merkle
  - Kevin Turcios
  - Shreejay Kurhade
  - Dhaval Gojiya
  - Tushar Mudi
  - ANGELUSD11
  - dependabot[bot]

#### Core typeagent package
- Deliver a new high-throughput streaming ingestion architecture and a new API
  `add_messages_streaming`, with stronger correctness and broader test coverage
  (#252, #266, #277).
- Move ingestion workflows onto the streaming path (notably email and VTT),
  simplifying duplicate filtering logic (#265, #267, #268, #271).
- Tune embedding model parameters per-model and add supporting benchmark (#228).
- Improve ingestion and query performance via batched SQLite writes,
  batched metadata lookups (removing N+1 paths), overlapped streaming pipeline
  stages, numpy-vectorized embedding lookup, and higher podcast ingest
  concurrency (#230, #232, #234, #248, #260).
- Replace ad-hoc retries with `stamina` and increase retries to better handle
  429s (#255, #245).
- Resolve Azure endpoint/URL parsing issues and regressions, with expanded
  parsing tests (#231, #239, #241, #244).
- Turn async provider accessor methods into sync properties for provider
  attributes (#256).
- Remove dead and obsolete internal code paths
  (`extract_knowledge_for_text_batch_q`, `create_async_openai_client`) (#254,
  #259).
- Rename `kplib.py` to `knowledge_schema.py` (#212).
- Remove runtime `black` dependency: switch to using `pprint`
  (#229, #235, #237).

#### Tools
- Add benchmark scripts under `tools/` (`benchmark_query.py`,
  `benchmark_vectorbase.py`, `benchmark_embeddings.py`,
  `repeat_embedding_benchmarks.py`) (#251, #228, #247).
- Improve streaming ingestion tool ergonomics in `ingest_email.py` and
  `ingest_vtt.py` with explicit batching/concurrency controls and clearer
  per-batch progress reporting (#265, #267, #268).
- Add utility to convert mbox files to `.eml` files (#220).
- Add new `chat_sessions.py` tool to browse/dump VS Code Copilot chat sessions
  (#226, #233).
- Add `make eval` target (#237).

#### Docs
- Document high-level ingestion methods in API docs (#261).

#### Testing and coverage
- Improved test coverage -- somewhat.
- Fixed coverage support on Windows (`.coveragerc`, `make.bat`) (#249, #253).

#### CI and security
- Harden CI workflow permissions and trigger model (`pull_request`, remove
  unnecessary `id-token: write`) (#214).
- Apply workflow security fixes reported by zizmor across CI/CodeQL and release
- workflows (#276).

### 0.4.0 (March 3)

Lots of improvements; the highlights are provider-agnostic model
configuration backed by pydantic_ai, email ingestion, and a major
infrastructure overhaul.

#### Core typeagent package
- Fixed a number of bugs that affected the core query algorithms
- Provider-agnostic chat and embedding model configuration via new
  `model_adapters` module backed by pydantic_ai (#200):
  - Use `provider:model` spec strings,
    e.g. `create_chat_model("openai:gpt-4o")`.
  - Replace `AsyncEmbeddingModel` with `IEmbedder`/`IEmbeddingModel`
    protocols and `CachingEmbeddingModel`.
  - Add `OPENAI_MODEL` and `OPENAI_EMBEDDING_MODEL` envvars
    to override the default chat and embedding models.
- Split `interfaces.py` into separate modules
  (`interfaces_core`, `_indexes`, `_search`, `_serialization`,
  `_storage`) (Bernhard Merkle, #118).
- Make remaining storage-provider APIs async
  (`get/set_conversation_metadata`, `is_source_ingested`, etc.) (#196).
- Fix listeners/recipients confusion in podcast metadata serialization (#174).
- Implement `SqliteRelatedTermsIndex.serialize()` (Rajiv Singh, #115).

#### Email
- New _tools/ingest_email.py_ tool to ingest email
  into a SQLite-backed conversation database (#111).
- Add _tools/mail/outlook_dump.py_ to dump Outlook/Microsoft 365 email
  via the Graph API (Bernhard Merkle, #199).
- Add _tools/mail/mbox_dump.py_ to convert mbox files for ingestion
  (Bernhard Merkle, #198).
- Consolidate mail dump tools under _tools/mail/_ with shared
  _README.md_ (Bernhard Merkle).
- Various ergonomic improvements and fixes (#162, #168, #170).

#### Tools
- Add conversation history to _tools/query.py_ for
  pronoun/reference resolution across multi-turn queries
  (Rajiv Singh, #117).
- Add _tools/load_json.py_ to load JSON-serialized index data
  into a SQLite database; remove `--podcast` flag from
  _tools/query.py_ (#164).

#### Docs
- Improve docs for Azure env vars (#175).
- Add AgentCon 2026 presentation and demo videos
  (Bernhard Merkle, #194, #202).
- VS Code / Pyright plugin setup instructions
  (Bernhard Merkle, #150).

#### Infrastructure
- Changes pyproject.toml to use uv more idiomatically.
  - Local devs write `uv sync` instead of `uv sync --extra dev`.
  - From PyPI (with uv or pip) you can use `typeagent[dev]`
    to install the dev dependencies with the package.
- Move _typeagent/_ to _src/typeagent/_ (Bernhard Merkle, #139).
- Move tests and test data to _tests/_ directory (Bernhard Merkle, #144).
- Move ancillary dirs into subdirs (Bernhard Merkle, #145).
- Introduce `isort` for import sorting.
- Make pyright error on unused variables and imports (#129).
- Add readline support on Windows (#152, Bernhard Merkle).
- Enhance release script to update _uv.lock_ and create release PR
  (Rajiv Singh, #169).

## 2025

### 0.3.3 (Nov 25)

General cleanup and fixes, and the following notable improvements:

#### Docs
- Document `AZURE_OPENAI_ENDPOINT_EMBEDDING` (Gwyneth Peña-Siguenza).
- Add link to PyBay 2025 talk video.

#### Core typeagent package
- Simplify `load_dotenv()`.
- Split embedding requests that are too large (Raphael Wirth).
- Overhaul conversation metadata in storage providers.
- Add extra optional keyword parameters to `create_conversation()`.
- Add `Quantifier` to ingestion schema in addition to `Quantity`.
- Extract knowledge concurrently (max 4 by default) (Kevin Turcios).
- Fixes for `get_multiple()` implementations.
- Tweak defaults in `ConversationSettings`.

#### MCP server
- Pass LLM requests to MCP client instead of calling the OpenAI API.
- Add `--database` option to MCP server.

#### Tools
- The _tools/query.py_ tool now supports `@`-commands. Try `@help`.
- Add _tools/ingest_podcast.py_ (a tool that ingests podcasts).

#### Testing
- Fix coverage support for MCP server test.
- Use an updated "Adrian Tchaikovsky" podcast data dump (Rob Gruen).
- Fix Windows testing issues. Run CI on Windows too (Raphael Wirth).
- Run tests in CI using secrets stored in repo (Rob Gruen).

#### Infrastructure
- Migrate package build from _setuptools_ to _uv_build_.
- Add `install-libatomic` target to `Makefile` (Bernhard Merkle).

### 0.3.2 (Oct 22)

Brown bag release!

- Put `black` back with the runtime dependencies (it's used for debug output).

### 0.3.1 (Oct 22)

- Limit dependencies to what's needed at runtime;
  dev dependencies can be installed separately with
  `uv sync --extra dev`.
- Add `endpoint_envvar` arg to `AsyncEmbeddingModel`
  to allow configuring a non-standard embedding service.

### 0.3.0 (Oct 17)

- First public release, for PyBay '25 talk
