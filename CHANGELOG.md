# Typeagent Change Log

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
