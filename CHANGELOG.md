# Typeagent Change Log

## 2025

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
