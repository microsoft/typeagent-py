# Pydantic AI Migration Plan

## Overview

typeagent-py currently uses **TypeChat** as its primary structured-output LLM
library and **pydantic_ai** as an adapter layer for multi-provider model
wiring.  This document proposes a phased plan for replacing TypeChat with
pydantic_ai Agents wherever it makes sense, while preserving the existing
query pipeline architecture.

---

## Current State

### Libraries in play

| Library | Role | Dependency |
|---|---|---|
| `typechat` | Structured LLM output (JSON schema → prompt → validate → repair) | production |
| `pydantic-ai-slim[openai]` | Model adapter layer, embedding adapter, `make_agent()` (unused) | dev-only |
| `openai` | Direct embedding API calls (`AsyncOpenAI` / `AsyncAzureOpenAI`) | production |
| `pydantic` | Dataclass serialization/validation, schema classes | production |

### TypeChat call sites (migration targets)

There are **3 TypeChat translator** patterns, each wrapping a
`pydantic.dataclass` schema:

| Call Site | Schema | Purpose |
|---|---|---|
| `convknowledge.py` → `KnowledgeExtractor.extract()` | `KnowledgeResponse` | Extract entities, actions, topics from messages |
| `searchlang.py` → `search_query_from_language()` | `SearchQuery` | Translate NL question → structured search query |
| `answers.py` → `generate_answer()` / `combine_answers()` | `AnswerResponse` | Generate answer from context |

These 3 translators are created in:
- `ConversationBase.query()` (`conversation_base.py` – lazy init of query + answer translators)
- `KnowledgeExtractor.__init__()` (`convknowledge.py` – knowledge translator)
- `EmailMemorySettings.__init__()` (`email_memory.py` – query + answer translators)
- `MCPTypeChatModel` usage in MCP server (`mcp/server.py`)

### pydantic_ai already used

- `model_adapters.py` — `PydanticAIChatModel` wraps a pydantic_ai `Model`
  back into TypeChat's `TypeChatLanguageModel` interface. This adapter is
  a stepping stone; once TypeChat is gone, it is unnecessary.
- `model_adapters.py` — `PydanticAIEmbeddingModel` wraps `pydantic_ai.Embedder`
  into `IEmbeddingModel`. This is clean and should be kept.
- `utils.py` — `make_agent()` creates a `pydantic_ai.Agent` with structured
  output but **is never called** anywhere in the codebase.

---

## Where Pydantic AI Services Make Sense

### 1. Structured LLM Output (replace TypeChat translators)

**Why**: pydantic_ai `Agent[None, T]` with `output_type=T` provides the same
structured-output flow TypeChat gives (schema → prompt → validate → retry)
but with:
- Native structured-output support (OpenAI JSON mode / tool calling)
- Built-in retries with configurable count
- No separate validator/translator/repair machinery
- Direct Pydantic model validation (our schemas already use `pydantic.dataclass`)
- Multi-provider support via `infer_model()` (25+ providers)
- Built-in logfire observability

**What to replace**:

| Current | Replacement |
|---|---|
| `TypeChatJsonTranslator[KnowledgeResponse]` | `Agent[None, KnowledgeResponse]` |
| `TypeChatJsonTranslator[SearchQuery]` | `Agent[None, SearchQuery]` |
| `TypeChatJsonTranslator[AnswerResponse]` | `Agent[None, AnswerResponse]` |
| `typechat.Result[T]` return type | native return or exception handling |
| `TypeChatLanguageModel` protocol | `pydantic_ai.models.Model` |
| `ModelWrapper` (Azure token refresh) | Handled by pydantic_ai provider |

### 2. Model Configuration / Provider Wiring

**Why**: `create_typechat_model()` manually parses env vars, constructs
OpenAI/Azure clients, handles token refresh. pydantic_ai's `infer_model()`
does all of this from a single `"provider:model"` string.

**Already partially done**: `model_adapters.create_chat_model()` wraps
`infer_model()` → `PydanticAIChatModel` → TypeChat interface.  After
removing TypeChat, this adapter layer collapses.

### 3. Embedding Model Abstraction

**Status**: Already done via `PydanticAIEmbeddingModel`. Keep as-is.

The existing `embeddings.py` (`OpenAIEmbeddingModel`) should remain
available as a fallback for direct OpenAI/Azure embedding calls with
fine-grained batching and token-aware truncation that pydantic_ai's
`Embedder` does not provide.

### 4. MCP Server LLM Routing

**Why**: `MCPTypeChatModel` converts TypeChat prompts to MCP sampling
messages. With pydantic_ai, this could be replaced by a custom
pydantic_ai `Model` implementation that routes through MCP, eliminating
the TypeChat intermediate format.

### 5. Observability / Tracing

**Why**: `setup_logfire()` manually instruments pydantic_ai and httpx.
With pydantic_ai as the primary LLM layer, logfire integration is
native — every agent call is automatically traced with input/output,
token usage, retries.

---

## Where Pydantic AI Does NOT Make Sense

- **Search execution** (`search.py`, `searchlib.py`): Pure index queries,
  no LLM calls. No change needed.
- **Data ingestion** (`podcast_ingest.py`, `transcript_ingest.py`,
  `email_import.py`): Regex/MIME parsing, no LLM calls. No change.
- **Core data structures** (`interfaces_core.py`, `kplib.py`): Pydantic
  dataclasses for entities, actions, etc. Keep as-is (they become
  pydantic_ai `output_type` directly).
- **Token counting / truncation** (`embeddings.py`): tiktoken-based logic
  specific to OpenAI. Keep as-is.

---

## Migration Plan

### Phase 0: Preparation

- [ ] Move `pydantic-ai-slim[openai]` from dev dependency to production
  dependency in `pyproject.toml`.
- [ ] Decide on Azure identity integration approach: pydantic_ai's
  `AzureProvider` vs keeping `AzureTokenProvider` wrapper.
- [ ] Add integration test fixtures that verify structured output with
  pydantic_ai Agents (can start from the unused `make_agent()` pattern).

### Phase 1: Agent Abstraction Layer

Build a thin abstraction that wraps pydantic_ai `Agent` to match the
current call patterns, making the rest of the migration mechanical.

- [ ] Create `aitools/agents.py` with:
  - `create_agent(model_spec, output_type, system_prompt, retries)` factory
  - Agent wrapper that returns `typechat.Result[T]` for backward compat
    during transition
  - Azure identity token refresh support (if not handled by provider)
- [ ] Update `IKnowledgeExtractor` protocol to accept either TypeChat or
  Agent-based implementations.

### Phase 2: Knowledge Extraction Migration

Replace `KnowledgeExtractor` in `convknowledge.py`.

- [ ] Create `PydanticAIKnowledgeExtractor` as alternative implementation of
  `IKnowledgeExtractor`.
- [ ] Port the custom system prompt from `create_request_prompt()` to
  pydantic_ai `system_prompt`.
- [ ] Wire through `ConversationSettings` so callers can choose the
  implementation.
- [ ] Validate output equivalence against existing TypeChat results on
  test podcasts.
- [ ] Remove `TypeChatJsonTranslator[KnowledgeResponse]` once validated.

### Phase 3: Search Query Translation Migration

Replace translator in `searchlang.py`.

- [ ] Create `SearchQueryAgent` that wraps `Agent[None, SearchQuery]`.
- [ ] Port `prompt_preamble` / time-range context to pydantic_ai
  `system_prompt` or `user_prompt` construction.
- [ ] Update `search_query_from_language()` to use the agent.
- [ ] Update `SearchQueryTranslator` type alias.
- [ ] Test against existing query test suite (`test_query.py`,
  `test_searchlib.py`, etc.).

### Phase 4: Answer Generation Migration

Replace translator in `answers.py`.

- [ ] Create `AnswerAgent` that wraps `Agent[None, AnswerResponse]`.
- [ ] Port `create_question_prompt()` and `create_context_prompt()` to
  pydantic_ai prompt sections.
- [ ] Update `generate_answer()` and `combine_answers()`.
- [ ] Test against existing answer tests.

### Phase 5: Wiring / Cleanup

- [ ] Update `ConversationBase.query()` to create pydantic_ai agents
  instead of TypeChat translators.
- [ ] Update `EmailMemorySettings` to use pydantic_ai agents.
- [ ] Port `MCPTypeChatModel` to a pydantic_ai custom `Model`
  implementation.
- [ ] Remove `PydanticAIChatModel` adapter (no longer needed).
- [ ] Remove `create_typechat_model()`, `create_translator()`, and
  `ModelWrapper` from `utils.py`.
- [ ] Remove unused `make_agent()` from `utils.py`.
- [ ] Remove `typechat` from production dependencies.
- [ ] Update `convsettings.py` to accept a `pydantic_ai.models.Model`
  (or `str` spec) for chat model configuration.
- [ ] Update all imports and type annotations.

### Phase 6: Enhanced Capabilities (Optional)

Once on pydantic_ai, these become easy to add:

- [ ] **Multi-provider support**: Switch models per task (e.g. cheaper
  model for knowledge extraction, stronger model for answers).
- [ ] **Streaming answers**: pydantic_ai supports streamed structured
  output for real-time UX.
- [ ] **Tool use in agents**: Give the answer agent tools to look up
  additional context on demand (agentic RAG).
- [ ] **Conversation memory**: Use pydantic_ai's message history for
  multi-turn query refinement.
- [ ] **Cost tracking**: pydantic_ai exposes token usage per call.

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Schema compatibility | All 3 schemas already use `pydantic.dataclass` — direct `output_type` use |
| Prompt regression | Port prompts 1:1; compare outputs on test corpus before switching |
| TypeChat repair loop loss | pydantic_ai `retries=N` provides equivalent retry; native structured output reduces need for repair |
| Azure identity auth | Test `AzureProvider` with identity tokens; keep `AzureTokenProvider` as fallback |
| MCP server compat | Build custom pydantic_ai `Model` for MCP sampling before removing TypeChat |
| `CamelCaseField` serialization | Verify pydantic_ai preserves alias config from the custom dataclass wrapper |

---

## Estimated Effort

| Phase | Effort | Dependencies |
|---|---|---|
| Phase 0 (Preparation) | 1 day | None |
| Phase 1 (Abstraction) | 2 days | Phase 0 |
| Phase 2 (Knowledge) | 2 days | Phase 1 |
| Phase 3 (Search Query) | 1-2 days | Phase 1 |
| Phase 4 (Answers) | 1-2 days | Phase 1 |
| Phase 5 (Cleanup) | 2 days | Phases 2-4 |
| Phase 6 (Enhanced) | Ongoing | Phase 5 |

**Total core migration: ~9-11 days**
