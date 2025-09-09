# TypeAgent Python - AI Coding Agent Instructions

## Project Overview

This is the **Python implementation of TypeAgent's Structured RAG** system - an experimental prototype exploring human-like memory with super-human precision for AI agents. The codebase is a Pythonic translation of the TypeScript `knowPro` packages, implementing a new approach to RAG that extracts structured information rather than relying solely on embeddings.

## Architecture Components

### Core Libraries (`typeagent/`)
- **`knowpro/`** - Structured RAG implementation with semantic ref indexing, query processing, and answer generation
- **`aitools/`** - Azure/OpenAI integration, embeddings, and vector storage
- **`storage/`** - SQLite-based storage providers for persistent conversation data
- **`mcp/`** - Model Context Protocol server implementation
- **`pydai/`** - Query schemas and structured prompting interfaces

### Key Workflow: 4-Stage Query Pipeline
The system processes queries through a standardized pipeline implemented in `tools/utool.py`:

1. **Stage 1**: Natural language → `SearchQuery` schema (via TypeChat translator)
2. **Stage 2**: `SearchQuery` → compiled search expressions
3. **Stage 3**: Execute search expressions → ranked results
4. **Stage 4**: Results + context → natural language answer

## Development Patterns

### Environment Setup
```bash
# Always use uv for dependency management
make venv              # Creates .venv with all dependencies
source .venv/bin/activate  # Manual activation if needed

# Essential commands
make all              # format, check, test, build
make demo             # Interactive query tool
make compare          # Batch evaluation against test data
```

### Environment Variables
Set these for Azure/OpenAI access:
- `AZURE_OPENAI_API_KEY` (or "identity" for managed identity)
- `AZURE_OPENAI_ENDPOINT` (must include api-version parameter)

Load via `utils.load_dotenv()` which looks for `../ts/.env` (shared with TypeScript)

### Testing & Debugging
- **Tests**: Use `pytest` with async fixtures from `test/fixtures.py`
- **Debug levels**: Tools support `--debug1/2/3/4` with values `none|diff|full|skip|nice`
- **Coverage**: `make test` includes coverage collection
- **Evaluation**: Use `tools/vizcmp.py` to compare evaluation results across runs

### Code Conventions (TypeScript → Python)
- **Classes**: 1:1 correspondence with TS interfaces/types
- **Fields**: `camelCase` → `snake_case` (e.g., `messageIndex` → `message_index`)
- **Interfaces**:
  - `I-named` → `Protocol` classes
  - Others → `@dataclass` or Pydantic models
- **Unions**: Preserve as Python unions, string literals become `Literal` types
- **Async**: All storage operations are async even for in-memory providers

### Import Architecture Rules
**Core Principle**: Never import a symbol from another module that imports it. Always import directly from the module that defines it.

- **Bad**: `from searchlang import SearchTermGroupTypes` (when searchlang imports it from interfaces)
- **Good**: `from interfaces import SearchTermGroupTypes` (where it's actually defined)

**Exceptions**:
- Modules with `__all__` - intentional public API
- Explicit export markers: `# For export` comments
- Re-export pattern: `from X import Y as Y`

This prevents circular imports and creates cleaner dependency graphs.

## Key Interfaces & Patterns

### Storage Provider Pattern
```python
# All storage operations are async
class IStorageProvider(Protocol):
    async def get_item(self, ordinal: int) -> T | None
    async def get_slice(self, start: int, end: int) -> list[T]
    # Sequential ordinals, auto-increment IDs
```

### Conversation Structure
```python
# Flattened from TypeScript version
IConversation:
  - message_collection: IMessageCollection
  - semantic_ref_collection: ISemanticRefCollection
  - indexes: IConversationSecondaryIndexes
  # Plus various search & filtering methods
```

### Schema Translation (Pydantic Models)
- Use `Field()` for validation, descriptions removed in recent refactoring
- Support both `default=None` patterns and required fields
- Entity types: concrete, specific (not generic like 'object', 'thing')

## Essential Tools & Entry Points

### `tools/utool.py` - Primary Development Tool
```python
# Interactive mode
make demo
python -m tools.utool

# Batch evaluation
make compare
python -m tools.utool --batch

# Debug specific stages
python -m tools.utool --debug3=nice "your query here"
```

### Evaluation Workflow
1. Create question/answer test data in `testdata/`
2. Run `make compare` to evaluate against known good results
3. Use `tools/vizcmp.py` to visualize score differences across algorithm variants
4. Results stored in `evals/eval-*.txt`

### MCP Server
```bash
.venv/bin/mcp dev typeagent/mcp/server.py
```

## Database & Indexing

### SQLite Schema (see `spec/` folder)
- **Messages**: Core content with chunks, timestamps, metadata
- **SemanticRefs**: Extracted entities, topics, relationships
- **Secondary Indexes**: Message text, properties, timestamps, related terms
- **Auto-increment IDs**: Sequential ordinals for efficient retrieval

### Index Types
- `MessageIndex` - Text search with embeddings
- `PropertyIndex` - Structured facet search (color:blue, author:Bach)
- `SecondaryIndexes` - Related terms, fuzzy matching, temporal ranges

## Integration Points

### Azure AI Integration
```python
# Auth patterns
from typeagent.aitools.auth import get_shared_token_provider
# Embeddings
from typeagent.aitools.embeddings import AsyncEmbeddingModel
# Utils
from typeagent.aitools.utils import load_dotenv, create_translator
```

### TypeChat Integration
```python
# Create translators for schema validation
model = convknowledge.create_typechat_model()
translator = utils.create_translator(model, YourSchemaClass)
```

## Common Debugging Scenarios

### Query Pipeline Issues
1. Check Stage 1 translation: `--debug1=full`
2. Examine compiled expressions: `--debug2=full`
3. Review search results: `--debug3=nice`
4. Inspect answer generation: `--debug4=nice`

### Schema Validation Failures
- Ensure `Field()` usage follows current patterns (check recent removals)
- Verify entity types are specific, not generic
- Check for proper `Literal` vs Union usage

### Performance Issues
- Use `--podcast` for smaller test datasets vs full indexes
- Check SQLite vs memory storage provider selection
- Monitor embedding model choice (test vs production)

## File Patterns

- **`*_schema.py`** - Pydantic models for TypeChat validation
- **`test_*.py`** - pytest-based tests with async fixtures
- **`interfaces.py`** - Protocol definitions (former TS interfaces)
- **`*index.py`** - Various indexing implementations
- **`serialization.py`** - Object persistence helpers

## Current State & TODOs

This is **active experimental code** with frequent refactoring. Key ongoing work:
- Moving from generic to specific entity types
- Flattening conversation architecture
- Improving query language precision
- Performance optimization for larger corpora
- Better integration between action/memory/planning systems

Focus on **working functionality over perfect code** - the goal is exploring Structured RAG concepts, not production software.
