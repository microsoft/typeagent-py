# TypeAgent Skills

This directory contains comprehensive Claude skills for working with the TypeAgent framework - a structured RAG (Retrieval-Augmented Generation) system for extracting and querying knowledge from conversations, podcasts, transcripts, and emails.

## Available Skills

### 📊 Core Operations

1. **[query-knowledge.md](query-knowledge.md)** - Query the knowledge base
   - Natural language queries
   - Multi-index search with citations
   - Thread and time-based filtering

2. **[extract-knowledge.md](extract-knowledge.md)** - Extract structured knowledge
   - LLM-powered entity, topic, and action extraction
   - Batch processing with concurrent extraction
   - Configurable extraction settings

3. **[search-indexes.md](search-indexes.md)** - Search across specialized indexes
   - 6 index types: semantic refs, properties, message text, timestamp, related terms, threads
   - Parallel multi-index search
   - Result fusion and ranking

### 📥 Data Ingestion

4. **[ingest-podcast.md](ingest-podcast.md)** - Ingest podcast episodes
   - Parse podcast transcripts with speaker detection
   - Automatic knowledge extraction
   - Thread-based organization

5. **[ingest-transcript.md](ingest-transcript.md)** - Ingest VTT transcripts
   - WebVTT format support
   - Timestamp preservation
   - Temporal indexing

6. **[import-emails.md](import-emails.md)** - Import Gmail emails
   - Gmail API integration
   - OAuth authentication
   - Email metadata tracking

### 🔧 Management & Configuration

7. **[manage-conversations.md](manage-conversations.md)** - Create and manage conversations
   - Conversation lifecycle management
   - Message handling
   - Thread organization

8. **[manage-storage.md](manage-storage.md)** - Manage storage backends
   - Memory vs SQLite backends
   - Data migration and backup
   - Storage configuration

9. **[run-mcp-server.md](run-mcp-server.md)** - Run MCP server
   - Model Context Protocol integration
   - Claude Desktop integration
   - MCP tool exposure

### 🧪 Testing & Development

10. **[run-tests.md](run-tests.md)** - Run test suite
    - Unit and integration tests
    - Coverage reporting
    - Type checking with Pyright

11. **[demo-workflow.md](demo-workflow.md)** - Complete demonstration workflows
    - End-to-end examples
    - Best practices
    - Performance benchmarks

## Quick Start

### 1. Query Knowledge Base

```bash
source .venv/bin/activate
python -m tools.query --interactive
```

### 2. Ingest Podcast

```bash
source .venv/bin/activate
python -m tools.ingest_podcast ./my-podcast.txt --database ./data/my-kb.db
```

### 3. Run Tests

```bash
source .venv/bin/activate
make test
```

## Architecture Overview

TypeAgent implements a sophisticated RAG system with:

- **Universal Message Format** - Supports conversations, podcasts, transcripts, emails
- **Six Specialized Indexes** - Optimized for different query patterns
- **Dual Storage Backends** - Memory (fast) and SQLite (persistent)
- **LLM-Powered Extraction** - GPT-4 for knowledge extraction
- **Multi-Index Search** - Parallel search with result fusion
- **Citation Tracking** - Source text location preservation

## The Six Index Types

1. **Semantic Reference Index** - Maps terms/keywords to extracted knowledge
2. **Property Index** - Maps properties (type, category, etc.) to references
3. **Message Text Index** - Embedding-based semantic similarity search
4. **Timestamp Index** - Temporal queries and time-range filtering
5. **Related Terms Index** - Alias resolution and fuzzy matching
6. **Conversation Threads** - Thread-based organization and filtering

## Common Workflows

### Build a Knowledge Base

```bash
# 1. Ingest data sources
python -m tools.ingest_podcast ./podcast1.txt -d ./data/kb.db
python -m tools.ingest_vtt ./meeting.vtt -d ./data/kb.db
python -m typeagent.emails.email_import -d ./data/kb.db

# 2. Query the knowledge base
python -m tools.query -d ./data/kb.db --query "What was discussed about AI?"
```

### Python API Usage

```python
from typeagent.knowpro import create_conversation

# Create conversation
conv = create_conversation(
    name="my-conv",
    backend="sqlite",
    data_dir="./data"
)

# Add messages
conv.add_message({
    "text": "AI is transforming software development.",
    "sender": "Expert"
})

# Query
result = conv.query("What was said about AI?")
print(result.answer)
```

### Run MCP Server for Claude

```bash
# Start server
make mcp

# Or with Python
python -m typeagent.mcp.server
```

Then configure Claude Desktop to use TypeAgent tools.

## Environment Setup

### Prerequisites

- Python 3.12+ (required for modern generic syntax)
- OpenAI API key (for LLM and embeddings)
- Optional: Azure OpenAI, Gmail API credentials

### Installation

```bash
# Install uv package manager (if needed)
make install-uv

# Create virtual environment and install dependencies
make venv

# Activate virtual environment
source .venv/bin/activate

# Run tests to verify installation
make test
```

### Environment Variables

Create a `.env` file:

```bash
# OpenAI (required)
OPENAI_API_KEY=sk-...

# Azure OpenAI (optional)
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...

# Configuration (optional)
TYPEAGENT_DATA_DIR=./data
TYPEAGENT_BACKEND=sqlite
```

## Development Commands

```bash
make venv          # Create virtual environment
make format        # Format code with Black
make check         # Type check with Pyright
make test          # Run pytest
make coverage      # Run tests with coverage
make demo          # Interactive demo
make mcp           # Run MCP server
make clean         # Clean build artifacts
```

## Skill Usage Patterns

Each skill document contains:

- **Overview** - What the skill does
- **Usage** - How to use it
- **Options** - Available parameters and flags
- **Examples** - Real-world usage examples
- **Implementation** - Technical details

To use a skill, open the corresponding `.md` file and follow the instructions.

## Storage Backends

### Memory Backend
- **Use for:** Testing, demos, temporary processing
- **Pros:** Very fast, no persistence overhead
- **Cons:** Data lost on exit, limited by RAM

### SQLite Backend
- **Use for:** Production, large datasets, multi-session
- **Pros:** Persistent, handles large data, reliable
- **Cons:** Slightly slower than memory

## Performance Tips

1. **Batch Processing** - Use larger batch sizes for knowledge extraction
2. **Backend Selection** - Memory for testing, SQLite for production
3. **Disable Extraction** - Skip knowledge extraction for faster ingestion
4. **Index Selection** - Enable only needed indexes
5. **Embedding Cache** - Reuse embeddings when possible

## Troubleshooting

### Python Version Error
```
SyntaxError: expected '('
```
**Solution:** Requires Python 3.12+. Run `make venv` to create environment with correct version.

### Import Errors
```
ImportError: cannot import name 'create_conversation'
```
**Solution:** Install package in editable mode:
```bash
source .venv/bin/activate
pip install -e .
```

### API Key Errors
```
OpenAI API key not found
```
**Solution:** Set `OPENAI_API_KEY` in `.env` file or environment.

## Contributing

When adding new functionality:
1. Add corresponding skill documentation
2. Include usage examples
3. Add tests
4. Update this README

## Resources

- **Documentation:** `/docs/` directory
- **Examples:** `/examples/` directory
- **Tests:** `/test/` directory
- **Tools:** `/tools/` directory

## License

See LICENSE file in repository root.

---

**Total Skills:** 11
**Lines of Documentation:** 1,776
**Test Coverage:** 357 tests
**Supported Python:** 3.12, 3.13, 3.14
