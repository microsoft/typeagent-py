# Query Knowledge Base

Query the TypeAgent knowledge base to retrieve information from processed conversations, podcasts, transcripts, or emails.

## Usage

This skill allows you to:
- Query the knowledge base using natural language
- Search for entities, topics, actions, and relationships
- Retrieve answers with citations and text locations
- Filter by conversation threads or time ranges

## How to use

When the user wants to query the knowledge base:

1. Run the query tool:
```bash
cd /home/user/typeagent-py
python -m tools.query [options]
```

## Options

- `--data-dir PATH` - Directory containing the knowledge base (default: ./data)
- `--backend {memory,sqlite}` - Storage backend to use (default: sqlite)
- `--interactive` - Run in interactive mode
- `--query "QUERY"` - Run a single query
- `--thread NAME` - Filter by conversation thread
- `--verbose` - Enable verbose output

## Examples

Interactive mode:
```bash
python -m tools.query --interactive
```

Single query:
```bash
python -m tools.query --query "What topics were discussed about AI?"
```

Query specific thread:
```bash
python -m tools.query --thread "podcast-1" --query "What was said about machine learning?"
```

## Implementation

The query system uses:
- Multi-index search across 6 specialized indexes
- LLM-powered query translation to structured search
- Result fusion and ranking algorithms
- Fallback to semantic similarity search
- Citation tracking with text locations
