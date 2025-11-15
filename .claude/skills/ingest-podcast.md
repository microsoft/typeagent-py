# Ingest Podcast

Process and ingest podcast episodes into the TypeAgent knowledge base with automatic knowledge extraction.

## Usage

This skill allows you to:
- Ingest podcast episodes from text/transcript files
- Extract speakers, topics, entities, and actions
- Build semantic indexes for intelligent querying
- Store in memory or SQLite backend

## How to use

When the user wants to ingest a podcast:

1. Run the ingestion tool:
```bash
cd /home/user/typeagent-py
python -m tools.ingest_podcast [options] <podcast_file>
```

## Options

- `<podcast_file>` - Path to the podcast transcript file (required)
- `--data-dir PATH` - Directory to store the knowledge base (default: ./data)
- `--backend {memory,sqlite}` - Storage backend (default: sqlite)
- `--thread NAME` - Conversation thread name for this podcast
- `--extract-knowledge` - Enable knowledge extraction (default: true)
- `--no-extract-knowledge` - Disable knowledge extraction
- `--batch-size N` - Batch size for concurrent extraction (default: 10)

## Examples

Ingest a podcast with knowledge extraction:
```bash
python -m tools.ingest_podcast ./podcasts/episode1.txt --thread "podcast-episode-1"
```

Ingest without knowledge extraction (faster):
```bash
python -m tools.ingest_podcast ./podcasts/episode2.txt --no-extract-knowledge
```

Custom data directory:
```bash
python -m tools.ingest_podcast ./podcasts/episode3.txt --data-dir ./my-data
```

## File Format

The podcast file should be a text file with speaker markers:
```
Speaker 1: Hello and welcome to the show!
Speaker 2: Thanks for having me.
Speaker 1: Today we're discussing AI...
```

## Implementation

The ingestion pipeline:
1. Parse podcast file with speaker detection
2. Convert to universal message format
3. Extract knowledge (entities, topics, actions) using LLMs
4. Build 6 specialized indexes for efficient retrieval
5. Store in chosen backend (memory or SQLite)
