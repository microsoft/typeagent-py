# Ingest Video/Audio Transcript

Process and ingest VTT (WebVTT) transcript files into the TypeAgent knowledge base.

## Usage

This skill allows you to:
- Ingest VTT/WebVTT transcript files from videos or audio
- Extract timestamps and speaker information
- Perform knowledge extraction on transcript content
- Enable temporal queries with timestamp indexing

## How to use

When the user wants to ingest a transcript:

1. Run the transcript ingestion tool:
```bash
cd /home/user/typeagent-py
python -m tools.ingest_vtt [options] <vtt_file>
```

## Options

- `<vtt_file>` - Path to the VTT transcript file (required)
- `--data-dir PATH` - Directory to store the knowledge base (default: ./data)
- `--backend {memory,sqlite}` - Storage backend (default: sqlite)
- `--thread NAME` - Conversation thread name for this transcript
- `--extract-knowledge` - Enable knowledge extraction (default: true)
- `--no-extract-knowledge` - Disable knowledge extraction
- `--batch-size N` - Batch size for concurrent extraction (default: 10)

## Examples

Ingest a VTT transcript:
```bash
python -m tools.ingest_vtt ./transcripts/meeting.vtt --thread "team-meeting-2024-01"
```

Ingest multiple transcripts to same thread:
```bash
python -m tools.ingest_vtt ./transcripts/part1.vtt --thread "conference"
python -m tools.ingest_vtt ./transcripts/part2.vtt --thread "conference"
```

## File Format

VTT (WebVTT) format:
```
WEBVTT

00:00:01.000 --> 00:00:05.000
Hello everyone, welcome to the meeting.

00:00:06.000 --> 00:00:12.000
Today we'll discuss the project timeline.
```

## Implementation

The transcript ingestion pipeline:
1. Parse VTT file to extract timestamps and text
2. Convert to universal message format with temporal metadata
3. Extract knowledge using LLM-powered extraction
4. Build temporal index for time-based queries
5. Build semantic indexes for content queries
6. Store in chosen backend
