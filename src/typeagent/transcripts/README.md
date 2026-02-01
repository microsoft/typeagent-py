# VTT Transcript Ingestion

This module provides functionality to ingest WebVTT (.vtt) transcript
files into the TypeAgent conversation system. It's designed to be
similar to the podcast ingest functionality but more general-purpose for
various types of transcripts.

## Features

- **WebVTT Format Support**: Ingest standard WebVTT subtitle/caption files
- **Speaker Detection**: Automatically extract speaker names from common patterns:
  - `SPEAKER: dialogue`
  - `[Speaker Name] dialogue`
  - `- Speaker: dialogue`
  - `(Speaker) dialogue`
- **Timestamp Preservation**: Maintains original WebVTT timing information
- **Message Merging**: Option to merge consecutive captions from the same speaker

## Usage

### Basic Ingestion

```python
from dotenv import load_dotenv

from typeagent.transcripts.transcript_ingest import ingest_vtt_transcript
from typeagent.knowpro.convsettings import ConversationSettings

# Load environment variables for API keys from .env file
load_dotenv()

# Create settings (tweak as needed)
settings = ConversationSettings()

# Ingest transcript
transcript = await ingest_vtt_transcript(
    vtt_file_path="my_transcript.vtt",
    settings=settings,
    transcript_name="My Transcript",
    merge_consecutive_same_speaker=True,
)

# Use the transcript
message_count = await transcript.messages.size()
print(f"Ingested {message_count} messages")
```

### Analyzing VTT Files

```python
from typeagent.transcripts.transcript_ingest import (
    get_transcript_duration,
    get_transcript_speakers,
    extract_speaker_from_text,
)

# Get basic information
duration = get_transcript_duration("transcript.vtt")
speakers = get_transcript_speakers("transcript.vtt")

print(f"Duration: {duration/60:.1f} minutes")
print(f"Speakers: {speakers}")

# Test speaker extraction
speaker, text = extract_speaker_from_text("NARRATOR: Once upon a time...")
print(f"Speaker: {speaker}, Text: {text}")
```

### In Tests

```python
import pytest
from fixtures import needs_auth, embedding_model

@pytest.mark.asyncio
async def test_my_transcript(needs_auth, embedding_model):
    settings = ConversationSettings(embedding_model)
    
    transcript = await ingest_vtt_transcript(
        "test.vtt", 
        settings,
        dbname="test.db",
    )
    
    assert await transcript.messages.size() > 0
```

## API Reference

### `ingest_vtt_transcript()`

```python
async def ingest_vtt_transcript(
    vtt_file_path: str,
    settings: ConversationSettings,
    transcript_name: str | None = None,
    start_date: Datetime | None = None,
    merge_consecutive_same_speaker: bool = True,
    dbname: str | None = None,
) -> Transcript:
```

**Parameters:**
- `vtt_file_path`: Path to the WebVTT file
- `settings`: Conversation settings with embedding model
- `transcript_name`: Name for the transcript (defaults to filename)
- `start_date`: Optional start date for timestamp generation
- `merge_consecutive_same_speaker`: Whether to merge consecutive captions from same speaker
- `dbname`: Database name for storage

**Returns:** `Transcript` object with ingested messages

### `get_transcript_duration(vtt_file_path: str) -> float`

Returns the total duration of the transcript in seconds.

### `get_transcript_speakers(vtt_file_path: str) -> set[str]`

Returns a set of all speakers found in the transcript.

### `extract_speaker_from_text(text: str) -> tuple[str | None, str]`

Extracts speaker name from text, returning `(speaker, remaining_text)`.

## WebVTT Format Support

The ingester supports standard WebVTT files with captions:

```webvtt
WEBVTT
Kind: captions
Language: en

00:00:07.599 --> 00:00:10.559
SPEAKER: Hello, this is a test.

00:00:10.560 --> 00:00:15.000
[Another Speaker] This is another line.
```

## Dependencies

- `webvtt-py`: For parsing WebVTT files
- Standard TypeAgent conversation infrastructure
