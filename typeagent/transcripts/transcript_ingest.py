# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
from datetime import timedelta

import webvtt

from ..knowpro.convsettings import ConversationSettings
from ..knowpro.interfaces import Datetime
from ..knowpro.universal_message import (
    UNIX_EPOCH,
    ConversationMessage,
    ConversationMessageMeta,
    format_timestamp_utc,
)
from ..storage.utils import create_storage_provider
from .transcript import Transcript


def webvtt_timestamp_to_seconds(timestamp: str) -> float:
    """Convert WebVTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = timestamp.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    else:
        return float(timestamp)


def extract_speaker_from_text(text: str) -> tuple[str | None, str]:
    """
    Extract speaker name from text if present.
    Returns tuple of (speaker_name, remaining_text).

    Handles patterns like:
    - "SPEAKER: text"
    - "Speaker Name: text"
    - "[Speaker] text"
    - "- Speaker: text"
    """
    text = text.strip()

    # Pattern 1: "SPEAKER:" or "Speaker Name:"
    speaker_colon_match = re.match(r"^([A-Z][A-Z\s]*?):\s*(.*)$", text)
    if speaker_colon_match:
        speaker = speaker_colon_match.group(1).strip()
        remaining = speaker_colon_match.group(2).strip()
        return speaker, remaining

    # Pattern 2: "[Speaker]" or "(Speaker)"
    bracket_match = re.match(r"^[\[\(]([^)\]]+)[\]\)]\s*(.*)$", text)
    if bracket_match:
        speaker = bracket_match.group(1).strip()
        remaining = bracket_match.group(2).strip()
        return speaker, remaining

    # Pattern 3: "- Speaker:"
    dash_match = re.match(r"^-\s*([^:]+):\s*(.*)$", text)
    if dash_match:
        speaker = dash_match.group(1).strip()
        remaining = dash_match.group(2).strip()
        return speaker, remaining

    # No speaker pattern found
    return None, text


def parse_voice_tags(raw_text: str) -> list[tuple[str | None, str]]:
    """
    Parse WebVTT voice tags from raw caption text.

    Returns a list of (speaker, text) tuples, one for each voice segment in the caption.

    WebVTT voice tags can be in the format:
    - <v Speaker>Text</v>
    - <v Speaker>Text (no closing tag)

    Multiple voice tags can exist in a single caption, and this function extracts all of them.

    Args:
        raw_text: Raw caption text that may contain <v> tags

    Returns:
        List of (speaker, text) tuples. If no voice tags found, returns [(None, raw_text)]
    """
    # Pattern to match <v Speaker>Text</v> or <v Speaker>Text
    # Captures speaker name and the text that follows until the next tag or end
    voice_pattern = r"<v\s+([^>]+)>([^<]*(?:</v>)?)"

    matches = list(re.finditer(voice_pattern, raw_text, re.IGNORECASE))

    if not matches:
        # No voice tags found, return the text as-is with no speaker
        return [(None, raw_text.strip())]

    results = []
    for match in matches:
        speaker = match.group(1).strip()
        text = match.group(2).strip()
        # Remove closing </v> tag if present
        text = re.sub(r"</v>\s*$", "", text, flags=re.IGNORECASE).strip()
        if text:  # Only add non-empty text
            results.append((speaker, text))

    return results if results else [(None, raw_text.strip())]


async def ingest_vtt_transcript(
    vtt_file_path: str,
    settings: ConversationSettings,
    transcript_name: str | None = None,
    start_date: Datetime | None = None,
    merge_consecutive_same_speaker: bool = True,
    use_text_based_speaker_detection: bool = False,
    dbname: str | None = None,
) -> Transcript:
    """
    Import a WebVTT transcript file into a Transcript object.

    Args:
        vtt_file_path: Path to the .vtt file
        settings: Conversation settings
        transcript_name: Name for the transcript (defaults to filename)
        start_date: Base datetime for absolute timestamps.
                   If None, uses Unix epoch (1970-01-01 00:00:00 UTC),
                   preserving relative timing while signaling that the actual
                   date is unknown (Unix "timestamp left at zero" convention).
        merge_consecutive_same_speaker: Whether to merge consecutive captions from same speaker
        use_text_based_speaker_detection: Whether to parse speaker names from text patterns (default: False)
                                          When False, only WebVTT <v> voice tags are used for speaker detection
        dbname: Database name

    Returns:
        Transcript object with imported data
    """
    # Parse the VTT file
    try:
        vtt = webvtt.read(vtt_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse VTT file {vtt_file_path}: {e}")

    if not transcript_name:
        transcript_name = os.path.splitext(os.path.basename(vtt_file_path))[0]

    # Use Unix epoch if no start_date provided (Easter egg!)
    base_date = start_date if start_date is not None else UNIX_EPOCH

    messages: list[ConversationMessage] = []
    current_speaker = None
    current_text_chunks = []
    current_start_time = None  # Keep for determining message start

    for caption in vtt:
        # Skip empty captions
        if not caption.text.strip():
            continue

        # Parse raw text for voice tags (handles multiple speakers per cue)
        raw_text = getattr(caption, "raw_text", caption.text)
        voice_segments = parse_voice_tags(raw_text)

        # Optionally fallback to text-based speaker detection for segments without speaker
        if use_text_based_speaker_detection:
            processed_segments = []
            for speaker, text in voice_segments:
                if speaker is None:
                    speaker, text = extract_speaker_from_text(text)
                processed_segments.append((speaker, text))
            voice_segments = processed_segments

        # Convert WebVTT timestamps
        start_time = caption.start

        # Process each voice segment in this caption
        for speaker, text in voice_segments:
            if not text.strip():
                continue

            # If we should merge consecutive captions from the same speaker
            if (
                merge_consecutive_same_speaker
                and speaker == current_speaker
                and current_text_chunks
            ):
                # Merge with current message (keep first caption's start time)
                current_text_chunks.append(text)
            else:
                # Save previous message if it exists
                if current_text_chunks and current_start_time is not None:
                    combined_text = " ".join(current_text_chunks).strip()
                    if combined_text:  # Only add non-empty messages
                        # Calculate absolute timestamp from WebVTT offset
                        offset_seconds = webvtt_timestamp_to_seconds(current_start_time)
                        timestamp = format_timestamp_utc(
                            base_date + timedelta(seconds=offset_seconds)
                        )

                        metadata = ConversationMessageMeta(
                            speaker=current_speaker,
                            recipients=[],  # Transcripts have no specific recipients
                        )
                        message = ConversationMessage(
                            text_chunks=[combined_text],
                            metadata=metadata,
                            timestamp=timestamp,  # Set during ingestion!
                        )
                        messages.append(message)

                # Start new message
                current_speaker = speaker
                current_text_chunks = [text] if text.strip() else []
                current_start_time = start_time

    # Don't forget the last message
    if current_text_chunks and current_start_time is not None:
        combined_text = " ".join(current_text_chunks).strip()
        if combined_text:
            # Calculate absolute timestamp from WebVTT offset
            offset_seconds = webvtt_timestamp_to_seconds(current_start_time)
            timestamp = format_timestamp_utc(
                base_date + timedelta(seconds=offset_seconds)
            )

            metadata = ConversationMessageMeta(
                speaker=current_speaker,
                recipients=[],
            )
            message = ConversationMessage(
                text_chunks=[combined_text],
                metadata=metadata,
                timestamp=timestamp,
            )
            messages.append(message)

    # Create storage provider
    provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        dbname,
        ConversationMessage,
    )
    # Attach provider to settings to prevent garbage collection
    settings.storage_provider = provider

    msg_coll = await provider.get_message_collection()
    semref_coll = await provider.get_semantic_ref_collection()

    # Create transcript first
    transcript = await Transcript.create(
        settings,
        name=transcript_name,
        tags=[transcript_name, "vtt-transcript"],
    )

    # Add messages with indexing to build embeddings (supports incremental import)
    await transcript.add_messages_with_indexing(messages)

    # No more generate_timestamps() - timestamps are set during ingestion!

    return transcript


def get_transcript_speakers(
    vtt_file_path: str, use_text_based_detection: bool = False
) -> set[str]:
    """
    Extract all unique speakers from a VTT file.

    Args:
        vtt_file_path: Path to the .vtt file
        use_text_based_detection: Whether to parse speaker names from text patterns (default: False)
                                   When False, only WebVTT <v> voice tags are used

    Returns:
        Set of speaker names found in the transcript
    """
    try:
        vtt = webvtt.read(vtt_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse VTT file {vtt_file_path}: {e}")

    speakers = set()
    for caption in vtt:
        # Parse raw text for voice tags (handles multiple speakers per cue)
        raw_text = getattr(caption, "raw_text", caption.text)
        voice_segments = parse_voice_tags(raw_text)

        # Optionally fallback to text-based speaker detection
        if use_text_based_detection:
            for speaker, text in voice_segments:
                if speaker is None:
                    speaker, _ = extract_speaker_from_text(text)
                if speaker:
                    speakers.add(speaker)
        else:
            for speaker, _ in voice_segments:
                if speaker:
                    speakers.add(speaker)

    return speakers


def get_transcript_duration(vtt_file_path: str) -> float:
    """
    Get the total duration of a VTT transcript in seconds.

    Args:
        vtt_file_path: Path to the .vtt file

    Returns:
        Duration in seconds
    """
    try:
        vtt = webvtt.read(vtt_file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse VTT file {vtt_file_path}: {e}")

    if not vtt:
        return 0.0

    # Find the last caption with content
    last_caption = None
    for caption in reversed(vtt):
        if caption.text.strip():
            last_caption = caption
            break

    if last_caption:
        return webvtt_timestamp_to_seconds(last_caption.end)
    else:
        return 0.0
