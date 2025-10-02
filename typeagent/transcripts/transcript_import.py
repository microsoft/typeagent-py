# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
from typing import Optional

import webvtt

from ..knowpro.convsettings import ConversationSettings
from ..knowpro.interfaces import Datetime
from ..storage.utils import create_storage_provider
from .transcript import Transcript, TranscriptMessage, TranscriptMessageMeta


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


async def import_vtt_transcript(
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
        start_date: Optional start date for timestamp generation
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

    messages: list[TranscriptMessage] = []
    current_speaker = None
    current_text_chunks = []
    current_start_time = None
    current_end_time = None

    for caption in vtt:
        # Skip empty captions
        if not caption.text.strip():
            continue

        # Get speaker from webvtt voice attribute
        speaker = getattr(caption, "voice", None)

        # Optionally fallback to text-based speaker detection
        if speaker is None and use_text_based_speaker_detection:
            # Fallback to text parsing for non-standard voice formats
            speaker, text = extract_speaker_from_text(caption.text)
        else:
            # Use the cleaned text (voice tags already stripped by webvtt-py)
            text = caption.text.strip()

        # Convert WebVTT timestamps
        start_time = caption.start
        end_time = caption.end

        # If we should merge consecutive captions from the same speaker
        if (
            merge_consecutive_same_speaker
            and speaker == current_speaker
            and current_text_chunks
        ):
            # Merge with current message
            current_text_chunks.append(text)
            current_end_time = end_time  # Update end time
        else:
            # Save previous message if it exists
            if current_text_chunks:
                combined_text = " ".join(current_text_chunks).strip()
                if combined_text:  # Only add non-empty messages
                    metadata = TranscriptMessageMeta(
                        speaker=current_speaker,
                        start_time=current_start_time,
                        end_time=current_end_time,
                    )
                    message = TranscriptMessage(
                        text_chunks=[combined_text], metadata=metadata
                    )
                    messages.append(message)

            # Start new message
            current_speaker = speaker
            current_text_chunks = [text] if text.strip() else []
            current_start_time = start_time
            current_end_time = end_time

    # Don't forget the last message
    if current_text_chunks:
        combined_text = " ".join(current_text_chunks).strip()
        if combined_text:
            metadata = TranscriptMessageMeta(
                speaker=current_speaker,
                start_time=current_start_time,
                end_time=current_end_time,
            )
            message = TranscriptMessage(text_chunks=[combined_text], metadata=metadata)
            messages.append(message)

    # Create storage provider
    provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        dbname,
        TranscriptMessage,
    )
    msg_coll = await provider.get_message_collection()
    semref_coll = await provider.get_semantic_ref_collection()
    if await msg_coll.size() or await semref_coll.size():
        raise RuntimeError(f"{dbname!r} already has messages or semantic refs.")

    await msg_coll.extend(messages)

    # Create transcript
    transcript = await Transcript.create(
        settings,
        name_tag=transcript_name,
        messages=msg_coll,
        tags=[transcript_name, "vtt-transcript"],
        semantic_refs=semref_coll,
    )

    # Generate timestamps if start_date provided
    if start_date:
        # Calculate duration from VTT timestamps if available
        if messages and messages[-1].metadata.end_time:
            last_end_seconds = webvtt_timestamp_to_seconds(
                messages[-1].metadata.end_time
            )
            duration_minutes = last_end_seconds / 60.0
        else:
            duration_minutes = 60.0  # Default fallback
        await transcript.generate_timestamps(start_date, duration_minutes)

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
        # Get speaker from webvtt voice attribute
        speaker = getattr(caption, "voice", None)

        # Optionally fallback to text-based speaker detection
        if speaker is None and use_text_based_detection:
            speaker, _ = extract_speaker_from_text(caption.text)

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
