#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
VTT Transcript Ingestion Tool

This script ingests WebVTT (.vtt) transcript files into a SQLite database
that can be queried using tools/query.py.

Usage:
    python tools/ingest_vtt.py input.vtt --database transcript.db
    python query.py --database transcript.db --query "What was discussed?"
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import webvtt

from typeagent.aitools import utils
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.storage.utils import create_storage_provider
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript_ingest import (
    get_transcript_duration,
    get_transcript_speakers,
    parse_voice_tags,
    webvtt_timestamp_to_seconds,
)
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)
from typeagent.knowpro.universal_message import (
    UNIX_EPOCH,
    format_timestamp_utc,
)
from typeagent.knowpro.convsettings import ConversationSettings


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the VTT ingestion tool."""
    parser = argparse.ArgumentParser(
        description="Ingest WebVTT transcript files into a database for querying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.vtt --database transcript.db
  %(prog)s file1.vtt file2.vtt -d transcript.db --name "Combined Transcript"
  %(prog)s lecture.vtt -d lecture.db --merge
        """,
    )

    parser.add_argument(
        "vtt_files",
        nargs="+",
        help="Path to one or more WebVTT (.vtt) files to ingest",
    )

    parser.add_argument(
        "-d",
        "--database",
        required=True,
        help="Path to the SQLite database file to create/use",
    )

    parser.add_argument(
        "-n",
        "--name",
        help="Name for the transcript (defaults to filename without extension)",
    )

    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge consecutive segments from the same speaker",
    )

    parser.add_argument(
        "--batchsize",
        type=int,
        default=None,
        help="Batch size for knowledge extraction (default: from settings)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )

    return parser


def vtt_timestamp_to_seconds(timestamp: str) -> float:
    """Convert VTT timestamp (HH:MM:SS.mmm) to seconds.

    Args:
        timestamp: VTT timestamp string

    Returns:
        Time in seconds as float
    """
    parts = timestamp.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        VTT timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


async def ingest_vtt_files(
    vtt_files: list[str],
    database: str,
    name: str | None = None,
    merge_consecutive: bool = False,
    verbose: bool = False,
    batchsize: int | None = None,
) -> None:
    """Ingest one or more VTT files into a database."""

    # Validate input files
    for vtt_file in vtt_files:
        if not os.path.exists(vtt_file):
            print(f"Error: VTT file '{vtt_file}' not found", file=sys.stderr)
            sys.exit(1)

    # Database must not exist (ensure clean start)
    if os.path.exists(database):
        print(
            f"Error: Database '{database}' already exists. Please remove it first or use a different filename.",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(f"Ingesting {len(vtt_files)} VTT file(s):")
        for vtt_file in vtt_files:
            print(f"  - {vtt_file}")
        print(f"Target database: {database}")

    # Analyze all VTT files
    if verbose:
        print("\nAnalyzing VTT files...")
    try:
        total_duration = 0.0
        all_speakers = set()
        for vtt_file in vtt_files:
            duration = get_transcript_duration(vtt_file)
            speakers = get_transcript_speakers(vtt_file)
            total_duration += duration
            all_speakers.update(speakers)

            if verbose:
                print(f"  {vtt_file}:")
                print(
                    f"    Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)"
                )
                print(f"    Speakers: {speakers if speakers else 'None detected'}")

        if verbose:
            print(
                f"\nTotal duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)"
            )
            print(
                f"All speakers: {len(all_speakers)} ({all_speakers if all_speakers else 'None detected'})"
            )
    except Exception as e:
        print(f"Error analyzing VTT files: {e}", file=sys.stderr)
        sys.exit(1)

    # Load environment for API access
    if verbose:
        print("Loading environment...")
    utils.load_dotenv()

    # Create conversation settings and storage provider
    if verbose:
        print("Setting up conversation settings...")
    try:
        embedding_model = AsyncEmbeddingModel()
        settings = ConversationSettings(embedding_model)

        # Create storage provider explicitly with the database
        storage_provider = await create_storage_provider(
            settings.message_text_index_settings,
            settings.related_term_index_settings,
            database,
            TranscriptMessage,
        )

        # Update settings to use our storage provider
        settings.storage_provider = storage_provider

        # Override batch size if specified
        if batchsize is not None:
            settings.semantic_ref_index_settings.batch_size = batchsize

        if verbose:
            print("Settings and storage provider configured")
    except Exception as e:
        print(f"Error creating settings: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine transcript name
    if not name:
        if len(vtt_files) == 1:
            name = Path(vtt_files[0]).stem
        else:
            name = "combined-transcript"

    # Import the transcripts
    if verbose:
        print(f"\nParsing VTT files and creating messages...")
    try:
        # Get collections from our storage provider
        msg_coll = await storage_provider.get_message_collection()
        semref_coll = await storage_provider.get_semantic_ref_collection()

        # Database should be empty (we checked it doesn't exist earlier)
        # But verify collections are empty just in case
        if await msg_coll.size() or await semref_coll.size():
            print(
                f"Error: Database already has data.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Process all VTT files and collect messages
        all_messages: list[TranscriptMessage] = []
        time_offset = 0.0  # Cumulative time offset for multiple files

        for file_idx, vtt_file in enumerate(vtt_files):
            if verbose:
                print(f"  Processing {vtt_file}...")
                if file_idx > 0:
                    print(f"    Time offset: {time_offset:.2f} seconds")

            # Parse VTT file
            try:
                vtt = webvtt.read(vtt_file)
            except Exception as e:
                print(
                    f"Error: Failed to parse VTT file {vtt_file}: {e}", file=sys.stderr
                )
                sys.exit(1)

            current_speaker = None
            current_text_chunks = []
            current_start_time = None
            file_max_end_time = 0.0  # Track the maximum end time in this file

            def save_current_message():
                """Helper to save the current message and add to all_messages."""
                if current_text_chunks and current_start_time is not None:
                    combined_text = " ".join(current_text_chunks).strip()
                    if combined_text:
                        # Calculate timestamp from WebVTT start time
                        offset_seconds = webvtt_timestamp_to_seconds(current_start_time)
                        timestamp = format_timestamp_utc(
                            UNIX_EPOCH + timedelta(seconds=offset_seconds)
                        )
                        metadata = TranscriptMessageMeta(
                            speaker=current_speaker,
                            recipients=[],
                        )
                        message = TranscriptMessage(
                            text_chunks=[combined_text],
                            metadata=metadata,
                            timestamp=timestamp,
                        )
                        all_messages.append(message)

            for caption in vtt:
                # Skip empty captions
                if not caption.text.strip():
                    continue

                # Parse raw text for voice tags (handles multiple speakers per cue)
                raw_text = getattr(caption, "raw_text", caption.text)
                voice_segments = parse_voice_tags(raw_text)

                # Convert WebVTT timestamps and apply offset for multi-file continuity
                start_time_seconds = (
                    vtt_timestamp_to_seconds(caption.start) + time_offset
                )
                end_time_seconds = vtt_timestamp_to_seconds(caption.end) + time_offset
                start_time = seconds_to_vtt_timestamp(start_time_seconds)

                # Track the maximum end time for this file
                if end_time_seconds > file_max_end_time:
                    file_max_end_time = end_time_seconds

                # Process each voice segment in this caption
                for speaker, text in voice_segments:
                    if not text.strip():
                        continue

                    # If we should merge consecutive segments from the same speaker
                    if (
                        merge_consecutive
                        and speaker == current_speaker
                        and current_text_chunks
                    ):
                        # Merge with current message
                        current_text_chunks.append(text)
                    else:
                        # Save previous message if it exists
                        save_current_message()

                        # Start new message
                        current_speaker = speaker
                        current_text_chunks = [text] if text.strip() else []
                        current_start_time = start_time

            # Don't forget the last message from this file
            save_current_message()

            if verbose:
                print(f"    Extracted {len(all_messages)} messages so far")
                if file_max_end_time > 0:
                    print(
                        f"    File time range: 0.00s to {file_max_end_time - time_offset:.2f}s (with offset: {time_offset:.2f}s to {file_max_end_time:.2f}s)"
                    )

            # Update time offset for next file: add 5 seconds gap
            if file_max_end_time > 0:
                time_offset = file_max_end_time + 5.0

        # Add all messages to the database in batches with indexing
        if verbose:
            print(f"\nAdding {len(all_messages)} total messages to database...")

        try:
            # Enable knowledge extraction for index building
            settings.semantic_ref_index_settings.auto_extract_knowledge = True

            if verbose:
                print(
                    f"    auto_extract_knowledge = {settings.semantic_ref_index_settings.auto_extract_knowledge}"
                )
                print(
                    f"    batch_size = {settings.semantic_ref_index_settings.batch_size}"
                )

            # Create a Transcript object
            transcript = await Transcript.create(
                settings,
                name=name,
                tags=[name, "vtt-transcript"],
            )

            # Process messages in batches
            batch_size = 10
            successful_count = 0
            start_time = time.time()

            print(
                f"  Processing {len(all_messages)} messages in batches of {batch_size}..."
            )

            for i in range(0, len(all_messages), batch_size):
                batch = all_messages[i : i + batch_size]
                batch_start = time.time()

                result = await transcript.add_messages_with_indexing(batch)

                successful_count += result.messages_added
                batch_time = time.time() - batch_start

                elapsed = time.time() - start_time
                print(
                    f"    {successful_count}/{len(all_messages)} messages | "
                    f"{await semref_coll.size()} refs | "
                    f"{batch_time:.1f}s/batch | "
                    f"{elapsed:.1f}s elapsed"
                )

            if verbose:
                semref_count = await semref_coll.size()
                print(f"  Successfully added {successful_count} messages")
                print(f"  Extracted {semref_count} semantic references")
            else:
                print(
                    f"Imported {successful_count} messages from {len(vtt_files)} file(s) to {database}"
                )

            print("All indexes built successfully")

        except BaseException as e:
            print(f"\nError: Failed to process messages: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()
            sys.exit(1)

    except Exception as e:
        print(f"Error importing transcripts: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Show usage information
    print()
    print("To query the transcript, use:")
    print(
        f"  python tools/query.py --database '{database}' --query 'Your question here'"
    )


def main():
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()

    # Run the ingestion
    asyncio.run(
        ingest_vtt_files(
            vtt_files=args.vtt_files,
            database=args.database,
            name=args.name,
            merge_consecutive=args.merge,
            batchsize=args.batchsize,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
