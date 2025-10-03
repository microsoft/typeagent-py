#!/usr/bin/env python3
"""
VTT Transcript Ingestion Tool

This script ingests WebVTT (.vtt) transcript files into a SQLite database
that can be queried using tools/utool.py.

Usage:
    python tools/ingest_vtt.py input.vtt --database transcript.db
    pyt    await ingest_vtt_file(
        args.vtt_file,
        args.database,
        name=args.name,
        start_date=args.start_date,
        merge_consecutive=not args.no_merge,
        use_text_speaker_detection=args.use_text_speaker_detection,
        build_index=args.build_index,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )utool.py --sqlite-db transcript.db --question "What was discussed?"
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

import webvtt

from typeagent.aitools import utils
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.storage.utils import create_storage_provider
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.transcripts.transcript_import import (
    extract_speaker_from_text,
    get_transcript_speakers,
    get_transcript_duration,
    parse_voice_tags,
)
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces import Datetime
from typeagent.knowpro import convknowledge
from typeagent.knowpro.messageutils import get_message_chunk_batch
from typeagent.storage.memory import semrefindex


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the VTT ingestion tool."""
    parser = argparse.ArgumentParser(
        description="Ingest WebVTT transcript files into a database for querying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.vtt --database transcript.db
  %(prog)s file1.vtt file2.vtt -d transcript.db --name "Combined Transcript"
  %(prog)s lecture.vtt -d lecture.db --start-date "2024-10-01T09:00:00"
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
        "--start-date",
        help="Start date/time for the transcript (ISO format: YYYY-MM-DDTHH:MM:SS)",
    )

    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Don't merge consecutive captions from the same speaker",
    )

    parser.add_argument(
        "--use-text-speaker-detection",
        action="store_true",
        help="Enable text-based speaker detection (e.g., 'SPEAKER:', '[Name]'). "
        "By default, only WebVTT <v> voice tags are used for speaker detection.",
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
    start_date: str | None = None,
    merge_consecutive: bool = True,
    use_text_speaker_detection: bool = False,
    verbose: bool = False,
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
            speakers = get_transcript_speakers(
                vtt_file, use_text_based_detection=use_text_speaker_detection
            )
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

        if verbose:
            print("Settings and storage provider configured")
    except Exception as e:
        print(f"Error creating settings: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse start date if provided
    start_datetime = None
    if start_date:
        try:
            start_datetime = Datetime.fromisoformat(start_date)
        except ValueError:
            print(
                f"Error: Invalid start date format '{start_date}'. Use ISO format: YYYY-MM-DDTHH:MM:SS",
                file=sys.stderr,
            )
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
            current_end_time = None
            file_max_end_time = 0.0  # Track the maximum end time in this file

            for caption in vtt:
                # Skip empty captions
                if not caption.text.strip():
                    continue

                # Parse raw text for voice tags (handles multiple speakers per cue)
                raw_text = getattr(caption, "raw_text", caption.text)
                voice_segments = parse_voice_tags(raw_text)

                # Optionally fallback to text-based speaker detection for segments without speaker
                if use_text_speaker_detection:
                    processed_segments = []
                    for speaker, text in voice_segments:
                        if speaker is None:
                            speaker, text = extract_speaker_from_text(text)
                        processed_segments.append((speaker, text))
                    voice_segments = processed_segments

                # Convert WebVTT timestamps and apply offset for multi-file continuity
                start_time_seconds = (
                    vtt_timestamp_to_seconds(caption.start) + time_offset
                )
                end_time_seconds = vtt_timestamp_to_seconds(caption.end) + time_offset
                start_time = seconds_to_vtt_timestamp(start_time_seconds)
                end_time = seconds_to_vtt_timestamp(end_time_seconds)

                # Track the maximum end time for this file
                if end_time_seconds > file_max_end_time:
                    file_max_end_time = end_time_seconds

                # Process each voice segment in this caption
                for speaker, text in voice_segments:
                    if not text.strip():
                        continue

                    # If we should merge consecutive captions from the same speaker
                    if (
                        merge_consecutive
                        and speaker == current_speaker
                        and current_text_chunks
                    ):
                        # Merge with current message
                        current_text_chunks.append(text)
                        current_end_time = end_time
                    else:
                        # Save previous message if it exists
                        if current_text_chunks:
                            combined_text = " ".join(current_text_chunks).strip()
                            if combined_text:
                                metadata = TranscriptMessageMeta(
                                    speaker=current_speaker,
                                    start_time=current_start_time,
                                    end_time=current_end_time,
                                )
                                message = TranscriptMessage(
                                    text_chunks=[combined_text], metadata=metadata
                                )
                                all_messages.append(message)

                        # Start new message
                        current_speaker = speaker
                        current_text_chunks = [text] if text.strip() else []
                        current_start_time = start_time
                        current_end_time = end_time

            # Don't forget the last message from this file
            if current_text_chunks:
                combined_text = " ".join(current_text_chunks).strip()
                if combined_text:
                    metadata = TranscriptMessageMeta(
                        speaker=current_speaker,
                        start_time=current_start_time,
                        end_time=current_end_time,
                    )
                    message = TranscriptMessage(
                        text_chunks=[combined_text], metadata=metadata
                    )
                    all_messages.append(message)

            if verbose:
                print(f"    Extracted {len(all_messages)} messages so far")
                if file_max_end_time > 0:
                    print(
                        f"    File time range: 0.00s to {file_max_end_time - time_offset:.2f}s (with offset: {time_offset:.2f}s to {file_max_end_time:.2f}s)"
                    )

            # Update time offset for next file: add 5 seconds gap
            if file_max_end_time > 0:
                time_offset = file_max_end_time + 5.0

        # Add all messages to the database
        if verbose:
            print(f"\nAdding {len(all_messages)} total messages to database...")
        await msg_coll.extend(all_messages)

        message_count = await msg_coll.size()
        if verbose:
            print(f"Successfully added {message_count} messages")
        else:
            print(
                f"Imported {message_count} messages from {len(vtt_files)} file(s) to {database}"
            )

        # Build all indexes (always)
        if verbose:
            print("\nBuilding indexes...")
            print("  Extracting knowledge (semantic refs)...")

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

            # Create a Transcript object to build indexes
            transcript = await Transcript.create(
                settings,
                name_tag=name,
                messages=msg_coll,
                semantic_refs=semref_coll,
                tags=[name, "vtt-transcript"],
            )

            semref_count_before = 0
            if verbose:
                print("  Building all indexes from messages and semantic refs...")
                semref_count_before = await semref_coll.size()
                print(f"    Semantic refs before: {semref_count_before}")

            # Extract knowledge with progress reporting
            knowledge_extractor = convknowledge.KnowledgeExtractor()
            batch_size = settings.semantic_ref_index_settings.batch_size

            # Get all batches
            batches = await get_message_chunk_batch(
                transcript.messages,
                0,  # Start from beginning
                batch_size,
            )

            total_batches = len(batches)
            messages_processed = 0
            last_report_time = time.time()

            print(f"  Processing {total_batches} batches (batch size: {batch_size})...")

            for batch_idx, batch in enumerate(batches):
                batch_start = time.time()

                # Process this batch
                await semrefindex.add_batch_to_semantic_ref_index(
                    transcript,
                    batch,
                    knowledge_extractor,
                    None,  # terms_added
                )

                messages_processed += len(batch)
                batch_time = time.time() - batch_start

                # Print progress every 10 messages (approximately)
                if messages_processed % 10 == 0 or batch_idx == total_batches - 1:
                    semref_count = await semref_coll.size()
                    elapsed = time.time() - last_report_time
                    print(
                        f"    {messages_processed}/{await transcript.messages.size()} chunks | "
                        f"{semref_count} refs | "
                        f"{batch_time:.1f}s/batch | "
                        f"{elapsed:.1f}s elapsed"
                    )
                    last_report_time = time.time()

            # Build remaining indexes (metadata-based semantic refs, secondary indexes, etc.)
            await transcript.add_metadata_to_index()
            if transcript.secondary_indexes is not None:
                # Build secondary indexes (message text index, related terms, etc.)
                from typeagent.knowpro import secindex

                await secindex.build_secondary_indexes(transcript, settings)

            if verbose:
                semref_count = await semref_coll.size()
                print(f"    Semantic refs after: {semref_count}")
                print(
                    f"  Extracted {semref_count - semref_count_before} new semantic references"
                )

            # Commit everything only after successful indexing
            if isinstance(storage_provider, SqliteStorageProvider):
                storage_provider.db.commit()
                if verbose:
                    print("\nAll data committed to database")

            print("All indexes built successfully")

        except Exception as e:
            print(f"\nError: Failed to build search indexes: {e}", file=sys.stderr)
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
        f"  python tools/utool.py --database '{database}' --question 'Your question here'"
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
            start_date=args.start_date,
            merge_consecutive=not args.no_merge,
            use_text_speaker_detection=args.use_text_speaker_detection,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
