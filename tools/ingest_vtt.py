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


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the VTT ingestion tool."""
    parser = argparse.ArgumentParser(
        description="Ingest WebVTT transcript files into a database for querying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.vtt --database transcript.db
  %(prog)s meeting.vtt -d meeting.db --name "Team Meeting"
  %(prog)s lecture.vtt -d lecture.db --start-date "2024-10-01T09:00:00"
        """,
    )

    parser.add_argument("vtt_file", help="Path to the WebVTT (.vtt) file to ingest")

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
        "--build-index",
        action="store_true",
        help="Build search indexes after ingestion (slower but enables full search)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing database if it exists",
    )

    return parser


async def ingest_vtt_file(
    vtt_file: str,
    database: str,
    name: str | None = None,
    start_date: str | None = None,
    merge_consecutive: bool = True,
    use_text_speaker_detection: bool = False,
    build_index: bool = False,
    verbose: bool = False,
    overwrite: bool = False,
) -> None:
    """Ingest a VTT file into a database."""

    # Validate input file
    if not os.path.exists(vtt_file):
        print(f"Error: VTT file '{vtt_file}' not found", file=sys.stderr)
        sys.exit(1)

    # Check if database already exists
    if os.path.exists(database) and not overwrite:
        print(
            f"Error: Database '{database}' already exists. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Remove existing database if overwriting
    if overwrite and os.path.exists(database):
        os.remove(database)
        if verbose:
            print(f"Removed existing database: {database}")

    if verbose:
        print(f"Ingesting VTT file: {vtt_file}")
        print(f"Target database: {database}")

    # Analyze the VTT file
    try:
        duration = get_transcript_duration(vtt_file)
        speakers = get_transcript_speakers(
            vtt_file, use_text_based_detection=use_text_speaker_detection
        )

        if verbose:
            print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
            print(
                f"Speakers found: {len(speakers)} ({speakers if speakers else 'None detected'})"
            )
    except Exception as e:
        print(f"Error analyzing VTT file: {e}", file=sys.stderr)
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
        name = Path(vtt_file).stem

    # Import the transcript
    if verbose:
        print(f"Parsing VTT file and creating messages...")
    try:
        # Get collections from our storage provider
        msg_coll = await storage_provider.get_message_collection()
        semref_coll = await storage_provider.get_semantic_ref_collection()

        # Step 0: Make sure tables are empty
        if await msg_coll.size() or await semref_coll.size():
            print(
                f"Error: Database already has data. Use --overwrite to replace.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Step 1: Parse VTT and insert messages into Messages table (once!)
        # Parse the VTT file directly instead of using import_vtt_transcript
        # to avoid creating a temporary storage provider

        try:
            vtt = webvtt.read(vtt_file)
        except Exception as e:
            print(f"Error: Failed to parse VTT file: {e}", file=sys.stderr)
            sys.exit(1)

        messages: list[TranscriptMessage] = []
        current_speaker = None
        current_text_chunks = []
        current_start_time = None
        current_end_time = None

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

            # Convert WebVTT timestamps
            start_time = caption.start
            end_time = caption.end

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
                message = TranscriptMessage(
                    text_chunks=[combined_text], metadata=metadata
                )
                messages.append(message)

        # Add messages to the database (once!)
        if verbose:
            print(f"Adding {len(messages)} messages to database...")
        await msg_coll.extend(messages)

        message_count = await msg_coll.size()
        if verbose:
            print(f"Successfully added {message_count} messages")
        else:
            print(f"Imported {message_count} messages to {database}")

        # Commit the transaction to ensure messages are saved
        if isinstance(storage_provider, SqliteStorageProvider):
            storage_provider.db.commit()
            if verbose:
                print("Messages committed to database")

        # Steps 2 & 3: Extract knowledge and build all indexes
        if build_index:
            if verbose:
                print("\nBuilding indexes...")

            if verbose:
                print("Step 2: Extracting knowledge (semantic refs)...")
            try:
                # Enable knowledge extraction for index building
                settings.semantic_ref_index_settings.auto_extract_knowledge = True

                if verbose:
                    print(
                        f"  auto_extract_knowledge = {settings.semantic_ref_index_settings.auto_extract_knowledge}"
                    )
                    print(
                        f"  batch_size = {settings.semantic_ref_index_settings.batch_size}"
                    )

                # Create a Transcript object to build indexes
                # Messages and semrefs are already in the database
                transcript = await Transcript.create(
                    settings,
                    name_tag=name,
                    messages=msg_coll,
                    semantic_refs=semref_coll,
                    tags=[name, "vtt-transcript"],
                )

                semref_count_before = 0
                if verbose:
                    print(
                        "Step 3: Building all indexes from messages and semantic refs..."
                    )
                    semref_count_before = await semref_coll.size()
                    print(f"  Semantic refs before build_index: {semref_count_before}")

                # Build the full index (extracts knowledge, builds semantic ref index, message text index, etc.)
                await transcript.build_index()

                # Commit all the index data
                if isinstance(storage_provider, SqliteStorageProvider):
                    storage_provider.db.commit()

                if verbose:
                    semref_count = await semref_coll.size()
                    print(f"  Semantic refs after build_index: {semref_count}")
                    print(
                        f"\nExtracted {semref_count - semref_count_before} new semantic references"
                    )
                    print("All indexes built successfully")
            except Exception as e:
                print(f"\nError: Failed to build search indexes: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                sys.exit(1)

    except Exception as e:
        print(f"Error importing transcript: {e}", file=sys.stderr)
        sys.exit(1)

    # Show usage information
    print()
    print("To query the transcript, use:")
    print(
        f"  python tools/utool.py --sqlite-db '{database}' --question 'Your question here'"
    )


def main():
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()

    # Run the ingestion
    asyncio.run(
        ingest_vtt_file(
            vtt_file=args.vtt_file,
            database=args.database,
            name=args.name,
            start_date=args.start_date,
            merge_consecutive=not args.no_merge,
            build_index=args.build_index,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
