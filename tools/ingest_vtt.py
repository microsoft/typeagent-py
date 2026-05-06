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
from collections.abc import AsyncIterator
from datetime import timedelta
import os
from pathlib import Path
import sys
import time

from dotenv import load_dotenv
import webvtt

from typeagent.aitools.model_adapters import create_embedding_model
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces import ConversationMetadata
from typeagent.knowpro.interfaces_core import AddMessagesResult
from typeagent.knowpro.universal_message import format_timestamp_utc, UNIX_EPOCH
from typeagent.storage.utils import create_storage_provider
from typeagent.transcripts.transcript import (
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)
from typeagent.transcripts.transcript_ingest import (
    get_transcript_duration,
    get_transcript_speakers,
    parse_voice_tags,
    webvtt_timestamp_to_seconds,
)


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the VTT ingestion tool."""
    parser = argparse.ArgumentParser(
        description="Ingest WebVTT transcript files into a database for querying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--concurrency",
        type=int,
        default=None,
        help="Max concurrent knowledge extractions (default: from settings)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of chunks per commit batch (default: 100)",
    )

    parser.add_argument(
        "--embedding-name",
        type=str,
        default=None,
        help="Embedding model name (default: text-embedding-ada-002)",
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
    concurrency: int | None = None,
    batch_size: int = 100,
    embedding_name: str | None = None,
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
    load_dotenv()

    # Determine transcript name before creating storage provider
    if not name:
        if len(vtt_files) == 1:
            name = Path(vtt_files[0]).stem
        else:
            name = "combined-transcript"

    # Create conversation settings and storage provider
    if verbose:
        print("Setting up conversation settings...")
    try:
        spec = embedding_name
        if spec and ":" not in spec:
            spec = f"openai:{spec}"
        embedding_model = create_embedding_model(spec)
        settings = ConversationSettings(embedding_model)

        # Create metadata with the conversation name
        metadata = ConversationMetadata(
            name_tag=name,
            tags=[name, "vtt-transcript"],
        )

        # Create storage provider explicitly with the database
        storage_provider = await create_storage_provider(
            settings.message_text_index_settings,
            settings.related_term_index_settings,
            database,
            TranscriptMessage,
            metadata=metadata,
        )

        # Update settings to use our storage provider
        settings.storage_provider = storage_provider

        # Override concurrency if specified
        if concurrency is not None:
            settings.semantic_ref_index_settings.concurrency = concurrency

        if verbose:
            print("Settings and storage provider configured")
    except Exception as e:
        print(f"Error creating settings: {e}", file=sys.stderr)
        sys.exit(1)

    # Import the transcripts
    if verbose:
        print(f"\nParsing VTT files and creating messages...")
    try:
        # Get collections from our storage provider
        msg_coll = storage_provider.messages
        semref_coll = storage_provider.semantic_refs

        # Database should be empty (we checked it doesn't exist earlier)
        # But verify collections are empty just in case
        if await msg_coll.size() or await semref_coll.size():
            print(
                f"Error: Database already has data.",
                file=sys.stderr,
            )
            sys.exit(1)

        msg_count = 0

        async def _message_stream() -> AsyncIterator[TranscriptMessage]:
            nonlocal msg_count
            time_offset = 0.0

            for file_idx, vtt_file in enumerate(vtt_files):
                if verbose:
                    print(f"  Processing {vtt_file}...")
                    if file_idx > 0:
                        print(f"    Time offset: {time_offset:.2f} seconds")

                try:
                    vtt = webvtt.read(vtt_file)
                except Exception as e:
                    print(
                        f"Error: Failed to parse VTT file {vtt_file}: {e}",
                        file=sys.stderr,
                    )
                    sys.exit(1)

                current_speaker: str | None = None
                current_text_chunks: list[str] = []
                current_start_time: str | None = None
                file_max_end_time = 0.0

                def _build_message() -> TranscriptMessage | None:
                    if current_text_chunks and current_start_time is not None:
                        combined_text = " ".join(current_text_chunks).strip()
                        if combined_text:
                            offset_seconds = webvtt_timestamp_to_seconds(
                                current_start_time
                            )
                            timestamp = format_timestamp_utc(
                                UNIX_EPOCH + timedelta(seconds=offset_seconds)
                            )
                            metadata = TranscriptMessageMeta(
                                speaker=current_speaker,
                                recipients=[],
                            )
                            return TranscriptMessage(
                                text_chunks=[combined_text],
                                metadata=metadata,
                                timestamp=timestamp,
                                source_id=f"{vtt_file}#{msg_count}",
                            )
                    return None

                for caption in vtt:
                    if not caption.text.strip():
                        continue

                    raw_text = getattr(caption, "raw_text", caption.text)
                    voice_segments = parse_voice_tags(raw_text)

                    start_time_seconds = (
                        vtt_timestamp_to_seconds(caption.start) + time_offset
                    )
                    end_time_seconds = (
                        vtt_timestamp_to_seconds(caption.end) + time_offset
                    )
                    start_time = seconds_to_vtt_timestamp(start_time_seconds)

                    if end_time_seconds > file_max_end_time:
                        file_max_end_time = end_time_seconds

                    for speaker, text in voice_segments:
                        if not text.strip():
                            continue

                        if (
                            merge_consecutive
                            and speaker == current_speaker
                            and current_text_chunks
                        ):
                            current_text_chunks.append(text)
                        else:
                            msg = _build_message()
                            if msg is not None:
                                msg_count += 1
                                yield msg

                            current_speaker = speaker
                            current_text_chunks = [text] if text.strip() else []
                            current_start_time = start_time

                # Last message from this file
                msg = _build_message()
                if msg is not None:
                    msg_count += 1
                    yield msg

                if file_max_end_time > 0:
                    time_offset = file_max_end_time + 5.0

        try:
            # Enable knowledge extraction for index building
            settings.semantic_ref_index_settings.auto_extract_knowledge = True

            if verbose:
                print(
                    f"    auto_extract_knowledge = {settings.semantic_ref_index_settings.auto_extract_knowledge}"
                )
                print(
                    f"    concurrency = {settings.semantic_ref_index_settings.concurrency}"
                )

            # Create a Transcript object
            transcript = await Transcript.create(
                settings,
                name=name,
                tags=[name, "vtt-transcript"],
            )

            start_time = time.time()
            last_batch_time = start_time

            counters: dict[str, int] = {
                "ingested": 0,
                "chunks": 0,
                "semrefs": 0,
                "batches": 0,
            }

            def on_batch_committed(result: AddMessagesResult) -> None:
                nonlocal last_batch_time
                counters["ingested"] += result.messages_added
                counters["chunks"] += result.chunks_added
                counters["semrefs"] += result.semrefs_added
                counters["batches"] += 1
                now = time.time()
                batch_secs = now - last_batch_time
                last_batch_time = now
                elapsed = now - start_time
                per_chunk = (
                    batch_secs / result.chunks_added if result.chunks_added else 0
                )
                parts = [
                    f"    {counters['ingested']} messages",
                    f"+{result.chunks_added} chunks",
                    f"+{result.semrefs_added} semrefs",
                ]
                print(
                    f"{' | '.join(parts)} | "
                    f"{batch_secs:.1f}s ({per_chunk:.2f}s/chunk) | "
                    f"{elapsed:.1f}s elapsed",
                    flush=True,
                )

            print(
                f"  Processing messages in batches of {batch_size}"
                f" (concurrency={settings.semantic_ref_index_settings.concurrency})..."
            )

            result: AddMessagesResult | None = None
            interrupted = False
            try:
                result = await transcript.add_messages_streaming(
                    _message_stream(),
                    batch_size=batch_size,
                    on_batch_committed=on_batch_committed,
                )
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True

            elapsed = time.time() - start_time
            if interrupted and counters["batches"] == 0:
                print()
                print("Interrupted before any batches were committed.")
                return

            messages_ingested = (
                result.messages_added if result is not None else counters["ingested"]
            )
            total_chunks = (
                result.chunks_added if result is not None else counters["chunks"]
            )
            semrefs_added = (
                result.semrefs_added if result is not None else counters["semrefs"]
            )
            overall_per_chunk = elapsed / total_chunks if total_chunks else 0

            if verbose:
                if interrupted:
                    print("Ingestion interrupted by user (^C).")
                print(f"  Successfully added {messages_ingested} messages")
                print(f"  Ingested {total_chunks} chunk(s)")
                print(f"  Extracted {semrefs_added} semantic references")
                print(f"  Total time: {elapsed:.1f}s")
                print(f"  Overall time per chunk: {overall_per_chunk:.2f}s/chunk")
            else:
                print(
                    f"Imported {messages_ingested} messages from {len(vtt_files)} file(s) to {database}"
                    f" ({total_chunks} chunks, {semrefs_added} refs, {elapsed:.1f}s,"
                    f" {overall_per_chunk:.2f}s/chunk)"
                )

            if not interrupted:
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
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            embedding_name=args.embedding_name,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
