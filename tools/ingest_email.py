#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Email Ingestion Tool

This script ingests email (.eml) files into a SQLite database
that can be queried using tools/query.py.

Usage:
    python tools/ingest_email.py -d email.db inbox_dump/
    python tools/ingest_email.py -d email.db message1.eml message2.eml
    python tools/ingest_email.py -d email.db inbox_dump/ --start-date 2023-01-01 --stop-date 2023-02-01
    python tools/ingest_email.py -d email.db inbox_dump/ --offset 10 --limit 5

    python tools/query.py --database email.db --query "What was discussed?"
"""

import argparse
import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Iterable

from dotenv import load_dotenv

from typeagent.aitools import utils
from typeagent.emails.email_import import (
    decode_encoded_words,
    email_matches_date_filter,
    import_email_from_file,
)
from typeagent.emails.email_memory import EmailMemory
from typeagent.emails.email_message import EmailMessage
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.interfaces_core import AddMessagesResult
from typeagent.storage.utils import create_storage_provider


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the email ingestion tool."""
    parser = argparse.ArgumentParser(
        description="Ingest email (.eml) files into a database for querying.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "filter pipeline:\n"
            "  1. --offset/--limit slice the input file list.\n"
            "  2. Already-ingested emails are always skipped.\n"
            "  3. --start-date/--stop-date narrow the date range (combinable).\n"
            "\n"
            "examples:\n"
            "  # Ingest all .eml files in a directory\n"
            "  python tools/ingest_email.py -d mail.db inbox/\n"
            "\n"
            "  # Ingest only January 2024 emails\n"
            "  python tools/ingest_email.py -d mail.db inbox/ "
            "--start-date 2024-01-01 --stop-date 2024-02-01\n"
            "\n"
            "  # Ingest the first 20 matching emails\n"
            "  python tools/ingest_email.py -d mail.db inbox/ --limit 20\n"
            "\n"
            "  # Skip the first 100, then ingest the next 50\n"
            "  python tools/ingest_email.py -d mail.db inbox/ "
            "--offset 100 --limit 50\n"
        ),
    )

    parser.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="One or more .eml files or directories containing .eml files",
    )

    parser.add_argument(
        "-d",
        "--database",
        required=True,
        help="Path to the SQLite database file to create/use",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose/debug output"
    )

    # Date filters
    parser.add_argument(
        "--start-date",
        metavar="DATE",
        help=(
            "Only include emails dated on or after DATE (YYYY-MM-DD, "
            "interpreted as local midnight). Combinable with --stop-date."
        ),
    )
    parser.add_argument(
        "--stop-date",
        metavar="DATE",
        help=(
            "Only include emails dated before DATE (YYYY-MM-DD, exclusive "
            "upper bound, local midnight). Combinable with --start-date."
        ),
    )

    # Pagination
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Skip the first N files in the input list "
            "(applied before any other filtering). Default: 0."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Process at most N files from the input list "
            "(applied before any other filtering). Default: no limit."
        ),
    )

    # Concurrency / batching
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of concurrent LLM extraction requests. "
            "Default: 4 (from ConversationSettings)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="Number of chunks per commit batch. Default: 100.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        metavar="N",
        help=(
            "Maximum number of text chunks to keep per email. "
            "Extra chunks are silently dropped. Default: 20."
        ),
    )

    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument combinations and exit on error."""
    errors: list[str] = []

    # --offset must be non-negative
    if args.offset < 0:
        errors.append("--offset must be a non-negative integer.")

    # --limit must be positive when given
    if args.limit is not None and args.limit <= 0:
        errors.append("--limit must be a positive integer.")

    # --concurrency must be positive when given
    if args.concurrency is not None and args.concurrency <= 0:
        errors.append("--concurrency must be a positive integer.")

    # --batch-size must be positive
    if args.batch_size <= 0:
        errors.append("--batch-size must be a positive integer.")

    # --max-chunks must be positive when given
    if args.max_chunks is not None and args.max_chunks <= 0:
        errors.append("--max-chunks must be a positive integer.")

    # --start-date must be before --stop-date when both are given
    if args.start_date and args.stop_date:
        start = _parse_date(args.start_date)
        stop = _parse_date(args.stop_date)
        if start >= stop:
            errors.append(
                f"--start-date ({args.start_date}) must be earlier than --stop-date ({args.stop_date})."
            )

    if errors:
        for err in errors:
            print(f"Error: {err}", file=sys.stderr)
        sys.exit(2)


def collect_eml_files(paths: list[str], verbose: bool) -> list[Path]:
    """Collect all .eml files from the given paths (files or directories)."""
    email_files: list[Path] = []

    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            print(f"Error: Path '{path}' not found", file=sys.stderr)
            sys.exit(1)

        if path.is_file():
            if path.suffix.lower() == ".eml":
                email_files.append(path)
            else:
                print(f"Error: Not an .eml file: {path}", file=sys.stderr)
                sys.exit(1)
        elif path.is_dir():
            eml_files = sorted(path.glob("*.eml"))
            if verbose:
                print(f"Found {len(eml_files)} .eml files in {path}")
            email_files.extend(eml_files)
        else:
            print(f"Error: Not a file or directory: {path}", file=sys.stderr)
            sys.exit(1)

    return email_files


def _parse_date(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD string into a timezone-aware datetime.

    The date is interpreted as 00:00:00 in the local timezone, so that
    ``--start-date 2024-01-15`` means the start of that day locally.
    """
    try:
        # astimezone() on a naive datetime assumes local time (Python 3.6+)
        return datetime.strptime(date_str, "%Y-%m-%d").astimezone()
    except ValueError:
        print(
            f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD.",
            file=sys.stderr,
        )
        sys.exit(1)


def _iter_emails(
    eml_paths: list[str],
    verbose: bool,
    offset: int = 0,
    limit: int | None = None,
) -> Iterable[tuple[str, Path, str]]:
    """Yield (source_id, file_path, label) from the given .eml paths.

    *offset* and *limit* slice the collected file list (like
    ``files[offset:offset+limit]``) before anything else happens.
    Does NOT parse the files; the caller imports only the emails it needs.
    """
    with utils.timelog("Collecting .eml files"):
        email_files = collect_eml_files(eml_paths, verbose)
    if not email_files:
        print("Error: No .eml files found", file=sys.stderr)
        sys.exit(1)
    total = len(email_files)
    if verbose:
        print(f"Found {total} .eml files")
    end = offset + limit if limit is not None else None
    email_files = email_files[offset:end]
    if verbose and (offset or limit is not None):
        print(f"After --offset={offset} --limit={limit}: {len(email_files)} files")
    sliced_total = len(email_files)
    for i, email_file in enumerate(email_files):
        label = f"[{i + 1}/{sliced_total}] {email_file}"
        yield str(email_file.resolve()), email_file, label


def _print_email_verbose(email: EmailMessage) -> None:
    """Print verbose details for an email."""
    print(f"    From: {decode_encoded_words(email.metadata.sender)}")
    if email.metadata.recipients:
        print(
            f"    To: {', '.join(decode_encoded_words(r) for r in email.metadata.recipients)}"
        )
    if email.metadata.cc:
        print(
            f"    Cc: {', '.join(decode_encoded_words(r) for r in email.metadata.cc)}"
        )
    if email.metadata.subject:
        print(
            f"    Subject: {decode_encoded_words(email.metadata.subject).replace('\n', '\\n')}"
        )
    print(f"    Date: {email.timestamp}")
    print(f"    Body chunks: {len(email.text_chunks)}")
    MAIL_PREVIEW_LEN = 80
    for chunk in email.text_chunks:
        preview = repr(chunk[: MAIL_PREVIEW_LEN + 1])[1:-1]
        if len(preview) > MAIL_PREVIEW_LEN:
            preview = preview[: MAIL_PREVIEW_LEN - 3] + "..."
        print(f"      {preview}")


async def _email_generator(
    email_entries: list[tuple[str, Path, str]],
    verbose: bool,
    start_date: datetime | None,
    stop_date: datetime | None,
    max_chunks: int | None,
    counters: dict[str, int],
    already_ingested: set[str],
) -> AsyncIterator[EmailMessage]:
    """Async generator that parses and yields EmailMessage objects.

    *email_entries* is a pre-collected list of ``(source_id, file_path, label)``
    tuples produced by :func:`_iter_emails`.

    *already_ingested* is the set of source_ids known to be in the DB at
    the start of this run (one bulk query).  Files in this set are skipped
    before parsing.

    *counters* is mutated in place to track ``parsed``, ``skipped``,
    ``date_skipped``, and ``failed`` counts for the caller's summary.
    """
    for source_id, email_file, label in email_entries:
        if source_id in already_ingested:
            counters["skipped"] += 1
            continue

        try:
            email = import_email_from_file(str(email_file))
        except Exception as e:
            counters["failed"] += 1
            print(
                f"Error parsing {source_id}: {e!r:.150s}",
                file=sys.stderr,
            )
            continue

        # Apply date filter
        if not email_matches_date_filter(email.timestamp, start_date, stop_date):
            counters["date_skipped"] += 1
            if verbose:
                print(f"{label}  [Outside date range, skipping]")
            continue

        if verbose:
            print(label)
            _print_email_verbose(email)

        # Truncate chunks if --max-chunks is set
        if max_chunks is not None and len(email.text_chunks) > max_chunks:
            if verbose:
                print(f"    Truncating {len(email.text_chunks)} chunks to {max_chunks}")
            email.text_chunks = email.text_chunks[:max_chunks]

        # Set source_id so streaming API handles dedup and tracking
        email.source_id = source_id
        counters["parsed"] += 1
        yield email


async def ingest_emails(
    eml_paths: list[str],
    database: str,
    verbose: bool = False,
    start_date: datetime | None = None,
    stop_date: datetime | None = None,
    offset: int = 0,
    limit: int | None = None,
    concurrency: int | None = None,
    batch_size: int = 100,
    max_chunks: int | None = 20,
) -> None:
    """Ingest email files into a database."""

    # Load environment for model API access
    if verbose:
        print("Loading environment...")
    load_dotenv()

    # Create conversation settings and storage provider
    if verbose:
        print("Setting up conversation settings...")

    settings = ConversationSettings()

    # Override concurrency if specified
    if concurrency is not None:
        settings.semantic_ref_index_settings.concurrency = concurrency

    settings.storage_provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        database,
        EmailMessage,
    )

    # Create EmailMemory
    email_memory = await EmailMemory.create(settings)

    if verbose:
        print(f"Target database: {database}")

    effective_concurrency = settings.semantic_ref_index_settings.concurrency
    if verbose:
        print(f"Concurrency: {effective_concurrency}")
        print(f"Batch size: {batch_size} chunks")

    # One bulk query: collect all source_ids, then ask the DB which are
    # already ingested.  This replaces N per-file is_source_ingested calls
    # with a single are_sources_ingested call.
    storage = settings.storage_provider
    email_entries = list(_iter_emails(eml_paths, verbose, offset, limit))
    all_source_ids = [sid for sid, _, _ in email_entries]
    already_ingested = await storage.are_sources_ingested(all_source_ids)
    if already_ingested and verbose:
        print(
            f"Pre-filter: {len(already_ingested)} of {len(all_source_ids)} already ingested"
        )

    if verbose:
        print("\nParsing and importing emails...")

    start_time = time.time()
    last_batch_time = start_time

    # Counters mutated by the generator and callback
    counters: dict[str, int] = {
        "parsed": 0,
        "skipped": 0,
        "date_skipped": 0,
        "failed": 0,
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
        per_chunk = batch_secs / result.chunks_added if result.chunks_added else 0
        parts = [
            f"  Batch {counters['batches']}:",
            f"+{result.messages_added} messages,",
            f"+{result.chunks_added} chunks,",
            f"+{result.semrefs_added} semrefs",
        ]
        print(
            f"{' '.join(parts)} | "
            f"{batch_secs:.1f}s ({per_chunk:.2f}s/chunk) | "
            f"{counters['ingested']} total ingested | "
            f"{elapsed:.1f}s elapsed",
            flush=True,
        )

    message_stream = _email_generator(
        email_entries,
        verbose,
        start_date,
        stop_date,
        max_chunks,
        counters,
        already_ingested,
    )

    result: AddMessagesResult | None = None
    interrupted = False
    try:
        result = await email_memory.add_messages_streaming(
            message_stream,
            batch_size=batch_size,
            on_batch_committed=on_batch_committed,
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True

    # Final summary
    elapsed = time.time() - start_time
    if interrupted and counters["batches"] == 0:
        print()
        print("Interrupted before any batches were committed.")
        return

    messages_ingested = (
        result.messages_added if result is not None else counters["ingested"]
    )
    total_chunks = result.chunks_added if result is not None else counters["chunks"]
    semrefs_added = result.semrefs_added if result is not None else counters["semrefs"]
    total_skipped = counters["skipped"]
    overall_per_chunk = elapsed / total_chunks if total_chunks else 0

    print()
    if verbose:
        if interrupted:
            print("Ingestion interrupted by user (^C).")
        print(f"Successfully ingested {messages_ingested} email(s)")
        print(f"Ingested {total_chunks} chunk(s)")
        if total_skipped:
            print(f"Skipped {total_skipped} already-ingested email(s)")
        if counters["date_skipped"]:
            print(f"Skipped {counters['date_skipped']} email(s) outside date range")
        if counters["failed"]:
            print(f"Failed to parse {counters['failed']} email(s)")
        print(f"Extracted {semrefs_added} semantic references")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Overall time per chunk: {overall_per_chunk:.2f}s/chunk")
    else:
        print(
            f"Ingested {messages_ingested} emails to {database} "
            f"({total_chunks} chunks, {semrefs_added} refs added, {elapsed:.1f}s, "
            f"{overall_per_chunk:.2f}s/chunk)"
        )
        if total_skipped:
            print(f"Skipped: {total_skipped} (already ingested)")
        if counters["date_skipped"]:
            print(f"Skipped: {counters['date_skipped']} (outside date range)")
        if counters["failed"]:
            print(f"Failed: {counters['failed']}")

    # Show usage information
    print()
    print("To query the emails, use:")
    print(
        f"  python tools/query.py --database '{database}' --query 'Your question here'"
    )


def main() -> None:
    """Main entry point."""
    parser = create_arg_parser()
    args = parser.parse_args()
    _validate_args(args)

    start_date = _parse_date(args.start_date) if args.start_date else None
    stop_date = _parse_date(args.stop_date) if args.stop_date else None

    asyncio.run(
        ingest_emails(
            eml_paths=args.paths,
            database=args.database,
            verbose=args.verbose,
            start_date=start_date,
            stop_date=stop_date,
            offset=args.offset,
            limit=args.limit,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
            max_chunks=args.max_chunks,
        )
    )


if __name__ == "__main__":
    main()
