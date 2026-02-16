#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Email Ingestion Tool

This script ingests email (.eml) files into a SQLite database
that can be queried using tools/query.py.

Usage:
    python tools/ingest_email.py -d email.db --eml inbox_dump/
    python tools/ingest_email.py -d email.db --eml message1.eml message2.eml
    python tools/ingest_email.py -d email.db --eml inbox_dump/ --after 2023-01-01 --before 2023-02-01
    python tools/ingest_email.py -d email.db --eml inbox_dump/ --offset 10 --limit 5

    python tools/query.py --database email.db --query "What was discussed?"
"""

"""
TODO

- Collect knowledge outside db transaction to reduce lock time
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys
import time
import traceback
from typing import Iterable

from dotenv import load_dotenv

import openai

from typeagent.aitools import utils
from typeagent.emails.email_import import (
    decode_encoded_words,
    import_email_from_file,
)
from typeagent.emails.email_memory import EmailMemory
from typeagent.emails.email_message import EmailMessage
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.storage.utils import create_storage_provider


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the email ingestion tool."""
    parser = argparse.ArgumentParser(
        description="Ingest email (.eml) files into a database for querying.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "filter pipeline:\n"
            "  1. Already-ingested emails are always skipped.\n"
            "  2. --after/--before narrow the date range (combinable).\n"
            "  3. --offset skips the first N qualifying emails.\n"
            "  4. --limit caps how many emails are ingested.\n"
            "\n"
            "examples:\n"
            "  # Ingest all .eml files in a directory\n"
            "  python tools/ingest_email.py -d mail.db --eml inbox/\n"
            "\n"
            "  # Ingest only January 2024 emails\n"
            "  python tools/ingest_email.py -d mail.db --eml inbox/ "
            "--after 2024-01-01 --before 2024-02-01\n"
            "\n"
            "  # Ingest the first 20 matching emails\n"
            "  python tools/ingest_email.py -d mail.db --eml inbox/ --limit 20\n"
            "\n"
            "  # Skip the first 100, then ingest the next 50\n"
            "  python tools/ingest_email.py -d mail.db --eml inbox/ "
            "--offset 100 --limit 50\n"
        ),
    )

    parser.add_argument(
        "--eml",
        nargs="+",
        metavar="PATH",
        required=True,
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
        "--after",
        metavar="DATE",
        help=(
            "Only include emails dated on or after DATE (YYYY-MM-DD, "
            "interpreted as local midnight). Combinable with --before."
        ),
    )
    parser.add_argument(
        "--before",
        metavar="DATE",
        help=(
            "Only include emails dated before DATE (YYYY-MM-DD, exclusive "
            "upper bound, local midnight). Combinable with --after."
        ),
    )

    # Pagination
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Skip the first N matching emails before ingesting "
            "(like SQL OFFSET). Default: 0."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Ingest at most N emails (like SQL LIMIT). "
            "Default: no limit (ingest all)."
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

    # --offset without --limit is allowed (skip first N, ingest the rest)
    # --limit without --offset is allowed (ingest at most N)

    # --after must be before --before when both are given
    if args.after and args.before:
        after = _parse_date(args.after)
        before = _parse_date(args.before)
        if after >= before:
            errors.append(
                f"--after ({args.after}) must be earlier than --before ({args.before})."
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

    The date is interpreted in the local timezone (not UTC), so that
    ``--after 2024-01-15`` means midnight local time.
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


def _email_matches_date_filter(
    timestamp: str | None,
    after: datetime | None,
    before: datetime | None,
) -> bool:
    """Check whether an email's ISO timestamp passes the date filters.

    Emails without a parseable timestamp are always included.
    """
    if timestamp is None:
        return True
    try:
        email_dt = datetime.fromisoformat(timestamp)
        # Treat offset-naive timestamps as local time for comparison
        if email_dt.tzinfo is None:
            email_dt = email_dt.astimezone()
    except ValueError:
        return True
    if after and email_dt < after:
        return False
    if before and email_dt >= before:
        return False
    return True


def _iter_emails(
    eml_paths: list[str],
    verbose: bool,
) -> Iterable[tuple[str, Path, str]]:
    """Yield (source_id, file_path, label) from the given .eml paths.

    Does NOT parse the files; the caller imports only the emails it needs.
    """
    with utils.timelog("Collecting .eml files"):
        email_files = collect_eml_files(eml_paths, verbose)
    if not email_files:
        print("Error: No .eml files found", file=sys.stderr)
        sys.exit(1)
    total = len(email_files)
    if verbose:
        print(f"Found {total} .eml files to ingest")
    for i, email_file in enumerate(email_files):
        label = f"[{i + 1}/{total}] {email_file}"
        yield str(email_file), email_file, label


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
    for chunk in email.text_chunks:
        VERBOSE_PREVIEW_LENGTH = 50
        preview = repr(chunk[: VERBOSE_PREVIEW_LENGTH + 1])[1:-1]
        if len(preview) > VERBOSE_PREVIEW_LENGTH:
            preview = preview[: VERBOSE_PREVIEW_LENGTH - 3] + "..."
        print(f"      {preview}")


async def ingest_emails(
    eml_paths: list[str],
    database: str,
    verbose: bool = False,
    after: datetime | None = None,
    before: datetime | None = None,
    offset: int = 0,
    limit: int | None = None,
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

    batch_size = settings.semantic_ref_index_settings.batch_size
    if verbose:
        print(f"Batch size: {batch_size}")
        print("\nParsing and importing emails...")

    success_count = 0
    failed_count = 0
    skipped_count = 0
    candidate_count = 0
    start_time = time.time()

    semref_coll = await settings.storage_provider.get_semantic_ref_collection()
    storage_provider = settings.storage_provider

    for source_id, email_file, label in _iter_emails(eml_paths, verbose):
        try:
            if verbose:
                print(label, end="", flush=True)

            # Parse the email only after confirming it hasn't been ingested
            email = import_email_from_file(str(email_file))

            # Apply date filter
            if not _email_matches_date_filter(email.timestamp, after, before):
                skipped_count += 1
                if verbose:
                    print("  [Outside date range, skipping]")
                continue

            # Apply offset/limit to qualifying candidates
            candidate_count += 1
            if candidate_count <= offset:
                skipped_count += 1
                if verbose:
                    print("  [Skipped by --offset]")
                continue
            if limit is not None and success_count >= limit:
                if verbose:
                    print("  [Reached --limit, stopping]")
                break

            # Check if already ingested by Message-ID
            email_id = email.metadata.id
            if verbose:
                print(f"  Email ID: {email_id}", end="")
            if email_id and (
                status := await storage_provider.get_source_status(email_id)
            ):
                skipped_count += 1
                if verbose:
                    print(f" [Previously {status}, skipping]")
                async with storage_provider:
                    await storage_provider.mark_source_ingested(source_id, status)
                continue
            else:
                if verbose:
                    print()

            if verbose:
                _print_email_verbose(email)

            # Ingest the email
            try:
                await email_memory.add_messages_with_indexing(
                    [email], source_ids=[source_id]
                )
                success_count += 1
            except openai.AuthenticationError as e:
                if verbose:
                    traceback.print_exc()
                sys.exit(f"Authentication error: {e!r}")

            # Print progress periodically
            if (success_count + failed_count) % batch_size == 0:
                elapsed = time.time() - start_time
                semref_count = await semref_coll.size()
                print(
                    f"\n{label} "
                    f"{success_count} imported | "
                    f"{failed_count} failed | "
                    f"{skipped_count} skipped | "
                    f"{semref_count} semrefs | "
                    f"{elapsed:.1f}s elapsed\n"
                )

        except Exception as e:
            failed_count += 1
            print(
                f"Error processing {source_id}: {e!r:.150s}",
                file=sys.stderr,
            )
            mod = e.__class__.__module__
            qual = e.__class__.__qualname__
            exc_name = qual if mod == "builtins" else f"{mod}.{qual}"
            async with storage_provider:
                await storage_provider.mark_source_ingested(source_id, exc_name)
            if verbose:
                traceback.print_exc(limit=10)

    # Final summary
    elapsed = time.time() - start_time
    semref_count = await semref_coll.size()

    print()
    if verbose:
        print(f"Successfully imported {success_count} email(s)")
        if skipped_count:
            print(f"Skipped {skipped_count} already-ingested email(s)")
        if failed_count:
            print(f"Failed to import {failed_count} email(s)")
        print(f"Extracted {semref_count} semantic references")
        print(f"Total time: {elapsed:.1f}s")
    else:
        print(
            f"Imported {success_count} emails to {database} "
            f"({semref_count} refs, {elapsed:.1f}s)"
        )
        if skipped_count:
            print(f"Skipped: {skipped_count} (already ingested)")
        if failed_count:
            print(f"Failed: {failed_count}")

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

    after = _parse_date(args.after) if args.after else None
    before = _parse_date(args.before) if args.before else None

    asyncio.run(
        ingest_emails(
            eml_paths=args.eml,
            database=args.database,
            verbose=args.verbose,
            after=after,
            before=before,
            offset=args.offset,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
