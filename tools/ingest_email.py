#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Email Ingestion Tool

This script ingests email (.eml) files or mbox files into a SQLite database
that can be queried using tools/query.py.

Usage:
    python tools/ingest_email.py -d email.db --eml inbox_dump/
    python tools/ingest_email.py -d email.db --eml message1.eml message2.eml
    python tools/ingest_email.py -d email.db --mbox mailbox.mbox
    python tools/ingest_email.py -d email.db --mbox mailbox.mbox --first 100
    python tools/ingest_email.py -d email.db --mbox mailbox.mbox --after 2023-01-01 --before 2023-02-01

    python tools/query.py --database email.db --query "What was discussed?"
"""

"""
TODO

- Catch auth errors and stop rather than marking as failed
- Collect knowledge outside db transaction to reduce lock time
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback
from typing import Iterable

from dotenv import load_dotenv

import openai

from typeagent.aitools import utils
from typeagent.emails.email_import import (
    count_emails_in_mbox,
    decode_encoded_words,
    import_email_from_file,
    import_emails_from_mbox,
)
from typeagent.emails.email_memory import EmailMemory
from typeagent.emails.email_message import EmailMessage
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.storage.utils import create_storage_provider


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the email ingestion tool."""
    parser = argparse.ArgumentParser(
        description="Ingest email (.eml) files or mbox files into a database for querying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--eml",
        nargs="+",
        metavar="PATH",
        help="One or more .eml files or directories containing .eml files",
    )
    source.add_argument(
        "--mbox",
        metavar="FILE",
        help="Path to an mbox file",
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
        help="Only include emails after this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--before",
        metavar="DATE",
        help="Only include emails before this date (YYYY-MM-DD)",
    )

    # Count filters (mbox only)
    count = parser.add_mutually_exclusive_group()
    count.add_argument(
        "--first",
        type=int,
        metavar="N",
        help="Only process the first N emails from an mbox file",
    )
    count.add_argument(
        "--last",
        type=int,
        metavar="N",
        help="Only process the last N emails from an mbox file",
    )

    return parser


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
    """Parse a YYYY-MM-DD string into a timezone-aware datetime."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
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
        # Make offset-naive timestamps UTC for comparison
        if email_dt.tzinfo is None:
            email_dt = email_dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    if after and email_dt < after:
        return False
    if before and email_dt > before:
        return False
    return True


def _iter_emails(
    eml_paths: list[str] | None,
    mbox_path: str | None,
    verbose: bool,
    first_n: int | None,
    last_n: int | None,
) -> Iterable[tuple[str, EmailMessage, str]]:
    """Yield (source_id, email, label) from the selected email sources.

    For --eml: parses each .eml file and yields it.
    For --mbox: iterates the mbox, applying --first/--last index range filtering.

    Date filtering is NOT applied here; the caller handles it.
    """
    if eml_paths:
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
            email = import_email_from_file(str(email_file))
            yield str(email_file), email, label

    if mbox_path:
        mbox = Path(mbox_path)
        if not mbox.exists():
            print(f"Error: mbox file not found: {mbox}", file=sys.stderr)
            sys.exit(1)
        mbox_total = count_emails_in_mbox(mbox_path)
        print(f"mbox file: {mbox} ({mbox_total} emails)")

        if last_n is not None:
            mbox_start = max(0, mbox_total - last_n)
            mbox_stop: int | None = None
        elif first_n is not None:
            mbox_start = 0
            mbox_stop = first_n
        else:
            mbox_start = 0
            mbox_stop = None

        for msg_index, email in import_emails_from_mbox(mbox_path):
            if msg_index < mbox_start:
                continue
            if mbox_stop is not None and msg_index >= mbox_stop:
                break
            source_id = f"{mbox_path}:{msg_index}"
            label = f"[{msg_index + 1}/{mbox_total}] {mbox_path}:{msg_index}"
            yield source_id, email, label


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
        VERBOSE_PREVIEW_LENGTH = 150
        preview = repr(chunk[: VERBOSE_PREVIEW_LENGTH + 1])[1:-1]
        if len(preview) > VERBOSE_PREVIEW_LENGTH:
            preview = preview[: VERBOSE_PREVIEW_LENGTH - 3] + "..."
        print(f"      {preview}")


async def ingest_emails(
    eml_paths: list[str] | None,
    mbox_path: str | None,
    database: str,
    verbose: bool = False,
    after: datetime | None = None,
    before: datetime | None = None,
    first_n: int | None = None,
    last_n: int | None = None,
) -> None:
    """Ingest email files into a database.

    Exactly one of eml_paths or mbox_path must be provided.
    """

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
    start_time = time.time()

    semref_coll = await settings.storage_provider.get_semantic_ref_collection()
    storage_provider = settings.storage_provider

    for source_id, email, label in _iter_emails(
        eml_paths, mbox_path, verbose, first_n, last_n
    ):
        try:
            if verbose:
                print(label, end="", flush=True)

            # Check if already ingested by source path/id
            if status := await storage_provider.get_source_status(source_id):
                skipped_count += 1
                if verbose:
                    print(f" [Previously {status}, skipping]")
                continue
            else:
                if verbose:
                    print()

            # Apply date filter
            if not _email_matches_date_filter(email.timestamp, after, before):
                skipped_count += 1
                if verbose:
                    print("  [Outside date range, skipping]")
                continue

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

    # Validate --first/--last are only used with --mbox
    if (args.first is not None or args.last is not None) and not args.mbox:
        parser.error("--first and --last can only be used with --mbox")

    after = _parse_date(args.after) if args.after else None
    before = _parse_date(args.before) if args.before else None

    asyncio.run(
        ingest_emails(
            eml_paths=args.eml,
            mbox_path=args.mbox,
            database=args.database,
            verbose=args.verbose,
            after=after,
            before=before,
            first_n=args.first,
            last_n=args.last,
        )
    )


if __name__ == "__main__":
    main()
