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
    python query.py --database email.db --query "What was discussed?"
"""

"""
TODO

- Find out why storage creation is slow when db is large (Issue #166)
- Use filename as source ID (so we skip parsing if already ingested)
- Catch auth errors and stop rather than marking as failed
- Collect knowledge outside db transaction to reduce lock time
"""

import argparse
import asyncio
from email.header import decode_header
from pathlib import Path
import sys
import time

from typeagent.aitools import utils
from typeagent.emails.email_import import import_email_from_file
from typeagent.emails.email_memory import EmailMemory
from typeagent.emails.email_message import EmailMessage
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.storage.utils import create_storage_provider


def create_arg_parser() -> argparse.ArgumentParser:
    """Create argument parser for the email ingestion tool."""
    parser = argparse.ArgumentParser(
        description="Ingest email (.eml) files into a database for querying",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "paths",
        nargs="+",
        help="Path to one or more .eml files or directories containing .eml files",
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

    return parser


def collect_email_files(paths: list[str], verbose: bool) -> list[Path]:
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
                print(f"Error: Skipping non-.eml file: {path}", file=sys.stderr)
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


def decode_encoded_word(s: str) -> str:
    """Decode an RFC 2047 encoded string."""
    if "=?utf-8?" not in s:
        return s  # Fast path for common case
    decoded_parts = decode_header(s)
    decoded_string = ""
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            decoded_string += part.decode(encoding or "utf-8", errors="replace")
        else:
            decoded_string += part
    return decoded_string


async def ingest_emails(
    paths: list[str],
    database: str,
    verbose: bool = False,
) -> None:
    """Ingest email files into a database."""

    # Collect all .eml files
    with utils.timelog("Collecting email files"):
        email_files = collect_email_files(paths, verbose)

    if not email_files:
        print("Error: No .eml files found", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Found {len(email_files)} email files in total to ingest")

    # Load environment for model API access
    if verbose:
        print("Loading environment...")
    utils.load_dotenv()

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

    # Parse and import emails
    if verbose:
        print("\nParsing and importing emails...")

    successful_count = 0
    failed_count = 0
    skipped_count = 0
    start_time = time.time()

    semref_coll = await settings.storage_provider.get_semantic_ref_collection()
    storage_provider = settings.storage_provider

    for i, email_file in enumerate(email_files):
        try:
            if verbose:
                print(f"[{i + 1}/{len(email_files)}] {email_file}", end="", flush=True)
            if status := storage_provider.get_source_status(str(email_file)):
                skipped_count += 1
                if verbose:
                    print(f" [Previously {status}, skipping]")
                continue
            else:
                if verbose:
                    print()

            email = import_email_from_file(str(email_file))
            source_id = email.metadata.id
            if verbose:
                print(f"  Email ID: {source_id}", end="")

            # Check if this email was already ingested
            if source_id and (status := storage_provider.get_source_status(source_id)):
                skipped_count += 1
                if verbose:
                    print(f" [Previously {status}, skipping]")
                async with storage_provider:
                    storage_provider.mark_source_ingested(str(email_file), status)
                continue
            else:
                if verbose:
                    print()

            if verbose:
                print(f"    From: {email.metadata.sender}")
                if email.metadata.subject:
                    print(
                        f"    Subject: {decode_encoded_word(email.metadata.subject).replace('\n', '\\n')}"
                    )
                print(f"    Date: {email.timestamp}")
                print(f"    Body chunks: {len(email.text_chunks)}")
                for chunk in email.text_chunks:
                    # Show first N chars of each decoded chunk
                    N = 150
                    chunk = decode_encoded_word(chunk)
                    preview = repr(chunk[: N + 1])[1:-1]
                    if len(preview) > N:
                        preview = preview[: N - 3] + "..."
                    print(f"      {preview}")

            # Pass source_id to mark as ingested atomically with the message
            await email_memory.add_messages_with_indexing(
                [email], source_ids=[str(email_file)]
            )  # This may raise, esp. if the knowledge extraction fails (see except below)
            successful_count += 1

            # Print progress periodically
            if (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                semref_count = await semref_coll.size()
                print(
                    f"\n[{i + 1}/{len(email_files)}] {successful_count} imported | "
                    f"{semref_count} refs | {elapsed:.1f}s elapsed\n"
                )

        except Exception as e:
            failed_count += 1
            print(f"Error processing {email_file}: {e}", file=sys.stderr)
            if verbose:
                import traceback

                traceback.print_exc(limit=10)

    # Final summary
    elapsed = time.time() - start_time
    semref_count = await semref_coll.size()

    print()
    if verbose:
        print(f"Successfully imported {successful_count} email(s)")
        if skipped_count:
            print(f"Skipped {skipped_count} already-ingested email(s)")
        if failed_count:
            print(f"Failed to import {failed_count} email(s)")
        print(f"Extracted {semref_count} semantic references")
        print(f"Total time: {elapsed:.1f}s")
    else:
        print(
            f"Imported {successful_count} emails to {database} "
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

    asyncio.run(
        ingest_emails(
            paths=args.paths,
            database=args.database,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()
