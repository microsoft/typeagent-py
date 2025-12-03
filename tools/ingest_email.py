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

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

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
                print(f"Warning: Skipping non-.eml file: {path}", file=sys.stderr)
        elif path.is_dir():
            eml_files = list(path.glob("*.eml"))
            if verbose:
                print(f"  Found {len(eml_files)} .eml files in {path}")
            email_files.extend(eml_files)
        else:
            print(f"Warning: Skipping special file: {path}", file=sys.stderr)

    return email_files


async def ingest_emails(
    paths: list[str],
    database: str,
    verbose: bool = False,
) -> None:
    """Ingest email files into a database."""

    # Collect all .eml files
    if verbose:
        print("Collecting email files...")
    email_files = collect_email_files(paths, verbose)

    if not email_files:
        print("Error: No .eml files found", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Found {len(email_files)} email files to ingest")

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
                print(f"\n[{i + 1}/{len(email_files)}] {email_file}")

            email = import_email_from_file(str(email_file))
            email_id = email.metadata.id

            # Check if this email was already ingested
            if email_id and storage_provider.is_source_ingested(email_id):
                skipped_count += 1
                if verbose:
                    print(f"    [Already ingested, skipping]")
                continue

            if verbose:
                print(f"    From: {email.metadata.sender}")
                if email.metadata.subject:
                    print(f"    Subject: {email.metadata.subject}")
                print(f"    Date: {email.timestamp}")
                print(f"    Body chunks: {len(email.text_chunks)}")
                for chunk in email.text_chunks:
                    # Show first 200 chars of each chunk
                    preview = chunk[:200].replace("\n", " ")
                    if len(chunk) > 200:
                        preview += "..."
                    print(f"      {preview}")

            # Pass source_id to mark as ingested atomically with the message
            source_ids = [email_id] if email_id else None
            await email_memory.add_messages_with_indexing(
                [email], source_ids=source_ids
            )
            successful_count += 1

            # Print progress periodically
            if not verbose and (i + 1) % batch_size == 0:
                elapsed = time.time() - start_time
                semref_count = await semref_coll.size()
                print(
                    f"  [{i + 1}/{len(email_files)}] {successful_count} imported | "
                    f"{semref_count} refs | {elapsed:.1f}s elapsed"
                )

        except Exception as e:
            failed_count += 1
            print(f"Error processing {email_file}: {e}", file=sys.stderr)
            if verbose:
                import traceback

                traceback.print_exc()

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
