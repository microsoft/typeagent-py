#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
JSON to SQLite Database Loader

This tool loads a JSON-serialized podcast database into a SQLite database
that can be queried using tools/query.py.

Usage:
    python tools/load_json.py <index_path> --database <db_file>
    python tools/load_json.py tests/testdata/Episode_53_AdrianTchaikovsky_index -d transcript.db

The index_path should exclude the "_data.json" suffix.
"""

import argparse
import asyncio
import os

from typeagent.aitools import utils
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.podcasts import podcast
from typeagent.storage.sqlite.provider import SqliteStorageProvider
from typeagent.storage.utils import create_storage_provider


async def load_json_to_database(
    podcast_file_prefix: str,
    dbname: str,
    verbose: bool = False,
) -> None:
    """Load JSON-serialized podcast data into a SQLite database.

    Args:
        podcast_file_prefix: Path to podcast index files (without "_data.json" suffix)
        dbname: Path to SQLite database file (must be empty)
        verbose: Whether to show verbose output
    """
    if verbose:
        print(f"Loading podcast from JSON: {podcast_file_prefix}")
        print(f"Target database: {dbname}")

    # Create settings and storage provider
    settings = ConversationSettings()
    settings.storage_provider = await create_storage_provider(
        settings.message_text_index_settings,
        settings.related_term_index_settings,
        dbname,
        podcast.PodcastMessage,
    )

    # Get the storage provider to check if database is empty
    provider = await settings.get_storage_provider()
    msgs = await provider.get_message_collection()

    # Check if database already has data
    msg_count = await msgs.size()
    if msg_count > 0:
        raise RuntimeError(
            f"Database '{dbname}' already contains {msg_count} messages. "
            "The database must be empty to load new data. "
            "Please use a different database file or remove the existing one."
        )

    # Load podcast from JSON files
    with utils.timelog(f"Loading podcast from {podcast_file_prefix!r}"):
        conversation = await podcast.Podcast.read_from_file(
            podcast_file_prefix, settings, dbname
        )
        if isinstance(provider, SqliteStorageProvider):
            provider.db.commit()

    # Print statistics
    if verbose:
        print(f"\nSuccessfully loaded podcast data:")
        print(f"  {await conversation.messages.size()} messages")
        print(f"  {await conversation.semantic_refs.size()} semantic refs")
        if conversation.semantic_ref_index:
            print(
                f"  {await conversation.semantic_ref_index.size()} semantic ref index entries"
            )

    print(f"\nDatabase created: {dbname}")
    print(f"\nTo query the database, use:")
    print(f"  python tools/query.py --database '{dbname}' --query 'Your question here'")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Load JSON-serialized podcast data into a SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/load_json.py tests/testdata/Episode_53_AdrianTchaikovsky_index -d podcast.db
  python tools/load_json.py path/to/index -d output.db -v

Note: The index path should exclude the "_data.json" suffix.
        """,
    )

    parser.add_argument(
        "index_path",
        help="Path to the podcast index files (excluding the '_data.json' suffix)",
    )

    parser.add_argument(
        "-d",
        "--database",
        required=True,
        help="Path to the SQLite database file (must be empty)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose output including statistics",
    )

    args = parser.parse_args()

    # Validate index file exists
    index_file = args.index_path + "_data.json"
    if not os.path.exists(index_file):
        raise SystemExit(
            f"Error: Podcast index file not found: {index_file}\n"
            f"Please verify the path exists and is accessible.\n"
            f"Note: The path should exclude the '_data.json' suffix."
        )

    # Load environment variables for API access
    utils.load_dotenv()

    # Run the loading process
    try:
        asyncio.run(
            load_json_to_database(
                args.index_path,
                args.database,
                args.verbose,
            )
        )
    except (RuntimeError, ValueError) as err:
        raise SystemExit(f"Error: {err}")


if __name__ == "__main__":
    main()
