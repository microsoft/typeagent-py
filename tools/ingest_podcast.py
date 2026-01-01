import argparse
import asyncio
import os

from util_testdata import EPISODE_53_TRANSCRIPT  # type: ignore[attr-defined]

from typeagent.aitools.utils import load_dotenv
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.podcasts.podcast_ingest import ingest_podcast

CHARS_PER_MINUTE = 1050  # My guess for average speech rate incl. overhead


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest a podcast transcript into a database"
    )
    parser.add_argument(
        "transcript", type=str, nargs="?", help="Transcript file to index"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress verbose output"
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default=None,
        help="Database file (default: use an in-memory database)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for message indexing (default 10)",
    )
    parser.add_argument(
        "--start-message",
        type=int,
        default=0,
        help="Message number (0-based) to start indexing, for restart after failure (default 0)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Output JSON file (default None), minus '_data.json' suffix",
    )

    args = parser.parse_args()
    if args.database is not None and args.json_output is not None:
        raise SystemExit("Please use at most one of --database and --json-output")
    if args.transcript is None:
        if os.path.exists(EPISODE_53_TRANSCRIPT):
            args.transcript = EPISODE_53_TRANSCRIPT
            print("Reading default transcript:", EPISODE_53_TRANSCRIPT)
        else:
            raise SystemExit("Please provide a transcript file to ingest")

    load_dotenv()

    try:
        settings = ConversationSettings()

        podcast = await ingest_podcast(
            args.transcript,
            settings,
            podcast_name=os.path.basename(args.transcript) or "Unnamed Podcast",
            start_date=None,
            length_minutes=os.path.getsize(args.transcript) / CHARS_PER_MINUTE,
            dbname=args.database,
            batch_size=args.batch_size,
            start_message=args.start_message,
            verbose=not args.quiet,
        )
    except (RuntimeError, ValueError) as err:
        raise SystemExit(repr(err))

    if args.json_output is not None:
        await podcast.write_to_file(args.json_output)
        if not args.quiet:
            print(f"Exported podcast to JSON file: {args.json_output}")


if __name__ == "__main__":
    asyncio.run(main())
