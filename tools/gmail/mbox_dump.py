# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mbox Dump Tool

Extract emails from an mbox file into individual .eml files.
Creates a folder with the same name as the mbox file and places
each email as a separate .eml file (1.eml, 2.eml, ...).

Usage:
    python tools/gmail/mbox_dump.py mailbox.mbox
    python tools/gmail/mbox_dump.py --url https://example.com/archive.mbox
    python tools/gmail/mbox_dump.py --url https://example.com/archive.mbox -o local.mbox
"""

import argparse
import mailbox
from pathlib import Path
import sys
from urllib.parse import urlparse
import urllib.request


def dump_mbox(mbox_path: str, output_dir: str | None = None) -> int:
    """Extract emails from an mbox file into individual .eml files.

    Args:
        mbox_path: Path to the mbox file.
        output_dir: Directory to write .eml files to. If None, a directory
            with the same name as the mbox file (without extension) is created
            alongside the mbox file.

    Returns:
        The number of emails extracted.
    """
    mbox_file = Path(mbox_path)
    if not mbox_file.exists():
        print(f"Error: mbox file not found: {mbox_file}", file=sys.stderr)
        sys.exit(1)

    if output_dir is None:
        out_path = mbox_file.parent / mbox_file.stem
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    mbox = mailbox.mbox(mbox_path)
    count = 0
    for i, message in enumerate(mbox):
        eml_path = out_path / f"{i + 1}.eml"
        eml_path.write_bytes(message.as_bytes())
        count += 1

    return count


def download_mbox(url: str, output_path: str) -> str:
    """Download an mbox file from a URL.

    Args:
        url: URL to download the mbox from.
        output_path: Path to save the downloaded mbox file.

    Returns:
        The path to the downloaded file.
    """
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Saved to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract emails from an mbox file into individual .eml files",
    )
    parser.add_argument(
        "mbox",
        nargs="?",
        help="Path to the mbox file to extract",
    )
    parser.add_argument(
        "--url",
        help="URL to download an mbox file from",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for downloaded mbox file (used with --url)",
    )
    args = parser.parse_args()

    if args.url:
        if args.output:
            mbox_path = args.output
        else:
            # Derive filename from URL
            url_path = urlparse(args.url).path
            filename = Path(url_path).name or "downloaded.mbox"
            mbox_path = filename
        download_mbox(args.url, mbox_path)
        if args.mbox is None:
            args.mbox = mbox_path

    if args.mbox is None:
        parser.error("either provide an mbox file path or use --url to download one")

    count = dump_mbox(args.mbox)
    out_dir = Path(args.mbox).parent / Path(args.mbox).stem
    print(f"Extracted {count} emails to {out_dir}/")


if __name__ == "__main__":
    main()
