#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Convert an mbox file into a directory of individual .eml files.

Usage:
    python tools/mbox_to_emls.py mailbox.mbox output_dir/
"""

import argparse
from email.utils import parsedate_to_datetime
import mailbox
import os
import sys


def make_filename(date_header: str | None, used_names: set[str]) -> str:
    """
    Generate a unique filename from the Date header.

    Format: YYYYMMDD_HHMMSS.eml, with _NNN suffix for disambiguation.
    Falls back to 'unknown_NNNNNN' if no valid date.
    """
    base = "unknown"
    if date_header:
        try:
            dt = parsedate_to_datetime(date_header)
            base = dt.strftime("%Y%m%d_%H%M%S")
        except (ValueError, TypeError):
            pass

    # Find a unique name
    if base == "unknown":
        # For unknown dates, use a 6-digit serial
        serial = 0
        while True:
            name = f"{base}_{serial:06d}.eml"
            if name not in used_names:
                used_names.add(name)
                return name
            serial += 1
    else:
        # Try without suffix first
        name = f"{base}.eml"
        if name not in used_names:
            used_names.add(name)
            return name
        # Add serial suffix for duplicates
        serial = 1
        while True:
            name = f"{base}_{serial:03d}.eml"
            if name not in used_names:
                used_names.add(name)
                return name
            serial += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert an mbox file to a directory of .eml files."
    )
    parser.add_argument("input", help="Path to the input .mbox file")
    parser.add_argument("output", help="Path to the output directory")
    args = parser.parse_args()

    mbox_path = args.input
    output_dir = args.output

    if not os.path.isfile(mbox_path):
        print(f"Error: Input file not found: {mbox_path}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    mbox = mailbox.mbox(mbox_path)
    used_names: set[str] = set()
    count = 0

    for message in mbox:
        date_header = message.get("Date")
        filename = make_filename(date_header, used_names)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "wb") as f:
            f.write(message.as_bytes())

        count += 1
        if count % 100 == 0:
            print(f"Processed {count} messages...", file=sys.stderr)

    print(f"Wrote {count} .eml files to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
