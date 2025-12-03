# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Demo script showing all quotequail API features."""

import email
from email.policy import default
from pathlib import Path
import sys

import quotequail

DEFAULT_FILENAME = "mail_dump/18f5057f38a9ef4b.eml"


def main():
    if len(sys.argv) < 2:
        print(f"Default filename: {DEFAULT_FILENAME}")
        filename = DEFAULT_FILENAME
    else:
        filename = sys.argv[1]

    # Parse the email
    msg = email.message_from_bytes(Path(filename).read_bytes(), policy=default)

    print(f"=== Email: {filename} ===")
    print(f"From: {msg['From']}")
    print(f"To: {msg['To']}")
    print(f"Subject: {msg['Subject']}")
    print(f"Date: {msg['Date']}")
    print()

    # Extract plain text and HTML parts
    plain_text = None
    html_text = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain" and plain_text is None:
                plain_text = part.get_content()
            elif content_type == "text/html" and html_text is None:
                html_text = part.get_content()
    else:
        content_type = msg.get_content_type()
        if content_type == "text/plain":
            plain_text = msg.get_content()
        elif content_type == "text/html":
            html_text = msg.get_content()

    # 1. quote() - Identify quoted sections in plain text
    if plain_text:
        print("=" * 60)
        print("1. quotequail.quote(text) - Identify quoted sections")
        print("=" * 60)
        result = quotequail.quote(plain_text)
        for i, (visible, text) in enumerate(result):
            status = "VISIBLE" if visible else "QUOTED/HIDDEN"
            preview = text[:200].replace("\n", "\\n")
            if len(text) > 200:
                preview += "..."
            print(f"  [{i}] {status}: {preview}")
        print()

    # 2. unwrap() - Parse forwarded/reply structure from plain text
    if plain_text:
        print("=" * 60)
        print("2. quotequail.unwrap(text) - Parse forwarded/reply structure")
        print("=" * 60)
        result = quotequail.unwrap(plain_text)
        if result:
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                print(f"  {key}: {value}")
        else:
            print("  (No wrapped/forwarded message detected)")
        print()

    # 3. quote_html() - Identify quoted sections in HTML
    if html_text:
        print("=" * 60)
        print("3. quotequail.quote_html(html) - Identify quoted sections in HTML")
        print("=" * 60)
        result = quotequail.quote_html(html_text)
        for i, (visible, text) in enumerate(result):
            status = "VISIBLE" if visible else "QUOTED/HIDDEN"
            # Strip HTML tags for preview
            preview = text[:300].replace("\n", "\\n")
            if len(text) > 300:
                preview += "..."
            print(f"  [{i}] {status}: {preview[:100]}...")
        print()

    # 4. unwrap_html() - Parse forwarded/reply structure from HTML
    if html_text:
        print("=" * 60)
        print(
            "4. quotequail.unwrap_html(html) - Parse forwarded/reply structure from HTML"
        )
        print("=" * 60)
        result = quotequail.unwrap_html(html_text)
        if result:
            for key, value in result.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                print(f"  {key}: {value}")
        else:
            print("  (No wrapped/forwarded message detected)")
        print()

    # Show raw text for reference
    if plain_text:
        print("=" * 60)
        print("RAW PLAIN TEXT (first 1000 chars)")
        print("=" * 60)
        print(plain_text[:1000])
        if len(plain_text) > 1000:
            print(f"... ({len(plain_text)} total chars)")


if __name__ == "__main__":
    main()
