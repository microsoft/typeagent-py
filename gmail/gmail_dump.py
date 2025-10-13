# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import time
from base64 import urlsafe_b64decode as b64d
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
OUT = Path("mail_dump")


def get_creds(creds_dir: Path):
    token_file = creds_dir / "token.json"
    client_secret_file = creds_dir / "client_secret.json"

    if token_file.exists():
        return Credentials.from_authorized_user_file(token_file, SCOPES)
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
    creds = flow.run_local_server(port=0)
    token_file.write_text(creds.to_json())
    return creds


def main():
    parser = argparse.ArgumentParser(
        description="Download Gmail messages as .eml files"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum number of messages to download (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Output directory for .eml files (default: mail_dump)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Gmail search query (default: empty, returns all messages)",
    )
    parser.add_argument(
        "--creds-dir",
        type=Path,
        default=Path("."),
        help="Directory containing client_secret.json and token.json (default: current directory)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(exist_ok=True)

    svc = build("gmail", "v1", credentials=get_creds(args.creds_dir))

    start_time = time.time()
    resp = (
        svc.users()
        .messages()
        .list(userId="me", maxResults=args.max_results, q=args.query)
        .execute()
    )
    count = 0
    for m in resp.get("messages", []):
        raw = (
            svc.users()
            .messages()
            .get(userId="me", id=m["id"], format="raw")
            .execute()["raw"]
        )
        Path(args.output_dir / f"{m['id']}.eml").write_bytes(b64d(raw.encode()))
        count += 1
    elapsed = time.time() - start_time
    print(f"Downloaded {count} messages to {args.output_dir} in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
