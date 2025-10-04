# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from base64 import urlsafe_b64decode as b64d
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CREDS = "client_secret.json"
TOKEN = Path("token.json")
OUT = Path("mail_dump"); OUT.mkdir(exist_ok=True)

def get_creds():
    if TOKEN.exists():
        return Credentials.from_authorized_user_file(TOKEN, SCOPES)
    flow = InstalledAppFlow.from_client_secrets_file(CREDS, SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN.write_text(creds.to_json())
    return creds

svc = build("gmail", "v1", credentials=get_creds())

resp = svc.users().messages().list(userId="me", maxResults=50, q="").execute()
for m in resp.get("messages", []):
    raw = svc.users().messages().get(userId="me", id=m["id"], format="raw").execute()["raw"]
    Path(OUT / f"{m['id']}.eml").write_bytes(b64d(raw.encode()))
print("Done.")
