# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from pathlib import Path
from datetime import datetime

from email import message_from_string
from email.utils import parsedate_to_datetime
from email.message import Message

from .email_message import EmailMessage, EmailMessageMeta

def import_emails_from_dir(dir_path: str) -> list[EmailMessage]:
    messages: list[EmailMessage] = []
    for file_path in Path(dir_path).iterdir():
        messages.append(import_email_from_file(str(file_path.resolve())));
    return messages

# Imports an email file (.eml) as a list of EmailMessage objects
def import_email_from_file(file_path: str) -> EmailMessage:
    email_string: str = ""
    with open(file_path, "r") as f:
        email_string = f.read()

    return import_email_string(email_string)

# Imports a single email MIME string and returns an EmailMessage object
def import_email_string(email_string: str) -> EmailMessage:
    msg: Message = message_from_string(email_string)
    email: EmailMessage = import_email_message(msg)
    return email

def import_forwarded_email_string(email_string: str) -> list[EmailMessage]:
    msg_parts = get_forwarded_email_parts(email_string)
    return [import_email_string(part) for part in msg_parts if len(part) > 0]

# Imports an email.message.Message object and returns an EmailMessage object
# If the message is a reply, returns only the latest response. 
def import_email_message(msg: Message) -> EmailMessage:
    # Extract metadata from
    email_meta = EmailMessageMeta(
        sender = msg.get("From", ""),
        recipients = _import_address_headers(msg.get_all("To", [])),
        cc = _import_address_headers(msg.get_all("Cc", [])),
        bcc = _import_address_headers(msg.get_all("Bcc", [])),
        subject=msg.get("Subject"))
    timestamp: str | None = None
    timestamp_date = msg.get("Date", None)
    if timestamp_date is not None:
        timestamp = parsedate_to_datetime(timestamp_date).isoformat()

    # Get email body. 
    # If the email was a reply, then ensure we only pick up the latest response
    body = _extract_email_body(msg)
    if body is None:
        body = "" 
    elif is_reply(msg):
        body = get_last_response_in_thread(body)
        
    if email_meta.subject is not None:
        body = email_meta.subject + "\n\n" + body

    email: EmailMessage = EmailMessage(
        metadata=email_meta, 
        text_chunks=[body], 
        timestamp=timestamp
    )
    return email

def is_reply(msg: Message) -> bool:
    return msg.get("In-Reply-To") is not None or msg.get("References") is not None

def is_forwarded(msg: Message) -> bool:
    subject = msg.get("Subject", "").upper()
    return subject.startswith("FW:") or subject.startswith("FWD:")

# Return all sub-parts of a forwarded email text in MIME format
def get_forwarded_email_parts(email_text: str) -> list[str]:
    # Forwarded emails often start with "From:" lines, so we can split on those
    split_delimiter = re.compile(r'(?=From:)', re.IGNORECASE)
    parts: list[str] = split_delimiter.split(email_text)
    return  _remove_empty(parts)

# Simple way to get the last response on an email thread in MIME format
def get_last_response_in_thread(email_text: str) -> str:
    if not email_text:
        return ""
    
    delimiters = [
        "From:",
        "Sent:",
        "To:",
        "Subject:",
        "-----Original Message-----",
        "----- Forwarded by",
        "________________________________________",
    ]

    first_delimiter_at = -1
    for delimiter in delimiters:
        index = email_text.find(delimiter)
        if index != -1 and (first_delimiter_at == -1 or index < first_delimiter_at):
            first_delimiter_at = index

    if first_delimiter_at > 0:
        email_text = email_text[:first_delimiter_at]

    email_text = email_text.strip()
    # Remove trailing line delimiters
    email_text = re.sub(r'[\r\n]_+\s*$', '', email_text)
    return email_text

# Extracts the plain text body from an email.message.Message object.
def _extract_email_body(msg: Message) -> str:
    """Extracts the plain text body from an email.message.Message object."""
    if msg.is_multipart():
        parts: list[str] = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and not part.get_filename():
                text: str = _decode_email_payload(part)
                if text:
                    parts.append(text)
        return "\n".join(parts)
    else:
        return _decode_email_payload(msg)
    
def _decode_email_payload(part: Message) -> str:
    """Decodes the payload of an email part to a string using its charset."""
    payload = part.get_payload(decode=True)
    if payload is None:
        # Try non-decoded payload (may be str)
        payload = part.get_payload(decode=False)
        if isinstance(payload, str):
            return payload
        return ""
    if isinstance(payload, bytes):
        return payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    if isinstance(payload, str):
        return payload
    return ""

def _import_address_headers(headers: list[str]) -> list[str]:
    if len(headers) == 0:
        return headers
    unique_addresses: set[str] = set()
    for header in headers:
        if header:
            addresses = _remove_empty(header.split(","))
            for address in addresses:
                unique_addresses.add(address)

    return list(unique_addresses)

def _remove_empty(strings: list[str]) -> list[str]:
    non_empty: list[str] = []
    for s in strings:
        s = s.strip()
        if len(s) > 0:
            non_empty.append(s)
    return non_empty
