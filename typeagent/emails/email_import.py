# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from pathlib import Path
from typing import Iterable

from email import message_from_string
from email.utils import parsedate_to_datetime
from email.message import Message

import quotequail

from .email_message import ChunkAttribution, EmailMessage, EmailMessageMeta


def import_emails_from_dir(
    dir_path: str, max_chunk_length: int = 4096
) -> Iterable[EmailMessage]:
    for file_path in Path(dir_path).iterdir():
        if file_path.is_file():
            yield import_email_from_file(str(file_path.resolve()), max_chunk_length)


# Imports an email file (.eml) as a list of EmailMessage objects
def import_email_from_file(
    file_path: str, max_chunk_length: int = 4096
) -> EmailMessage:
    email_string: str = ""
    with open(file_path, "r") as f:
        email_string = f.read()

    email = import_email_string(email_string, max_chunk_length)
    email.src_url = file_path
    return email


# Imports a single email MIME string and returns an EmailMessage object
def import_email_string(
    email_string: str, max_chunk_length: int = 4096
) -> EmailMessage:
    msg: Message = message_from_string(email_string)
    email: EmailMessage = import_email_message(msg, max_chunk_length)
    return email


def import_forwarded_email_string(
    email_string: str, max_chunk_length: int = 4096
) -> list[EmailMessage]:
    msg_parts = get_forwarded_email_parts(email_string)
    return [
        import_email_string(part, max_chunk_length)
        for part in msg_parts
        if len(part) > 0
    ]


# Imports an email.message.Message object and returns an EmailMessage object
def import_email_message(msg: Message, max_chunk_length: int) -> EmailMessage:
    # Extract metadata
    email_meta = EmailMessageMeta(
        sender=msg.get("From", ""),
        recipients=_import_address_headers(msg.get_all("To", [])),
        cc=_import_address_headers(msg.get_all("Cc", [])),
        bcc=_import_address_headers(msg.get_all("Bcc", [])),
        subject=msg.get("Subject"),
        id=msg.get("Message-ID", None),
    )
    timestamp: str | None = None
    timestamp_date = msg.get("Date", None)
    if timestamp_date is not None:
        timestamp = parsedate_to_datetime(timestamp_date).isoformat()

    # Get email body (plain text only)
    body = _extract_email_body(msg)
    if body is None:
        body = ""

    # Parse body into quoted and non-quoted chunks using quotequail
    text_chunks: list[str] = []
    chunk_info: list[ChunkAttribution] = []

    # Subject is always the first chunk
    if email_meta.subject:
        text_chunks.append(email_meta.subject)
        chunk_info.append(None)

    # Use quotequail to identify quoted sections
    quote_results = quotequail.quote(body)

    # Process each quotequail chunk
    pending_attribution: str | None = None
    for visible, text in quote_results:
        text = text.rstrip()
        if not text:
            continue

        if visible:
            # Original content - check for attribution at the end
            text, attribution = _split_attribution(text)
            if attribution:
                pending_attribution = attribution
            if not text:
                continue

            # Split into sub-chunks if needed
            sub_chunks = _text_to_chunks(text, max_chunk_length)
            for sub_chunk in sub_chunks:
                text_chunks.append(sub_chunk)
                chunk_info.append(None)
        else:
            # Quoted content: use pending attribution or " " for unknown
            attr = pending_attribution if pending_attribution else " "
            sub_chunks = _text_to_chunks(text, max_chunk_length)
            for sub_chunk in sub_chunks:
                text_chunks.append(sub_chunk)
                chunk_info.append(attr)
            pending_attribution = None

    # Drop trailing signature chunks (original text that looks like a signature)
    while text_chunks and chunk_info and chunk_info[-1] is None:
        if _is_signature(text_chunks[-1]):
            text_chunks.pop()
            chunk_info.pop()
        else:
            break

    # Drop trailing quoted chunks (keep quoted only if followed by original)
    while text_chunks and chunk_info and chunk_info[-1] is not None:
        text_chunks.pop()
        chunk_info.pop()

    # Strip trailing Nones and set to None if all original
    while chunk_info and chunk_info[-1] is None:
        chunk_info.pop()
    email_meta.chunk_info = chunk_info if chunk_info else None
    email: EmailMessage = EmailMessage(
        metadata=email_meta, text_chunks=text_chunks, timestamp=timestamp
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
    split_delimiter = re.compile(r"(?=From:)", re.IGNORECASE)
    parts: list[str] = split_delimiter.split(email_text)
    return _remove_empty_strings(parts)


# Attribution patterns for "On ... wrote:" style headers
# Supports English, German, French, and Outlook-style headers
_ATTRIBUTION_PATTERNS = [
    # English: "On Dec 1, 2025, John Smith <john@example.com> wrote:"
    re.compile(r"^On .+?wrote:\s*$", re.IGNORECASE | re.MULTILINE),
    # German: "Am 1. Dezember 2025 schrieb John Smith:"
    re.compile(r"^Am .+?schrieb .+?:\s*$", re.IGNORECASE | re.MULTILINE),
    # French: "Le 1 décembre 2025 à 10:00, John Smith a écrit :"
    re.compile(r"^Le .+?a écrit\s*:\s*$", re.IGNORECASE | re.MULTILINE),
    # Outlook-style header block start
    re.compile(r"^From:\s*.+$", re.IGNORECASE | re.MULTILINE),
]

# Patterns to extract the name/email from attribution
_ATTRIBUTION_NAME_PATTERNS = [
    re.compile(r"(?:On .+?,\s*)?(.+?)\s*(?:<[^>]+>)?\s*wrote:\s*$", re.IGNORECASE),
    re.compile(r"Am .+?schrieb\s+(.+?):\s*$", re.IGNORECASE),
    re.compile(r"Le .+?,\s*(.+?)\s*a écrit\s*:\s*$", re.IGNORECASE),
]

# Signature patterns (from email_reply_parser)
_SIGNATURE_PATTERNS = [
    re.compile(r"^(--|__|-\w)", re.MULTILINE),
    re.compile(r"^Sent from my", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Get Outlook for", re.IGNORECASE | re.MULTILINE),
]


def _split_attribution(text: str) -> tuple[str, str | None]:
    """Split text into content and trailing attribution.

    Returns (content, attribution) where attribution is the name of the person
    being quoted, or None if no attribution was found.
    """
    lines = text.rstrip().splitlines()
    if not lines:
        return text, None

    # Check last few lines for attribution pattern
    for num_lines in range(min(3, len(lines)), 0, -1):
        last_lines = "\n".join(lines[-num_lines:])
        for pattern in _ATTRIBUTION_PATTERNS:
            match = pattern.search(last_lines)
            if match:
                attr_text = match.group(0)
                # Try to extract the name
                for name_pattern in _ATTRIBUTION_NAME_PATTERNS:
                    name_match = name_pattern.search(attr_text)
                    if name_match:
                        content = "\n".join(lines[:-num_lines]).rstrip()
                        attr_name = name_match.group(1).strip()
                        print(f"=== {content=} {attr_name=}")
                        return content, attr_name
                # Matched attribution but couldn't extract name
                content = "\n".join(lines[:-num_lines]).rstrip()
                attr_name = " "
                print(f"=== {content=} {attr_name=} ===")
                return content, attr_name

    return text, None


def _is_signature(text: str) -> bool:
    """Check if text looks like an email signature."""
    for pattern in _SIGNATURE_PATTERNS:
        if pattern.search(text):
            return True
    return False


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
            addresses = _remove_empty_strings(header.split(","))
            for address in addresses:
                unique_addresses.add(address)

    return list(unique_addresses)


def _remove_empty_strings(strings: list[str]) -> list[str]:
    non_empty: list[str] = []
    for s in strings:
        s = s.strip()
        if len(s) > 0:
            non_empty.append(s)
    return non_empty


def _text_to_chunks(text: str, max_chunk_length: int) -> list[str]:
    if len(text) < max_chunk_length:
        return [text]

    paragraphs = _splitIntoParagraphs(text)
    return list(_merge_chunks(paragraphs, "\n\n", max_chunk_length))


def _splitIntoParagraphs(text: str) -> list[str]:
    return _remove_empty_strings(re.split(r"\n{2,}", text))


def _merge_chunks(
    chunks: Iterable[str], separator: str, max_chunk_length: int
) -> Iterable[str]:
    sep_length = len(separator)
    cur_chunk: str = ""
    for new_chunk in chunks:
        cur_length = len(cur_chunk)
        new_length = len(new_chunk)
        if new_length > max_chunk_length:
            # Truncate
            new_chunk = new_chunk[0:max_chunk_length]
            new_length = len(new_chunk)

        if cur_length + (new_length + sep_length) > max_chunk_length:
            if cur_length > 0:
                yield cur_chunk
            cur_chunk = new_chunk
        else:
            cur_chunk += separator
            cur_chunk += new_chunk

    if (len(cur_chunk)) > 0:
        yield cur_chunk
