# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from pathlib import Path
from typing import Iterable

from email import message_from_string
from email.utils import parsedate_to_datetime
from email.message import Message

from .email_message import EmailMessage, EmailMessageMeta


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
# If the message is a reply, returns only the latest response.
def import_email_message(msg: Message, max_chunk_length: int) -> EmailMessage:
    # Extract metadata from
    email_meta = EmailMessageMeta(
        sender=msg.get("From", ""),
        recipients=_import_address_headers(msg.get_all("To", [])),
        cc=_import_address_headers(msg.get_all("Cc", [])),
        bcc=_import_address_headers(msg.get_all("Bcc", [])),
        subject=msg.get("Subject"),  # TODO: Remove newlines
        id=msg.get("Message-ID", None),
    )
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

    body_chunks = _text_to_chunks(body, max_chunk_length)
    email: EmailMessage = EmailMessage(
        metadata=email_meta, text_chunks=body_chunks, timestamp=timestamp
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


# Precompiled regex for reply/forward delimiters and quoted reply headers
_THREAD_DELIMITERS = re.compile(
    "|".join(
        [
            r"^from: .+$",  # From: someone
            r"^sent: .+$",  # Sent: ...
            r"^to: .+$",  # To: ...
            r"^subject: .+$",  # Subject: ...
            r"^-{2,}\s*Original Message\s*-{2,}$",  # -----Original Message-----
            r"^-{2,}\s*Forwarded by.*$",  # ----- Forwarded by
            r"^_{5,}$",  # _________
            r"^on .+wrote:\s*(?:\r?\n\s*)+>",  # On ... wrote: followed by quoted text
        ]
    ),
    re.IGNORECASE | re.MULTILINE,
)

# Precompiled regex for trailing line delimiters (underscores, dashes, equals, spaces)
_TRAILING_LINE_DELIMITERS = re.compile(r"[\r\n][_\-= ]+\s*$")

# Pattern to detect "On <date> <user> wrote:" header for inline replies
_INLINE_REPLY_HEADER = re.compile(
    r"^on\s+.+\s+wrote:\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Pattern to match quoted lines (starting with > possibly with leading whitespace)
_QUOTED_LINE = re.compile(r"^\s*>")

# Pattern to detect email signature markers
_SIGNATURE_MARKER = re.compile(r"^--\s*$", re.MULTILINE)


def is_inline_reply(email_text: str) -> bool:
    """
    Detect if an email contains inline replies (responses interspersed with quotes).

    An inline reply has:
    1. An "On ... wrote:" header
    2. Quoted lines (starting with >) interspersed with non-quoted response lines
    """
    if not email_text:
        return False

    # Must have the "On ... wrote:" header
    header_match = _INLINE_REPLY_HEADER.search(email_text)
    if not header_match:
        return False

    # Check content after the header for mixed quoted/non-quoted lines
    content_after_header = email_text[header_match.end() :]
    lines = content_after_header.split("\n")

    has_quoted = False
    has_non_quoted_after_quoted = False

    for line in lines:
        # Check for signature marker
        if _SIGNATURE_MARKER.match(line):
            break

        stripped = line.strip()
        if not stripped:
            continue

        if _QUOTED_LINE.match(line):
            has_quoted = True
        elif has_quoted:
            # Non-quoted line after we've seen quoted lines = inline reply
            has_non_quoted_after_quoted = True
            break

    return has_quoted and has_non_quoted_after_quoted


def extract_inline_reply(email_text: str, include_context: bool = False) -> str:
    """
    Extract reply content from an email with inline responses.

    For emails where the author responds inline to quoted text, this extracts
    the non-quoted portions (the actual replies).

    Args:
        email_text: The full email body text
        include_context: If True, include abbreviated quoted context before each reply

    Returns:
        The extracted reply text. If include_context is True, quoted lines are
        prefixed with "[quoted]" to show what's being replied to.
    """
    if not email_text:
        return ""

    # Find the "On ... wrote:" header
    header_match = _INLINE_REPLY_HEADER.search(email_text)
    if not header_match:
        # No inline reply pattern, return as-is
        return email_text

    # Get preamble (content before the "On ... wrote:" header)
    preamble = email_text[: header_match.start()].strip()

    # Process content after header
    content_after_header = email_text[header_match.end() :]
    lines = content_after_header.split("\n")

    result_parts: list[str] = []
    if preamble:
        result_parts.append(preamble)

    current_reply_lines: list[str] = []
    current_quoted_lines: list[str] = []
    in_signature = False

    for line in lines:
        # Check for signature marker
        if _SIGNATURE_MARKER.match(line):
            in_signature = True
            # Flush any pending reply
            if current_reply_lines:
                if include_context and current_quoted_lines:
                    result_parts.append(_summarize_quoted(current_quoted_lines))
                result_parts.append("\n".join(current_reply_lines))
                current_reply_lines = []
                current_quoted_lines = []
            continue

        if in_signature:
            # Skip signature content
            continue

        if _QUOTED_LINE.match(line):
            # This is a quoted line
            if current_reply_lines:
                # Flush the current reply block
                if include_context and current_quoted_lines:
                    result_parts.append(_summarize_quoted(current_quoted_lines))
                result_parts.append("\n".join(current_reply_lines))
                current_reply_lines = []
                current_quoted_lines = []
            # Accumulate quoted lines for context (only if needed)
            if include_context:
                # Strip the leading > and any space after it
                unquoted = re.sub(r"^\s*>\s?", "", line)
                current_quoted_lines.append(unquoted)
        else:
            # Non-quoted line - part of the reply
            stripped = line.strip()
            if stripped or current_reply_lines:
                # Include non-empty lines, or preserve blank lines within a reply block
                current_reply_lines.append(line.rstrip())

    # Flush any remaining reply
    if current_reply_lines:
        if include_context and current_quoted_lines:
            result_parts.append(_summarize_quoted(current_quoted_lines))
        result_parts.append("\n".join(current_reply_lines))

    result = "\n\n".join(part for part in result_parts if part.strip())
    return _strip_trailing_delimiters(result)


def _summarize_quoted(quoted_lines: list[str]) -> str:
    """Create a brief summary of quoted content for context."""
    # Join and truncate to provide context
    text = " ".join(line.strip() for line in quoted_lines if line.strip())
    if len(text) > 100:
        text = text[:97] + "..."
    return f"[In reply to: {text}]"


def _strip_trailing_delimiters(text: str) -> str:
    """Remove trailing line delimiters (underscores, dashes, equals, spaces)."""
    text = text.strip()
    return _TRAILING_LINE_DELIMITERS.sub("", text)


# Simple way to get the last response on an email thread in MIME format
def get_last_response_in_thread(email_text: str) -> str:
    """
    Extract the latest response from an email thread.

    Handles two patterns:
    1. Top-posted replies: New content at top, quoted thread at bottom
    2. Inline replies: Responses interspersed with quoted text

    For inline replies, only the reply portions (non-quoted text) are extracted.
    """
    if not email_text:
        return ""

    # Check for inline reply pattern first
    if is_inline_reply(email_text):
        return extract_inline_reply(email_text, include_context=False)

    # Fall back to original behavior for bottom-posted replies
    match = _THREAD_DELIMITERS.search(email_text)
    if match:
        email_text = email_text[: match.start()]

    return _strip_trailing_delimiters(email_text)


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
