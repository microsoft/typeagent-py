# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for email filtering logic and email parsing edge cases."""

from datetime import datetime, timezone

# Import the filtering helpers from the tool module.
import importlib
import importlib.util
from pathlib import Path
import sys

from typeagent.emails.email_import import import_email_string

_tools_dir = str(Path(__file__).resolve().parent.parent / "tools")
_spec = importlib.util.spec_from_file_location(
    "ingest_email", Path(_tools_dir) / "ingest_email.py"
)
assert _spec and _spec.loader
_ingest_mod = importlib.util.module_from_spec(_spec)
sys.modules["ingest_email"] = _ingest_mod
_spec.loader.exec_module(_ingest_mod)

from ingest_email import (  # type: ignore[import-untyped]
    _email_matches_date_filter,
)

# ===========================================================================
# Tests for _email_matches_date_filter
# ===========================================================================


class TestEmailMatchesDateFilter:
    """Tests for the _email_matches_date_filter helper in ingest_email.py."""

    def _utc(self, year: int, month: int, day: int) -> datetime:
        return datetime(year, month, day, tzinfo=timezone.utc)

    def test_no_filters(self) -> None:
        """All emails pass when no filters are set."""
        assert _email_matches_date_filter("2024-01-15T10:00:00+00:00", None, None)

    def test_none_timestamp_always_passes(self) -> None:
        """Emails without a timestamp are always included."""
        assert _email_matches_date_filter(
            None, self._utc(2024, 1, 1), self._utc(2024, 12, 31)
        )

    def test_invalid_timestamp_always_passes(self) -> None:
        """Emails with unparseable timestamps are always included."""
        assert _email_matches_date_filter(
            "not-a-date", self._utc(2024, 1, 1), self._utc(2024, 12, 31)
        )

    def test_start_date_filter_includes(self) -> None:
        """Email on or after the start_date passes."""
        start = self._utc(2024, 1, 15)
        assert _email_matches_date_filter("2024-01-15T00:00:00+00:00", start, None)
        assert _email_matches_date_filter("2024-01-16T00:00:00+00:00", start, None)

    def test_start_date_filter_excludes(self) -> None:
        """Email before the start_date is excluded."""
        start = self._utc(2024, 1, 15)
        assert not _email_matches_date_filter("2024-01-14T23:59:59+00:00", start, None)

    def test_stop_date_filter_includes(self) -> None:
        """Email before the stop_date passes."""
        stop = self._utc(2024, 2, 1)
        assert _email_matches_date_filter("2024-01-31T23:59:59+00:00", None, stop)

    def test_stop_date_filter_excludes(self) -> None:
        """Email on or after the stop_date is excluded (exclusive upper bound)."""
        stop = self._utc(2024, 2, 1)
        assert not _email_matches_date_filter("2024-02-01T00:00:00+00:00", None, stop)

    def test_date_range(self) -> None:
        """Email within [start_date, stop_date) passes; outside fails."""
        start = self._utc(2024, 1, 1)
        stop = self._utc(2024, 2, 1)
        # Inside
        assert _email_matches_date_filter("2024-01-15T12:00:00+00:00", start, stop)
        # Before range
        assert not _email_matches_date_filter("2023-12-31T23:59:59+00:00", start, stop)
        # At upper bound (exclusive)
        assert not _email_matches_date_filter("2024-02-01T00:00:00+00:00", start, stop)

    def test_naive_timestamp_treated_as_local(self) -> None:
        """Offset-naive timestamps should be treated as local time."""
        from datetime import datetime as dt

        # Build the filter boundary in local time so the test is TZ-independent
        local_tz = dt.now().astimezone().tzinfo
        start = datetime(2024, 1, 15, tzinfo=local_tz)
        assert _email_matches_date_filter("2024-01-15T00:00:00", start, None)
        assert not _email_matches_date_filter("2024-01-14T23:59:59", start, None)

    def test_different_timezone(self) -> None:
        """Timestamps with non-UTC offsets are compared correctly."""
        # 2024-01-15T00:00:00+05:00 is 2024-01-14T19:00:00 UTC
        start = self._utc(2024, 1, 15)
        assert not _email_matches_date_filter("2024-01-15T00:00:00+05:00", start, None)
        # 2024-01-15T10:00:00+05:00 is 2024-01-15T05:00:00 UTC
        assert _email_matches_date_filter("2024-01-15T10:00:00+05:00", start, None)


# ===========================================================================
# Tests for email encoding edge cases
# ===========================================================================


_EMAIL_WITH_ENCODED_HEADER = """\
From: =?utf-8?b?SsO8cmdlbg==?= <juergen@example.com>\r
To: recipient@example.com\r
Subject: =?utf-8?q?M=C3=BCnchen_weather?=\r
Date: Mon, 01 Jan 2024 10:00:00 +0000\r
Message-ID: <encoded@example.com>\r
\r
Hello from Munich!\r
"""


class TestEncodingEdgeCases:
    def test_encoded_header_sender(self) -> None:
        """RFC 2047 encoded sender should be decoded to a string, not raise."""
        email = import_email_string(_EMAIL_WITH_ENCODED_HEADER)
        assert isinstance(email.metadata.sender, str)

    def test_encoded_header_subject(self) -> None:
        """RFC 2047 encoded subject should be decoded to a string."""
        email = import_email_string(_EMAIL_WITH_ENCODED_HEADER)
        assert isinstance(email.metadata.subject, str)


_EMAIL_WITH_UNKNOWN_CHARSET = """\
From: test@example.com\r
To: recipient@example.com\r
Subject: Unknown charset test\r
Date: Mon, 01 Jan 2024 10:00:00 +0000\r
Message-ID: <charset@example.com>\r
MIME-Version: 1.0\r
Content-Type: text/plain; charset="iso-8859-8-i"\r
Content-Transfer-Encoding: base64\r
\r
SGVsbG8gV29ybGQ=\r
"""


class TestUnknownCharset:
    def test_unknown_charset_does_not_crash(self) -> None:
        """An email with an unknown charset should be decoded without raising."""
        email = import_email_string(_EMAIL_WITH_UNKNOWN_CHARSET)
        body = " ".join(email.text_chunks)
        assert "Hello World" in body or len(body) > 0


# ===========================================================================
# Tests for mbox with missing / malformed date
# ===========================================================================

_EMAIL_NO_DATE = """\
From: test@example.com\r
To: recipient@example.com\r
Subject: No date header\r
Message-ID: <nodate@example.com>\r
\r
This email has no Date header.\r
"""


class TestMissingDate:
    def test_email_without_date_has_none_timestamp(self) -> None:
        email = import_email_string(_EMAIL_NO_DATE)
        assert email.timestamp is None

    def test_email_without_date_passes_date_filter(self) -> None:
        """Emails without timestamps should always pass the date filter."""
        assert _email_matches_date_filter(
            None, datetime(2024, 1, 1, tzinfo=timezone.utc), None
        )


# ===========================================================================
# Tests for import_email_string (also exercised by mbox, but directly tested)
# ===========================================================================

_SIMPLE_EMAIL = """\
From: alice@example.com\r
To: bob@example.com\r
Subject: Test\r
Date: Mon, 01 Jan 2024 10:00:00 +0000\r
Message-ID: <simple@example.com>\r
\r
Hello Bob!\r
"""

_MULTIPART_EMAIL = """\
From: alice@example.com\r
To: bob@example.com\r
Subject: Multipart\r
Date: Mon, 01 Jan 2024 10:00:00 +0000\r
MIME-Version: 1.0\r
Content-Type: multipart/alternative; boundary="boundary"\r
\r
--boundary\r
Content-Type: text/plain\r
\r
Plain text body\r
--boundary\r
Content-Type: text/html\r
\r
<p>HTML body</p>\r
--boundary--\r
"""


class TestImportEmailString:
    def test_simple_email(self) -> None:
        email = import_email_string(_SIMPLE_EMAIL)
        assert "alice@example.com" in email.metadata.sender
        assert email.metadata.subject is not None
        assert "Test" in email.metadata.subject
        assert email.metadata.id == "<simple@example.com>"
        assert email.timestamp is not None
        assert len(email.text_chunks) > 0

    def test_multipart_email(self) -> None:
        email = import_email_string(_MULTIPART_EMAIL)
        # Should extract the plain text part
        body = " ".join(email.text_chunks)
        assert "Plain text body" in body
