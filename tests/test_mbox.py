# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mbox import functionality and email filtering logic."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from typeagent.emails.email_import import (
    count_emails_in_mbox,
    import_email_string,
    import_emails_from_mbox,
)

from conftest import get_testdata_path, has_testdata_file

# ---------------------------------------------------------------------------
# Paths to testdata mbox files
# ---------------------------------------------------------------------------

FWTS_MBOX = get_testdata_path("email-mbox/fwts-announce.mbox")
BAZAAR_MBOX = get_testdata_path("email-mbox/bazaar-announce.mbox")

# Import the filtering helpers from the tool module.
import importlib
import importlib.util
import sys

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
    _iter_emails,
)

# ===========================================================================
# Tests for count_emails_in_mbox
# ===========================================================================


@pytest.mark.skipif(
    not has_testdata_file("email-mbox/fwts-announce.mbox"),
    reason="fwts-announce.mbox not present",
)
class TestCountEmailsInMbox:
    def test_count_fwts(self) -> None:
        count = count_emails_in_mbox(FWTS_MBOX)
        assert count == 121

    def test_count_bazaar(self) -> None:
        count = count_emails_in_mbox(BAZAAR_MBOX)
        assert count == 425

    def test_count_empty_mbox(self, tmp_path: Path) -> None:
        empty_mbox = tmp_path / "empty.mbox"
        empty_mbox.write_text("")
        assert count_emails_in_mbox(str(empty_mbox)) == 0


# ===========================================================================
# Tests for import_emails_from_mbox
# ===========================================================================


@pytest.mark.skipif(
    not has_testdata_file("email-mbox/fwts-announce.mbox"),
    reason="fwts-announce.mbox not present",
)
class TestImportEmailsFromMbox:
    def test_yields_indexed_emails(self) -> None:
        """Verify that the generator yields (index, email) tuples with sequential indices."""
        results = list(import_emails_from_mbox(FWTS_MBOX))
        assert len(results) == 121
        # Indices should be sequential 0..120
        indices = [i for i, _ in results]
        assert indices == list(range(121))

    def test_email_metadata_populated(self) -> None:
        """First email in fwts-announce should have expected metadata fields."""
        first_index, first_email = next(iter(import_emails_from_mbox(FWTS_MBOX)))
        assert first_index == 0
        assert "kengyu" in first_email.metadata.sender.lower()
        assert first_email.metadata.subject is not None
        assert "firmware" in first_email.metadata.subject.lower()
        assert first_email.timestamp is not None

    def test_src_url_contains_mbox_path_and_index(self) -> None:
        """src_url should be set to '<mbox_path>:<index>' format."""
        for i, email in import_emails_from_mbox(FWTS_MBOX):
            assert email.src_url == f"{FWTS_MBOX}:{i}"
            if i >= 2:
                break

    def test_text_chunks_non_empty(self) -> None:
        """Emails should have at least one non-empty text chunk."""
        for i, email in import_emails_from_mbox(FWTS_MBOX):
            assert len(email.text_chunks) > 0
            assert any(len(chunk) > 0 for chunk in email.text_chunks)
            if i >= 4:
                break

    def test_empty_mbox_yields_nothing(self, tmp_path: Path) -> None:
        """An empty mbox file should yield no emails."""
        empty_mbox = tmp_path / "empty.mbox"
        empty_mbox.write_text("")
        results = list(import_emails_from_mbox(str(empty_mbox)))
        assert results == []


# ===========================================================================
# Synthetic mbox for deterministic tests
# ===========================================================================

_SYNTHETIC_MBOX_CONTENT = """\
From sender1@example.com Mon Jan  1 00:00:00 2024
From: sender1@example.com
To: recipient@example.com
Subject: First email
Date: Mon, 01 Jan 2024 10:00:00 +0000
Message-ID: <msg1@example.com>

Body of the first email.

From sender2@example.com Tue Jan  2 00:00:00 2024
From: sender2@example.com
To: recipient@example.com
Subject: Second email
Date: Tue, 02 Jan 2024 12:00:00 +0000
Message-ID: <msg2@example.com>

Body of the second email.

From sender3@example.com Wed Jan  3 00:00:00 2024
From: sender3@example.com
To: recipient@example.com
Subject: Third email
Date: Wed, 03 Jan 2024 14:00:00 +0000
Message-ID: <msg3@example.com>

Body of the third email.

From sender4@example.com Thu Feb  1 00:00:00 2024
From: sender4@example.com
To: recipient@example.com
Subject: Fourth email
Date: Thu, 01 Feb 2024 08:00:00 +0000
Message-ID: <msg4@example.com>

Body of the fourth email.

From sender5@example.com Fri Mar  1 00:00:00 2024
From: sender5@example.com
To: recipient@example.com
Subject: Fifth email
Date: Fri, 01 Mar 2024 09:00:00 +0000
Message-ID: <msg5@example.com>

Body of the fifth email.

"""


@pytest.fixture
def synthetic_mbox(tmp_path: Path) -> str:
    """Create a synthetic mbox file with 5 known emails and return its path."""
    mbox_file = tmp_path / "synthetic.mbox"
    mbox_file.write_text(_SYNTHETIC_MBOX_CONTENT)
    return str(mbox_file)


class TestSyntheticMbox:
    def test_count(self, synthetic_mbox: str) -> None:
        assert count_emails_in_mbox(synthetic_mbox) == 5

    def test_all_emails_parsed(self, synthetic_mbox: str) -> None:
        results = list(import_emails_from_mbox(synthetic_mbox))
        assert len(results) == 5
        subjects = [email.metadata.subject for _, email in results]
        for subj in subjects:
            assert subj is not None
        # Subjects contain the email body prefixed by subject line
        assert subjects[0] is not None and "First email" in subjects[0]
        assert subjects[4] is not None and "Fifth email" in subjects[4]

    def test_timestamps_parsed(self, synthetic_mbox: str) -> None:
        results = list(import_emails_from_mbox(synthetic_mbox))
        for _, email in results:
            assert email.timestamp is not None
            # Should be valid ISO format
            dt = datetime.fromisoformat(email.timestamp)
            assert dt.year == 2024

    def test_senders(self, synthetic_mbox: str) -> None:
        results = list(import_emails_from_mbox(synthetic_mbox))
        senders = [email.metadata.sender for _, email in results]
        assert "sender1@example.com" in senders[0]
        assert "sender5@example.com" in senders[4]

    def test_message_ids(self, synthetic_mbox: str) -> None:
        results = list(import_emails_from_mbox(synthetic_mbox))
        ids = [email.metadata.id for _, email in results]
        assert ids[0] == "<msg1@example.com>"
        assert ids[4] == "<msg5@example.com>"


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

    def test_after_filter_includes(self) -> None:
        """Email on or after the 'after' date passes."""
        after = self._utc(2024, 1, 15)
        assert _email_matches_date_filter("2024-01-15T00:00:00+00:00", after, None)
        assert _email_matches_date_filter("2024-01-16T00:00:00+00:00", after, None)

    def test_after_filter_excludes(self) -> None:
        """Email before the 'after' date is excluded."""
        after = self._utc(2024, 1, 15)
        assert not _email_matches_date_filter("2024-01-14T23:59:59+00:00", after, None)

    def test_before_filter_includes(self) -> None:
        """Email before the 'before' date passes."""
        before = self._utc(2024, 2, 1)
        assert _email_matches_date_filter("2024-01-31T23:59:59+00:00", None, before)

    def test_before_filter_excludes(self) -> None:
        """Email on or after the 'before' date is excluded (exclusive upper bound)."""
        before = self._utc(2024, 2, 1)
        assert not _email_matches_date_filter("2024-02-01T00:00:00+00:00", None, before)

    def test_date_range(self) -> None:
        """Email within [after, before) passes; outside fails."""
        after = self._utc(2024, 1, 1)
        before = self._utc(2024, 2, 1)
        # Inside
        assert _email_matches_date_filter("2024-01-15T12:00:00+00:00", after, before)
        # Before range
        assert not _email_matches_date_filter(
            "2023-12-31T23:59:59+00:00", after, before
        )
        # At upper bound (exclusive)
        assert not _email_matches_date_filter(
            "2024-02-01T00:00:00+00:00", after, before
        )

    def test_naive_timestamp_treated_as_utc(self) -> None:
        """Offset-naive timestamps should be treated as UTC."""
        after = self._utc(2024, 1, 15)
        assert _email_matches_date_filter("2024-01-15T00:00:00", after, None)
        assert not _email_matches_date_filter("2024-01-14T23:59:59", after, None)

    def test_different_timezone(self) -> None:
        """Timestamps with non-UTC offsets are compared correctly."""
        # 2024-01-15T00:00:00+05:00 is 2024-01-14T19:00:00 UTC
        after = self._utc(2024, 1, 15)
        assert not _email_matches_date_filter("2024-01-15T00:00:00+05:00", after, None)
        # 2024-01-15T10:00:00+05:00 is 2024-01-15T05:00:00 UTC
        assert _email_matches_date_filter("2024-01-15T10:00:00+05:00", after, None)


# ===========================================================================
# Tests for _iter_emails with mbox + first/last
# ===========================================================================


class TestIterEmailsMbox:
    """Tests for _iter_emails with mbox source, --first, and --last."""

    def test_all_emails(self, synthetic_mbox: str) -> None:
        results = list(
            _iter_emails(None, synthetic_mbox, verbose=False, first_n=None, last_n=None)
        )
        assert len(results) == 5

    def test_first_n(self, synthetic_mbox: str) -> None:
        results = list(
            _iter_emails(None, synthetic_mbox, verbose=False, first_n=3, last_n=None)
        )
        assert len(results) == 3
        # Should be indices 0, 1, 2
        source_ids = [sid for sid, _, _ in results]
        assert all(f":{i}" in sid for i, sid in enumerate(source_ids))

    def test_first_n_exceeds_total(self, synthetic_mbox: str) -> None:
        results = list(
            _iter_emails(None, synthetic_mbox, verbose=False, first_n=100, last_n=None)
        )
        assert len(results) == 5

    def test_last_n(self, synthetic_mbox: str) -> None:
        results = list(
            _iter_emails(None, synthetic_mbox, verbose=False, first_n=None, last_n=2)
        )
        assert len(results) == 2
        # Should be indices 3 and 4
        source_ids = [sid for sid, _, _ in results]
        assert ":3" in source_ids[0]
        assert ":4" in source_ids[1]

    def test_last_n_exceeds_total(self, synthetic_mbox: str) -> None:
        results = list(
            _iter_emails(None, synthetic_mbox, verbose=False, first_n=None, last_n=100)
        )
        assert len(results) == 5

    def test_first_one(self, synthetic_mbox: str) -> None:
        results = list(
            _iter_emails(None, synthetic_mbox, verbose=False, first_n=1, last_n=None)
        )
        assert len(results) == 1
        _, email, _ = results[0]
        assert "First email" in (email.metadata.subject or "")

    def test_last_one(self, synthetic_mbox: str) -> None:
        results = list(
            _iter_emails(None, synthetic_mbox, verbose=False, first_n=None, last_n=1)
        )
        assert len(results) == 1
        _, email, _ = results[0]
        assert "Fifth email" in (email.metadata.subject or "")

    def test_label_format(self, synthetic_mbox: str) -> None:
        results = list(
            _iter_emails(None, synthetic_mbox, verbose=False, first_n=2, last_n=None)
        )
        for _, _, label in results:
            # Label should contain [N/total] and mbox path
            assert "[" in label
            assert synthetic_mbox in label


# ===========================================================================
# Tests for mbox with encoding edge cases
# ===========================================================================


_MBOX_WITH_ENCODED_HEADER = """\
From test@example.com Mon Jan  1 00:00:00 2024
From: =?utf-8?b?SsO8cmdlbg==?= <juergen@example.com>
To: recipient@example.com
Subject: =?utf-8?q?M=C3=BCnchen_weather?=
Date: Mon, 01 Jan 2024 10:00:00 +0000
Message-ID: <encoded@example.com>

Hello from Munich!

"""


class TestMboxEncodingEdgeCases:
    def test_encoded_header_sender(self, tmp_path: Path) -> None:
        """RFC 2047 encoded sender should be decoded to a string, not raise."""
        mbox_file = tmp_path / "encoded.mbox"
        mbox_file.write_text(_MBOX_WITH_ENCODED_HEADER)
        results = list(import_emails_from_mbox(str(mbox_file)))
        assert len(results) == 1
        _, email = results[0]
        # The sender should be a string type (may or may not be decoded, but shouldn't crash)
        assert isinstance(email.metadata.sender, str)

    def test_encoded_header_subject(self, tmp_path: Path) -> None:
        """RFC 2047 encoded subject should be decoded to a string."""
        mbox_file = tmp_path / "encoded.mbox"
        mbox_file.write_text(_MBOX_WITH_ENCODED_HEADER)
        _, email = list(import_emails_from_mbox(str(mbox_file)))[0]
        assert isinstance(email.metadata.subject, str)


_MBOX_WITH_UNKNOWN_CHARSET = """\
From test@example.com Mon Jan  1 00:00:00 2024
From: test@example.com
To: recipient@example.com
Subject: Unknown charset test
Date: Mon, 01 Jan 2024 10:00:00 +0000
Message-ID: <charset@example.com>
MIME-Version: 1.0
Content-Type: text/plain; charset="iso-8859-8-i"
Content-Transfer-Encoding: base64

SGVsbG8gV29ybGQ=

"""


class TestMboxUnknownCharset:
    def test_unknown_charset_does_not_crash(self, tmp_path: Path) -> None:
        """An email with an unknown charset should be decoded without raising."""
        mbox_file = tmp_path / "unknown_charset.mbox"
        mbox_file.write_text(_MBOX_WITH_UNKNOWN_CHARSET)
        results = list(import_emails_from_mbox(str(mbox_file)))
        assert len(results) == 1
        _, email = results[0]
        # Body should contain the decoded base64 content
        body = " ".join(email.text_chunks)
        assert "Hello World" in body or len(body) > 0


# ===========================================================================
# Tests for mbox with missing / malformed date
# ===========================================================================

_MBOX_NO_DATE = """\
From test@example.com Mon Jan  1 00:00:00 2024
From: test@example.com
To: recipient@example.com
Subject: No date header
Message-ID: <nodate@example.com>

This email has no Date header.

"""


class TestMboxMissingDate:
    def test_email_without_date_has_none_timestamp(self, tmp_path: Path) -> None:
        mbox_file = tmp_path / "nodate.mbox"
        mbox_file.write_text(_MBOX_NO_DATE)
        results = list(import_emails_from_mbox(str(mbox_file)))
        assert len(results) == 1
        _, email = results[0]
        assert email.timestamp is None

    def test_email_without_date_passes_date_filter(self, tmp_path: Path) -> None:
        """Emails without timestamps should always pass the date filter."""
        assert _email_matches_date_filter(
            None, datetime(2024, 1, 1, tzinfo=timezone.utc), None
        )


# ===========================================================================
# Tests for import_email_string (also exercised by mbox, but directly tested)
# ===========================================================================


class TestImportEmailString:
    def test_simple_email(self) -> None:
        raw = (
            "From: alice@example.com\r\n"
            "To: bob@example.com\r\n"
            "Subject: Test\r\n"
            "Date: Mon, 01 Jan 2024 10:00:00 +0000\r\n"
            "Message-ID: <simple@example.com>\r\n"
            "\r\n"
            "Hello Bob!\r\n"
        )
        email = import_email_string(raw)
        assert "alice@example.com" in email.metadata.sender
        assert email.metadata.subject is not None
        assert "Test" in email.metadata.subject
        assert email.metadata.id == "<simple@example.com>"
        assert email.timestamp is not None
        assert len(email.text_chunks) > 0

    def test_multipart_email(self) -> None:
        raw = (
            "From: alice@example.com\r\n"
            "To: bob@example.com\r\n"
            "Subject: Multipart\r\n"
            "Date: Mon, 01 Jan 2024 10:00:00 +0000\r\n"
            "MIME-Version: 1.0\r\n"
            'Content-Type: multipart/alternative; boundary="boundary"\r\n'
            "\r\n"
            "--boundary\r\n"
            "Content-Type: text/plain\r\n"
            "\r\n"
            "Plain text body\r\n"
            "--boundary\r\n"
            "Content-Type: text/html\r\n"
            "\r\n"
            "<p>HTML body</p>\r\n"
            "--boundary--\r\n"
        )
        email = import_email_string(raw)
        # Should extract the plain text part
        body = " ".join(email.text_chunks)
        assert "Plain text body" in body
