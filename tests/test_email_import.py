# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.emails.email_import import (
    is_inline_reply,
    parse_email_chunks,
)


class TestIsInlineReply:
    def test_empty_text(self) -> None:
        assert is_inline_reply("") is False

    def test_no_header(self) -> None:
        text = "Just a regular email with no quoted content."
        assert is_inline_reply(text) is False

    def test_top_posted_reply(self) -> None:
        # This has "On ... wrote:" but all quotes are at the bottom, no interleaving
        text = """\
Thanks for the info!

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:

> Here is some quoted text.
> More quoted text.
> Even more.
"""
        assert is_inline_reply(text) is False

    def test_inline_reply(self) -> None:
        text = """\
I've given my replies in line with the quoted text.

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Quoted blah.

That is clearly BS.

> Quoted blah blah.

Here I must agree.

> More quoted text.

-- 
Guido van Rossum
"""
        assert is_inline_reply(text) is True

    def test_inline_reply_no_preamble(self) -> None:
        text = """\
On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> First quote.

My first response.

> Second quote.

My second response.
"""
        assert is_inline_reply(text) is True


class TestParseEmailChunks:
    def test_empty_text(self) -> None:
        assert parse_email_chunks("") == []

    def test_no_inline_pattern(self) -> None:
        text = "Just a regular email."
        result = parse_email_chunks(text)
        assert len(result) == 1
        assert result[0] == ("Just a regular email.", None)

    def test_basic_inline_reply(self) -> None:
        text = """\
I've given my replies in line with the quoted text.

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Quoted blah.

That is clearly BS.

> Quoted blah blah.

Here I must agree.

> More quoted text.

-- 
Guido van Rossum
"""
        result = parse_email_chunks(text)
        # Should have: preamble (original), quoted, reply, quoted, reply, quoted
        texts = [chunk[0] for chunk in result]
        # sources = [chunk[1] for chunk in result]

        # Check we have all the content
        assert any("I've given my replies" in t for t in texts)
        assert any("That is clearly BS" in t for t in texts)
        assert any("Here I must agree" in t for t in texts)
        assert any("Quoted blah" in t for t in texts)

        # Original content should have None source
        for text, source in result:
            if "I've given my replies" in text or "That is clearly BS" in text:
                assert source is None

        # Quoted content should have the person's name
        for text, source in result:
            if "Quoted blah" in text:
                assert source == "Someone"

        # Signature should NOT be included
        assert not any("Guido van Rossum" in t for t in texts)

    def test_extracts_quoted_person_name(self) -> None:
        text = """\
On Mon, Dec 10, 2020 at 10:30 AM John Doe wrote:
> Is Python good?

Yes, absolutely!

> What about JavaScript?

It has its uses.
"""
        result = parse_email_chunks(text)

        # Find quoted chunks - they should have "John Doe" as source
        quoted_chunks = [(t, s) for t, s in result if s is not None]
        assert len(quoted_chunks) == 2
        for text, source in quoted_chunks:
            assert source == "John Doe"

    def test_preserves_preamble(self) -> None:
        text = """\
Here's my preamble before the inline replies.

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Question?

Answer!
"""
        result = parse_email_chunks(text)
        texts = [chunk[0] for chunk in result]

        assert any("preamble" in t for t in texts)
        assert any("Answer" in t for t in texts)

    def test_strips_trailing_delimiters(self) -> None:
        text = """\
On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Question?

Answer!
_______________
"""
        result = parse_email_chunks(text)
        # Last non-quoted chunk should not end with underscores
        original_chunks = [t for t, s in result if s is None]
        assert len(original_chunks) > 0
        assert not original_chunks[-1].endswith("_")

    def test_quoted_content_is_unabbreviated(self) -> None:
        text = """\
On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> This is a very long quoted line that should be preserved in full.
> And this is another line that continues the quote.
> Even more content here.

My response.
"""
        result = parse_email_chunks(text)

        # Find the quoted chunk
        quoted = [t for t, s in result if s is not None]
        assert len(quoted) == 1
        # Full content should be preserved
        assert "very long quoted line" in quoted[0]
        assert "another line" in quoted[0]
        assert "Even more content" in quoted[0]
