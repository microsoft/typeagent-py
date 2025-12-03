# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.emails.email_import import (
    extract_inline_reply,
    get_last_response_in_thread,
    is_inline_reply,
)


class TestIsInlineReply:
    def test_empty_text(self) -> None:
        assert is_inline_reply("") is False

    def test_no_header(self) -> None:
        text = "Just a regular email with no quoted content."
        assert is_inline_reply(text) is False

    def test_bottom_posted_reply(self) -> None:
        # This has "On ... wrote:" but all quotes are at the bottom, no interleaving
        text = """Thanks for the info!

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:

> Here is some quoted text.
> More quoted text.
> Even more."""
        assert is_inline_reply(text) is False

    def test_inline_reply(self) -> None:
        text = """I've given my replies in line with the quoted text.

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Quoted blah.

That is clearly BS.

> Quoted blah blah.

Here I must agree.

> More quoted text.

-- 
Guido van Rossum"""
        assert is_inline_reply(text) is True

    def test_inline_reply_no_preamble(self) -> None:
        text = """On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> First quote.

My first response.

> Second quote.

My second response."""
        assert is_inline_reply(text) is True


class TestExtractInlineReply:
    def test_empty_text(self) -> None:
        assert extract_inline_reply("") == ""

    def test_no_inline_pattern(self) -> None:
        text = "Just a regular email."
        assert extract_inline_reply(text) == "Just a regular email."

    def test_basic_inline_reply(self) -> None:
        text = """I've given my replies in line with the quoted text.

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Quoted blah.

That is clearly BS.

> Quoted blah blah.

Here I must agree.

> More quoted text.

-- 
Guido van Rossum"""
        result = extract_inline_reply(text)
        assert "I've given my replies in line" in result
        assert "That is clearly BS." in result
        assert "Here I must agree." in result
        # Quoted content should be removed
        assert "Quoted blah" not in result
        # Signature should be removed
        assert "Guido van Rossum" not in result

    def test_inline_reply_with_context(self) -> None:
        text = """On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Is Python good?

Yes, absolutely!

> What about JavaScript?

It has its uses."""
        result = extract_inline_reply(text, include_context=True)
        assert "Yes, absolutely!" in result
        assert "It has its uses." in result
        assert "[In reply to:" in result
        assert "Python" in result

    def test_preserves_preamble(self) -> None:
        text = """Here's my preamble before the inline replies.

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Question?

Answer!"""
        result = extract_inline_reply(text)
        assert "Here's my preamble" in result
        assert "Answer!" in result

    def test_strips_trailing_delimiters(self) -> None:
        text = """On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Question?

Answer!
_______________"""
        result = extract_inline_reply(text)
        assert result.endswith("Answer!")


class TestGetLastResponseInThread:
    def test_empty_text(self) -> None:
        assert get_last_response_in_thread("") == ""

    def test_simple_text(self) -> None:
        assert get_last_response_in_thread("Hello world") == "Hello world"

    def test_bottom_posted_reply(self) -> None:
        text = """This is my response.

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:

> Original message here.
> More original."""
        result = get_last_response_in_thread(text)
        assert result == "This is my response."

    def test_inline_reply(self) -> None:
        text = """Preamble.

On Mon, Dec 10, 2020 at 10:30 AM Someone wrote:
> Quote 1.

Reply 1.

> Quote 2.

Reply 2.

-- 
Signature"""
        result = get_last_response_in_thread(text)
        assert "Preamble" in result
        assert "Reply 1" in result
        assert "Reply 2" in result
        assert "Quote" not in result
        assert "Signature" not in result

    def test_original_message_delimiter(self) -> None:
        text = """My response.

-----Original Message-----
From: Someone
The original content."""
        result = get_last_response_in_thread(text)
        assert result == "My response."

    def test_forwarded_delimiter(self) -> None:
        text = """My thoughts on this.

----- Forwarded by Someone -----
Original forwarded content."""
        result = get_last_response_in_thread(text)
        assert result == "My thoughts on this."
