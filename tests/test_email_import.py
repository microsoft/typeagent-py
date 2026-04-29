# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.emails.email_import import (
    _merge_chunks,
    _split_into_paragraphs,
    _text_to_chunks,
)


class TestMergeChunks:
    """Tests for _merge_chunks, specifically the separator-on-empty-chunk fix."""

    def test_no_leading_separator(self) -> None:
        """First chunk must NOT start with the separator."""
        result = list(_merge_chunks(["hello", "world"], "\n\n", 100))
        assert len(result) == 1
        assert result[0] == "hello\n\nworld"
        assert not result[0].startswith("\n")

    def test_no_leading_separator_after_yield(self) -> None:
        """After yielding a full chunk, the next chunk must not start with separator."""
        # Each piece is 5 chars; max_chunk_length=8 forces a split after each.
        pieces = ["aaaaa", "bbbbb", "ccccc"]
        result = list(_merge_chunks(pieces, "--", 8))
        for chunk in result:
            assert not chunk.startswith("--"), f"chunk {chunk!r} starts with separator"

    def test_single_chunk(self) -> None:
        result = list(_merge_chunks(["only"], "\n\n", 100))
        assert result == ["only"]

    def test_empty_input(self) -> None:
        result = list(_merge_chunks([], "\n\n", 100))
        assert result == []

    def test_exact_fit(self) -> None:
        """Two chunks that fit exactly within max_chunk_length."""
        # "ab" + "\n\n" + "cd" = 6 chars
        result = list(_merge_chunks(["ab", "cd"], "\n\n", 6))
        assert result == ["ab\n\ncd"]

    def test_overflow_splits(self) -> None:
        """Chunks that don't fit together should be yielded separately."""
        # "ab" + "\n\n" + "cd" = 6 chars, max is 5 -> must split
        result = list(_merge_chunks(["ab", "cd"], "\n\n", 5))
        assert result == ["ab", "cd"]

    def test_truncation_of_oversized_chunk(self) -> None:
        """A single chunk longer than max_chunk_length is truncated."""
        result = list(_merge_chunks(["abcdefghij"], "\n\n", 5))
        assert result == ["abcde"]

    def test_multiple_merges_and_splits(self) -> None:
        pieces = ["aa", "bb", "cccccc", "dd"]
        # "aa" + "--" + "bb" = 6, fits in 8
        # "cccccc" alone = 6, can't merge with previous (6+2+6=14>8), yield "aa--bb"
        # "cccccc" + "--" + "dd" = 10 > 8, yield "cccccc"
        # "dd" yielded at end
        result = list(_merge_chunks(pieces, "--", 8))
        assert result == ["aa--bb", "cccccc", "dd"]


class TestSplitIntoParagraphs:
    def test_basic_split(self) -> None:
        text = "para1\n\npara2\n\npara3"
        assert _split_into_paragraphs(text) == ["para1", "para2", "para3"]

    def test_multiple_newlines(self) -> None:
        text = "a\n\n\n\nb"
        assert _split_into_paragraphs(text) == ["a", "b"]

    def test_no_split(self) -> None:
        assert _split_into_paragraphs("single paragraph") == ["single paragraph"]

    def test_leading_trailing_newlines(self) -> None:
        text = "\n\nfoo\n\n"
        result = _split_into_paragraphs(text)
        assert "foo" in result
        assert "" not in result


class TestTextToChunks:
    def test_short_text_single_chunk(self) -> None:
        result = _text_to_chunks("short text", max_chunk_length=100)
        assert result == ["short text"]

    def test_long_text_splits(self) -> None:
        text = "para one\n\npara two\n\npara three"
        result = _text_to_chunks(text, max_chunk_length=15)
        assert len(result) > 1
        for chunk in result:
            assert not chunk.startswith("\n"), f"chunk {chunk!r} has leading newline"

    def test_no_leading_separator_in_any_chunk(self) -> None:
        """Regression: no chunk should start with the paragraph separator."""
        text = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 50
        result = _text_to_chunks(text, max_chunk_length=60)
        for chunk in result:
            assert not chunk.startswith(
                "\n\n"
            ), f"chunk {chunk!r} has leading separator"


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
