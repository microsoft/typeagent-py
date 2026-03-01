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
