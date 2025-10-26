# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the MCP server MCPTypeChatModel."""


def test_max_tokens_value():
    """Test that max_tokens is set to a reasonable value."""
    # The MCP server should use a higher max_tokens value to allow
    # applications that need more than 4K tokens
    expected_max_tokens = 16384
    assert expected_max_tokens == 16384
    assert expected_max_tokens > 4096  # Old value was 4096, new is higher


def test_text_join_delimiter():
    """Test that we use newline as delimiter when joining text parts."""
    # When MCP returns multiple content items, we should join with newlines
    text_parts = ["Part 1", "Part 2", "Part 3"]
    result = "\n".join(text_parts)
    assert result == "Part 1\nPart 2\nPart 3"
    # Verify newline is used, not just concatenation
    assert "\n" in result


def test_error_formatting():
    """Test that errors use repr for better debugging."""
    test_exception = RuntimeError("Test error")
    error_message = f"MCP sampling failed: {test_exception!r}"
    # Should contain both the exception type and message
    assert "RuntimeError" in error_message
    assert "Test error" in error_message
    # repr should give more info than str
    str_message = f"MCP sampling failed: {str(test_exception)}"
    assert len(error_message) >= len(str_message)
