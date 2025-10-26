# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end tests for the MCP server."""

import pytest

from mcp.types import TextContent

from fixtures import really_needs_auth


@pytest.mark.asyncio
async def test_mcp_server_query_conversation(really_needs_auth):
    """Test the query_conversation tool end-to-end using MCP client."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Configure server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "typeagent.mcp.server"],
        env=None,
    )

    # Create client session and connect to server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]

            # Verify query_conversation tool exists
            assert (
                "query_conversation" in tool_names
            ), f"query_conversation tool not found. Available tools: {tool_names}"

            # Call the query_conversation tool
            result = await session.call_tool(
                "query_conversation",
                arguments={"question": "What is the title of this podcast?"},
            )

            # Verify response structure
            assert len(result.content) > 0, "Expected non-empty response"

            # Type narrow the content to TextContent
            content_item = result.content[0]
            assert isinstance(
                content_item, TextContent
            ), f"Expected TextContent, got {type(content_item)}"
            response_text = content_item.text

            # Parse response (it should be JSON with success, answer, time_used)
            import json

            response_data = json.loads(response_text)
            assert "success" in response_data
            assert "answer" in response_data
            assert "time_used" in response_data

            # If successful, answer should be non-empty
            if response_data["success"]:
                assert len(response_data["answer"]) > 0


@pytest.mark.asyncio
async def test_mcp_server_empty_question(really_needs_auth):
    """Test the query_conversation tool with an empty question."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Configure server parameters
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "typeagent.mcp.server"],
        env=None,
    )

    # Create client session and connect to server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # Call with empty question
            result = await session.call_tool(
                "query_conversation",
                arguments={"question": ""},
            )

            # Verify response
            assert len(result.content) > 0

            # Type narrow the content to TextContent
            content_item = result.content[0]
            assert isinstance(
                content_item, TextContent
            ), f"Expected TextContent, got {type(content_item)}"
            response_text = content_item.text

            import json

            response_data = json.loads(response_text)
            assert response_data["success"] is False
            assert "No question provided" in response_data["answer"]
