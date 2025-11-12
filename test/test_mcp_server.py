# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end tests for the MCP server."""

import os
import sys
from typing import Any

import pytest
from mcp import StdioServerParameters
from mcp.client.session import ClientSession as ClientSessionType
from mcp.shared.context import RequestContext
from mcp.types import CreateMessageRequestParams, CreateMessageResult, TextContent

from fixtures import really_needs_auth


@pytest.fixture
def server_params() -> StdioServerParameters:
    """Create MCP server parameters with minimal environment."""
    env = {}
    if "COVERAGE_PROCESS_START" in os.environ:
        env["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_PROCESS_START"]

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "typeagent.mcp.server"],
        env=env,
    )


async def sampling_callback(
    context: RequestContext[ClientSessionType, Any, Any],
    params: CreateMessageRequestParams,
) -> CreateMessageResult:
    """Sampling callback that uses OpenAI to generate responses."""
    # Use OpenAI to generate a response
    from openai.types.chat import ChatCompletionMessageParam

    from typeagent.aitools.utils import create_async_openai_client

    client = create_async_openai_client()

    # Convert MCP SamplingMessage to OpenAI format
    messages: list[ChatCompletionMessageParam] = []
    for msg in params.messages:
        # Handle TextContent
        content: str
        if isinstance(msg.content, TextContent):
            content = msg.content.text
        else:
            raise ValueError(
                f"Unsupported content type in sampling message: {type(msg.content)}"
            )

        # MCP roles are "user" or "assistant", which are compatible with OpenAI
        if msg.role == "user":
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "assistant", "content": content})

    # Add system prompt if provided
    if params.systemPrompt:
        messages.insert(0, {"role": "system", "content": params.systemPrompt})

    # Call OpenAI
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=params.maxTokens,
        temperature=params.temperature if params.temperature is not None else 1.0,
    )

    # Convert response to MCP format
    return CreateMessageResult(
        role="assistant",
        content=TextContent(
            type="text", text=response.choices[0].message.content or ""
        ),
        model=response.model,
        stopReason="endTurn",
    )


@pytest.mark.asyncio
async def test_mcp_server_query_conversation_slow(
    really_needs_auth, server_params: StdioServerParameters
):
    """Test the query_conversation tool end-to-end using MCP client."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    # Create client session and connect to server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=sampling_callback
        ) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            tool_names = [tool.name for tool in tools_result.tools]

            # Verify query_conversation tool exists
            assert "query_conversation" in tool_names

            # Call the query_conversation tool
            result = await session.call_tool(
                "query_conversation",
                arguments={"question": "Who is Kevin Scott?"},
            )

            # Verify response structure
            assert len(result.content) > 0, "Expected non-empty response"

            # Type narrow the content to TextContent
            content_item = result.content[0]
            assert isinstance(content_item, TextContent)
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

            assert response_data["success"] is True, response_data


@pytest.mark.asyncio
async def test_mcp_server_empty_question(server_params: StdioServerParameters):
    """Test the query_conversation tool with an empty question."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    # Create client session and connect to server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(
            read, write, sampling_callback=sampling_callback
        ) as session:
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
            assert isinstance(content_item, TextContent)
            response_text = content_item.text

            import json

            response_data = json.loads(response_text)
            assert response_data["success"] is False
            assert "No question provided" in response_data["answer"]
