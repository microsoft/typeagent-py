# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from fixtures import really_needs_auth  # type: ignore

from typeagent.aitools.utils import create_async_openai_client


@pytest.mark.asyncio
async def test_why_is_sky_blue(really_needs_auth: None):
    """Test that chat agent responds correctly to 'why is the sky blue?'"""

    # Create an async OpenAI client
    try:
        client = create_async_openai_client()
    except RuntimeError as e:
        if "Neither OPENAI_API_KEY nor AZURE_OPENAI_API_KEY was provided." in str(e):
            pytest.skip("API keys not configured")
        raise

    # Send the user request
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "why is the sky blue?",
            }
        ],
        temperature=0,
    )

    # Get the response message
    msg = response.choices[0].message.content
    assert msg is not None, "Chat agent didn't respond"

    print(f"Chat agent response: {msg}")

    # Check that the response contains the expected keyword
    assert (
        "scatter" in msg.lower()
    ), "Chat agent didn't respond with the expected message about scattering."
