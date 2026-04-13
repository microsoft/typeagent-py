# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from typeagent.aitools.model_adapters import create_chat_model


@pytest.mark.asyncio
async def test_why_is_sky_blue(really_needs_auth: None):
    """Test that chat agent responds correctly to 'why is the sky blue?'

    Uses create_chat_model (the pydantic-ai code path) so this test exercises
    the same Azure provider wiring as the rest of the codebase.
    """
    model = create_chat_model()

    response = await model._model.request(
        [ModelRequest(parts=[UserPromptPart(content="why is the sky blue?")])],
        None,
        ModelRequestParameters(),
    )

    text_parts = [p.content for p in response.parts if isinstance(p, TextPart)]
    msg = "".join(text_parts)
    assert msg, "Chat agent didn't respond"

    print(f"Chat agent response: {msg}")

    # Check that the response contains the expected keyword
    assert (
        "scatter" in msg.lower()
    ), "Chat agent didn't respond with the expected message about scattering."
