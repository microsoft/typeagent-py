# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pytest
from contextlib import redirect_stdout
from io import StringIO

import typeagent.aitools.utils as utils


def test_timelog():
    buf = StringIO()
    with redirect_stdout(buf):
        with utils.timelog("test block"):
            pass
    out = buf.getvalue()
    assert out.endswith("s -- test block\n")


def test_pretty_print():
    # Use a simple object and check output is formatted by black
    obj = {"a": 1}
    buf = StringIO()
    with redirect_stdout(buf):
        utils.pretty_print(obj)
    out = buf.getvalue()
    # Should be valid Python and contain the dict
    assert out == '{"a": 1}\n', out


def test_load_dotenv():
    # Call load_dotenv and check for at least one expected key
    utils.load_dotenv()
    assert "OPENAI_API_KEY" in os.environ or "AZURE_OPENAI_API_KEY" in os.environ


def test_create_translator():
    import typechat

    class DummyModel(typechat.TypeChatLanguageModel):
        async def complete(self, *args, **kwargs) -> typechat.Result:
            return typechat.Failure("dummy response")

    import pydantic.dataclasses

    @pydantic.dataclasses.dataclass
    class DummySchema:
        pass

    # This will raise if the environment or typechat is not set up correctly
    translator = utils.create_translator(DummyModel(), DummySchema)
    assert hasattr(translator, "model")
