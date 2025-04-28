# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile

import pytest

from typeagent.aitools import auth


@pytest.fixture(scope="session")
def needs_auth():
    auth.load_dotenv()

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as dir:
        yield dir
