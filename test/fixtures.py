# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import tempfile

import pytest

from typeagent.aitools import utils


@pytest.fixture(scope="session")
def needs_auth():
    utils.load_dotenv()


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as dir:
        yield dir
