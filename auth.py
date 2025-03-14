#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility to set AZURE_OPENAI_API_KEY to a valid token.

Usage: eval `./auth.py`

NOTE: The token is only valid for a short time.
"""

import sys

from azure.identity import DeviceCodeCredential

save_stdout = sys.stdout
sys.stdout = sys.stderr

# TODO: Do something non-interactive.
credential = DeviceCodeCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")

sys.stdout = save_stdout
print(f"export AZURE_OPENAI_API_KEY={token.token}")
