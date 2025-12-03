# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Shared type definitions for knowpro modules to avoid circular imports.
"""

from typing import Any, Generic, Literal, TypedDict, TypeVar

# --- Shared TypedDicts and type aliases ---

TMessageData = TypeVar("TMessageData")

class ConversationDataWithIndexes(TypedDict, Generic[TMessageData]):
    messages: list[TMessageData]
    relatedTermsIndexData: dict[str, Any] | None
    messageIndexData: dict[str, Any] | None
    # Add other fields as needed

SearchTermGroupTypes = Any

class Tag(TypedDict):
    knowledge_type: Literal["tag"]
    text: str

class Topic(TypedDict):
    knowledge_type: Literal["topic"]
    text: str

# Add any other shared types here as needed
