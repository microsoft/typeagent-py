# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Shared type helpers used to break circular imports in knowpro."""

from typing import Any, Generic, NotRequired, TypedDict, TypeVar

TMessageData = TypeVar("TMessageData")


class ConversationDataWithIndexes(TypedDict, Generic[TMessageData]):
    """Serializable conversation payload with index metadata."""

    nameTag: str
    messages: list[TMessageData]
    tags: list[str]
    semanticRefs: list[Any] | None
    semanticIndexData: NotRequired[Any]
    relatedTermsIndexData: NotRequired[Any]
    threadData: NotRequired[Any]
    messageIndexData: NotRequired[Any]


# When importing from modules that cannot depend on knowpro.interfaces,
# fall back to ``Any`` to avoid circular references while keeping type checkers
# satisfied.
SearchTermGroupTypes = Any

__all__ = [
    "ConversationDataWithIndexes",
    "SearchTermGroupTypes",
]
