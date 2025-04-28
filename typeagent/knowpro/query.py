# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import Protocol

from .collections import PropertyTermSet, TermSet, TextRangesInScope
from .interfaces import (
    IConversation,
    IPropertyToSemanticRefIndex,
    ITimestampToTextRangeIndex,
)


def is_conversation_searchable(conversation: IConversation) -> bool:
    return (
        conversation.semantic_ref_index is not None
        and conversation.semantic_refs is not None
    )


@dataclass
class QueryEvalContext:
    conversation: IConversation
    property_index: IPropertyToSemanticRefIndex | None = None
    timestamp_index: ITimestampToTextRangeIndex | None = None
    matched_terms: TermSet = field(init=False, default_factory=TermSet)
    match_property_terms: PropertyTermSet = field(
        init=False, default_factory=PropertyTermSet
    )
    text_ranges_in_scope: TextRangesInScope | None = field(
        init=False, default_factory=TextRangesInScope
    )

    def __post_init__(self):
        if not is_conversation_searchable(self.conversation):
            raise ValueError(
                f"{self.conversation.name_tag} "
                + "is not initialized and cannot be searched."
            )


class IQueryOpExpr[T](Protocol):
    def eval(self, context: QueryEvalContext) -> T:
        raise NotImplementedError
