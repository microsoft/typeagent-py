# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import typechat

from .interfaces import (
    DateRange,
    Datetime,
    IConversation,
    IMessage,
    ITermToSemanticRefIndex,
)


def get_time_range_prompt_section_for_conversation[
    TMessage: IMessage, TIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TIndex],
) -> typechat.PromptSection | None:
    time_range = get_time_range_for_conversation(conversation)
    if time_range is not None:
        start = time_range.start.replace(tzinfo=None).isoformat()
        end = (
            time_range.end.replace(tzinfo=None).isoformat() if time_range.end else "now"
        )
        return typechat.PromptSection(
            role="system",
            content=f"ONLY IF user request explicitly asks for time ranges, "
            f'THEN use the CONVERSATION TIME RANGE: "{start} to {end}"',
        )


def get_time_range_for_conversation[
    TMessage: IMessage, TIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TIndex],
) -> DateRange | None:
    messages = conversation.messages
    if len(messages) > 0:
        start = messages[0].timestamp
        if start is not None:
            end = messages[-1].timestamp
            return DateRange(
                start=Datetime.fromisoformat(start),
                end=Datetime.fromisoformat(end) if end else None,
            )
    return None
