# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field

import typechat

from . import kplib
from ..aitools.utils import create_typechat_model  # Re-export for backward compat

# Re-export: callers may still do ``convknowledge.create_typechat_model()``.
__all__ = ["create_typechat_model", "KnowledgeExtractor"]


@dataclass
class KnowledgeExtractor:
    model: typechat.TypeChatLanguageModel = field(default_factory=create_typechat_model)
    max_chars_per_chunk: int = 2048
    merge_action_knowledge: bool = (
        False  # TODO: Implement merge_action_knowledge_into_response
    )
    # Not in the signature:
    translator: typechat.TypeChatJsonTranslator[kplib.KnowledgeResponse] = field(
        init=False
    )

    def __post_init__(self):
        self.translator = self.create_translator(self.model)

    # TODO: Use max_chars_per_chunk and merge_action_knowledge.

    async def extract(self, message: str) -> typechat.Result[kplib.KnowledgeResponse]:
        result = await self.translator.translate(message)
        if isinstance(result, typechat.Success):
            if self.merge_action_knowledge:
                self.merge_action_knowledge_into_response(result.value)
        else:
            result.message += f" -- MESSAGE={message!r}"
        return result

    def create_translator(
        self, model: typechat.TypeChatLanguageModel
    ) -> typechat.TypeChatJsonTranslator[kplib.KnowledgeResponse]:
        schema = kplib.KnowledgeResponse
        type_name = "KnowledgeResponse"
        validator = typechat.TypeChatValidator[kplib.KnowledgeResponse](schema)
        translator = typechat.TypeChatJsonTranslator[kplib.KnowledgeResponse](
            model, validator, kplib.KnowledgeResponse
        )
        schema_text = translator.schema_str.rstrip()

        def create_request_prompt(intent: str) -> str:
            return (
                f"You are a service that translates user messages in a conversation "
                + f'into JSON objects of type "{type_name}" '
                + f"according to the following TypeScript definitions:\n"
                + f"```\n"
                + f"{schema_text}\n"
                + f"```\n"
                + f"The following are messages in a conversation:\n"
                + f'"""\n'
                + f"{intent}\n"
                + f'"""\n'
                + f"The following is the user request translated into a JSON object "
                + f"with 2 spaces of indentation and no properties with the value undefined:\n"
            )

        translator._create_request_prompt = create_request_prompt
        return translator

    def merge_action_knowledge_into_response(
        self, knowledge: kplib.KnowledgeResponse
    ) -> None:
        """Merge action knowledge into a single knowledge object."""
        raise NotImplementedError("TODO")
