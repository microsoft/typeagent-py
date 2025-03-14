# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
import os

import typechat

from . import kplib


def create_typechat_model() -> typechat.TypeChatLanguageModel:
    return typechat.create_language_model(dict(os.environ))


@dataclass
class KnowledgeExtractor:

    def __init__(self, model: typechat.TypeChatLanguageModel | None = None):
        if model is None:
            model = create_typechat_model()
        assert model is not None
        self.model = model
        self.translator = self.create_translator(self.model)

    async def extract(self, message: str) -> kplib.KnowledgeResponse | None:
        result: typechat.Result[kplib.KnowledgeResponse] = await self.extract_knowledge(
            message
        )
        if isinstance(result, typechat.Success):
            return result.value
        else:
            return None

    async def extract_knowledge(
        self, message: str
    ) -> typechat.Result[kplib.KnowledgeResponse]:
        result = await self.translator.translate(message)
        # TODO
        # if isinstance(result, typechat.Success):
        #     self.merge_action_knowledge(result.data)
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
        schema_text = translator._schema_str

        def create_request_prompt(intent: str) -> str:
            return (
                f'You are a service that translates user messages in a conversation into JSON objects of type "{type_name}" according to the following TypeScript definitions:\n'
                + f"```\n{schema_text}```\n"
                + f"The following are messages in a conversation:\n"
                + f'"""\n{intent}\n"""\n'
                + f"The following is the user request translated into a JSON object with 2 spaces of indentation and no properties with the value undefined:\n"
            )

        translator._create_request_prompt = create_request_prompt
        return translator
