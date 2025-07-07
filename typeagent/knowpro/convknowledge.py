# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from dataclasses import dataclass, field
import os

import typechat

from ..aitools import auth
from . import kplib


# TODO: Move ModelWrapper and create_typechat_model() to aitools package.


class ModelWrapper(typechat.TypeChatLanguageModel):
    def __init__(
        self,
        base_model: typechat.TypeChatLanguageModel,
        token_provider: auth.AzureTokenProvider,
    ):
        self.base_model = base_model
        self.token_provider = token_provider

    async def complete(
        self, prompt: str | list[typechat.PromptSection]
    ) -> typechat.Result[str]:
        if self.token_provider.needs_refresh():
            loop = asyncio.get_running_loop()
            api_key = await loop.run_in_executor(
                None, self.token_provider.refresh_token
            )
            env: dict[str, str | None] = dict(os.environ)
            key_name = "AZURE_OPENAI_API_KEY"
            env[key_name] = api_key
            self.base_model = typechat.create_language_model(env)
        return await self.base_model.complete(prompt)


def create_typechat_model() -> typechat.TypeChatLanguageModel:
    env: dict[str, str | None] = dict(os.environ)
    key_name = "AZURE_OPENAI_API_KEY"
    key = env.get(key_name)
    shared_token_provider: auth.AzureTokenProvider | None = None
    if key is not None and key.lower() == "identity":
        shared_token_provider = auth.get_shared_token_provider()
        env[key_name] = shared_token_provider.get_token()
    model = typechat.create_language_model(env)
    if shared_token_provider is not None:
        model = ModelWrapper(model, shared_token_provider)
    return model


@dataclass
class KnowledgeExtractor:
    model: typechat.TypeChatLanguageModel = field(default_factory=create_typechat_model)
    max_chars_per_chunk: int = 2048
    merge_action_knowledge: bool = True
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
        schema_text = translator.schema_str.rstrip()  # type: ignore  # Must access internal.

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

        translator._create_request_prompt = create_request_prompt  # type: ignore  # Must overwrite internal.
        return translator

    def merge_action_knowledge_into_response(
        self, knowledge: kplib.KnowledgeResponse
    ) -> None:
        """Merge action knowledge into a single knowledge object."""
        raise NotImplementedError("TODO")
