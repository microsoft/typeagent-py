# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from dataclasses import dataclass
import json
import copy
from pydantic.dataclasses import dataclass as pydantic_dataclass
import typechat
from ..aitools import utils
from ..knowpro import (
    secindex,
    convknowledge,
    search_query_schema,
    searchlang,
    answer_response_schema,
    answers,
)
from ..knowpro.convsettings import ConversationSettings
from ..knowpro.interfaces import (
    IConversation,
    IConversationSecondaryIndexes,
    IMessage,
    IMessageCollection,
    ISemanticRefCollection,
    ITermToSemanticRefIndex,
    Term,
)
from ..knowpro.conversation_base import ConversationBase
from ..storage.memory import semrefindex
from typeagent.storage.sqlite.provider import SqliteStorageProvider

from .email_message import EmailMessage


class EmailMemorySettings:
    def __init__(self, conversation_settings: ConversationSettings) -> None:
        self.language_model = convknowledge.create_typechat_model()
        self.query_translator = utils.create_translator(
            self.language_model, search_query_schema.SearchQuery
        )
        self.answer_translator = utils.create_translator(
            self.language_model, answer_response_schema.AnswerResponse
        )
        self.conversation_settings = conversation_settings
        self.conversation_settings.semantic_ref_index_settings.auto_extract_knowledge = (
            True
        )


@dataclass
class EmailMemory(ConversationBase[EmailMessage]):
    def __init__(self, settings, name, tags):
        super().__init__(settings, name, tags)
        self.noise_terms: set[str] = set()

    @staticmethod
    def create_lang_search_options() -> searchlang.LanguageSearchOptions:
        return searchlang.LanguageSearchOptions(
            compile_options=EmailMemory.create_lang_search_compile_options(),
            exact_match=False,
            max_knowledge_matches=50,
            max_message_matches=25,
        )

    @staticmethod
    def create_lang_search_compile_options() -> searchlang.LanguageQueryCompileOptions:
        return searchlang.LanguageQueryCompileOptions(
            apply_scope=True, exact_scope=False, verb_scope=True, term_filter=None
        )

    @staticmethod
    def create_answer_context_options() -> answers.AnswerContextOptions:
        return answers.AnswerContextOptions(
            entities_top_k=50, topics_top_k=50, messages_top_k=None, chunking=None
        )

    @classmethod
    async def create(
        cls,
        settings: ConversationSettings,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> "EmailMemory":
        instance = await super().create(settings, name, tags)
        await instance._configure_memory()
        return instance

    async def query(
        self,
        question: str,
        search_options: searchlang.LanguageSearchOptions | None = None,
        answer_options: answers.AnswerContextOptions | None = None,
    ) -> str:
        return await super().query(
            question,
            self._adjust_search_options(search_options),
            (
                answer_options
                if answer_options is not None
                else EmailMemory.create_answer_context_options()
            ),
        )

    # Search email memory using language
    async def query_debug(
        self,
        search_text: str,
        query_translator: typechat.TypeChatJsonTranslator[
            search_query_schema.SearchQuery
        ],
        debug_context: searchlang.LanguageSearchDebugContext | None = None,
    ) -> typechat.Result[list[searchlang.ConversationSearchResult]]:
        return await searchlang.search_conversation_with_language(
            self,
            query_translator,
            search_text,
            self._adjust_search_options(None),
            None,
            debug_context,
        )

    async def _configure_memory(self):
        # Adjust settings to support knowledge extraction from message ext
        self.settings.semantic_ref_index_settings.auto_extract_knowledge = True
        # Add aliases for all the ways in which people can say 'send' and 'received'
        await _add_synonyms_file_as_aliases(self, "emailVerbs.json", clean=True)
        # Remove common terms used in email search that can make retrieval noisy
        _add_noise_words_from_file(self.noise_terms, "noiseTerms.txt")

    def _adjust_search_options(
        self, options: searchlang.LanguageSearchOptions | None
    ) -> searchlang.LanguageSearchOptions:
        # TODO: should actually clone the object the caller passed
        if options is None:
            options = EmailMemory.create_lang_search_options()

        if options.compile_options is None:
            options.compile_options = EmailMemory.create_lang_search_compile_options()
        else:
            # Copy for modification
            options.compile_options = copy.copy(options.compile_options)

        options.compile_options.term_filter = lambda term: self._is_searchable_term(
            term
        )
        return options

    def _is_searchable_term(self, term: str) -> bool:
        is_searchable = term not in self.noise_terms
        return is_searchable


#
# TODO: Migrate some variation of these into a shared API
#


# Load synonyms from a file and add them as aliases
async def _add_synonyms_file_as_aliases(
    conversation: ConversationBase, file_name: str, clean: bool
) -> None:
    secondary_indexes = conversation.secondary_indexes
    assert secondary_indexes is not None
    assert secondary_indexes.term_to_related_terms_index is not None

    aliases = secondary_indexes.term_to_related_terms_index.aliases
    synonym_file = os.path.join(os.path.dirname(__file__), file_name)
    if not os.path.exists(synonym_file):
        return

    with open(synonym_file) as f:
        data: list[dict] = json.load(f)
    if data:
        storage_provider = conversation.settings.storage_provider
        async with storage_provider:
            if clean:
                await aliases.clear()
            for obj in data:
                text = obj.get("term")
                synonyms = obj.get("relatedTerms")
                if text and synonyms:
                    related_term = Term(text=text.lower())
                    for synonym in synonyms:
                        await aliases.add_related_term(synonym.lower(), related_term)


def _add_noise_words_from_file(
    noise: set[str],
    file_name: str,
) -> None:
    noise_file = os.path.join(os.path.dirname(__file__), file_name)
    if not os.path.exists(noise_file):
        return

    with open(noise_file) as f:
        words = f.readlines()
    for word in words:
        word = word.strip()
        if len(word) > 0:
            noise.add(word)
