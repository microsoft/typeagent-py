# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from dataclasses import dataclass
import json
from pydantic.dataclasses import dataclass as pydantic_dataclass
import typechat
from ..aitools import utils
from ..knowpro import (
    secindex,
    convknowledge,
    search_query_schema,
    searchlang,
    answer_response_schema,
    answers
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
from ..storage.memory import (
    semrefindex,
)
from typeagent.storage.sqlite.provider import SqliteStorageProvider
        
from .email_message import EmailMessage

class EmailMemorySettings:
    def __init__(self, conversation_settings: ConversationSettings) -> None:
        self.language_model = convknowledge.create_typechat_model()
        self.query_translator = utils.create_translator(
            self.language_model, 
            search_query_schema.SearchQuery
        )
        self.answer_translator = utils.create_translator(
            self.language_model, 
            answer_response_schema.AnswerResponse
        )
        self.conversation_settings = conversation_settings
        self.conversation_settings.semantic_ref_index_settings.auto_extract_knowledge = True


@dataclass
class EmailMemory(IConversation[EmailMessage, ITermToSemanticRefIndex]):
    settings: EmailMemorySettings
    name_tag: str
    messages: IMessageCollection[EmailMessage]
    semantic_refs: ISemanticRefCollection
    tags: list[str]
    semantic_ref_index: ITermToSemanticRefIndex
    secondary_indexes: IConversationSecondaryIndexes[EmailMessage] | None

    noise_terms: set[str]

    @classmethod
    async def create(
        cls,
        settings: ConversationSettings,
        name_tag: str | None = None,
        messages: IMessageCollection[EmailMessage] | None = None,
        semantic_refs: ISemanticRefCollection | None = None,
        semantic_ref_index: ITermToSemanticRefIndex | None = None,
        tags: list[str] | None = None,
        secondary_indexes: IConversationSecondaryIndexes[EmailMessage] | None = None,
    ) -> "EmailMemory":

        noise_terms = set()
        storage_provider = await settings.get_storage_provider()
        email_memory = cls(
            EmailMemorySettings(settings),
            name_tag or "",
            messages or await storage_provider.get_message_collection(),
            semantic_refs or await storage_provider.get_semantic_ref_collection(),
            tags if tags is not None else [],
            semantic_ref_index or await storage_provider.get_semantic_ref_index(),
            secondary_indexes
            or await secindex.ConversationSecondaryIndexes.create(
                storage_provider, settings.related_term_index_settings
            ),
            noise_terms
        )

        # Add aliases for all the ways in which people can say 'send' and 'received'
        await _add_synonyms_file_as_aliases(email_memory, "emailVerbs.json")  
        # Remove common terms used in email search that can make retrieval noisy
        _add_noise_words_from_file(email_memory.noise_terms, "noiseTerms.txt")
        email_memory.noise_terms.add("email")
        email_memory.noise_terms.add("message")

        return email_memory

    # Add an email message to the memory.     
    async def add_message(self, message: EmailMessage) -> None:    
        await self.messages.append(message)
        self._commit()

    # Build an index using ALL messages in the memory
    async def build_index(
        self,
    ) -> None:
        await semrefindex.add_metadata_to_index(
            self.messages,
            self.semantic_refs,
            self.semantic_ref_index,
        )
        assert (
            self.settings is not None
        ), "Settings must be initialized before building index"

        self._commit()
        await semrefindex.build_semantic_ref(self, self.settings.conversation_settings)
        self._commit()
        await secindex.build_transient_secondary_indexes(self, self.settings.conversation_settings)
        self._commit()

    # Search email memory using language
    async def search_with_language(
        self,
        search_text: str,
        options: searchlang.LanguageSearchOptions | None = None,
        lang_search_filter: searchlang.LanguageSearchFilter | None = None,
        debug_context: searchlang.LanguageSearchDebugContext | None = None
    ) -> typechat.Result[list[searchlang.ConversationSearchResult]]:
        return await searchlang.search_conversation_with_language(
            self, 
            self.settings.query_translator,
            search_text,
            self._adjust_search_options(options),
            lang_search_filter,
            debug_context
        )
    
    async def get_answer_with_language(
        self,
        question: str,
        search_options: searchlang.LanguageSearchOptions | None = None,
        lang_search_filter: searchlang.LanguageSearchFilter |  None = None,
        answer_context_options: answers.AnswerContextOptions | None = None,
    ) -> typechat.Result[tuple[list[answers.AnswerResponse], answers.AnswerResponse]]:
        search_results = await self.search_with_language(
            question,
            search_options,
            lang_search_filter,
            None)        
        if isinstance(search_results, typechat.Failure):
            return search_results

        if answer_context_options is None:
            answer_context_options = EmailMemory.create_answer_context_options()
            
        answer = await answers.generate_answers(
                self.settings.answer_translator,
                search_results.value,
                self,
                question,
                answer_context_options,
            )
        return typechat.Success(answer)
        
    @staticmethod
    def create_lang_search_options() -> searchlang.LanguageSearchOptions :
        return searchlang.LanguageSearchOptions(
            compile_options = EmailMemory.create_lang_search_compile_options(),
            exact_match=False,
            max_knowledge_matches=50,
            max_message_matches=25,
        )

    @staticmethod
    def create_lang_search_compile_options() -> searchlang.LanguageQueryCompileOptions :
        return searchlang.LanguageQueryCompileOptions(
                apply_scope=True,
                exact_scope=False, 
                verb_scope=True, 
                term_filter=None 
            )
    
    @staticmethod
    def create_answer_context_options() -> answers.AnswerContextOptions:
        return answers.AnswerContextOptions(
            entities_top_k=50, 
            topics_top_k=50, 
            messages_top_k=None, 
            chunking=None
        )

    def _get_secondary_indexes(self) -> IConversationSecondaryIndexes[EmailMessage]:
        """Get secondary indexes, asserting they are initialized."""
        assert (
            self.secondary_indexes is not None
        ), "Use await f.create() to create an initialized instance"
        return self.secondary_indexes
    
    def _commit(self):
        provider = self.settings.conversation_settings.storage_provider
        if isinstance(provider, SqliteStorageProvider):
            provider.db.commit()

    def _adjust_search_options(self, options: searchlang.LanguageSearchOptions | None):
        if options is None:
            options = EmailMemory.create_lang_search_options()

        if options.compile_options is None:
            options.compile_options = EmailMemory.create_lang_search_compile_options()

        options.compile_options.term_filter = lambda term: self._is_searchable_term(term)
        return options
        
    def _is_searchable_term(self, term: str) -> bool:
        is_searchable = term not in self.noise_terms
        return is_searchable
    
#  
# TODO: Migrate some variation of these into a shared API
#

# Load synonyms from a file and add them as aliases 
async def _add_synonyms_file_as_aliases(conversation: IConversation, file_name: str) -> None:
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
        if (len(word) > 0):
            noise.add(word) 
