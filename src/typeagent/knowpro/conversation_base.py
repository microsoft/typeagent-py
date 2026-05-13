# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for conversations with incremental indexing support."""

from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, Protocol, Self, TypeVar

import typechat

from . import (
    answer_response_schema,
    answers,
    convknowledge,
)
from . import (
    search_query_schema,
    searchlang,
    secindex,
)
from . import knowledge_schema as kplib
from ..aitools import model_adapters, utils
from ..aitools.embeddings import NormalizedEmbedding
from ..storage.memory import semrefindex
from .convsettings import ConversationSettings
from .interfaces import (
    AddMessagesResult,
    IConversation,
    IConversationSecondaryIndexes,
    IMessage,
    IMessageCollection,
    IndexingStartPoints,
    ISemanticRefCollection,
    IStorageProvider,
    ITermToSemanticRefIndex,
    MessageOrdinal,
    Topic,
)
from .interfaces_core import TextLocation
from .messageutils import get_all_message_chunk_locations

TMessage = TypeVar("TMessage", bound=IMessage)


class _ChunkCommitResult(Protocol):
    """Neutral chunk commit payload shape used by pipeline batch commit."""

    chunk_id: TextLocation
    chunk_count: int
    extracted_knowledge: kplib.KnowledgeResponse | None
    chunk_embedding: NormalizedEmbedding | None
    related_terms: list[str] | None
    related_term_embeddings: list[NormalizedEmbedding] | None


@dataclass(init=False)
class ConversationBase(
    Generic[TMessage], IConversation[TMessage, ITermToSemanticRefIndex]
):
    """Base class for conversations with incremental indexing support."""

    settings: ConversationSettings
    storage_provider: IStorageProvider[TMessage]
    name_tag: str
    tags: list[str]
    messages: IMessageCollection[TMessage]
    semantic_refs: ISemanticRefCollection
    semantic_ref_index: ITermToSemanticRefIndex
    secondary_indexes: IConversationSecondaryIndexes[TMessage] | None

    # Private cached translators
    _query_translator: (
        typechat.TypeChatJsonTranslator[search_query_schema.SearchQuery] | None
    ) = None
    _answer_translator: (
        typechat.TypeChatJsonTranslator[answer_response_schema.AnswerResponse] | None
    ) = None

    def __init__(
        self,
        settings: ConversationSettings,
        name: str,
        tags: list[str],
    ):
        """Initialize conversation with storage provider.

        Collections and indexes are obtained from the settings' storage provider
        by the create() factory method.
        """
        self.settings = settings
        self.name_tag = name
        self.tags = tags

    @classmethod
    async def create(
        cls,
        settings: ConversationSettings,
        name: str | None = None,
        tags: list[str] | None = None,
    ) -> Self:
        """Create a fully initialized conversation instance."""
        storage_provider = await settings.get_storage_provider()
        instance = cls(
            settings,
            name or "",
            tags if tags is not None else [],
        )
        instance.storage_provider = storage_provider
        instance.messages = storage_provider.messages
        instance.semantic_refs = storage_provider.semantic_refs
        instance.semantic_ref_index = storage_provider.semantic_ref_index
        instance.secondary_indexes = secindex.ConversationSecondaryIndexes(
            storage_provider, settings.related_term_index_settings
        )
        return instance

    def _get_secondary_indexes(self) -> IConversationSecondaryIndexes[TMessage]:
        """Get secondary indexes, asserting they are initialized."""
        assert (
            self.secondary_indexes is not None
        ), f"Use await {self.__class__.__name__}.create() to create an initialized instance"
        return self.secondary_indexes

    async def add_metadata_to_index(self) -> None:
        """Add metadata knowledge to the semantic reference index."""
        await semrefindex.add_metadata_to_index(
            self.messages,
            self.semantic_refs,
            self.semantic_ref_index,
        )

    async def add_messages_with_indexing(
        self,
        messages: list[TMessage],
        *,
        source_ids: list[str] | None = None,
    ) -> AddMessagesResult:
        """
        Add messages and build all indexes incrementally in a single transaction.

        For SQLite: Either completely succeeds (all changes committed) or raises
        exception (all changes rolled back).

        For in-memory: No rollback support; partial changes may remain on error.

        Args:
            messages: Messages to add
            source_ids: Optional explicit list of source IDs to mark as ingested,
                one per message. When ``None`` (the default), each message's
                ``source_id`` attribute is used instead — messages whose
                ``source_id`` is ``None`` are silently skipped.  These are marked
                within the same transaction, so if the indexing fails, the source
                IDs won't be marked as ingested (for SQLite storage).

        Returns:
            Result with counts of messages/semrefs added

        Raises:
            Exception: Any error
        """
        storage = await self.settings.get_storage_provider()
        if source_ids is not None:
            if len(source_ids) != len(messages):
                raise ValueError(
                    f"Length of source_ids {len(source_ids)} "
                    f"must match length of messages {len(messages)}"
                )

        async with storage:
            # Mark source IDs as ingested (will be rolled back on error)
            sids = (
                source_ids
                if source_ids is not None
                else [m.source_id for m in messages if m.source_id is not None]
            )
            if sids:
                await storage.mark_sources_ingested_batch(sids)

            start_points = IndexingStartPoints(
                message_count=await self.messages.size(),
                semref_count=await self.semantic_refs.size(),
            )

            await self.messages.extend(messages)

            await self._add_metadata_knowledge_incremental(start_points.message_count)

            if self.settings.semantic_ref_index_settings.auto_extract_knowledge:
                # Add LLM-extracted knowledge
                await self._add_llm_knowledge_incremental(
                    messages, start_points.message_count
                )

            await self._update_secondary_indexes_incremental(start_points)

            messages_added = await self.messages.size() - start_points.message_count
            chunks_added = sum(len(m.text_chunks) for m in messages[:messages_added])
            result = AddMessagesResult(
                messages_added=messages_added,
                chunks_added=chunks_added,
                semrefs_added=await self.semantic_refs.size()
                - start_points.semref_count,
            )

            # Update the updated_at timestamp
            await storage.update_conversation_timestamps(
                updated_at=datetime.now(timezone.utc)
            )

            return result

    async def add_messages_streaming(
        self,
        messages: AsyncIterable[TMessage],
        *,
        batch_size: int = 100,
        on_batch_committed: Callable[[AddMessagesResult], None] | None = None,
        skip_failed_messages: bool = False,
    ) -> AddMessagesResult:
        """Delegate to the pipelined add_messages implementation."""
        from . import add_messages

        return await add_messages.add_messages_streaming(
            self,
            messages,
            batch_size=batch_size,
            on_batch_committed=on_batch_committed,
            skip_failed_messages=skip_failed_messages,
        )

    async def _commit_batch_from_chunk_results(
        self,
        storage: IStorageProvider[TMessage],
        messages_batch: list[TMessage],
        chunk_results: Sequence[_ChunkCommitResult],
    ) -> AddMessagesResult:
        """Commit one pipeline batch using precomputed extraction and embeddings."""
        if not messages_batch:
            return AddMessagesResult()

        async with storage:
            start_points = IndexingStartPoints(
                message_count=await self.messages.size(),
                semref_count=await self.semantic_refs.size(),
            )

            await self.messages.extend(messages_batch)

            source_ids = [
                m.source_id for m in messages_batch if m.source_id is not None
            ]
            if source_ids:
                await storage.mark_sources_ingested_batch(source_ids)

            await self._add_metadata_knowledge_incremental(start_points.message_count)

            knowledge_items: list[
                tuple[MessageOrdinal, int, kplib.KnowledgeResponse]
            ] = []
            fuzzy_terms: list[str] = []
            fuzzy_term_embeddings: list[NormalizedEmbedding] = []

            for result in chunk_results:
                if result.chunk_count == 0:
                    continue

                if result.chunk_embedding is None:
                    raise ValueError(
                        "Chunk result missing chunk embedding for "
                        f"message={result.chunk_id.message_ordinal}, "
                        f"chunk={result.chunk_id.chunk_ordinal}"
                    )

                if result.extracted_knowledge is None:
                    raise ValueError(
                        "Chunk result missing extracted knowledge for "
                        f"message={result.chunk_id.message_ordinal}, "
                        f"chunk={result.chunk_id.chunk_ordinal}"
                    )
                knowledge_items.append(
                    (
                        result.chunk_id.message_ordinal,
                        result.chunk_id.chunk_ordinal,
                        result.extracted_knowledge,
                    )
                )

                if (
                    result.related_terms is None
                    or result.related_term_embeddings is None
                ):
                    raise ValueError(
                        "Chunk result missing related-term embeddings for "
                        f"message={result.chunk_id.message_ordinal}, "
                        f"chunk={result.chunk_id.chunk_ordinal}"
                    )
                if len(result.related_terms) != len(result.related_term_embeddings):
                    raise ValueError(
                        "related_terms and related_term_embeddings length mismatch for "
                        f"message={result.chunk_id.message_ordinal}, "
                        f"chunk={result.chunk_id.chunk_ordinal}: "
                        f"{len(result.related_terms)} != "
                        f"{len(result.related_term_embeddings)}"
                    )
                fuzzy_terms.extend(result.related_terms)
                fuzzy_term_embeddings.extend(result.related_term_embeddings)

            await semrefindex.add_knowledge_batch_to_semantic_ref_index(
                self,
                knowledge_items,
            )

            await self._update_secondary_indexes_incremental_with_embeddings(
                start_points,
                messages_batch,
                fuzzy_terms,
                fuzzy_term_embeddings,
            )

            await storage.update_conversation_timestamps(
                updated_at=datetime.now(timezone.utc)
            )

            messages_added = await self.messages.size() - start_points.message_count
            chunks_added = sum(
                len(message.text_chunks) for message in messages_batch[:messages_added]
            )
            return AddMessagesResult(
                messages_added=messages_added,
                chunks_added=chunks_added,
                semrefs_added=await self.semantic_refs.size()
                - start_points.semref_count,
            )

    async def _update_secondary_indexes_incremental_with_embeddings(
        self,
        start_points: IndexingStartPoints,
        new_messages: list[TMessage],
        related_terms: list[str],
        related_term_embeddings: list[NormalizedEmbedding],
    ) -> None:
        """Update secondary indexes using precomputed embeddings when available."""
        if self.secondary_indexes is None:
            return

        from ..storage.memory import propindex

        await propindex.add_to_property_index(self, start_points.semref_count)

        await self._add_timestamps_for_messages(
            new_messages,
            start_points.message_count,
        )

        term_to_related = self.secondary_indexes.term_to_related_terms_index
        if term_to_related is not None:
            fuzzy_index = term_to_related.fuzzy_index
            if fuzzy_index is not None and related_terms:
                await fuzzy_index.add_terms_with_embeddings(
                    related_terms,
                    related_term_embeddings,
                )

    async def _add_metadata_knowledge_incremental(
        self,
        start_from_message_ordinal: int,
    ) -> None:
        """Extract metadata knowledge from messages starting at ordinal."""
        messages_slice = await self.messages.get_slice(
            start_from_message_ordinal,
            999_999_999,
        )
        await semrefindex.add_metadata_to_index_from_list(
            messages_slice,
            self.semantic_refs,
            self.semantic_ref_index,
            start_from_message_ordinal,
        )

    async def _add_llm_knowledge_incremental(
        self,
        messages: list[TMessage],
        start_from_message_ordinal: int,
    ) -> None:
        """Extract LLM knowledge from messages starting at ordinal."""
        settings = self.settings.semantic_ref_index_settings
        if not settings.auto_extract_knowledge:
            return

        knowledge_extractor = (
            settings.knowledge_extractor or convknowledge.KnowledgeExtractor()
        )

        text_locations = get_all_message_chunk_locations(
            messages,
            start_from_message_ordinal,
        )
        await semrefindex.add_batch_to_semantic_ref_index_from_list(
            self,
            messages,
            text_locations,
            knowledge_extractor,
            concurrency=settings.concurrency,
        )

    async def _update_secondary_indexes_incremental(
        self,
        start_points: IndexingStartPoints,
    ) -> None:
        """Update all secondary indexes with new data."""
        if self.secondary_indexes is None:
            return

        from ..storage.memory import propindex

        await propindex.add_to_property_index(self, start_points.semref_count)

        new_messages = await self.messages.get_slice(
            start_points.message_count,
            999_999_999,
        )
        await self._add_timestamps_for_messages(
            new_messages,
            start_points.message_count,
        )

        await self._update_related_terms_incremental(start_points.semref_count)

        # Update message text index with new messages
        await self._update_message_index_incremental(
            new_messages,
            start_points.message_count,
        )

    async def _add_timestamps_for_messages(
        self,
        messages: list[TMessage],
        start_ordinal: MessageOrdinal,
    ) -> None:
        """Add timestamps for new messages to the timestamp index."""
        if (
            self.secondary_indexes is None
            or self.secondary_indexes.timestamp_index is None
        ):
            return

        timestamp_data: list[tuple[MessageOrdinal, str]] = []
        for i, msg in enumerate(messages, start_ordinal):
            if msg.timestamp:
                timestamp_data.append((i, msg.timestamp))

        if timestamp_data:
            await self.secondary_indexes.timestamp_index.add_timestamps(timestamp_data)

    async def _update_related_terms_incremental(
        self,
        start_from_semref_ordinal: int,
    ) -> None:
        """Update related terms index with new semantic refs."""
        if (
            self.secondary_indexes is None
            or self.secondary_indexes.term_to_related_terms_index is None
        ):
            return

        new_semrefs = await self.semantic_refs.get_slice(
            start_from_semref_ordinal,
            999_999_999,
        )

        fuzzy_index = self.secondary_indexes.term_to_related_terms_index.fuzzy_index
        if fuzzy_index is not None and new_semrefs:
            new_terms = set()
            for semref in new_semrefs:
                knowledge = semref.knowledge
                if isinstance(knowledge, kplib.ConcreteEntity):
                    new_terms.add(knowledge.name.lower())
                elif isinstance(knowledge, Topic):
                    new_terms.add(knowledge.text.lower())
                elif isinstance(knowledge, kplib.Action):
                    for verb in knowledge.verbs:
                        new_terms.add(verb.lower())

            if new_terms:
                await fuzzy_index.add_terms(list(new_terms))

    async def _update_message_index_incremental(
        self,
        new_messages: list[TMessage],
        start_ordinal: MessageOrdinal,
    ) -> None:
        """Update message text index with new messages."""
        if (
            self.secondary_indexes is None
            or self.secondary_indexes.message_index is None
        ):
            return

        # The message index add_messages handles the ordinal tracking internally
        await self.secondary_indexes.message_index.add_messages(new_messages)

    # Use options to customize number of messages to match, topK etc.
    async def query(
        self,
        question: str,
        search_options: searchlang.LanguageSearchOptions | None = None,
        answer_options: answers.AnswerContextOptions | None = None,
    ) -> str:
        """
        Run an end-to-end query on the conversation.

        This method performs a natural language search and generates an answer
        based on the conversation content.

        Args:
            question: The natural language question to answer

        Returns:
            A natural language answer string. If the answer cannot be determined,
            returns an explanation of why no answer was found.

        Example:
            >>> answer = await conv.query("What topics were discussed?")
            >>> print(answer)
        """
        # Create translators lazily (once per conversation instance)
        if self._query_translator is None:
            model = model_adapters.create_chat_model(retrier=self.settings.chat_retrier)
            self._query_translator = utils.create_translator(
                model, search_query_schema.SearchQuery
            )
        if self._answer_translator is None:
            model = model_adapters.create_chat_model(retrier=self.settings.chat_retrier)
            self._answer_translator = utils.create_translator(
                model, answer_response_schema.AnswerResponse
            )

        # Stage 1-3: Search the conversation with the natural language query
        if search_options is None:
            search_options = searchlang.LanguageSearchOptions(
                compile_options=searchlang.LanguageQueryCompileOptions(
                    exact_scope=False,
                    verb_scope=True,
                    term_filter=None,
                    apply_scope=True,
                ),
                exact_match=False,
                max_message_matches=25,
            )

        result = await searchlang.search_conversation_with_language(
            self,
            self._query_translator,
            question,
            search_options,
        )

        if isinstance(result, typechat.Failure):
            return f"Search failed: {result.message}"

        search_results = result.value

        # Stage 4: Generate answer from search results
        if answer_options is None:
            answer_options = answers.AnswerContextOptions(
                entities_top_k=50, topics_top_k=50, messages_top_k=None, chunking=None
            )

        _, combined_answer = await answers.generate_answers(
            self._answer_translator,
            search_results,
            self,
            question,
            options=answer_options,
        )

        match combined_answer.type:
            case "NoAnswer":
                return f"No answer found: {combined_answer.why_no_answer or 'Unable to find relevant information'}"
            case "Answered":
                return combined_answer.answer or "No answer provided"
            case _:  # Cannot happen in type-checked code
                return f"Unexpected answer type: {combined_answer.type}"
