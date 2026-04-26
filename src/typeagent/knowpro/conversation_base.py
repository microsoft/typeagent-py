# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for conversations with incremental indexing support."""

from collections.abc import AsyncIterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Generic, Self, TypeVar

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
from .knowledge import extract_knowledge_from_text_batch
from .messageutils import get_all_message_chunk_locations

TMessage = TypeVar("TMessage", bound=IMessage)


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
        instance.messages = await storage_provider.get_message_collection()
        instance.semantic_refs = await storage_provider.get_semantic_ref_collection()
        instance.semantic_ref_index = await storage_provider.get_semantic_ref_index()
        instance.secondary_indexes = await secindex.ConversationSecondaryIndexes.create(
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
            if source_ids is not None:
                for sid in source_ids:
                    await storage.mark_source_ingested(sid)
            else:
                for msg in messages:
                    if msg.source_id is not None:
                        await storage.mark_source_ingested(msg.source_id)

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

            result = AddMessagesResult(
                messages_added=await self.messages.size() - start_points.message_count,
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
    ) -> AddMessagesResult:
        """Add messages from an async iterable, committing in batches.

        Unlike ``add_messages_with_indexing`` which processes all messages in a
        single transaction, this method buffers messages into batches of
        ``batch_size``, processes each batch in its own transaction, and commits
        after every batch.  This makes it suitable for ingesting large streams
        (millions of messages) where a single all-or-nothing transaction would
        be impractical.

        **Source-ID tracking**: each message's ``source_id`` (if not ``None``)
        is checked before ingestion.  Already-ingested sources are silently
        skipped.  Newly ingested sources are marked within the same transaction.

        **Extraction failures**: when knowledge extraction returns a
        ``Failure`` for a chunk, the failure is recorded via
        ``storage.record_chunk_failure`` and processing continues with the
        remaining chunks.  Raised exceptions (HTTP errors, timeouts, etc.)
        are treated as systemic and stop the run immediately — the current
        batch is rolled back and the exception propagates.

        Args:
            messages: An async iterable of messages to ingest.
            batch_size: Number of messages per commit batch.

        Returns:
            Cumulative ``AddMessagesResult`` across all committed batches.
        """
        storage = await self.settings.get_storage_provider()
        total_messages_added = 0
        total_semrefs_added = 0

        batch: list[TMessage] = []
        async for msg in messages:
            batch.append(msg)
            if len(batch) >= batch_size:
                result = await self._ingest_batch_streaming(storage, batch)
                total_messages_added += result.messages_added
                total_semrefs_added += result.semrefs_added
                batch = []

        # Flush remaining messages
        if batch:
            result = await self._ingest_batch_streaming(storage, batch)
            total_messages_added += result.messages_added
            total_semrefs_added += result.semrefs_added

        return AddMessagesResult(
            messages_added=total_messages_added,
            semrefs_added=total_semrefs_added,
        )

    async def _ingest_batch_streaming(
        self,
        storage: IStorageProvider[TMessage],
        batch: list[TMessage],
    ) -> AddMessagesResult:
        """Process and commit a single batch within a transaction.

        Messages whose ``source_id`` is already ingested are filtered out.
        Extraction ``Failure``\\s are recorded as chunk failures.
        """
        # Filter out already-ingested sources
        filtered: list[TMessage] = []
        for msg in batch:
            if msg.source_id is not None and await storage.is_source_ingested(
                msg.source_id
            ):
                continue
            filtered.append(msg)

        if not filtered:
            return AddMessagesResult(messages_added=0, semrefs_added=0)

        async with storage:
            start_points = IndexingStartPoints(
                message_count=await self.messages.size(),
                semref_count=await self.semantic_refs.size(),
            )

            await self.messages.extend(filtered)

            # Mark source IDs as ingested (rolled back on error)
            for msg in filtered:
                if msg.source_id is not None:
                    await storage.mark_source_ingested(msg.source_id)

            await self._add_metadata_knowledge_incremental(start_points.message_count)

            if self.settings.semantic_ref_index_settings.auto_extract_knowledge:
                await self._add_llm_knowledge_streaming(
                    storage, filtered, start_points.message_count
                )

            await self._update_secondary_indexes_incremental(start_points)

            await storage.update_conversation_timestamps(
                updated_at=datetime.now(timezone.utc)
            )

            return AddMessagesResult(
                messages_added=await self.messages.size() - start_points.message_count,
                semrefs_added=await self.semantic_refs.size()
                - start_points.semref_count,
            )

    async def _add_llm_knowledge_streaming(
        self,
        storage: IStorageProvider[TMessage],
        messages: list[TMessage],
        start_from_message_ordinal: int,
    ) -> None:
        """Extract LLM knowledge, recording failures instead of raising.

        On ``Failure``: records a chunk failure via the storage provider and
        continues.  On a raised exception: lets it propagate (the caller's
        ``async with storage`` will roll back the transaction).
        """
        settings = self.settings.semantic_ref_index_settings
        knowledge_extractor = (
            settings.knowledge_extractor or convknowledge.KnowledgeExtractor()
        )

        text_locations = get_all_message_chunk_locations(
            messages, start_from_message_ordinal
        )
        if not text_locations:
            return

        start_ordinal = text_locations[0].message_ordinal
        text_batch: list[str] = []
        for tl in text_locations:
            list_index = tl.message_ordinal - start_ordinal
            text_batch.append(
                messages[list_index].text_chunks[tl.chunk_ordinal].strip()
            )

        knowledge_results = await extract_knowledge_from_text_batch(
            knowledge_extractor,
            text_batch,
            settings.concurrency,
        )
        for i, knowledge_result in enumerate(knowledge_results):
            tl = text_locations[i]
            if isinstance(knowledge_result, typechat.Failure):
                await storage.record_chunk_failure(
                    tl.message_ordinal,
                    tl.chunk_ordinal,
                    type(knowledge_result).__name__,
                    knowledge_result.message[:500],
                )
                continue
            await semrefindex.add_knowledge_to_semantic_ref_index(
                self,
                tl.message_ordinal,
                tl.chunk_ordinal,
                knowledge_result.value,
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
            model = model_adapters.create_chat_model()
            self._query_translator = utils.create_translator(
                model, search_query_schema.SearchQuery
            )
        if self._answer_translator is None:
            model = model_adapters.create_chat_model()
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
