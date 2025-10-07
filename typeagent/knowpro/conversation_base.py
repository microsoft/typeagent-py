# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base class for conversations with incremental indexing support."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from . import convknowledge, kplib, secindex
from ..storage.memory import semrefindex
from .convsettings import ConversationSettings
from .interfaces import (
    AddMessagesResult,
    IConversation,
    IConversationSecondaryIndexes,
    IMessage,
    IMessageCollection,
    ISemanticRefCollection,
    ITermToSemanticRefIndex,
    IndexingStartPoints,
    MessageOrdinal,
    Topic,
)

TMessage = TypeVar("TMessage", bound=IMessage)


@dataclass
class ConversationBase(
    Generic[TMessage], IConversation[TMessage, ITermToSemanticRefIndex]
):
    """Base class for conversations with incremental indexing support."""

    settings: ConversationSettings
    name_tag: str
    messages: IMessageCollection[TMessage]
    semantic_refs: ISemanticRefCollection
    tags: list[str]
    semantic_ref_index: ITermToSemanticRefIndex
    secondary_indexes: IConversationSecondaryIndexes[TMessage] | None

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
    ) -> AddMessagesResult:
        """
        Add messages and build all indexes incrementally in a single transaction.

        For SQLite: Either completely succeeds (all changes committed) or raises
        exception (all changes rolled back).

        For in-memory: No rollback support; partial changes may remain on error.

        Args:
            messages: Messages to add

        Returns:
            Result with counts of messages/semrefs added

        Raises:
            BaseException: Any error
        """
        storage = await self.settings.get_storage_provider()

        await storage.begin_transaction()

        try:
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

            await storage.commit_transaction()

            return result

        except BaseException:
            await storage.rollback_transaction()
            raise

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

        # Get batches of text locations from the message list
        from .messageutils import get_message_chunk_batch_from_list

        batches = get_message_chunk_batch_from_list(
            messages,
            start_from_message_ordinal,
            settings.batch_size,
        )
        for text_location_batch in batches:
            await semrefindex.add_batch_to_semantic_ref_index_from_list(
                self,
                messages,
                text_location_batch,
                knowledge_extractor,
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
