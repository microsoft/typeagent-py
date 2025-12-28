# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations  # TODO: Avoid

from collections.abc import AsyncIterable, Callable

from typechat import Failure

from ...knowpro import convknowledge, kplib, secindex
from ...knowpro.convsettings import ConversationSettings
from ...knowpro.convsettings import SemanticRefIndexSettings
from ...knowpro.interfaces import (
    # Interfaces.
    IConversation,
    IKnowledgeExtractor,
    IMessage,
    ISemanticRefCollection,
    ITermToSemanticRefIndex,
    # Other imports.
    Knowledge,
    KnowledgeType,
    MessageOrdinal,
    SemanticRefOrdinal,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    TermToSemanticRefIndexItemData,
    TermToSemanticRefIndexData,
    TextLocation,
    TextRange,
    Topic,
)
from ...knowpro.messageutils import (
    get_message_chunk_batch,
    text_range_from_message_chunk,
)
from ...knowpro.knowledge import extract_knowledge_from_text_batch


def text_range_from_location(
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> TextRange:
    return TextRange(
        start=TextLocation(message_ordinal, chunk_ordinal),
        end=None,
    )


type KnowledgeValidator = Callable[
    [
        KnowledgeType,  # knowledge_type
        Knowledge,  # knowledge
    ],
    bool,
]


async def add_batch_to_semantic_ref_index[
    TMessage: IMessage, TTermToSemanticRefIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TTermToSemanticRefIndex],
    batch: list[TextLocation],
    knowledge_extractor: IKnowledgeExtractor,
    terms_added: set[str] | None = None,
) -> None:
    messages = conversation.messages

    text_batch = [
        (await messages.get_item(tl.message_ordinal))
        .text_chunks[tl.chunk_ordinal]
        .strip()
        for tl in batch
    ]

    knowledge_results = await extract_knowledge_from_text_batch(
        knowledge_extractor,
        text_batch,
        len(text_batch),
    )
    for i, knowledge_result in enumerate(knowledge_results):
        if isinstance(knowledge_result, Failure):
            raise RuntimeError(
                f"Knowledge extraction failed: {knowledge_result.message}"
            )
        text_location = batch[i]
        knowledge = knowledge_result.value
        await add_knowledge_to_semantic_ref_index(
            conversation,
            text_location.message_ordinal,
            text_location.chunk_ordinal,
            knowledge,
            terms_added,
        )


async def add_batch_to_semantic_ref_index_from_list[
    TMessage: IMessage, TTermToSemanticRefIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TTermToSemanticRefIndex],
    messages: list[TMessage],
    batch: list[TextLocation],
    knowledge_extractor: IKnowledgeExtractor,
    terms_added: set[str] | None = None,
) -> None:
    """
    Add a batch of knowledge to semantic ref index, extracting from provided message list.

    Args:
        conversation: The conversation containing semantic refs and index
        messages: List of messages containing the text to extract from
        batch: List of text locations (ordinals) to process
        knowledge_extractor: Extractor for LLM-based knowledge extraction
        terms_added: Optional set to track newly added terms
    """
    # Get the starting ordinal of the message list
    if not batch:
        return
    start_ordinal = batch[0].message_ordinal

    # Extract text from the messages list
    text_batch = []
    for tl in batch:
        # Calculate index in the list from the ordinal
        list_index = tl.message_ordinal - start_ordinal
        if list_index < 0 or list_index >= len(messages):
            raise IndexError(
                f"Message ordinal {tl.message_ordinal} out of range for list starting at {start_ordinal}"
            )
        message = messages[list_index]
        text = message.text_chunks[tl.chunk_ordinal].strip()
        text_batch.append(text)

    knowledge_results = await extract_knowledge_from_text_batch(
        knowledge_extractor,
        text_batch,
        len(text_batch),
    )
    for i, knowledge_result in enumerate(knowledge_results):
        if isinstance(knowledge_result, Failure):
            raise RuntimeError(
                f"Knowledge extraction failed: {knowledge_result.message}"
            )
        text_location = batch[i]
        knowledge = knowledge_result.value
        await add_knowledge_to_semantic_ref_index(
            conversation,
            text_location.message_ordinal,
            text_location.chunk_ordinal,
            knowledge,
            terms_added,
        )


async def add_entity_to_index(
    entity: kplib.ConcreteEntity,
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> None:
    ref_ordinal = await semantic_refs.size()
    await semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=ref_ordinal,
            range=text_range_from_location(message_ordinal, chunk_ordinal),
            knowledge=entity,
        )
    )
    await semantic_ref_index.add_term(entity.name, ref_ordinal)
    # Add each type as a separate term.
    for type in entity.type:
        await semantic_ref_index.add_term(type, ref_ordinal)
    # Add every facet name as a separate term.
    if entity.facets:
        for facet in entity.facets:
            await add_facet(facet, ref_ordinal, semantic_ref_index)


async def add_term_to_index(
    index: ITermToSemanticRefIndex,
    term: str,
    semantic_ref_ordinal: SemanticRefOrdinal,
    terms_added: set[str] | None = None,
) -> None:
    """Add a term to the semantic reference index.

    Args:
        index: The index to add the term to
        term: The term to add
        semantic_ref_ordinal: Ordinal of the semantic reference
        terms_added: Optional set to track terms added to the index
    """
    term = await index.add_term(term, semantic_ref_ordinal)
    if terms_added is not None:
        terms_added.add(term)


async def add_entity(
    entity: kplib.ConcreteEntity,
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int,
    terms_added: set[str] | None = None,
) -> None:
    """Add an entity to the semantic reference index.

    Args:
        entity: The concrete entity to add
        semantic_refs: Collection of semantic references to add to
        semantic_ref_index: Index to add terms to
        message_ordinal: Ordinal of the message containing the entity
        chunk_ordinal: Ordinal of the chunk within the message
        terms_added: Optional set to track terms added to the index
    """
    semantic_ref_ordinal = await semantic_refs.size()
    await semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=semantic_ref_ordinal,
            range=text_range_from_message_chunk(message_ordinal, chunk_ordinal),
            knowledge=entity,
        )
    )
    await add_term_to_index(
        semantic_ref_index,
        entity.name,
        semantic_ref_ordinal,
        terms_added,
    )

    # Add each type as a separate term
    for type_name in entity.type:
        await add_term_to_index(
            semantic_ref_index, type_name, semantic_ref_ordinal, terms_added
        )

    # Add every facet name as a separate term
    if entity.facets:
        for facet in entity.facets:
            await add_facet(facet, semantic_ref_ordinal, semantic_ref_index)


async def add_facet(
    facet: kplib.Facet | None,
    semantic_ref_ordinal: SemanticRefOrdinal,
    semantic_ref_index: ITermToSemanticRefIndex,
    terms_added: set[str] | None = None,
) -> None:
    if facet is not None:
        await add_term_to_index(
            semantic_ref_index,
            facet.name,
            semantic_ref_ordinal,
            terms_added,
        )
        if facet.value is not None:
            await add_term_to_index(
                semantic_ref_index,
                str(facet.value),
                semantic_ref_ordinal,
                terms_added,
            )
        # semantic_ref_index.add_term(facet.name, ref_ordinal)
        # semantic_ref_index.add_term(str(facet), ref_ordinal)


async def add_topic(
    topic: Topic,
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int,
    terms_added: set[str] | None = None,
) -> None:
    """Add a topic to the semantic reference index.

    Args:
        topic: The topic to add
        semantic_refs: Collection of semantic references to add to
        semantic_ref_index: Index to add terms to
        message_ordinal: Ordinal of the message containing the topic
        chunk_ordinal: Ordinal of the chunk within the message
        terms_added: Optional set to track terms added to the index
    """
    semantic_ref_ordinal = await semantic_refs.size()
    await semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=semantic_ref_ordinal,
            range=text_range_from_message_chunk(message_ordinal, chunk_ordinal),
            knowledge=topic,
        )
    )

    await add_term_to_index(
        semantic_ref_index,
        topic.text,
        semantic_ref_ordinal,
        terms_added,
    )


async def add_action(
    action: kplib.Action,
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int,
    terms_added: set[str] | None = None,
) -> None:
    """Add an action to the semantic reference index.

    Args:
        action: The action to add
        semantic_refs: Collection of semantic references to add to
        semantic_ref_index: Index to add terms to
        message_ordinal: Ordinal of the message containing the action
        chunk_ordinal: Ordinal of the chunk within the message
        terms_added: Optional set to track terms added to the index
    """
    semantic_ref_ordinal = await semantic_refs.size()
    await semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=semantic_ref_ordinal,
            range=text_range_from_message_chunk(message_ordinal, chunk_ordinal),
            knowledge=action,
        )
    )

    await add_term_to_index(
        semantic_ref_index,
        " ".join(action.verbs),
        semantic_ref_ordinal,
        terms_added,
    )

    if action.subject_entity_name != "none":
        await add_term_to_index(
            semantic_ref_index,
            action.subject_entity_name,
            semantic_ref_ordinal,
            terms_added,
        )

    if action.object_entity_name != "none":
        await add_term_to_index(
            semantic_ref_index,
            action.object_entity_name,
            semantic_ref_ordinal,
            terms_added,
        )

    if action.indirect_object_entity_name != "none":
        await add_term_to_index(
            semantic_ref_index,
            action.indirect_object_entity_name,
            semantic_ref_ordinal,
            terms_added,
        )

    if action.params:
        for param in action.params:
            if isinstance(param, str):
                await add_term_to_index(
                    semantic_ref_index,
                    param,
                    semantic_ref_ordinal,
                    terms_added,
                )
            else:
                await add_term_to_index(
                    semantic_ref_index,
                    param.name,
                    semantic_ref_ordinal,
                    terms_added,
                )
                if isinstance(param.value, str):
                    await add_term_to_index(
                        semantic_ref_index,
                        param.value,
                        semantic_ref_ordinal,
                        terms_added,
                    )

    await add_facet(
        action.subject_entity_facet,
        semantic_ref_ordinal,
        semantic_ref_index,
        terms_added,
    )


# TODO: add_tag
# TODO:L KnowledgeValidator


async def add_knowledge_to_semantic_ref_index(
    conversation: IConversation,
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int,
    knowledge: kplib.KnowledgeResponse,
    terms_added: set[str] | None = None,
) -> None:
    """Add knowledge to the semantic reference index of a conversation.

    Args:
        conversation: The conversation to add knowledge to
        message_ordinal: Ordinal of the message containing the knowledge
        chunk_ordinal: Ordinal of the chunk within the message
        knowledge: Knowledge response containing entities, actions and topics
        terms_added: Optional set to track terms added to the index
    """
    verify_has_semantic_ref_index(conversation)

    semantic_refs = conversation.semantic_refs
    assert semantic_refs is not None
    semantic_ref_index = conversation.semantic_ref_index
    assert semantic_ref_index is not None

    for entity in knowledge.entities:
        if validate_entity(entity):
            await add_entity(
                entity,
                semantic_refs,
                semantic_ref_index,
                message_ordinal,
                chunk_ordinal,
                terms_added,
            )

    for action in knowledge.actions:
        await add_action(
            action,
            semantic_refs,
            semantic_ref_index,
            message_ordinal,
            chunk_ordinal,
            terms_added,
        )

    for inverse_action in knowledge.inverse_actions:
        await add_action(
            inverse_action,
            semantic_refs,
            semantic_ref_index,
            message_ordinal,
            chunk_ordinal,
            terms_added,
        )

    for topic in knowledge.topics:
        topic_obj = Topic(text=topic)
        await add_topic(
            topic_obj,
            semantic_refs,
            semantic_ref_index,
            message_ordinal,
            chunk_ordinal,
            terms_added,
        )


def validate_entity(entity: kplib.ConcreteEntity) -> bool:
    return bool(entity.name)


async def add_topic_to_index(
    topic: Topic | str,
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> None:
    if isinstance(topic, str):
        topic = Topic(text=topic)
    ref_ordinal = await semantic_refs.size()
    await semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=ref_ordinal,
            range=text_range_from_location(message_ordinal, chunk_ordinal),
            knowledge=topic,
        )
    )
    await semantic_ref_index.add_term(topic.text, ref_ordinal)


async def add_action_to_index(
    action: kplib.Action,
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: int,
    chunk_ordinal: int = 0,
) -> None:
    ref_ordinal = await semantic_refs.size()
    await semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=ref_ordinal,
            range=text_range_from_location(message_ordinal, chunk_ordinal),
            knowledge=action,
        )
    )
    await semantic_ref_index.add_term(" ".join(action.verbs), ref_ordinal)
    if action.subject_entity_name != "none":
        await semantic_ref_index.add_term(action.subject_entity_name, ref_ordinal)
    if action.object_entity_name != "none":
        await semantic_ref_index.add_term(action.object_entity_name, ref_ordinal)
    if action.indirect_object_entity_name != "none":
        await semantic_ref_index.add_term(
            action.indirect_object_entity_name, ref_ordinal
        )
    if action.params:
        for param in action.params:
            if isinstance(param, str):
                await semantic_ref_index.add_term(param, ref_ordinal)
            else:
                await semantic_ref_index.add_term(param.name, ref_ordinal)
                if isinstance(param.value, str):
                    await semantic_ref_index.add_term(param.value, ref_ordinal)
    await add_facet(action.subject_entity_facet, ref_ordinal, semantic_ref_index)


async def add_knowledge_to_index(
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    knowledge: kplib.KnowledgeResponse,
) -> None:
    for entity in knowledge.entities:
        await add_entity_to_index(
            entity, semantic_refs, semantic_ref_index, message_ordinal
        )
    for action in knowledge.actions:
        await add_action_to_index(
            action, semantic_refs, semantic_ref_index, message_ordinal
        )
    for inverse_action in knowledge.inverse_actions:
        await add_action_to_index(
            inverse_action, semantic_refs, semantic_ref_index, message_ordinal
        )
    for topic in knowledge.topics:
        await add_topic_to_index(
            topic, semantic_refs, semantic_ref_index, message_ordinal
        )


async def add_metadata_to_index[TMessage: IMessage](
    messages: AsyncIterable[TMessage],
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    knowledge_validator: KnowledgeValidator | None = None,
) -> None:
    # Find the highest message ordinal already processed
    # by checking existing semantic refs
    start_from_ordinal = 0
    existing_ref_count = await semantic_refs.size()
    if existing_ref_count > 0:
        # Get the last semantic ref to find the highest processed message ordinal
        last_ref = await semantic_refs.get_item(existing_ref_count - 1)
        if last_ref.range and last_ref.range.start:
            start_from_ordinal = last_ref.range.start.message_ordinal + 1

    i = 0
    async for msg in messages:
        # Skip messages that were already processed
        if i < start_from_ordinal:
            i += 1
            continue

        knowledge_response = msg.get_knowledge()
        for entity in knowledge_response.entities:
            if knowledge_validator is None or knowledge_validator("entity", entity):
                await add_entity_to_index(entity, semantic_refs, semantic_ref_index, i)
        for action in knowledge_response.actions:
            if knowledge_validator is None or knowledge_validator("action", action):
                await add_action_to_index(action, semantic_refs, semantic_ref_index, i)
        for topic_response in knowledge_response.topics:
            topic = Topic(text=topic_response)
            if knowledge_validator is None or knowledge_validator("topic", topic):
                await add_topic_to_index(topic, semantic_refs, semantic_ref_index, i)
        i += 1


async def add_metadata_to_index_from_list[TMessage: IMessage](
    messages: list[TMessage],
    semantic_refs: ISemanticRefCollection,
    semantic_ref_index: ITermToSemanticRefIndex,
    start_from_ordinal: MessageOrdinal,
    knowledge_validator: KnowledgeValidator | None = None,
) -> None:
    """Extract metadata knowledge from a list of messages starting at ordinal."""
    for i, msg in enumerate(messages, start_from_ordinal):
        knowledge_response = msg.get_knowledge()
        for entity in knowledge_response.entities:
            if knowledge_validator is None or knowledge_validator("entity", entity):
                await add_entity_to_index(entity, semantic_refs, semantic_ref_index, i)
        for action in knowledge_response.actions:
            if knowledge_validator is None or knowledge_validator("action", action):
                await add_action_to_index(action, semantic_refs, semantic_ref_index, i)
        for topic_response in knowledge_response.topics:
            topic = Topic(text=topic_response)
            if knowledge_validator is None or knowledge_validator("topic", topic):
                await add_topic_to_index(topic, semantic_refs, semantic_ref_index, i)


class TermToSemanticRefIndex(ITermToSemanticRefIndex):
    _map: dict[str, list[ScoredSemanticRefOrdinal]]

    def __init__(self):
        super().__init__()
        self._map = {}

    async def size(self) -> int:
        return len(self._map)

    async def get_terms(self) -> list[str]:
        return list(self._map)

    async def clear(self) -> None:
        self._clear()

    def _clear(self) -> None:
        self._map.clear()

    async def add_term(
        self,
        term: str,
        semantic_ref_ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
    ) -> str:
        if not term:
            return term
        if not isinstance(semantic_ref_ordinal, ScoredSemanticRefOrdinal):
            semantic_ref_ordinal = ScoredSemanticRefOrdinal(semantic_ref_ordinal, 1.0)
        term = self._prepare_term(term)
        existing = self._map.get(term)
        if existing is not None:
            existing.append(semantic_ref_ordinal)
        else:
            self._map[term] = [semantic_ref_ordinal]
        return term

    async def lookup_term(self, term: str) -> list[ScoredSemanticRefOrdinal] | None:
        return self._map.get(self._prepare_term(term)) or []

    async def remove_term(
        self, term: str, semantic_ref_ordinal: SemanticRefOrdinal
    ) -> None:
        term = self._prepare_term(term)
        if term in self._map:
            # Remove only the specific semantic ref ordinal, not the entire term
            scored_refs = self._map[term]
            self._map[term] = [
                ref
                for ref in scored_refs
                if ref.semantic_ref_ordinal != semantic_ref_ordinal
            ]
            # Clean up empty terms
            if not self._map[term]:
                del self._map[term]

    async def serialize(self) -> TermToSemanticRefIndexData:
        items: list[TermToSemanticRefIndexItemData] = []
        for term, scored_semantic_ref_ordinals in self._map.items():
            items.append(
                TermToSemanticRefIndexItemData(
                    term=term,
                    semanticRefOrdinals=[
                        s.serialize() for s in scored_semantic_ref_ordinals
                    ],
                )
            )
        return TermToSemanticRefIndexData(items=items)

    async def deserialize(self, data: TermToSemanticRefIndexData) -> None:
        self._clear()
        for index_item_data in data["items"]:
            term = index_item_data.get("term")
            term = self._prepare_term(term)
            scored_refs_data = index_item_data["semanticRefOrdinals"]
            scored_refs = [
                ScoredSemanticRefOrdinal.deserialize(s) for s in scored_refs_data
            ]
            self._map[term] = scored_refs

    def _prepare_term(self, term: str) -> str:
        return term.lower()


# ...


async def build_semantic_ref[TMessage: IMessage](
    conversation: IConversation[TMessage, ITermToSemanticRefIndex],
    conversation_settings: ConversationSettings,
) -> None:
    await build_semantic_ref_index(
        conversation,
        conversation_settings.semantic_ref_index_settings,
    )
    if conversation.semantic_ref_index is not None:
        await secindex.build_secondary_indexes(
            conversation,
            conversation_settings,
        )


async def build_semantic_ref_index[TM: IMessage](
    conversation: IConversation[TM, ITermToSemanticRefIndex],
    settings: SemanticRefIndexSettings,
) -> None:
    # For LLM-based knowledge extraction, we need to track separately from metadata extraction
    # For now, always start from 0 to process all messages
    # TODO: Implement proper tracking of which messages have had LLM extraction
    await add_to_semantic_ref_index(conversation, settings, 0)


async def add_to_semantic_ref_index[
    TMessage: IMessage, TTermToSemanticRefIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TTermToSemanticRefIndex],
    settings: SemanticRefIndexSettings,
    message_ordinal_start_at: MessageOrdinal,
    terms_added: set[str] | None = None,
) -> None:
    """Add semantic references to the conversation's semantic reference index."""

    # Only create knowledge extractor if auto extraction is enabled
    knowledge_extractor = None
    if settings.auto_extract_knowledge:
        knowledge_extractor = (
            settings.knowledge_extractor or convknowledge.KnowledgeExtractor()
        )

        # Process messages in batches for LLM knowledge extraction
        batches = await get_message_chunk_batch(
            conversation.messages,
            message_ordinal_start_at,
            settings.batch_size,
        )
        for text_location_batch in batches:
            await add_batch_to_semantic_ref_index(
                conversation,
                text_location_batch,
                knowledge_extractor,
                terms_added,
            )


def verify_has_semantic_ref_index(conversation: IConversation) -> None:
    if conversation.secondary_indexes is None or conversation.semantic_refs is None:
        raise ValueError("Conversation does not have an index")


async def dump(
    semantic_ref_index: TermToSemanticRefIndex, semantic_refs: ISemanticRefCollection
) -> None:
    print("semantic_ref_index = {")
    for k, v in semantic_ref_index._map.items():
        print(f"    {k!r}: {v},")
    print("}\n")
    print("semantic_refs = []")
    async for semantic_ref in semantic_refs:
        print(f"    {semantic_ref},")
    print("]\n")
