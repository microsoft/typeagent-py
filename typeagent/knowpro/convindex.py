# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from typing import Callable

import typechat

from .interfaces import (
    # Interfaces.
    IConversation,
    IMessage,
    ITermToSemanticRefIndex,
    # Other imports.
    IndexingEventHandlers,
    IndexingResults,
    Knowledge,
    KnowledgeType,
    MessageOrdinal,
    SemanticRefOrdinal,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    TermToSemanticRefIndexItemData,
    TermToSemanticRefIndexData,
    TextIndexingResult,
    TextLocation,
    TextRange,
    Topic,
)
from . import convknowledge, importing, kplib, secindex


def text_range_from_message_chunk(
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> TextRange:
    return TextRange(
        start=TextLocation(message_ordinal, chunk_ordinal),
        end=None,
    )


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


def default_knowledge_validator(
    knowledg_Type: KnowledgeType,
    knowledge: Knowledge,
) -> bool:
    return True


def add_entity_to_index(
    entity: kplib.ConcreteEntity,
    semantic_refs: list[SemanticRef],
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> None:
    ref_ordinal = len(semantic_refs)
    semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=ref_ordinal,
            range=text_range_from_location(message_ordinal, chunk_ordinal),
            knowledge_type="entity",
            knowledge=entity,
        )
    )
    semantic_ref_index.add_term(entity.name, ref_ordinal)
    # Add each type as a separate term.
    for type in entity.type:
        semantic_ref_index.add_term(type, ref_ordinal)
    # Add every facet name as a separate term.
    if entity.facets:
        for facet in entity.facets:
            add_facet(facet, ref_ordinal, semantic_ref_index)


def add_facet(
    facet: kplib.Facet | None,
    ref_ordinal: int,
    semantic_ref_index: ITermToSemanticRefIndex,
) -> None:
    if facet is not None:
        semantic_ref_index.add_term(facet.name, ref_ordinal)
        semantic_ref_index.add_term(str(facet), ref_ordinal)


def add_topic_to_index(
    topic: Topic | str,
    semantic_refs: list[SemanticRef],
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    chunk_ordinal: int = 0,
) -> None:
    if isinstance(topic, str):
        topic = Topic(text=topic)
    ref_ordinal = len(semantic_refs)
    semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=ref_ordinal,
            range=text_range_from_location(message_ordinal, chunk_ordinal),
            knowledge_type="topic",
            knowledge=topic,
        )
    )
    semantic_ref_index.add_term(topic.text, ref_ordinal)


def add_action_to_index(
    action: kplib.Action,
    semantic_refs: list[SemanticRef],
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: int,
    chunk_ordinal: int = 0,
) -> None:
    ref_ordinal = len(semantic_refs)
    semantic_refs.append(
        SemanticRef(
            semantic_ref_ordinal=ref_ordinal,
            range=text_range_from_location(message_ordinal, chunk_ordinal),
            knowledge_type="action",
            knowledge=action,
        )
    )
    semantic_ref_index.add_term(" ".join(action.verbs), ref_ordinal)
    if action.subject_entity_name != "none":
        semantic_ref_index.add_term(action.subject_entity_name, ref_ordinal)
    if action.object_entity_name != "none":
        semantic_ref_index.add_term(action.object_entity_name, ref_ordinal)
    if action.indirect_object_entity_name != "none":
        semantic_ref_index.add_term(action.indirect_object_entity_name, ref_ordinal)
    if action.params:
        for param in action.params:
            if isinstance(param, str):
                semantic_ref_index.add_term(param, ref_ordinal)
            else:
                semantic_ref_index.add_term(param.name, ref_ordinal)
                if isinstance(param.value, str):
                    semantic_ref_index.add_term(param.value, ref_ordinal)
    add_facet(action.subject_entity_facet, ref_ordinal, semantic_ref_index)


def add_knowledge_to_index(
    semantic_refs: list[SemanticRef],
    semantic_ref_index: ITermToSemanticRefIndex,
    message_ordinal: MessageOrdinal,
    knowledge: kplib.KnowledgeResponse,
) -> None:
    for entity in knowledge.entities:
        add_entity_to_index(entity, semantic_refs, semantic_ref_index, message_ordinal)
    for action in knowledge.actions:
        add_action_to_index(action, semantic_refs, semantic_ref_index, message_ordinal)
    for inverse_action in knowledge.inverse_actions:
        add_action_to_index(
            inverse_action, semantic_refs, semantic_ref_index, message_ordinal
        )
    for topic in knowledge.topics:
        add_topic_to_index(topic, semantic_refs, semantic_ref_index, message_ordinal)


def add_metadata_to_index[TMessage: IMessage](
    messages: list[TMessage],
    semantic_refs: list[SemanticRef],
    semantic_ref_index: ITermToSemanticRefIndex,
    knowledge_validator: KnowledgeValidator | None = None,
) -> None:
    if knowledge_validator is None:
        knowledge_validator = default_knowledge_validator
    for i, msg in enumerate(messages):
        knowledge_response = msg.get_knowledge()
        for entity in knowledge_response.entities:
            if knowledge_validator("entity", entity):
                add_entity_to_index(entity, semantic_refs, semantic_ref_index, i)
        for action in knowledge_response.actions:
            if knowledge_validator("action", action):
                add_action_to_index(action, semantic_refs, semantic_ref_index, i)
        for topic_response in knowledge_response.topics:
            topic = Topic(text=topic_response)
            if knowledge_validator("topic", topic):
                add_topic_to_index(topic, semantic_refs, semantic_ref_index, i)


@dataclass
class ConversationIndex(ITermToSemanticRefIndex):
    _map: dict[str, list[ScoredSemanticRefOrdinal]]

    def __init__(self, data: TermToSemanticRefIndexData | None = None):
        self._map = {}
        if data:
            self.deserialize(data)

    def __len__(self) -> int:
        return len(self._map)

    # Needed because otherwise an empty index would be falsy.
    def __bool__(self) -> bool:
        return True

    def get_terms(self) -> list[str]:
        return list(self._map)

    def clear(self) -> None:
        self._map.clear()

    def add_term(
        self,
        term: str,
        semantic_ref_ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
    ) -> None:
        if not term:
            return
        if not isinstance(semantic_ref_ordinal, ScoredSemanticRefOrdinal):
            semantic_ref_ordinal = ScoredSemanticRefOrdinal(semantic_ref_ordinal, 1.0)
        term = self._prepare_term(term)
        existing = self._map.get(term)
        if existing is not None:
            existing.append(semantic_ref_ordinal)
        else:
            self._map[term] = [semantic_ref_ordinal]

    def lookup_term(self, term: str) -> list[ScoredSemanticRefOrdinal] | None:
        return self._map.get(self._prepare_term(term)) or []

    def remove_term(self, term: str, semantic_ref_ordinal: SemanticRefOrdinal) -> None:
        self._map.pop(self._prepare_term(term), None)

    def remove_term_if_empty(self, term: str) -> None:
        """Clean up a term if it has lost its last semantic reference."""
        term = self._prepare_term(term)
        if term in self._map and len(self._map[term]) == 0:
            self._map.pop(term)

    def serialize(self) -> TermToSemanticRefIndexData:
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

    def deserialize(self, data: TermToSemanticRefIndexData) -> None:
        self.clear()
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


async def build_conversation_index[TMessage: IMessage](
    conversation: IConversation[TMessage, ConversationIndex],
    conversation_settings: importing.ConversationSettings,
    event_handler: IndexingEventHandlers | None = None,
) -> IndexingResults:
    result = IndexingResults()
    result.semantic_refs = await build_semantic_ref_index(
        conversation, None, event_handler
    )
    if (
        result.semantic_refs
        and not result.semantic_refs.error
        and conversation.semantic_ref_index
    ):
        result.secondary_index_results = await secindex.build_secondary_indexes(
            conversation,  # type: ignore  # TODO
            conversation_settings,
            event_handler,
        )
    return result


async def build_semantic_ref_index[TM: IMessage](
    conversation: IConversation[TM, ConversationIndex],
    extractor: convknowledge.KnowledgeExtractor | None = None,
    event_handler: IndexingEventHandlers | None = None,
) -> TextIndexingResult:
    semantic_ref_index = conversation.semantic_ref_index
    if semantic_ref_index is None:
        conversation.semantic_ref_index = semantic_ref_index = ConversationIndex()

    semantic_refs = conversation.semantic_refs
    if semantic_refs is None:
        conversation.semantic_refs = semantic_refs = []

    if extractor is None:
        extractor = convknowledge.KnowledgeExtractor()

    indexing_result = TextIndexingResult()

    for message_ordinal, message in enumerate(conversation.messages):
        if event_handler and event_handler.on_message_started:
            if not event_handler.on_message_started(message_ordinal):
                break
        chunk_ordinal = 0
        # Only one chunk per message for now.
        text = message.text_chunks[chunk_ordinal]
        # TODO: retries (but beware that TypeChat already retries).
        match await extractor.extract(text):
            case typechat.Failure(error):
                indexing_result.error = f"Failed to extract knowledge from message {message_ordinal} ({text!r}): {error}"
                break
            case typechat.Success(knowledge):
                pass
        if (
            knowledge.entities
            or knowledge.actions
            or knowledge.inverse_actions
            or knowledge.topics
        ):
            add_knowledge_to_index(
                semantic_refs,
                semantic_ref_index,
                message_ordinal,
                knowledge,
            )
        completed_chunk = TextLocation(message_ordinal, chunk_ordinal)
        indexing_result.completed_upto = completed_chunk
        if event_handler and event_handler.on_knowledge_extracted:
            if not event_handler.on_knowledge_extracted(completed_chunk, knowledge):
                break

    # dump(semantic_ref_index, semantic_refs)

    return indexing_result


def dump(
    semantic_ref_index: ConversationIndex, semantic_refs: list[SemanticRef]
) -> None:
    print("semantic_ref_index = {")
    for k, v in semantic_ref_index._map.items():
        print(f"    {k!r}: {v},")
    print("}\n")
    print("semantic_refs = {")
    for semantic_ref in semantic_refs:
        print(f"    {semantic_ref},")
    print("}\n")
