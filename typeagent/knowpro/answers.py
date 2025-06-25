# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import black
import typechat

from .answer_context_schema import AnswerContext, RelevantKnowledge, RelevantMessage
from .answer_response_schema import AnswerResponse
from .collections import Scored, get_top_k
from .interfaces import (
    IConversation,
    IMessage,
    ISemanticRefCollection,
    ITermToSemanticRefIndex,
    Knowledge,
    KnowledgeType,
    MessageOrdinal,
    ScoredMessageOrdinal,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    SemanticRefSearchResult,
    Topic,
)
from .kplib import ConcreteEntity, Facet
from .search import ConversationSearchResult


@dataclass
class AnswerContextOptions:
    entities_top_k: int | None = None
    topics_top_k: int | None = None
    messages_top_k: int | None = None
    chunking: bool | None = None


async def generate_answers(
    translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    search_results: list[ConversationSearchResult],
    conversation: IConversation,
    orig_query_text: str,
    options: AnswerContextOptions | None = None,
) -> tuple[list[AnswerResponse], AnswerResponse]:  # (all answers, combined answer)
    all_answers: list[AnswerResponse] = []
    good_answers: list[str] = []
    for i, search_result in enumerate(search_results):
        for j, result in enumerate(search_results):
            answer = await generate_answer(translator, result, conversation, options)
            all_answers.append(answer)
            match answer.type:
                case "Answered":
                    assert answer.answer is not None, "Answered answer must not be None"
                    good = answer.answer.strip()
                    if good:
                        good_answers.append(good)
                case "NoAnswer":
                    pass
                case _:
                    raise ValueError(f"Unexpected answer type: {answer.type}")
    combined_answer: AnswerResponse | None = None
    if len(good_answers) >= 2:
        combined_answer = await combine_answers(
            translator, good_answers, orig_query_text
        )
    elif len(good_answers) == 1:
        combined_answer = AnswerResponse(type="Answered", answer=good_answers[0])
    else:
        combined_answer = AnswerResponse(
            type="NoAnswer", whyNoAnswer="No good answers found."
        )
    return all_answers, combined_answer


async def generate_answer[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    search_result: ConversationSearchResult,
    conversation: IConversation[TMessage, TIndex],
    options: AnswerContextOptions | None = None,
) -> AnswerResponse:
    assert search_result.raw_query_text is not None, "Raw query text must not be None"
    context = make_context(search_result, conversation, options)
    request = f"{create_question_prompt(search_result.raw_query_text)}\n{create_context_prompt(context)}"
    # print("+" * 80)
    # print(request)
    # print("+" * 80)
    result = await translator.translate(request)
    if isinstance(result, typechat.Failure):
        return AnswerResponse(type="NoAnswer", answer=None, whyNoAnswer=result.message)
    else:
        return result.value


def create_question_prompt(question: str) -> str:
    prompt = [
        "The following is a user question:",
        "===",
        question,
        "===",
        "- The included [ANSWER CONTEXT] contains information that MAY be relevant to answering the question.",
        "- Answer the user question PRECISELY using ONLY relevant topics, entities, actions, messages and time ranges/timestamps found in [ANSWER CONTEXT].",
        "- Return 'NoAnswer' if unsure or if the topics and entity names/types in the question are not in [ANSWER CONTEXT].",
        "- Use the 'name', 'type' and 'facets' properties of the provided JSON entities to identify those highly relevant to answering the question.",
        "- When asked for lists, ensure the the list contents answer the question and nothing else.",
        "E.g. for the question 'List all books': List only the books in [ANSWER CONTEXT].",
        "- Use direct quotes only when needed or asked. Otherwise answer in your own words.",
        "- Your answer is readable and complete, with appropriate formatting: line breaks, numbered lists, bullet points etc.",
    ]
    return "\n".join(prompt)


def create_context_prompt(context: AnswerContext) -> str:
    # TODO: Use a more compact representation of the context than JSON.
    prompt = [
        "[ANSWER CONTEXT]",
        "===",
        black.format_str(str(dictify(context)), mode=black.FileMode(line_length=200)),
        "===",
    ]
    return "\n".join(prompt)


def dictify(object: object) -> Any:
    """Convert an object to a dictionary, recursively."""
    # NOTE: Can't use dataclasses.asdict() because not every object is a dataclass.
    if ann := getattr(object.__class__, "__annotations__", None):
        return {
            k: dictify(v) for k in ann if (v := getattr(object, k, None)) is not None
        }
    elif isinstance(object, dict):
        return {k: dictify(v) for k, v in object.items() if v is not None}
    elif isinstance(object, list):
        return [dictify(item) for item in object]
    elif hasattr(object, "__dict__"):
        return {
            k: dictify(v) for k, v in object.__dict__.items() if v is not None
        }  #  if not k.startswith("_")
    else:
        if isinstance(object, float) and object.is_integer():
            return int(object)
        else:
            return object


def make_context[TMessage: IMessage, TIndex: ITermToSemanticRefIndex](
    search_result: ConversationSearchResult,
    conversation: IConversation[TMessage, TIndex],
    options: AnswerContextOptions | None = None,
) -> AnswerContext:
    context = AnswerContext([], [], [])

    if search_result.message_matches:
        context.messages = get_relevant_messages_for_answer(
            conversation,
            search_result.message_matches,
            options and options.messages_top_k,
        )

    for ktype, knowledge in search_result.knowledge_matches.items():
        match ktype:
            case "entity":
                context.entities = get_relevant_entities_for_answer(
                    conversation,
                    knowledge,
                    options and options.entities_top_k,
                )
            case "topic":
                context.topics = get_relevant_topics_for_answer(
                    conversation,
                    knowledge,
                    options and options.topics_top_k,
                )
            case _:
                pass  # TODO: Actions and tags (once we support them)?

    return context


type MergedFacets = dict[str, list[str]]


# NOT a dataclass -- an optional merge-in attribute for MergedEntity etc.
class MergedKnowledge:
    source_message_ordinals: set[MessageOrdinal] | None = None


@dataclass
class MergedTopic(MergedKnowledge):
    topic: Topic


@dataclass
class MergedEntity(MergedKnowledge):
    name: str
    type: list[str]
    facets: MergedFacets | None = None


def get_relevant_messages_for_answer[
    TMessage: IMessage, TIndex: ITermToSemanticRefIndex
](
    conversation: IConversation[TMessage, TIndex],
    message_matches: list[ScoredMessageOrdinal],
    top_k: int | None = None,
) -> list[RelevantMessage]:
    relevant_messages = []

    for scored_msg_ord in message_matches:
        msg = conversation.messages[scored_msg_ord.message_ordinal]
        if not msg.text_chunks:
            continue
        # TODO: Type-safety: In TS, message.metadata has source/dest attrs. Copy that.
        relevant_messages.append(
            RelevantMessage(
                from_=msg.speaker,  # type: ignore  # It's a PodcastMessage
                to=msg.listeners,  # type: ignore  # It's a PodcastMessage
                timestamp=msg.timestamp,  # type: ignore  # It's a PodcastMessage
                messageText=(
                    msg.text_chunks[0] if len(msg.text_chunks) == 1 else msg.text_chunks
                ),
            )
        )
        if top_k and len(relevant_messages) >= top_k:
            break

    return relevant_messages


def get_relevant_topics_for_answer(
    conversation: IConversation,
    search_result: SemanticRefSearchResult,
    top_k: int | None = None,
) -> list[RelevantKnowledge]:
    assert conversation.semantic_refs is not None, "Semantic refs must not be None"
    scored_topics: Iterable[Scored[SemanticRef]] = (
        get_scored_semantic_refs_from_ordinals_iter(
            conversation.semantic_refs,
            search_result.semantic_ref_matches,
            "topic",
        )
    )
    merged_topics = merge_scored_topics(scored_topics, True)
    candidate_topics: Iterable[Scored[MergedTopic]] = merged_topics.values()
    if top_k and len(merged_topics) > top_k:
        candidate_topics = get_top_k(candidate_topics, top_k)

    relevant_topics: list[RelevantKnowledge] = []

    for scored_value in candidate_topics:
        merged_topic = scored_value.item
        relevant_topics.append(
            create_relevant_knowledge(
                conversation,
                merged_topic.topic,
                merged_topic.source_message_ordinals,
            )
        )

    return relevant_topics


def merge_scored_topics(
    scored_topics: Iterable[Scored[SemanticRef]],
    merge_ordinals: bool,
) -> dict[str, Scored[MergedTopic]]:
    merged_topics: dict[str, Scored[MergedTopic]] = {}

    for scored_topic in scored_topics:
        assert isinstance(scored_topic.item.knowledge, Topic)
        topic = scored_topic.item.knowledge
        existing = merged_topics.get(topic.text)
        if existing is not None:
            assert existing.item.topic.text == topic.text
            # Merge scores.
            if existing.score < scored_topic.score:
                existing.score = scored_topic.score
        else:
            existing = Scored(
                item=MergedTopic(topic=topic),
                score=scored_topic.score,
            )
            merged_topics[topic.text] = existing
        if merge_ordinals:
            merge_message_ordinals(existing.item, scored_topic.item)

    return merged_topics


def get_relevant_entities_for_answer(
    conversation: IConversation,
    search_result: SemanticRefSearchResult,
    top_k: int | None = None,
) -> list[RelevantKnowledge]:
    assert conversation.semantic_refs is not None, "Semantic refs must not be None"
    merged_entities = merge_scored_concrete_entities(
        get_scored_semantic_refs_from_ordinals_iter(
            conversation.semantic_refs,
            search_result.semantic_ref_matches,
            "entity",
        ),
        merge_ordinals=True,
    )
    candidate_entities = merged_entities.values()
    if top_k and len(merged_entities) > top_k:
        candidate_entities = get_top_k(candidate_entities, top_k)

    relevant_entities: list[RelevantKnowledge] = []

    for scored_value in candidate_entities:
        merged_entity = scored_value.item
        relevane_entity = create_relevant_knowledge(
            conversation,
            merged_to_concrete_entity(merged_entity),
            merged_entity.source_message_ordinals,
        )
        relevant_entities.append(relevane_entity)

    return relevant_entities


def create_relevant_knowledge(
    conversation: IConversation,
    knowledge: Knowledge,
    source_message_ordinals: set[MessageOrdinal] | None = None,
) -> RelevantKnowledge:
    relevant_knowledge = RelevantKnowledge(knowledge)

    if source_message_ordinals:
        # TODO: getEnclosingDateRangeForMessages(...) etc.
        relevant_knowledge.timeRange = None
        relevant_knowledge.origin = None
        relevant_knowledge.audience = None
    return relevant_knowledge


def get_scored_semantic_refs_from_ordinals_iter(
    semantic_refs: ISemanticRefCollection,
    semantic_ref_matches: list[ScoredSemanticRefOrdinal],
    knowledge_type: KnowledgeType,
) -> Iterator[Scored[SemanticRef]]:
    for semantic_ref_match in semantic_ref_matches:
        semantic_ref = semantic_refs[semantic_ref_match.semantic_ref_ordinal]
        if semantic_ref.knowledge_type == knowledge_type:
            yield Scored(
                item=semantic_ref,
                score=semantic_ref_match.score,
            )


def merge_scored_concrete_entities(
    scored_entities: Iterable[Scored[SemanticRef]],
    merge_ordinals: bool,
) -> dict[str, Scored[MergedEntity]]:
    merged_entities: dict[str, Scored[MergedEntity]] = {}

    for scored_entity in scored_entities:
        assert isinstance(scored_entity.item.knowledge, ConcreteEntity)
        merged_entity = concrete_to_merged_entity(
            scored_entity.item.knowledge,
        )
        existing = merged_entities.get(merged_entity.name)
        if existing is not None:
            assert existing.item.name == merged_entity.name
            # Merge type list.
            if not existing.item.type:
                existing.item.type = merged_entity.type
            elif merged_entity.type:
                existing.item.type = sorted(
                    set(existing.item.type) | set(merged_entity.type)
                )
            # Merge facet dicts.
            if not existing.item.facets:
                existing.item.facets = merged_entity.facets
            elif merged_entity.facets:
                for name, value in merged_entity.facets.items():
                    existing.item.facets.setdefault(name, []).extend(value)
            # Merge scores.
            if existing.score < scored_entity.score:
                existing.score = scored_entity.score
        else:
            existing = Scored(
                item=merged_entity,
                score=scored_entity.score,
            )
            merged_entities[merged_entity.name] = existing
        if existing and merge_ordinals:
            merge_message_ordinals(existing.item, scored_entity.item)

    return merged_entities


def merge_message_ordinals(merged_entity: MergedKnowledge, sr: SemanticRef) -> None:
    if merged_entity.source_message_ordinals is None:
        merged_entity.source_message_ordinals = set()
    merged_entity.source_message_ordinals.add(sr.range.start.message_ordinal)


def concrete_to_merged_entity(
    entity: ConcreteEntity,
) -> MergedEntity:
    return MergedEntity(
        name=entity.name.lower(),
        type=sorted(tp.lower() for tp in entity.type),
        facets=facets_to_merged_facets(entity.facets) if entity.facets else None,
    )


def merged_to_concrete_entity(merged_entity: MergedEntity) -> ConcreteEntity:
    entity = ConcreteEntity(name=merged_entity.name, type=merged_entity.type)
    if merged_entity.facets:
        entity.facets = merged_facets_to_facets(merged_entity.facets)
    return entity


def facets_to_merged_facets(facets: list[Facet]) -> MergedFacets:
    merged_facets: MergedFacets = {}
    for facet in facets:
        name = facet.name.lower()
        value = str(facet).lower()
        merged_facets.setdefault(name, []).append(value)
    return merged_facets


def merged_facets_to_facets(merged_facets: MergedFacets) -> list[Facet]:
    facets: list[Facet] = []
    for facet_name, facet_values in merged_facets.items():
        if facet_values:
            facets.append(Facet(name=facet_name, value="; ".join(facet_values)))
    return facets


async def combine_answers(
    translator: typechat.TypeChatJsonTranslator[AnswerResponse],
    answers: list[str],
    original_query_text: str,
) -> AnswerResponse:
    """Combine multiple answers into a single answer."""
    if not answers:
        return AnswerResponse(type="NoAnswer", whyNoAnswer="No answers provided.")
    if len(answers) == 1:
        return AnswerResponse(type="Answered", answer=answers[0])
    request_parts = [
        "The following are multiple partial answers to the same question.",
        "Combine the partial answers into a single answer to the original question.",
        "Don't just concatenate the answers, but blend them into a single accurate and precise answer.",
        "",
        "*** Original Question ***",
        original_query_text,
        "*** Partial answers ***",
        "===",
    ]
    for answer in answers:
        request_parts.append(answer.strip())
        request_parts.append("===")
    request = "\n".join(request_parts)
    result = await translator.translate(request)
    if isinstance(result, typechat.Failure):
        return AnswerResponse(type="NoAnswer", whyNoAnswer=result.message)
    else:
        return result.value
