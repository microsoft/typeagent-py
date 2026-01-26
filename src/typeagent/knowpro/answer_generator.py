# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Answer generation utilities based on AnswerContext."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import typechat

from ..aitools import utils
from .answer_context import (
    answer_context_to_string,
    AnswerContextChunkBuilder,
    AnswerContextOptions,
)
from .answer_context_schema import AnswerContext
from .answer_response_schema import AnswerResponse
from .answers import (
    get_relevant_entities_for_answer,
    get_relevant_messages_for_answer,
    get_relevant_topics_for_answer,
)
from .convknowledge import create_typechat_model
from .interfaces import IConversation
from .search import ConversationSearchResult
from .searchlib import create_multiple_choice_question

type AnswerTranslator = typechat.TypeChatJsonTranslator[AnswerResponse]

type ProcessProgress = Callable[
    [AnswerContext, int, typechat.Result[AnswerResponse]], None
]


def create_answer_translator(
    model: typechat.TypeChatLanguageModel,
) -> AnswerTranslator:
    return utils.create_translator(model, AnswerResponse)


class IAnswerGenerator(Protocol):
    """Protocol for answer generators."""

    settings: "AnswerGeneratorSettings"

    async def generate_answer(
        self, question: str, context: AnswerContext | str
    ) -> typechat.Result[AnswerResponse]: ...

    async def combine_partial_answers(
        self, question: str, responses: Sequence[AnswerResponse | None]
    ) -> typechat.Result[AnswerResponse]: ...


@dataclass
class AnswerGeneratorSettings:
    answer_generator_model: typechat.TypeChatLanguageModel
    answer_combiner_model: typechat.TypeChatLanguageModel
    max_chars_in_budget: int
    concurrency: int
    fast_stop: bool
    model_instructions: list[typechat.PromptSection] | None = None
    include_context_schema: bool | None = None


def create_answer_generator_settings(
    model: typechat.TypeChatLanguageModel | None = None,
) -> AnswerGeneratorSettings:
    answer_generator_model = model or create_typechat_model()
    answer_combiner_model = create_typechat_model()
    return AnswerGeneratorSettings(
        answer_generator_model=answer_generator_model,
        answer_combiner_model=answer_combiner_model,
        max_chars_in_budget=4096 * 4,
        concurrency=2,
        fast_stop=True,
    )


async def generate_answer(
    conversation: IConversation,
    generator: IAnswerGenerator,
    question: str,
    search_results: ConversationSearchResult | list[ConversationSearchResult],
    progress: ProcessProgress | None = None,
    context_options: AnswerContextOptions | None = None,
) -> typechat.Result[AnswerResponse]:
    if not isinstance(search_results, list):
        return await _generate_answer_from_search_result(
            conversation,
            generator,
            question,
            search_results,
            progress,
            context_options,
        )
    if not search_results:
        return typechat.Failure("No search results")
    if len(search_results) == 1:
        return await _generate_answer_from_search_result(
            conversation,
            generator,
            question,
            search_results[0],
            progress,
            context_options,
        )

    partial_results = await _map_async(
        search_results,
        generator.settings.concurrency,
        lambda sr: _generate_answer_from_search_result(
            conversation,
            generator,
            question,
            sr,
            progress,
            context_options,
        ),
    )
    partial_responses: list[AnswerResponse] = []
    for result in partial_results:
        if isinstance(result, typechat.Failure):
            return result
        partial_responses.append(result.value)
    return await generator.combine_partial_answers(question, partial_responses)


async def generate_answer_in_chunks(
    answer_generator: IAnswerGenerator,
    question: str,
    chunks: list[AnswerContext],
    progress: ProcessProgress | None = None,
) -> typechat.Result[list[AnswerResponse]]:
    if not chunks:
        return typechat.Success([])
    if len(chunks) == 1:
        return await _run_single_chunk(answer_generator, question, chunks, progress)

    chunk_answers: list[AnswerResponse] = []
    knowledge_chunks = _get_knowledge_chunks(chunks)
    has_knowledge_answer = False
    if knowledge_chunks:
        knowledge_answers = await _run_generate_answers(
            answer_generator, question, knowledge_chunks, progress
        )
        if isinstance(knowledge_answers, typechat.Failure):
            return knowledge_answers
        chunk_answers.extend(knowledge_answers.value)
        has_knowledge_answer = any(a.type == "Answered" for a in chunk_answers)

    if not has_knowledge_answer or not answer_generator.settings.fast_stop:
        message_chunks = [
            chunk for chunk in chunks if chunk.messages is not None and chunk.messages
        ]
        message_answers = await _run_generate_answers(
            answer_generator, question, message_chunks
        )
        if isinstance(message_answers, typechat.Failure):
            return message_answers
        chunk_answers.extend(message_answers.value)

    return typechat.Success(chunk_answers)


async def generate_multiple_choice_answer(
    conversation: IConversation,
    generator: IAnswerGenerator,
    question: str,
    answer_choices: list[str],
    search_results: ConversationSearchResult | list[ConversationSearchResult],
    progress: ProcessProgress | None = None,
    context_options: AnswerContextOptions | None = None,
) -> typechat.Result[AnswerResponse]:
    question = create_multiple_choice_question(question, answer_choices)
    return await generate_answer(
        conversation,
        generator,
        question,
        search_results,
        progress,
        context_options,
    )


class AnswerGenerator(IAnswerGenerator):
    def __init__(self, settings: AnswerGeneratorSettings | None = None) -> None:
        self.settings = settings or create_answer_generator_settings()
        self.answer_translator = create_answer_translator(
            self.settings.answer_generator_model
        )
        self.context_schema = _create_context_schema(
            self.settings.answer_generator_model
        )
        self.context_type_name = "AnswerContext"

    async def generate_answer(
        self, question: str, context: AnswerContext | str
    ) -> typechat.Result[AnswerResponse]:
        context_content = (
            context if isinstance(context, str) else answer_context_to_string(context)
        )
        if context_content and len(context_content) > self.settings.max_chars_in_budget:
            context_content = trim_string_length(
                context_content, self.settings.max_chars_in_budget
            )
        prompt_parts = [
            create_question_prompt(question),
            create_context_prompt(
                self.context_type_name,
                self.context_schema if self.settings.include_context_schema else "",
                context_content,
            ),
        ]
        prompt_text = "\n\n".join(prompt_parts)
        return await self.answer_translator.translate(
            prompt_text, prompt_preamble=self.settings.model_instructions
        )

    async def combine_partial_answers(
        self, question: str, responses: Sequence[AnswerResponse | None]
    ) -> typechat.Result[AnswerResponse]:
        if len(responses) == 1:
            response = responses[0]
            if response is not None:
                return typechat.Success(response)
            return typechat.Failure("No answer")

        answer_text = ""
        why_no_answer: str | None = None
        answer_count = 0
        for partial_answer in responses:
            if partial_answer is None:
                continue
            if partial_answer.type == "Answered":
                answer_count += 1
                if partial_answer.answer:
                    answer_text += f"{partial_answer.answer}\n"
            else:
                why_no_answer = why_no_answer or partial_answer.why_no_answer

        if answer_text:
            if answer_count > 1:
                answer_text = trim_string_length(
                    answer_text, self.settings.max_chars_in_budget
                )
                rewritten = await rewrite_text(
                    self.settings.answer_combiner_model, answer_text, question
                )
                if not rewritten:
                    return typechat.Failure("rewrite_answer failed")
                answer_text = rewritten
            return typechat.Success(AnswerResponse(type="Answered", answer=answer_text))

        return typechat.Success(
            AnswerResponse(type="NoAnswer", why_no_answer=why_no_answer or "")
        )


def split_context_into_chunks(
    context: AnswerContext, max_chars_per_chunk: int
) -> list[AnswerContext]:
    chunk_builder = AnswerContextChunkBuilder(context, max_chars_per_chunk)
    return list(chunk_builder.get_chunks())


async def answer_context_from_search_result(
    conversation: IConversation,
    search_result: ConversationSearchResult,
    options: AnswerContextOptions | None = None,
) -> AnswerContext:
    context = AnswerContext()
    for knowledge_type, knowledge in search_result.knowledge_matches.items():
        match knowledge_type:
            case "entity":
                context.entities = await get_relevant_entities_for_answer(
                    conversation,
                    knowledge,
                    options.entities_top_k if options else None,
                )
            case "topic":
                context.topics = await get_relevant_topics_for_answer(
                    conversation,
                    knowledge,
                    options.topics_top_k if options else None,
                )
            case _:
                continue

    if search_result.message_matches:
        context.messages = await get_relevant_messages_for_answer(
            conversation,
            search_result.message_matches,
            options.messages_top_k if options else None,
        )

    return context


async def _generate_answer_from_search_result(
    conversation: IConversation,
    generator: IAnswerGenerator,
    question: str,
    search_result: ConversationSearchResult,
    progress: ProcessProgress | None = None,
    context_options: AnswerContextOptions | None = None,
) -> typechat.Result[AnswerResponse]:
    context = await answer_context_from_search_result(
        conversation, search_result, context_options
    )
    context_content = answer_context_to_string(context)
    chunking = (
        True
        if context_options is None or context_options.chunking is None
        else context_options.chunking
    )
    if not chunking or len(context_content) <= generator.settings.max_chars_in_budget:
        return await generator.generate_answer(question, context_content)

    chunks = split_context_into_chunks(context, generator.settings.max_chars_in_budget)
    chunk_responses = await generate_answer_in_chunks(
        generator, question, chunks, progress
    )
    if isinstance(chunk_responses, typechat.Failure):
        return chunk_responses
    return await generator.combine_partial_answers(question, chunk_responses.value)


async def _run_single_chunk(
    answer_generator: IAnswerGenerator,
    question: str,
    chunks: list[AnswerContext],
    progress: ProcessProgress | None,
) -> typechat.Result[list[AnswerResponse]]:
    response = await answer_generator.generate_answer(question, chunks[0])
    if progress:
        progress(chunks[0], 0, response)
    if isinstance(response, typechat.Failure):
        return response
    return typechat.Success([response.value])


def _get_knowledge_chunks(chunks: Iterable[AnswerContext]) -> list[AnswerContext]:
    structured_chunks: list[AnswerContext] = []
    for chunk in chunks:
        knowledge_chunk: AnswerContext | None = None
        if chunk.entities:
            knowledge_chunk = knowledge_chunk or AnswerContext()
            knowledge_chunk.entities = chunk.entities
        if chunk.topics:
            knowledge_chunk = knowledge_chunk or AnswerContext()
            knowledge_chunk.topics = chunk.topics
        if chunk.messages:
            knowledge_chunk = knowledge_chunk or AnswerContext()
            knowledge_chunk.messages = chunk.messages
        if knowledge_chunk is not None:
            structured_chunks.append(knowledge_chunk)
    return structured_chunks


async def _run_generate_answers(
    answer_generator: IAnswerGenerator,
    question: str,
    chunks: list[AnswerContext],
    progress: ProcessProgress | None = None,
) -> typechat.Result[list[AnswerResponse]]:
    if not chunks:
        return typechat.Success([])

    results = await _map_async(
        chunks,
        answer_generator.settings.concurrency,
        lambda chunk: answer_generator.generate_answer(question, chunk),
        progress,
        lambda _chunk, _index, response: not (
            isinstance(response, typechat.Success) and response.value.type == "Answered"
        ),
    )
    return _flatten_results(results)


def _create_context_schema(model: typechat.TypeChatLanguageModel) -> str:
    validator = typechat.TypeChatValidator[AnswerContext](AnswerContext)
    translator = typechat.TypeChatJsonTranslator[AnswerContext](
        model, validator, AnswerContext
    )
    return translator.schema_str.rstrip()


def create_question_prompt(question: str) -> str:
    prompt_lines = [
        "The following is a user question:",
        "===",
        question,
        "",
        "===",
        "- The included [ANSWER CONTEXT] contains information that MAY be relevant to answering the question.",
        "- Answer the user question PRECISELY using ONLY information EXPLICITLY provided in the topics, entities, actions, messages and time ranges/timestamps found in [ANSWER CONTEXT]",
        "- Return 'NoAnswer' if you are unsure, , if the answer is not explicitly in [ANSWER CONTEXT], or if the topics or {entity names, types and facets} in the question are not found in [ANSWER CONTEXT].",
        "- Use the 'name', 'type' and 'facets' properties of the provided JSON entities to identify those highly relevant to answering the question.",
        "- 'origin' and 'audience' fields contain the names of entities involved in communication about the knowledge",
        "**Important:** Communicating DOES NOT imply associations such as authorship, ownership etc. E.g. origin: [X] telling audience [Y, Z] communicating about a book does not imply authorship.",
        "- When asked for lists, ensure the list contents answer the question and nothing else. E.g. for the question 'List all books': List only the books in [ANSWER CONTEXT].",
        "- Use direct quotes only when needed or asked. Otherwise answer in your own words.",
        "- Your answer is readable and complete, with appropriate formatting: line breaks, numbered lists, bullet points etc.",
    ]
    return "\n".join(prompt_lines)


def create_context_prompt(type_name: str, schema: str, context: str) -> str:
    content = ""
    if schema:
        content += (
            "[ANSWER CONTEXT] for answering user questions is a JSON object of type "
            f"{type_name} according to the following TypeScript definitions:\n"
            f"```\n{schema}\n```\n"
        )
    content += f"[ANSWER CONTEXT]\n===\n{context}\n===\n"
    return content


def trim_string_length(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars]


async def rewrite_text(
    model: typechat.TypeChatLanguageModel, text: str, question: str
) -> str | None:
    prompt = [
        typechat.PromptSection(
            role="system",
            content=(
                "Rewrite the partial answers into a single concise answer that "
                "directly addresses the original question."
            ),
        ),
        typechat.PromptSection(
            role="user",
            content=(
                f"Question:\n{question}\n\nPartial answers:\n{text}\n\n"
                "Rewrite into a single answer:"
            ),
        ),
    ]
    result = await model.complete(prompt)
    if isinstance(result, typechat.Failure):
        return None
    return result.value.strip()


async def _map_async[TItem, TResult](
    items: list[TItem],
    concurrency: int,
    worker: Callable[[TItem], Awaitable[typechat.Result[TResult]]],
    progress: Callable[[TItem, int, typechat.Result[TResult]], None] | None = None,
    should_continue: (
        Callable[[TItem, int, typechat.Result[TResult]], bool] | None
    ) = None,
) -> list[typechat.Result[TResult]]:
    if not items:
        return []

    queue: asyncio.Queue[tuple[int, TItem]] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))

    results: list[typechat.Result[TResult] | None] = [None] * len(items)
    stop_event = asyncio.Event()

    async def run_worker() -> None:
        while True:
            if stop_event.is_set():
                return
            try:
                index, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await worker(item)
                results[index] = result
                if progress:
                    progress(item, index, result)
                if should_continue and not should_continue(item, index, result):
                    stop_event.set()
            finally:
                queue.task_done()

    worker_count = max(1, min(concurrency, len(items)))
    tasks = [asyncio.create_task(run_worker()) for _ in range(worker_count)]
    await asyncio.gather(*tasks)

    return [result for result in results if result is not None]


def _flatten_results[TResult](
    results: list[typechat.Result[TResult]],
) -> typechat.Result[list[TResult]]:
    values: list[TResult] = []
    for result in results:
        if isinstance(result, typechat.Failure):
            return result
        values.append(result.value)
    return typechat.Success(values)
