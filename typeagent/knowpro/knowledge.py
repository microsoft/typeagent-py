# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio

from typechat import Result, TypeChatLanguageModel

from . import convknowledge
from . import kplib
from .interfaces import IKnowledgeExtractor


def create_knowledge_extractor(
    chat_model: TypeChatLanguageModel | None = None,
) -> convknowledge.KnowledgeExtractor:
    """Create a knowledge extractor using the given Chat Model."""
    chat_model = chat_model or convknowledge.create_typechat_model()
    extractor = convknowledge.KnowledgeExtractor(
        chat_model, max_chars_per_chunk=4096, merge_action_knowledge=False
    )
    return extractor


async def extract_knowledge_from_text(
    knowledge_extractor: IKnowledgeExtractor,
    text: str,
    max_retries: int,
) -> Result[kplib.KnowledgeResponse]:
    """Extract knowledge from a single text input with retries."""
    # TODO: Add a retry mechanism to handle transient errors.
    return await knowledge_extractor.extract(text)


async def batch_producer(
    q: asyncio.Queue[tuple[int, str] | None],
    text_batch: list[str],
    concurrency: int,
) -> None:
    for index, text in enumerate(text_batch):
        await q.put((index, text))
    for _ in range(concurrency):
        await q.put(None)


async def batch_worker(
    q: asyncio.Queue[tuple[int, str] | None],
    knowledge_extractor: IKnowledgeExtractor,
    results: dict[int, Result[kplib.KnowledgeResponse]],
    max_retries: int,
) -> None:
    while item := await q.get():
        index, text = item
        result = await extract_knowledge_from_text(
            knowledge_extractor, text, max_retries
        )
        results[index] = result


async def extract_knowledge_from_text_batch(
    knowledge_extractor: IKnowledgeExtractor,
    text_batch: list[str],
    concurrency: int = 2,
    max_retries: int = 3,
) -> list[Result[kplib.KnowledgeResponse]]:
    """Extract knowledge from a batch of text inputs concurrently."""
    if not text_batch:
        return []

    q: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue(
        maxsize=2 * concurrency + 2
    )
    results: dict[int, Result[kplib.KnowledgeResponse]] = {}

    async with asyncio.TaskGroup() as tg:
        tg.create_task(batch_producer(q, text_batch, concurrency))
        for _ in range(concurrency):
            tg.create_task(batch_worker(q, knowledge_extractor, results, max_retries))

    return [results[i] for i in range(len(text_batch))]


def merge_concrete_entities(
    entities: list[kplib.ConcreteEntity],
) -> list[kplib.ConcreteEntity]:
    """Merge a list of concrete entities into a single list of merged entities."""
    raise NotImplementedError("TODO")
    # merged_entities = concrete_to_merged_entities(entities)

    # merged_concrete_entities = []
    # for merged_entity in merged_entities.values():
    #     merged_concrete_entities.append(merged_to_concrete_entity(merged_entity))
    # return merged_concrete_entities


def merge_topics(topics: list[str]) -> list[str]:
    """Merge a list of topics into a unique list of topics."""
    # TODO: Preserve order of first occurrence?
    merged_topics = set(topics)
    return list(merged_topics)


async def extract_knowledge_for_text_batch_q(
    knowledge_extractor: convknowledge.KnowledgeExtractor,
    text_batch: list[str],
    concurrency: int = 2,
    max_retries: int = 3,
) -> list[Result[kplib.KnowledgeResponse]]:
    """Extract knowledge for a batch of text inputs using a task queue."""
    raise NotImplementedError("TODO")
    # TODO: BatchTask etc.
    # task_batch = [BatchTask(task=text) for text in text_batch]

    # await run_in_batches(
    #     task_batch,
    #     lambda text: extract_knowledge_from_text(knowledge_extractor, text, max_retries),
    #     concurrency,
    # )

    # results = []
    # for task in task_batch:
    #     results.append(task.result if task.result else Failure("No result"))
    # return results
