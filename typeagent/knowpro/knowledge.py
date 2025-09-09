# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typechat import Result, TypeChatLanguageModel

from . import convknowledge
from . import kplib


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
    knowledge_extractor: convknowledge.KnowledgeExtractor,
    text: str,
    max_retries: int,
) -> Result[kplib.KnowledgeResponse]:
    """Extract knowledge from a single text input with retries."""
    # TODO: Add a retry mechanism to handle transient errors.
    return await knowledge_extractor.extract(text)


async def extract_knowledge_from_text_batch(
    knowledge_extractor: convknowledge.KnowledgeExtractor,
    text_batch: list[str],
    concurrency: int = 2,
    max_retries: int = 3,
) -> list[Result[kplib.KnowledgeResponse]]:
    """Extract knowledge from a batch of text inputs concurrently."""
    # TODO: Use concurrency.
    results: list[Result[kplib.KnowledgeResponse]] = []
    for text in text_batch:
        result = await extract_knowledge_from_text(
            knowledge_extractor, text, max_retries
        )
        results.append(result)
    return results


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
