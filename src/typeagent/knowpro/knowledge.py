# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from dataclasses import dataclass

from typechat import Result, TypeChatLanguageModel

from . import convknowledge, kplib
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
        for _ in range(concurrency):
            tg.create_task(batch_worker(q, knowledge_extractor, results, max_retries))

        for index, text in enumerate(text_batch):
            await q.put((index, text))
        for _ in range(concurrency):
            await q.put(None)

    return [results[i] for i in range(len(text_batch))]


def merge_concrete_entities(
    entities: list[kplib.ConcreteEntity],
) -> list[kplib.ConcreteEntity]:
    """Merge a list of concrete entities by name, combining types and facets.

    Entities with the same name are merged:
    - Types are combined into a sorted unique list
    - Facets with the same name have their values concatenated with "; "
    """
    if not entities:
        return []

    # Build a dict of merged entities keyed by name
    merged: dict[str, _MergedEntity] = {}

    for entity in entities:
        name_key = entity.name
        existing = merged.get(name_key)

        if existing is None:
            # First occurrence - create new merged entity
            merged[name_key] = _MergedEntity(
                name=entity.name,
                types=set(entity.type),
                facets=_facets_to_merged(entity.facets) if entity.facets else {},
            )
        else:
            # Merge into existing
            existing.types.update(entity.type)
            if entity.facets:
                merged_facets = _facets_to_merged(entity.facets)
                for facet_name, facet_values in merged_facets.items():
                    existing.facets.setdefault(facet_name, []).extend(facet_values)

    # Convert merged entities back to ConcreteEntity
    result = []
    for merged_entity in merged.values():
        concrete = kplib.ConcreteEntity(
            name=merged_entity.name,
            type=sorted(merged_entity.types),
        )
        if merged_entity.facets:
            concrete.facets = _merged_to_facets(merged_entity.facets)
        result.append(concrete)

    return result


@dataclass
class _MergedEntity:
    """Internal helper for merging entities."""

    name: str
    types: set[str]
    facets: dict[str, list[str]]


def _facets_to_merged(facets: list[kplib.Facet]) -> dict[str, list[str]]:
    """Convert a list of Facets to a merged facets dict."""
    merged: dict[str, list[str]] = {}
    for facet in facets:
        name = facet.name
        value = str(facet.value) if facet.value else ""
        merged.setdefault(name, []).append(value)
    return merged


def _merged_to_facets(merged_facets: dict[str, list[str]]) -> list[kplib.Facet]:
    """Convert a merged facets dict back to a list of Facets."""
    facets = []
    for name, values in merged_facets.items():
        if values:
            # Deduplicate values while preserving order
            unique_values = list(dict.fromkeys(values))
            facets.append(kplib.Facet(name=name, value="; ".join(unique_values)))
    return facets


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
