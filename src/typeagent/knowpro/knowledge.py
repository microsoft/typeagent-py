# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from collections.abc import Callable
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


@dataclass
class _MergedEntity:
    """Internal helper for merging entities."""

    name: str
    types: set[str]
    facets: dict[str, set[str]]


def merge_concrete_entities(
    entities: list[kplib.ConcreteEntity],
    normalize: Callable[[str], str] = str.lower,
) -> list[kplib.ConcreteEntity]:
    """Merge a list of concrete entities by name, combining types and facets.

    Entities with the same name (after normalization) are merged:
    - Names, types, and facet names/values are normalized for matching
    - Types are combined into a sorted unique list (normalized)
    - Facets with the same name have their unique values concatenated with "; "

    Args:
        entities: List of entities to merge.
        normalize: Function to normalize strings for matching. Defaults to
            str.lower for case-insensitive matching. Pass lambda x: x to
            preserve original casing.

    Note:
        By default, this function normalizes all text to lowercase, matching
        the TypeScript implementation in knowledgeMerge.ts. Facet values are
        converted to strings during merging. Complex types like Quantity and
        Quantifier use their __str__ representation (e.g., "5 kg" or "many items").

    Returns:
        A list of merged entities sorted by name for deterministic ordering.
    """
    if not entities:
        return []

    # Build a dict of merged entities keyed by normalized name
    merged: dict[str, _MergedEntity] = {}

    for entity in entities:
        name_key = normalize(entity.name)
        existing = merged.get(name_key)

        if existing is None:
            # First occurrence - create new merged entity
            merged[name_key] = _MergedEntity(
                name=name_key,
                types=set(normalize(t) for t in entity.type),
                facets=(
                    _facets_to_merged(entity.facets, normalize) if entity.facets else {}
                ),
            )
        else:
            # Merge into existing
            existing.types.update(normalize(t) for t in entity.type)
            if entity.facets:
                _merge_facets(existing.facets, entity.facets, normalize)

    # Convert merged entities back to ConcreteEntity, sorted by name
    result = []
    for merged_entity in sorted(merged.values(), key=lambda e: e.name):
        concrete = kplib.ConcreteEntity(
            name=merged_entity.name,
            type=sorted(merged_entity.types),
        )
        if merged_entity.facets:
            concrete.facets = _merged_to_facets(merged_entity.facets)
        result.append(concrete)

    return result


def _add_facet_to_merged(
    merged: dict[str, set[str]],
    facet: kplib.Facet,
    normalize: Callable[[str], str],
) -> None:
    """Add a single facet to a merged facets dict."""
    name = normalize(facet.name)
    value = normalize(str(facet.value)) if facet.value else ""
    merged.setdefault(name, set()).add(value)


def _facets_to_merged(
    facets: list[kplib.Facet],
    normalize: Callable[[str], str],
) -> dict[str, set[str]]:
    """Convert a list of Facets to a merged facets dict.

    Facet names and values are normalized for merging.
    """
    merged: dict[str, set[str]] = {}
    for facet in facets:
        _add_facet_to_merged(merged, facet, normalize)
    return merged


def _merge_facets(
    existing: dict[str, set[str]],
    facets: list[kplib.Facet],
    normalize: Callable[[str], str],
) -> None:
    """Merge facets into an existing facets dict."""
    for facet in facets:
        _add_facet_to_merged(existing, facet, normalize)


def _merged_to_facets(merged_facets: dict[str, set[str]]) -> list[kplib.Facet]:
    """Convert a merged facets dict back to a list of Facets."""
    facets = []
    for name, values in sorted(merged_facets.items()):
        if values:
            facets.append(kplib.Facet(name=name, value="; ".join(sorted(values))))
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
