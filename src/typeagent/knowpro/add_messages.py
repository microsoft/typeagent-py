# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""New modular implementation of add_messages_streaming with pipelined architecture."""

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass

import typechat

from . import knowledge_schema as kplib
from ..aitools.embeddings import IEmbeddingModel, NormalizedEmbedding
from ..storage.memory.semrefindex import collect_action_terms, collect_entity_terms
from .interfaces_core import IKnowledgeExtractor, IMessage, TextLocation


@dataclass
class PipelineStopState:
    """Shared stop marker for pipeline stages.

    A message ordinal greater than or equal to ``stop_at_message_id`` is
    considered out-of-scope for further processing.
    """

    stop_at_message_id: int = 10**100


@dataclass
class ProducerState:
    """Mutable producer state shared with orchestrator/reporting."""

    next_message_id: int
    produced_messages: int = 0
    produced_chunks: int = 0
    exception: Exception | None = None


@dataclass
class ChunkWorkItem[TMessage: IMessage]:
    """One chunk scheduled by the producer for worker processing."""

    chunk_id: TextLocation
    message_id: int
    chunk_ordinal: int
    chunk_count: int
    chunk_text: str
    message: TMessage


async def _producer_task[TMessage: IMessage](
    messages: AsyncIterable[TMessage],
    chunk_queue: asyncio.Queue[ChunkWorkItem[TMessage] | None],
    worker_count: int,
    stop_state: PipelineStopState,
    producer_state: ProducerState,
) -> None:
    """Read input messages and enqueue chunk work items.

    The producer stops enqueueing once it reaches ``stop_at_message_id``.
    It always sends one sentinel per worker, even if the input iterator raises.
    """
    try:
        async for message in messages:
            message_id = producer_state.next_message_id
            if message_id >= stop_state.stop_at_message_id:
                break

            chunk_count = len(message.text_chunks)
            for chunk_ordinal, chunk_text in enumerate(message.text_chunks):
                if message_id >= stop_state.stop_at_message_id:
                    break
                await chunk_queue.put(
                    ChunkWorkItem[TMessage](
                        chunk_id=TextLocation(message_id, chunk_ordinal),
                        message_id=message_id,
                        chunk_ordinal=chunk_ordinal,
                        chunk_count=chunk_count,
                        chunk_text=chunk_text,
                        message=message,
                    )
                )
                producer_state.produced_chunks += 1

            producer_state.produced_messages += 1
            producer_state.next_message_id += 1
    except Exception as exc:
        producer_state.exception = exc
    finally:
        for _ in range(worker_count):
            await chunk_queue.put(None)


@dataclass
class ChunkProcessingResult:
    """Result of processing a single chunk through extraction and embeddings.

    Attributes:
        chunk_id: Message/chunk location for the processed chunk
        message_id: Global message ordinal
        extracted_knowledge: Extracted KnowledgeResponse, or None if extraction failed/wasn't run
        chunk_embedding: Normalized embedding vector for the message chunk
        related_terms: Lowercased related-term texts extracted from knowledge
        related_term_embeddings: Embeddings for related_terms (same order)
        error: Exception from the first failing operation, or None if successful
    """

    chunk_id: TextLocation
    message_id: int
    extracted_knowledge: kplib.KnowledgeResponse | None = None
    chunk_embedding: NormalizedEmbedding | None = None
    related_terms: list[str] | None = None
    related_term_embeddings: list[NormalizedEmbedding] | None = None
    error: Exception | None = None

    @property
    def success(self) -> bool:
        """True if both extraction and embedding succeeded."""
        return (
            self.extracted_knowledge is not None
            and self.chunk_embedding is not None
            and self.related_terms is not None
            and self.related_term_embeddings is not None
            and self.error is None
        )


def _collect_related_terms_for_fuzzy_index(
    knowledge: kplib.KnowledgeResponse,
) -> list[str]:
    """Collect canonical related-term texts for the fuzzy related-terms index.

    These terms are derived from the same knowledge that feeds semantic refs.
    We lowercase and deduplicate while preserving order to match index behavior.
    """
    seen: set[str] = set()
    related_terms: list[str] = []

    def _add_term(term: str) -> None:
        canonical = term.strip().lower()
        if canonical and canonical not in seen:
            seen.add(canonical)
            related_terms.append(canonical)

    for entity in knowledge.entities:
        for term in collect_entity_terms(entity):
            _add_term(term)

    for action in list(knowledge.actions) + list(knowledge.inverse_actions):
        for term in collect_action_terms(action):
            _add_term(term)

    for topic in knowledge.topics:
        _add_term(topic)

    return related_terms


async def process_chunk_with_extraction_and_embeddings(
    chunk_id: TextLocation,
    message_id: int,
    chunk_text: str,
    knowledge_extractor: IKnowledgeExtractor,
    message_embedding_model: IEmbeddingModel,
    related_terms_embedding_model: IEmbeddingModel | None = None,
) -> ChunkProcessingResult:
    """Process a single text chunk through knowledge extraction and embeddings.

    Runs both knowledge extraction and embedding in a single function call,
    capturing the first failure and stopping processing if an error occurs.

    Extraction runs first; if it fails, embedding work is skipped.

    Chunk embeddings are computed uncached, while related-term embeddings are
    computed using cache-aware model calls.

    Args:
        chunk_id: Message/chunk location for this chunk
        message_id: Global message ordinal (1-based in SQLite context)
        chunk_text: Text content of the chunk (stripped)
        knowledge_extractor: IKnowledgeExtractor instance for LLM extraction
        message_embedding_model: Embedding model for chunk text embeddings
        related_terms_embedding_model: Optional embedding model for related-term
            embeddings. If None, message_embedding_model is used.

    Returns:
        ChunkProcessingResult with knowledge, chunk embedding, related-term
        embeddings, or an error from the first failed operation.
    """
    result = ChunkProcessingResult(chunk_id=chunk_id, message_id=message_id)

    # Step 1: Extract knowledge
    try:
        knowledge_result = await knowledge_extractor.extract(chunk_text)
        if isinstance(knowledge_result, typechat.Success):
            result.extracted_knowledge = knowledge_result.value
        else:
            # Extraction returned a Failure; treat as error and stop
            result.error = RuntimeError(
                f"Knowledge extraction failed: {knowledge_result.message}"
            )
            return result
    except Exception as e:
        # Extraction raised an exception; stop processing
        result.error = e
        return result

    result.related_terms = _collect_related_terms_for_fuzzy_index(
        result.extracted_knowledge
    )

    related_model = related_terms_embedding_model or message_embedding_model

    # Step 2: Generate embeddings (only if extraction succeeded)
    try:
        result.chunk_embedding = await message_embedding_model.get_embedding_nocache(
            chunk_text
        )
        if result.related_terms:
            rel_embeddings = await related_model.get_embeddings(
                result.related_terms,
            )
            result.related_term_embeddings = [e for e in rel_embeddings]
        else:
            result.related_term_embeddings = []
    except Exception as e:
        # Embedding failed; record error and return
        result.error = e
        return result

    return result
