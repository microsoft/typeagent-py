# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""New modular implementation of add_messages_streaming with pipelined architecture."""

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass

import typechat

from . import knowledge_schema as kplib
from ..aitools.embeddings import IEmbeddingModel, NormalizedEmbedding
from ..storage.memory.semrefindex import collect_action_terms, collect_entity_terms
from .interfaces_core import IKnowledgeExtractor, IMessage, MessageOrdinal, TextLocation

type ChunkOrdinal = int


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

    next_message_id: MessageOrdinal
    produced_messages: int = 0
    produced_chunks: int = 0
    exception: Exception | None = None


@dataclass
class ChunkWorkItem[TMessage: IMessage]:
    """One chunk scheduled by the producer for worker processing."""

    chunk_id: TextLocation
    chunk_count: int
    chunk_text: str
    message: TMessage


async def _producer_task[TMessage: IMessage](
    messages: AsyncIterable[TMessage],
    chunk_queue: asyncio.Queue[ChunkWorkItem[TMessage] | None],
    stop_state: PipelineStopState,
    producer_state: ProducerState,
) -> None:
    """Read input messages and enqueue chunk work items.

    The producer stops enqueueing once it reaches ``stop_at_message_id``.
    It always sends a sentinel to shut down the dispatcher, even if the
    input iterator raises.
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
        await chunk_queue.put(None)


async def _dispatcher_task[TMessage: IMessage](
    chunk_queue: asyncio.Queue[ChunkWorkItem[TMessage] | None],
    result_queue: asyncio.Queue[ChunkProcessingResult[TMessage] | None],
    stop_state: PipelineStopState,
    knowledge_extractor: IKnowledgeExtractor,
    message_embedding_model: IEmbeddingModel,
    related_terms_embedding_model: IEmbeddingModel | None = None,
    concurrency: int = 4,
) -> None:
    """Dispatch chunk work items to bounded per-item worker tasks.

    Reads work items from ``chunk_queue`` until it receives a ``None``
    sentinel, then awaits all in-flight tasks via a TaskGroup and puts a
    ``None`` sentinel on ``result_queue`` to signal the reassembler.

    Concurrency is bounded by a semaphore so at most ``concurrency`` worker
    tasks run simultaneously.  Chunks at or beyond ``stop_at_message_id`` are
    skipped and reported as error results so the reassembler can account for
    them deterministically.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _process_one(work_item: ChunkWorkItem[TMessage]) -> None:
        try:
            stop_at = stop_state.stop_at_message_id
            if work_item.chunk_id.message_ordinal >= stop_at:
                result: ChunkProcessingResult[TMessage] = ChunkProcessingResult(
                    chunk_id=work_item.chunk_id,
                    chunk_count=work_item.chunk_count,
                    message=work_item.message,
                    error=RuntimeError(
                        "Chunk skipped because stop_at_message_id "
                        f"is {stop_at} and message_id is "
                        f"{work_item.chunk_id.message_ordinal}"
                    ),
                )
            else:
                result = await process_chunk_with_extraction_and_embeddings(
                    chunk_id=work_item.chunk_id,
                    chunk_text=work_item.chunk_text,
                    chunk_count=work_item.chunk_count,
                    message=work_item.message,
                    knowledge_extractor=knowledge_extractor,
                    message_embedding_model=message_embedding_model,
                    related_terms_embedding_model=related_terms_embedding_model,
                )
                if result.error is not None:
                    stop_state.stop_at_message_id = min(
                        stop_state.stop_at_message_id,
                        work_item.chunk_id.message_ordinal,
                    )
            await result_queue.put(result)
        finally:
            sem.release()

    async with asyncio.TaskGroup() as tg:
        while True:
            item = await chunk_queue.get()
            if item is None:
                break
            await sem.acquire()
            tg.create_task(_process_one(item))

    await result_queue.put(None)


@dataclass
class ChunkProcessingResult[TMessage: IMessage]:
    """Result of processing a single chunk through extraction and embeddings.

    Attributes:
        chunk_id: Message/chunk location for the processed chunk
        extracted_knowledge: Extracted KnowledgeResponse, or None if extraction failed/wasn't run
        chunk_embedding: Normalized embedding vector for the message chunk
        related_terms: Lowercased related-term texts extracted from knowledge
        related_term_embeddings: Embeddings for related_terms (same order)
        error: Exception from the first failing operation, or None if successful
    """

    chunk_id: TextLocation
    chunk_count: int
    message: TMessage
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


async def process_chunk_with_extraction_and_embeddings[TMessage: IMessage](
    chunk_id: TextLocation,
    chunk_text: str,
    chunk_count: int,
    message: TMessage,
    knowledge_extractor: IKnowledgeExtractor,
    message_embedding_model: IEmbeddingModel,
    related_terms_embedding_model: IEmbeddingModel | None = None,
) -> ChunkProcessingResult[TMessage]:
    """Process a single text chunk through knowledge extraction and embeddings.

    Runs both knowledge extraction and embedding in a single function call,
    capturing the first failure and stopping processing if an error occurs.

    Extraction runs first; if it fails, embedding work is skipped.

    Chunk embeddings are computed uncached, while related-term embeddings are
    computed using cache-aware model calls.

    Args:
        chunk_id: Message/chunk location for this chunk
        chunk_text: Text content of the chunk (stripped)
        knowledge_extractor: IKnowledgeExtractor instance for LLM extraction
        message_embedding_model: Embedding model for chunk text embeddings
        related_terms_embedding_model: Optional embedding model for related-term
            embeddings. If None, message_embedding_model is used.

    Returns:
        ChunkProcessingResult with knowledge, chunk embedding, related-term
        embeddings, or an error from the first failed operation.
    """
    result = ChunkProcessingResult(
        chunk_id=chunk_id,
        chunk_count=chunk_count,
        message=message,
    )

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


@dataclass
class MessageAssembly[TMessage: IMessage]:
    """In-memory chunk accumulation for one message."""

    message_id: MessageOrdinal
    chunk_count: int
    message: TMessage
    chunks: dict[ChunkOrdinal, ChunkProcessingResult[TMessage]]
    has_error: bool = False

    def is_complete(self) -> bool:
        return len(self.chunks) == self.chunk_count


@dataclass
class ReassemblerResult:
    """Progress and counters produced by the reassembler stage."""

    first_uncommitted_ordinal: MessageOrdinal
    messages_committed: int = 0
    chunks_committed: int = 0
    chunk_failures: int = 0
    buffered_messages: int = 0


async def _reassembler_task[TMessage: IMessage](
    result_queue: asyncio.Queue[ChunkProcessingResult[TMessage] | None],
    stop_state: PipelineStopState,
    first_uncommitted_ordinal: MessageOrdinal,
    target_commit_chunk_count: int,
    commit_batch: Callable[
        [list[TMessage], list[ChunkProcessingResult[TMessage]]],
        Awaitable[None],
    ],
    on_batch_committed: Callable[[int, int], None] | None = None,
) -> ReassemblerResult:
    """Reassemble chunks into messages and commit only consecutive complete ones.

    This stage consumes worker results until it sees a ``None`` sentinel.
    It never commits out-of-order messages: if message N is incomplete or failed,
    messages N+1 and later remain buffered.
    """
    state = ReassemblerResult(first_uncommitted_ordinal=first_uncommitted_ordinal)
    assemblies: dict[MessageOrdinal, MessageAssembly[TMessage]] = {}

    staged_messages: list[TMessage] = []
    staged_results: list[ChunkProcessingResult[TMessage]] = []
    staged_chunks = 0

    async def _commit_if_needed(force: bool = False) -> None:
        nonlocal staged_chunks
        if not staged_messages:
            return
        if not force and staged_chunks < target_commit_chunk_count:
            return
        msg_count = len(staged_messages)
        chunk_count = staged_chunks
        await commit_batch(staged_messages, staged_results)
        state.messages_committed += msg_count
        state.chunks_committed += chunk_count
        if on_batch_committed is not None:
            on_batch_committed(msg_count, chunk_count)
        staged_messages.clear()
        staged_results.clear()
        staged_chunks = 0

    async def _drain_consecutive_complete() -> None:
        nonlocal staged_chunks
        while True:
            assembly = assemblies.get(state.first_uncommitted_ordinal)
            if assembly is None:
                return
            if not assembly.is_complete() or assembly.has_error:
                return

            ordered_chunk_ordinals = sorted(assembly.chunks)
            ordered_results = [assembly.chunks[i] for i in ordered_chunk_ordinals]
            staged_messages.append(assembly.message)
            staged_results.extend(ordered_results)
            staged_chunks += len(ordered_results)

            del assemblies[state.first_uncommitted_ordinal]
            state.first_uncommitted_ordinal += 1
            await _commit_if_needed()

    try:
        while True:
            item = await result_queue.get()
            if item is None:
                break

            chunk_ordinal = item.chunk_id.chunk_ordinal
            message_id = item.chunk_id.message_ordinal

            try:
                if chunk_ordinal < 0 or chunk_ordinal >= item.chunk_count:
                    raise RuntimeError(
                        f"Invalid chunk ordinal: message_id={message_id}, "
                        f"chunk_ordinal={chunk_ordinal}, chunk_count={item.chunk_count}"
                    )

                existing = assemblies.get(message_id)
                if existing is None:
                    existing = MessageAssembly[TMessage](
                        message_id=message_id,
                        chunk_count=item.chunk_count,
                        message=item.message,
                        chunks={},
                    )
                    assemblies[message_id] = existing
                elif existing.chunk_count != item.chunk_count:
                    raise RuntimeError(
                        f"Mismatched chunk count for message: message_id={message_id}, "
                        f"expected={existing.chunk_count}, got={item.chunk_count}"
                    )

                if chunk_ordinal in existing.chunks:
                    raise RuntimeError(
                        f"Duplicate chunk: message_id={message_id}, "
                        f"chunk_ordinal={chunk_ordinal}, chunk_count={item.chunk_count}"
                    )

                existing.chunks[chunk_ordinal] = item
            except Exception:
                # On validation error, set stop flag and re-raise
                # The finally block will drain and commit consecutive complete messages
                stop_state.stop_at_message_id = min(
                    stop_state.stop_at_message_id, message_id
                )
                raise

            if item.error is not None:
                existing.has_error = True
                state.chunk_failures += 1
                stop_state.stop_at_message_id = min(
                    stop_state.stop_at_message_id, message_id
                )

            await _drain_consecutive_complete()
    finally:
        # Always drain and commit consecutive complete messages before raising
        await _drain_consecutive_complete()
        await _commit_if_needed(force=True)

    state.buffered_messages = len(assemblies)
    return state
