# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""New modular implementation of add_messages_streaming with pipelined architecture."""

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

import typechat

from . import knowledge_schema as kplib
from ..aitools.embeddings import IEmbeddingModel, NormalizedEmbedding
from ..storage.memory.semrefindex import collect_action_terms, collect_entity_terms
from .interfaces import AddMessagesResult
from .interfaces_core import IKnowledgeExtractor, IMessage, MessageOrdinal, TextLocation

__all__ = ["add_messages_streaming"]

if TYPE_CHECKING:
    from .conversation_base import ConversationBase

type ChunkOrdinal = int

_EMPTY_KNOWLEDGE = kplib.KnowledgeResponse(
    entities=[], actions=[], inverse_actions=[], topics=[]
)


class NoOpKnowledgeExtractor:
    """No-op extractor used when auto_extract_knowledge is False."""

    async def extract(self, message: str) -> typechat.Result[kplib.KnowledgeResponse]:
        return typechat.Success(_EMPTY_KNOWLEDGE)


@dataclass
class PipelineStopState:
    """Shared stop marker for pipeline stages.

    A message ordinal greater than or equal to ``stop_at_message_id`` is
    considered out-of-scope for further processing.

    ``exception`` holds the error from the lowest-ordinal message that
    caused the stop, so the orchestrator can re-raise it after the pipeline
    drains.
    """

    stop_at_message_id: int = 10**100
    exception: Exception | None = None


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
    result_queue: asyncio.Queue["ChunkProcessingResult[TMessage] | None"],
    shutdown_event: asyncio.Event | None,
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
            if shutdown_event is not None and shutdown_event.is_set():
                break

            chunk_count = len(message.text_chunks)
            if chunk_count == 0:
                # Zero-chunk message: nothing for the dispatcher to process.
                # Emit a zero-chunk result directly to the reassembler.
                await result_queue.put(
                    ChunkProcessingResult[TMessage](
                        chunk_id=TextLocation(message_id, 0),
                        chunk_count=0,
                        message=message,
                    )
                )
                producer_state.produced_messages += 1
                producer_state.next_message_id += 1
                continue

            for chunk_ordinal, chunk_text in enumerate(message.text_chunks):
                if message_id >= stop_state.stop_at_message_id:
                    break
                if shutdown_event is not None and shutdown_event.is_set():
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
    result_queue: asyncio.Queue["ChunkProcessingResult[TMessage] | None"],
    stop_state: PipelineStopState,
    knowledge_extractor: IKnowledgeExtractor,
    embedding_model: IEmbeddingModel,
    concurrency: int,
    skip_failed_messages: bool,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Dispatch chunk work items to bounded per-item worker tasks.

    Reads work items from ``chunk_queue`` until it receives a ``None``
    sentinel, then awaits all in-flight tasks via a TaskGroup and puts a
    ``None`` sentinel on ``result_queue`` to signal the reassembler.

    Concurrency is bounded by a semaphore so at most ``concurrency`` worker
    tasks run simultaneously.  Chunks at or beyond ``stop_at_message_id`` are
    skipped and reported as error results so the reassembler can account for
    them deterministically.

    Args:
        skip_failed_messages: If True, don't halt producer on extraction/embedding
            failures; continue processing. If False, halt on first failure.
        shutdown_event: If set, stop processing new chunks and let the pipeline drain.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _process_one(work_item: ChunkWorkItem[TMessage]) -> None:
        try:
            stop_at = stop_state.stop_at_message_id
            if (
                work_item.chunk_id.message_ordinal >= stop_at
                or shutdown_event is not None
                and shutdown_event.is_set()
            ):
                result: "ChunkProcessingResult[TMessage]" = ChunkProcessingResult(
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
                    embedding_model=embedding_model,
                )
                if result.error is not None:
                    if not skip_failed_messages:
                        new_stop = min(
                            stop_state.stop_at_message_id,
                            work_item.chunk_id.message_ordinal,
                        )
                        if new_stop < stop_state.stop_at_message_id:
                            stop_state.stop_at_message_id = new_stop
                        if stop_state.exception is None:
                            stop_state.exception = result.error
        finally:
            sem.release()

        await result_queue.put(result)

    async with asyncio.TaskGroup() as tg:
        while not (shutdown_event is not None and shutdown_event.is_set()):
            item = await chunk_queue.get()
            if item is None:
                break
            await sem.acquire()
            tg.create_task(_process_one(item))
        else:
            # Shutdown was set: drain remaining items so the producer's put()
            # calls can unblock and it can send the None sentinel.
            while True:
                item = await chunk_queue.get()
                if item is None:
                    break

    await result_queue.put(None)


@dataclass
class ChunkProcessingResult[TMessage: IMessage]:
    """Result of processing a single chunk through extraction and embeddings.

    Attributes:
        chunk_id: Message/chunk location for the processed chunk.
        chunk_count: Total number of chunks in the message that owns this chunk.
        message: Original message object containing this chunk.
        extracted_knowledge: Extracted KnowledgeResponse if extraction succeeded, else None.
        chunk_embedding: Normalized embedding vector for the message chunk, or None if extraction or embedding failed.
        related_terms: Lowercased, deduplicated related-term texts extracted from knowledge.
        related_term_embeddings: Embeddings for related_terms in the same order, or [] when there are no related terms.
        error: Exception from the first failing operation, or None if extraction and embedding succeeded.
    """

    chunk_id: TextLocation
    chunk_count: int
    message: TMessage
    extracted_knowledge: kplib.KnowledgeResponse | None = None
    chunk_embedding: NormalizedEmbedding | None = None
    related_terms: list[str] | None = None
    related_term_embeddings: list[NormalizedEmbedding] | None = None
    error: Exception | None = None


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

    for action in chain(knowledge.actions, knowledge.inverse_actions):
        for term in collect_action_terms(action):
            _add_term(term)

    for topic in knowledge.topics:
        _add_term(topic)

    return related_terms


# "Public", imported by tests
async def process_chunk_with_extraction_and_embeddings[TMessage: IMessage](
    chunk_id: TextLocation,
    chunk_text: str,
    chunk_count: int,
    message: TMessage,
    knowledge_extractor: IKnowledgeExtractor,
    embedding_model: IEmbeddingModel,
) -> ChunkProcessingResult[TMessage]:
    """Process a single text chunk through knowledge extraction and embeddings.

    Runs extraction/related-term embedding and chunk embedding concurrently,
    capturing the first failure and stopping processing if an error occurs.

    Chunk embeddings are computed uncached; related-term embeddings use
    cache-aware model calls on the same embedding model.

    Args:
        chunk_id: Message/chunk location for this chunk.
        chunk_text: Text content of the chunk (stripped).
        chunk_count: Total number of chunks in the message.
        message: Original message object containing this chunk.
        knowledge_extractor: IKnowledgeExtractor instance for LLM extraction.
        embedding_model: Embedding model for chunk and related-term embeddings.

    Returns:
        ChunkProcessingResult with knowledge, chunk embedding, related-term
        embeddings, or an error from the first failed operation.
    """
    result = ChunkProcessingResult(
        chunk_id=chunk_id, chunk_count=chunk_count, message=message
    )
    sem = asyncio.Semaphore(1)  # Avoid concurrent embedding requests

    async def _extract_knowledge_and_related_embeddings() -> None:
        knowledge_result = await knowledge_extractor.extract(chunk_text)
        if isinstance(knowledge_result, typechat.Failure):
            raise RuntimeError(
                f"Knowledge extraction failed: {knowledge_result.message}"
            )
        result.extracted_knowledge = knowledge_result.value

        result.related_terms = _collect_related_terms_for_fuzzy_index(
            result.extracted_knowledge
        )
        if result.related_terms:
            async with sem:
                rel_embeddings = await embedding_model.get_embeddings(
                    result.related_terms
                )
            result.related_term_embeddings = list(rel_embeddings)
        else:
            result.related_term_embeddings = []

    async def _generate_chunk_embedding() -> None:
        async with sem:
            result.chunk_embedding = await embedding_model.get_embedding_nocache(
                chunk_text
            )

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(_extract_knowledge_and_related_embeddings())
            tg.create_task(_generate_chunk_embedding())
    except Exception as error:
        while isinstance(error, ExceptionGroup) and len(error.exceptions) == 1:
            error = error.exceptions[0]
        result.error = error

    return result


@dataclass
class MessageAssembly[TMessage: IMessage]:
    """In-memory chunk accumulation for one message."""

    message_id: MessageOrdinal
    chunk_count: int
    message: TMessage
    chunks: dict[ChunkOrdinal, ChunkProcessingResult[TMessage]]
    has_error: bool = False
    first_error_msg: str = "Unknown error"

    def is_complete(self) -> bool:
        return len(self.chunks) == self.chunk_count


@dataclass
class ReassemblerResult:
    """Progress and counters produced by the reassembler stage."""

    first_uncommitted_ordinal: MessageOrdinal
    messages_committed: int = 0
    chunks_committed: int = 0
    chunk_failures: int = 0
    messages_skipped: int = 0
    buffered_messages: int = 0


async def _reassembler_task[TMessage: IMessage](
    result_queue: asyncio.Queue[ChunkProcessingResult[TMessage] | None],
    stop_state: PipelineStopState,
    first_uncommitted_ordinal: MessageOrdinal,
    target_commit_chunk_count: int,
    commit_batch: Callable[
        [list[TMessage], list[ChunkProcessingResult[TMessage]]], Awaitable[None]
    ],
    skip_failed_messages: bool,
) -> ReassemblerResult:
    """Reassemble chunks into messages and commit only consecutive complete ones.

    This stage consumes worker results until it sees a ``None`` sentinel.
    It never commits out-of-order messages: if message N is incomplete or failed,
    messages N+1 and later remain buffered.

    Args:
        skip_failed_messages: If True, skip messages that fail extraction/embedding
            and continue processing. If False, halt processing on first failure.
    """
    state = ReassemblerResult(first_uncommitted_ordinal=first_uncommitted_ordinal)
    assemblies: dict[MessageOrdinal, MessageAssembly[TMessage]] = {}

    staged_messages: list[TMessage] = []
    staged_results: list[ChunkProcessingResult[TMessage]] = []
    staged_chunks = 0

    async def _commit_if_needed(force: bool = False) -> None:
        nonlocal staged_chunks, staged_messages, staged_results
        if not staged_messages:
            return
        if not force and staged_chunks < target_commit_chunk_count:
            return
        pending_messages = staged_messages
        pending_results = staged_results
        msg_count = len(pending_messages)
        chunk_count = staged_chunks

        # Clear staged state before awaiting commit/callback paths so a post-commit
        # exception cannot trigger a duplicate retry during final drain.
        staged_messages = []
        staged_results = []
        staged_chunks = 0

        await commit_batch(pending_messages, pending_results)
        state.messages_committed += msg_count
        state.chunks_committed += chunk_count

    async def _drain_consecutive_complete(force: bool = False) -> None:
        nonlocal staged_chunks
        while True:
            assembly = assemblies.get(state.first_uncommitted_ordinal)
            if assembly is None:
                await _commit_if_needed(force)
                return
            if not assembly.is_complete():
                await _commit_if_needed(force)
                return
            if assembly.has_error:
                if skip_failed_messages:
                    print(
                        f"Skipping message {state.first_uncommitted_ordinal} "
                        f"due to chunk processing error: {assembly.first_error_msg}"
                    )
                    del assemblies[state.first_uncommitted_ordinal]
                    state.first_uncommitted_ordinal += 1
                    state.messages_skipped += 1
                    # Continue to check the next message
                    continue
                else:
                    # Stop at failed message; halt processing
                    await _commit_if_needed(force)
                    return

            # Pre-flush: if staging this message would push staged_chunks past
            # the target, commit the current batch first.
            if (
                staged_messages
                and staged_chunks + assembly.chunk_count > target_commit_chunk_count
            ):
                await _commit_if_needed(force=True)

            ordered_chunk_ordinals = sorted(assembly.chunks)
            ordered_results = [assembly.chunks[i] for i in ordered_chunk_ordinals]
            staged_messages.append(assembly.message)
            staged_results.extend(ordered_results)
            staged_chunks += len(ordered_results)

            del assemblies[state.first_uncommitted_ordinal]
            state.first_uncommitted_ordinal += 1
            await _commit_if_needed(force)

    try:
        while True:
            item = await result_queue.get()
            if item is None:
                break

            chunk_ordinal = item.chunk_id.chunk_ordinal
            message_id = item.chunk_id.message_ordinal

            validation_error: str | None = None
            assembly = assemblies.get(message_id)
            if item.chunk_count == 0:
                # Zero-chunk message: create an immediately-complete assembly.
                if assembly is None:
                    assembly = MessageAssembly[TMessage](
                        message_id=message_id,
                        chunk_count=0,
                        message=item.message,
                        chunks={},
                    )
                    assemblies[message_id] = assembly
            elif chunk_ordinal < 0 or chunk_ordinal >= item.chunk_count:
                validation_error = (
                    f"Invalid chunk ordinal: message_id={message_id}, "
                    f"chunk_ordinal={chunk_ordinal}, chunk_count={item.chunk_count}"
                )
            elif assembly is None:
                assembly = MessageAssembly[TMessage](
                    message_id=message_id,
                    chunk_count=item.chunk_count,
                    message=item.message,
                    chunks={},
                )
                assemblies[message_id] = assembly
            elif assembly.chunk_count != item.chunk_count:
                validation_error = (
                    f"Mismatched chunk count for message: message_id={message_id}, "
                    f"expected={assembly.chunk_count}, got={item.chunk_count}"
                )
            elif chunk_ordinal in assembly.chunks:
                validation_error = (
                    f"Duplicate chunk: message_id={message_id}, "
                    f"chunk_ordinal={chunk_ordinal}, chunk_count={item.chunk_count}"
                )

            if validation_error is not None:
                stop_state.stop_at_message_id = min(
                    stop_state.stop_at_message_id, message_id
                )
                raise RuntimeError(validation_error)

            assert assembly is not None

            if item.chunk_count > 0:
                assembly.chunks[chunk_ordinal] = item

            if item.error is not None:
                if not assembly.has_error:
                    assembly.first_error_msg = str(item.error)
                assembly.has_error = True
                state.chunk_failures += 1
                if not skip_failed_messages:
                    stop_state.stop_at_message_id = min(
                        stop_state.stop_at_message_id, message_id
                    )

            await _drain_consecutive_complete()
    finally:
        # Always drain and commit consecutive complete messages before raising
        await _drain_consecutive_complete(force=True)

    state.buffered_messages = len(assemblies)
    return state


async def add_messages_streaming[TMessage: IMessage](
    conv: "ConversationBase[TMessage]",
    messages: AsyncIterable[TMessage],
    *,
    batch_size: int = 100,
    on_batch_committed: Callable[[AddMessagesResult], None] | None = None,
    skip_failed_messages: bool = False,
    shutdown_event: asyncio.Event | None = None,
) -> AddMessagesResult:
    """Ingest messages through a producer/dispatcher/reassembler pipeline.

    The function preserves message commit order while processing chunk extraction
    and embedding concurrently. Batches are committed only for consecutive,
    complete, non-failing messages.

    Args:
        conv: Conversation receiving the new messages.
        messages: Async iterable of messages to ingest.
        batch_size: Target number of chunks per commit batch.
        on_batch_committed: Optional callback invoked after each committed batch
            with that batch's AddMessagesResult.
        skip_failed_messages: If True, skip messages that fail extraction or
            embedding and continue processing. If False (default), halt on
            first failure and raise an exception.

    Returns:
        AddMessagesResult aggregating all committed batches, including count
        of messages_skipped when skip_failed_messages is True.

    Raises:
        Exception: If a single failure occurs during production, processing,
            reassembly, or commit (when skip_failed_messages is False).
        ExceptionGroup: If multiple distinct failures are observed across
            pipeline stages (when skip_failed_messages is False).
    """
    from . import convknowledge

    settings = conv.settings
    sem_ref_settings = settings.semantic_ref_index_settings
    storage = await settings.get_storage_provider()
    if sem_ref_settings.auto_extract_knowledge:
        knowledge_extractor: IKnowledgeExtractor = (
            sem_ref_settings.knowledge_extractor or convknowledge.KnowledgeExtractor()
        )
    else:
        knowledge_extractor = NoOpKnowledgeExtractor()
    embedding_model = settings.embedding_model

    initial_message_id: MessageOrdinal = await conv.messages.size()

    total = AddMessagesResult()

    def _accumulate(result: AddMessagesResult) -> None:
        total.messages_added += result.messages_added
        total.semrefs_added += result.semrefs_added
        total.chunks_added += result.chunks_added
        if on_batch_committed:
            on_batch_committed(result)

    async def _commit_batch(
        messages_batch: list[TMessage],
        chunk_results: list[ChunkProcessingResult[TMessage]],
    ) -> None:
        result = await conv._commit_batch_from_chunk_results(
            storage, messages_batch, chunk_results
        )
        _accumulate(result)

    chunk_queue: asyncio.Queue[ChunkWorkItem[TMessage] | None] = asyncio.Queue(
        maxsize=sem_ref_settings.concurrency * 2
    )
    result_queue: asyncio.Queue[ChunkProcessingResult[TMessage] | None] = asyncio.Queue(
        maxsize=sem_ref_settings.concurrency * 2
    )
    stop_state = PipelineStopState()
    producer_state = ProducerState(next_message_id=initial_message_id)

    task_exceptions: list[Exception] = []
    reassembler_task: asyncio.Task[ReassemblerResult] | None = None
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                _producer_task(
                    messages,
                    chunk_queue,
                    stop_state,
                    producer_state,
                    result_queue,
                    shutdown_event=shutdown_event,
                )
            )
            tg.create_task(
                _dispatcher_task(
                    chunk_queue,
                    result_queue,
                    stop_state,
                    knowledge_extractor,
                    embedding_model,
                    concurrency=sem_ref_settings.concurrency,
                    skip_failed_messages=skip_failed_messages,
                    shutdown_event=shutdown_event,
                )
            )
            reassembler_task = tg.create_task(
                _reassembler_task(
                    result_queue,
                    stop_state,
                    first_uncommitted_ordinal=initial_message_id,
                    target_commit_chunk_count=batch_size,
                    commit_batch=_commit_batch,
                    skip_failed_messages=skip_failed_messages,
                )
            )
    except ExceptionGroup as eg:
        task_exceptions.extend(eg.exceptions)
    except Exception as exc:
        task_exceptions.append(exc)

    if producer_state.exception is not None:
        task_exceptions.append(producer_state.exception)

    if stop_state.exception is not None and not skip_failed_messages:
        task_exceptions.append(stop_state.exception)

    if task_exceptions:
        distinct_exceptions: list[Exception] = []
        for exc in task_exceptions:
            if exc not in distinct_exceptions:
                distinct_exceptions.append(exc)

        if len(distinct_exceptions) == 1:
            raise distinct_exceptions[0]
        raise ExceptionGroup("add_messages_streaming failed", distinct_exceptions)

    # Collect messages_skipped from reassembler result if skip_failed_messages is True
    if skip_failed_messages and reassembler_task is not None:
        try:
            reassembler_result = reassembler_task.result()
            total.messages_skipped = reassembler_result.messages_skipped
        except Exception:
            # reassembler_task result may not be available if task group failed
            pass

    return total
