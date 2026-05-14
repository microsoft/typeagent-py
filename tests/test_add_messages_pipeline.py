# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import numpy as np
import pytest

import typechat

from typeagent.aitools.embeddings import NormalizedEmbedding, NormalizedEmbeddings
from typeagent.knowpro import knowledge_schema as kplib
from typeagent.knowpro.add_messages import (
    _collect_related_terms_for_fuzzy_index,
    _dispatcher_task,
    _producer_task,
    _reassembler_task,
    ChunkProcessingResult,
    ChunkWorkItem,
    PipelineStopState,
    process_chunk_with_extraction_and_embeddings,
    ProducerState,
)
from typeagent.knowpro.interfaces_core import (
    DeletionInfo,
    IMessageMetadata,
    TextLocation,
)


@dataclass
class _Message:
    text_chunks: list[str]
    tags: list[str] = field(default_factory=list)
    timestamp: str | None = None
    deletion_info: DeletionInfo | None = None
    metadata: IMessageMetadata | None = None
    source_id: str | None = None

    def get_knowledge(self) -> kplib.KnowledgeResponse:
        return _empty_knowledge()


class _SequenceExtractor:
    def __init__(
        self, outputs: list[typechat.Result[kplib.KnowledgeResponse] | Exception]
    ) -> None:
        self._outputs = outputs
        self.calls: list[str] = []

    async def extract(self, message: str) -> typechat.Result[kplib.KnowledgeResponse]:
        self.calls.append(message)
        output = self._outputs[len(self.calls) - 1]
        if isinstance(output, Exception):
            raise output
        return output


class _StubEmbeddingModel:
    def __init__(
        self,
        *,
        chunk_error: Exception | None = None,
        related_error: Exception | None = None,
    ) -> None:
        self.chunk_error = chunk_error
        self.related_error = related_error
        self.chunk_calls: list[str] = []
        self.related_calls: list[list[str]] = []
        self._cache: dict[str, NormalizedEmbedding] = {}

    @property
    def model_name(self) -> str:
        return "test-embedding"

    def add_embedding(self, key: str, embedding: NormalizedEmbedding) -> None:
        self._cache[key] = embedding

    async def get_embedding_nocache(self, input: str) -> NormalizedEmbedding:
        self.chunk_calls.append(input)
        if self.chunk_error is not None:
            raise self.chunk_error
        return _embedding([1.0, 0.0])

    async def get_embeddings_nocache(self, input: list[str]) -> NormalizedEmbeddings:
        self.related_calls.append(input)
        if self.related_error is not None:
            raise self.related_error
        return np.array([_embedding([0.0, 1.0]) for _ in input], dtype=np.float32)

    async def get_embedding(self, key: str) -> NormalizedEmbedding:
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        embedding = await self.get_embedding_nocache(key)
        self._cache[key] = embedding
        return embedding

    async def get_embeddings(self, keys: list[str]) -> NormalizedEmbeddings:
        if not keys:
            raise ValueError("Cannot embed an empty list")
        output: list[NormalizedEmbedding] = []
        missing: list[str] = []
        for key in keys:
            cached = self._cache.get(key)
            if cached is None:
                missing.append(key)
            else:
                output.append(cached)
        if missing:
            fresh = await self.get_embeddings_nocache(missing)
            for index, key in enumerate(missing):
                self._cache[key] = fresh[index]
        return np.array([self._cache[key] for key in keys], dtype=np.float32)


class _FailingAsyncMessages:
    def __init__(self, messages: list[_Message], error: Exception) -> None:
        self.messages = messages
        self.error = error

    def __aiter__(self) -> AsyncIterator[_Message]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[_Message]:
        for message in self.messages:
            yield message
        raise self.error


class _StopMutatingChunks(list[str]):
    """Iterable that lowers stop_at_message_id after yielding first chunk."""

    def __init__(self, stop_state: PipelineStopState, chunks: list[str]) -> None:
        super().__init__(chunks)
        self._stop_state = stop_state

    def __iter__(self):
        for index, chunk in enumerate(super().__iter__()):
            if index == 1:
                self._stop_state.stop_at_message_id = 0
            yield chunk


def _embedding(values: list[float]) -> NormalizedEmbedding:
    return np.array(values, dtype=np.float32)


def _empty_knowledge() -> kplib.KnowledgeResponse:
    return kplib.KnowledgeResponse(
        entities=[], actions=[], inverse_actions=[], topics=[]
    )


def _knowledge_with_terms() -> kplib.KnowledgeResponse:
    entity = kplib.ConcreteEntity(
        name=" Alice ",
        type=["Person"],
        facets=[kplib.Facet(name="Role", value="Engineer")],
    )
    action = kplib.Action(
        verbs=["Mentors"],
        verb_tense="present",
        subject_entity_name="Alice",
        object_entity_name="Bob",
    )
    return kplib.KnowledgeResponse(
        entities=[entity],
        actions=[action],
        inverse_actions=[],
        topics=["ALICE", "  Mentorship "],
    )


async def _drain_result_queue(
    queue: asyncio.Queue[ChunkProcessingResult[_Message] | None],
) -> list[ChunkProcessingResult[_Message] | None]:
    items: list[ChunkProcessingResult[_Message] | None] = []
    while True:
        item = await queue.get()
        items.append(item)
        if item is None:
            return items


@pytest.mark.asyncio
async def test_collect_related_terms_lowercases_dedupes_and_preserves_order() -> None:
    knowledge = _knowledge_with_terms()

    terms = _collect_related_terms_for_fuzzy_index(knowledge)

    assert terms[0] == "alice"
    assert "mentorship" in terms
    assert len(terms) == len(set(terms))


@pytest.mark.asyncio
async def test_process_chunk_success_with_related_terms() -> None:
    extractor = _SequenceExtractor([typechat.Success(_knowledge_with_terms())])
    message_model = _StubEmbeddingModel()

    result = await process_chunk_with_extraction_and_embeddings(
        chunk_id=TextLocation(0, 0),
        chunk_text="hello",
        chunk_count=1,
        message=_Message(["hello"]),
        knowledge_extractor=extractor,
        embedding_model=message_model,
    )

    assert result.error is None
    assert result.extracted_knowledge is not None
    assert result.chunk_embedding is not None
    assert result.related_terms is not None
    assert result.related_term_embeddings is not None
    assert len(result.related_terms) == len(result.related_term_embeddings)
    assert message_model.chunk_calls == ["hello"]
    assert len(message_model.related_calls) == 1


@pytest.mark.asyncio
async def test_process_chunk_extraction_failure_returns_error() -> None:
    """A Failure result from the extractor sets error and skips embedding."""
    extractor = _SequenceExtractor([typechat.Failure("bad extraction")])
    message_model = _StubEmbeddingModel()

    result = await process_chunk_with_extraction_and_embeddings(
        chunk_id=TextLocation(0, 0),
        chunk_text="hello",
        chunk_count=1,
        message=_Message(["hello"]),
        knowledge_extractor=extractor,
        embedding_model=message_model,
    )

    assert isinstance(result.error, RuntimeError)
    assert "bad extraction" in str(result.error)
    assert result.extracted_knowledge is None
    assert message_model.chunk_calls == []


@pytest.mark.asyncio
async def test_process_chunk_extraction_exception_returns_error() -> None:
    extractor = _SequenceExtractor([RuntimeError("extract boom")])
    message_model = _StubEmbeddingModel()

    result = await process_chunk_with_extraction_and_embeddings(
        chunk_id=TextLocation(0, 0),
        chunk_text="hello",
        chunk_count=1,
        message=_Message(["hello"]),
        knowledge_extractor=extractor,
        embedding_model=message_model,
    )

    assert isinstance(result.error, RuntimeError)
    assert "extract boom" in str(result.error)


@pytest.mark.asyncio
async def test_process_chunk_chunk_embedding_exception_returns_error() -> None:
    extractor = _SequenceExtractor([typechat.Success(_empty_knowledge())])
    message_model = _StubEmbeddingModel(chunk_error=RuntimeError("embed boom"))

    result = await process_chunk_with_extraction_and_embeddings(
        chunk_id=TextLocation(0, 0),
        chunk_text="hello",
        chunk_count=1,
        message=_Message(["hello"]),
        knowledge_extractor=extractor,
        embedding_model=message_model,
    )

    assert isinstance(result.error, RuntimeError)
    assert "embed boom" in str(result.error)


@pytest.mark.asyncio
async def test_process_chunk_related_term_embedding_exception_returns_error() -> None:
    extractor = _SequenceExtractor([typechat.Success(_knowledge_with_terms())])
    message_model = _StubEmbeddingModel(related_error=RuntimeError("related boom"))

    result = await process_chunk_with_extraction_and_embeddings(
        chunk_id=TextLocation(0, 0),
        chunk_text="hello",
        chunk_count=1,
        message=_Message(["hello"]),
        knowledge_extractor=extractor,
        embedding_model=message_model,
    )

    assert isinstance(result.error, RuntimeError)
    assert "related boom" in str(result.error)


@pytest.mark.asyncio
async def test_producer_enqueues_all_chunks_and_sentinel() -> None:
    messages = [_Message(["a", "b"]), _Message(["c"])]
    queue: asyncio.Queue[ChunkWorkItem[_Message] | None] = asyncio.Queue()
    stop_state = PipelineStopState()
    producer_state = ProducerState(next_message_id=0)

    async def _iter_messages() -> AsyncIterator[_Message]:
        for message in messages:
            yield message

    result_queue = asyncio.Queue()
    await _producer_task(
        _iter_messages(), queue, stop_state, producer_state, result_queue, None
    )

    items: list[ChunkWorkItem[_Message] | None] = [
        await queue.get(),
        await queue.get(),
        await queue.get(),
        await queue.get(),
    ]

    assert [item.chunk_id for item in items[:-1] if item is not None] == [
        TextLocation(0, 0),
        TextLocation(0, 1),
        TextLocation(1, 0),
    ]
    assert items[-1] is None
    assert producer_state.produced_messages == 2
    assert producer_state.produced_chunks == 3
    assert producer_state.exception is None


@pytest.mark.asyncio
async def test_producer_stops_at_stop_marker() -> None:
    queue: asyncio.Queue[ChunkWorkItem[_Message] | None] = asyncio.Queue()
    stop_state = PipelineStopState(stop_at_message_id=1)
    producer_state = ProducerState(next_message_id=0)

    async def _iter_messages() -> AsyncIterator[_Message]:
        yield _Message(["a"])
        yield _Message(["b"])

    result_queue = asyncio.Queue()
    await _producer_task(
        _iter_messages(), queue, stop_state, producer_state, result_queue, None
    )

    first = await queue.get()
    sentinel = await queue.get()

    assert first is not None
    assert first.chunk_id == TextLocation(0, 0)
    assert sentinel is None
    assert producer_state.produced_messages == 1


@pytest.mark.asyncio
async def test_producer_sets_exception_and_still_sends_sentinel() -> None:
    queue: asyncio.Queue[ChunkWorkItem[_Message] | None] = asyncio.Queue()
    stop_state = PipelineStopState()
    producer_state = ProducerState(next_message_id=0)

    failing_iter = _FailingAsyncMessages([_Message(["a"])], RuntimeError("input boom"))

    result_queue = asyncio.Queue()
    await _producer_task(
        failing_iter, queue, stop_state, producer_state, result_queue, None
    )

    first = await queue.get()
    sentinel = await queue.get()

    assert first is not None
    assert first.chunk_id == TextLocation(0, 0)
    assert sentinel is None
    assert isinstance(producer_state.exception, RuntimeError)
    assert "input boom" in str(producer_state.exception)


@pytest.mark.asyncio
async def test_producer_breaks_inside_chunk_loop_when_stop_marker_changes() -> None:
    queue: asyncio.Queue[ChunkWorkItem[_Message] | None] = asyncio.Queue()
    stop_state = PipelineStopState()
    producer_state = ProducerState(next_message_id=0)

    message = _Message(["a", "b", "c"])
    message.text_chunks = _StopMutatingChunks(stop_state, ["a", "b", "c"])

    async def _iter_messages() -> AsyncIterator[_Message]:
        yield message

    result_queue = asyncio.Queue()
    await _producer_task(
        _iter_messages(), queue, stop_state, producer_state, result_queue, None
    )

    first = await queue.get()
    sentinel = await queue.get()

    assert first is not None
    assert first.chunk_id == TextLocation(0, 0)
    assert sentinel is None
    assert producer_state.produced_chunks == 1


@pytest.mark.asyncio
async def test_dispatcher_stops_on_sentinel_and_emits_result_sentinel() -> None:
    chunk_queue: asyncio.Queue[ChunkWorkItem[_Message] | None] = asyncio.Queue()
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    message = _Message(["hello"])
    await chunk_queue.put(
        ChunkWorkItem(
            chunk_id=TextLocation(0, 0),
            chunk_count=1,
            chunk_text="hello",
            message=message,
        )
    )
    await chunk_queue.put(None)

    extractor = _SequenceExtractor([typechat.Success(_empty_knowledge())])
    model = _StubEmbeddingModel()

    await _dispatcher_task(
        chunk_queue,
        result_queue,
        stop_state,
        extractor,
        model,
        concurrency=2,
        skip_failed_messages=False,
    )

    items = await _drain_result_queue(result_queue)
    assert len(items) == 2
    assert items[-1] is None
    assert items[0] is not None
    assert items[0].error is None


@pytest.mark.asyncio
async def test_dispatcher_extraction_failure_lowers_stop() -> None:
    """A Failure from the extractor sets error and lowers stop_at_message_id."""
    chunk_queue: asyncio.Queue[ChunkWorkItem[_Message] | None] = asyncio.Queue()
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    m0 = _Message(["first"])
    m1 = _Message(["second"])
    await chunk_queue.put(
        ChunkWorkItem(
            chunk_id=TextLocation(0, 0), chunk_count=1, chunk_text="first", message=m0
        )
    )
    await chunk_queue.put(
        ChunkWorkItem(
            chunk_id=TextLocation(1, 0), chunk_count=1, chunk_text="second", message=m1
        )
    )
    await chunk_queue.put(None)

    extractor = _SequenceExtractor(
        [typechat.Failure("first failed"), typechat.Success(_empty_knowledge())]
    )
    model = _StubEmbeddingModel()

    await _dispatcher_task(
        chunk_queue,
        result_queue,
        stop_state,
        extractor,
        model,
        concurrency=1,
        skip_failed_messages=False,
    )

    items = await _drain_result_queue(result_queue)
    first = items[0]
    second = items[1]

    assert first is not None
    assert isinstance(first.error, RuntimeError)
    assert "first failed" in str(first.error)

    # Second chunk is skipped because stop_at_message_id was lowered to 0.
    assert second is not None
    assert second.error is not None

    assert stop_state.stop_at_message_id == 0
    assert extractor.calls == ["first"]


@pytest.mark.asyncio
async def test_dispatcher_extraction_failure_skips_and_keeps_processing() -> None:
    chunk_queue: asyncio.Queue[ChunkWorkItem[_Message] | None] = asyncio.Queue()
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    m0 = _Message(["first"])
    m1 = _Message(["second"])
    await chunk_queue.put(
        ChunkWorkItem(
            chunk_id=TextLocation(0, 0), chunk_count=1, chunk_text="first", message=m0
        )
    )
    await chunk_queue.put(
        ChunkWorkItem(
            chunk_id=TextLocation(1, 0), chunk_count=1, chunk_text="second", message=m1
        )
    )
    await chunk_queue.put(None)

    extractor = _SequenceExtractor(
        [typechat.Failure("first failed"), typechat.Success(_empty_knowledge())]
    )
    model = _StubEmbeddingModel()

    await _dispatcher_task(
        chunk_queue,
        result_queue,
        stop_state,
        extractor,
        model,
        concurrency=1,
        skip_failed_messages=True,
    )

    items = await _drain_result_queue(result_queue)
    first = items[0]
    second = items[1]

    assert first is not None
    assert isinstance(first.error, RuntimeError)
    assert "first failed" in str(first.error)

    assert second is not None
    assert second.error is None

    assert stop_state.stop_at_message_id == 10**100
    assert extractor.calls == ["first", "second"]


def _chunk_result(
    message: _Message,
    message_ordinal: int,
    chunk_ordinal: int,
    chunk_count: int,
    *,
    error: Exception | None = None,
) -> ChunkProcessingResult[_Message]:
    return ChunkProcessingResult(
        chunk_id=TextLocation(message_ordinal, chunk_ordinal),
        chunk_count=chunk_count,
        message=message,
        error=error,
    )


@pytest.mark.asyncio
async def test_reassembler_commits_out_of_order_after_gap_is_filled() -> None:
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    m0 = _Message(["m0"])
    m1 = _Message(["m1"])
    await result_queue.put(_chunk_result(m1, 1, 0, 1))
    await result_queue.put(_chunk_result(m0, 0, 0, 1))
    await result_queue.put(None)

    committed_batches: list[tuple[int, int]] = []

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        committed_batches.append((len(messages), len(results)))

    state = await _reassembler_task(
        result_queue,
        stop_state,
        first_uncommitted_ordinal=0,
        target_commit_chunk_count=10,
        commit_batch=_commit,
        on_batch_committed=None,
        skip_failed_messages=False,
    )

    assert committed_batches == [(2, 2)]
    assert state.messages_committed == 2
    assert state.chunks_committed == 2
    assert state.first_uncommitted_ordinal == 2
    assert state.buffered_messages == 0


@pytest.mark.asyncio
async def test_reassembler_marks_failure_and_blocks_later_commits() -> None:
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    m0 = _Message(["m0"])
    m1 = _Message(["m1"])
    await result_queue.put(_chunk_result(m0, 0, 0, 1, error=RuntimeError("boom")))
    await result_queue.put(_chunk_result(m1, 1, 0, 1))
    await result_queue.put(None)

    commit_calls = 0

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1

    state = await _reassembler_task(
        result_queue,
        stop_state,
        first_uncommitted_ordinal=0,
        target_commit_chunk_count=1,
        commit_batch=_commit,
        on_batch_committed=None,
        skip_failed_messages=False,
    )

    assert commit_calls == 0
    assert state.chunk_failures == 1
    assert state.messages_committed == 0
    assert state.buffered_messages == 2
    assert stop_state.stop_at_message_id == 0


@pytest.mark.asyncio
async def test_reassembler_skips_failed_message_and_commits_later_messages() -> None:
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    m0 = _Message(["m0"])
    m1 = _Message(["m1"])
    await result_queue.put(_chunk_result(m0, 0, 0, 1, error=RuntimeError("boom")))
    await result_queue.put(_chunk_result(m1, 1, 0, 1))
    await result_queue.put(None)

    committed_batches: list[tuple[int, int]] = []

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        committed_batches.append((len(messages), len(results)))

    state = await _reassembler_task(
        result_queue,
        stop_state,
        first_uncommitted_ordinal=0,
        target_commit_chunk_count=1,
        commit_batch=_commit,
        on_batch_committed=None,
        skip_failed_messages=True,
    )

    assert committed_batches == [(1, 1)]
    assert state.chunk_failures == 1
    assert state.messages_skipped == 1
    assert state.messages_committed == 1
    assert state.chunks_committed == 1
    assert state.buffered_messages == 0
    assert state.first_uncommitted_ordinal == 2
    assert stop_state.stop_at_message_id == 10**100


@pytest.mark.asyncio
async def test_reassembler_force_commits_small_staged_tail() -> None:
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    message = _Message(["m0"])
    await result_queue.put(_chunk_result(message, 0, 0, 1))
    await result_queue.put(None)

    commit_calls = 0

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1

    state = await _reassembler_task(
        result_queue,
        stop_state,
        first_uncommitted_ordinal=0,
        target_commit_chunk_count=99,
        commit_batch=_commit,
        on_batch_committed=None,
        skip_failed_messages=False,
    )

    assert commit_calls == 1
    assert state.messages_committed == 1
    assert state.chunks_committed == 1


@pytest.mark.asyncio
async def test_reassembler_raises_on_invalid_chunk_ordinal_and_sets_stop_marker() -> (
    None
):
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    message = _Message(["m0", "m0b"])
    await result_queue.put(_chunk_result(message, 3, 2, 2))
    await result_queue.put(None)

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        return None

    with pytest.raises(RuntimeError, match="Invalid chunk ordinal"):
        await _reassembler_task(
            result_queue,
            stop_state,
            first_uncommitted_ordinal=0,
            target_commit_chunk_count=1,
            commit_batch=_commit,
            on_batch_committed=None,
            skip_failed_messages=False,
        )

    assert stop_state.stop_at_message_id == 3


@pytest.mark.asyncio
async def test_reassembler_raises_on_duplicate_chunk_and_sets_stop_marker() -> None:
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    message = _Message(["m1-a", "m1-b"])
    await result_queue.put(_chunk_result(message, 5, 0, 2))
    await result_queue.put(_chunk_result(message, 5, 0, 2))
    await result_queue.put(None)

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        return None

    with pytest.raises(RuntimeError, match="Duplicate chunk"):
        await _reassembler_task(
            result_queue,
            stop_state,
            first_uncommitted_ordinal=0,
            target_commit_chunk_count=1,
            commit_batch=_commit,
            on_batch_committed=None,
            skip_failed_messages=False,
        )

    assert stop_state.stop_at_message_id == 5


@pytest.mark.asyncio
async def test_reassembler_on_batch_committed_callback_is_invoked() -> None:
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    message = _Message(["m0"])
    await result_queue.put(_chunk_result(message, 0, 0, 1))
    await result_queue.put(None)

    callback_calls: list[tuple[int, int]] = []

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        return None

    await _reassembler_task(
        result_queue,
        stop_state,
        first_uncommitted_ordinal=0,
        target_commit_chunk_count=1,
        commit_batch=_commit,
        on_batch_committed=lambda msg_count, chunk_count: callback_calls.append(
            (msg_count, chunk_count)
        ),
        skip_failed_messages=False,
    )

    assert callback_calls == [(1, 1)]


@pytest.mark.asyncio
async def test_reassembler_raises_on_mismatched_chunk_count_and_sets_stop_marker() -> (
    None
):
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    message = _Message(["m0-a", "m0-b", "m0-c"])
    await result_queue.put(_chunk_result(message, 4, 0, 2))
    await result_queue.put(_chunk_result(message, 4, 1, 3))
    await result_queue.put(None)

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        return None

    with pytest.raises(RuntimeError, match="Mismatched chunk count"):
        await _reassembler_task(
            result_queue,
            stop_state,
            first_uncommitted_ordinal=0,
            target_commit_chunk_count=1,
            commit_batch=_commit,
            on_batch_committed=None,
            skip_failed_messages=False,
        )

    assert stop_state.stop_at_message_id == 4


@pytest.mark.asyncio
async def test_reassembler_handles_existing_assembly_non_duplicate_chunk() -> None:
    result_queue = asyncio.Queue()
    stop_state = PipelineStopState()

    message = _Message(["m0-a", "m0-b"])
    await result_queue.put(_chunk_result(message, 0, 0, 2))
    await result_queue.put(_chunk_result(message, 0, 1, 2))
    await result_queue.put(None)

    commit_calls = 0

    async def _commit(
        messages: list[_Message], results: list[ChunkProcessingResult[_Message]]
    ) -> None:
        nonlocal commit_calls
        commit_calls += 1

    state = await _reassembler_task(
        result_queue,
        stop_state,
        first_uncommitted_ordinal=0,
        target_commit_chunk_count=1,
        commit_batch=_commit,
        on_batch_committed=None,
        skip_failed_messages=False,
    )

    assert commit_calls == 1
    assert state.messages_committed == 1
    assert state.chunks_committed == 2
