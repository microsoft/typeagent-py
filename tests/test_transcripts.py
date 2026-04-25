# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import timedelta
import os

import pytest
import webvtt

from typeagent.aitools.embeddings import IEmbeddingModel
from typeagent.aitools.model_adapters import create_test_embedding_model
from typeagent.knowpro.convsettings import ConversationSettings
from typeagent.knowpro.universal_message import format_timestamp_utc, UNIX_EPOCH
from typeagent.storage.memory.collections import (
    MemoryMessageCollection,
    MemorySemanticRefCollection,
)
from typeagent.storage.memory.semrefindex import TermToSemanticRefIndex
from typeagent.transcripts.transcript import (
    split_speaker_name,
    Transcript,
    TranscriptMessage,
    TranscriptMessageMeta,
)
from typeagent.transcripts.transcript_ingest import (
    extract_speaker_from_text,
    get_transcript_duration,
    get_transcript_speakers,
    parse_voice_tags,
    webvtt_timestamp_to_seconds,
)

from conftest import CONFUSE_A_CAT_VTT, has_testdata_file, PARROT_SKETCH_VTT


def test_extract_speaker_from_text():
    """Test speaker extraction from various text formats."""
    test_cases = [
        ("SPEAKER: Hello world", "SPEAKER", "Hello world"),
        ("[John] This is a test", "John", "This is a test"),
        ("- Mary: Another test", "Mary", "Another test"),
        ("Just plain text without speaker", None, "Just plain text without speaker"),
        ("VETERINARIAN: How can I help you?", "VETERINARIAN", "How can I help you?"),
        (
            "(Dr. Smith) Let me examine the patient",
            "Dr. Smith",
            "Let me examine the patient",
        ),
        ("", None, ""),
        ("NARRATOR: Once upon a time...", "NARRATOR", "Once upon a time..."),
    ]

    for input_text, expected_speaker, expected_text in test_cases:
        speaker, text = extract_speaker_from_text(input_text)
        assert (
            speaker == expected_speaker
        ), f"Speaker mismatch for '{input_text}': got {speaker}, expected {expected_speaker}"
        assert (
            text == expected_text
        ), f"Text mismatch for '{input_text}': got {text}, expected {expected_text}"


def test_webvtt_timestamp_conversion():
    """Test conversion of WebVTT timestamps to seconds."""
    test_cases = [
        ("00:00:07.599", 7.599),
        ("00:01:30.000", 90.0),
        ("01:05:45.123", 3945.123),
        ("10.5", 10.5),
        ("01:30", 90.0),
    ]

    for timestamp, expected_seconds in test_cases:
        result = webvtt_timestamp_to_seconds(timestamp)
        assert (
            abs(result - expected_seconds) < 0.001
        ), f"Timestamp conversion failed for {timestamp}: got {result}, expected {expected_seconds}"


@pytest.mark.skipif(
    not has_testdata_file("Confuse-A-Cat.vtt"),
    reason="Test VTT file not found",
)
def test_get_transcript_info():
    """Test getting basic information from a VTT file."""
    vtt_file = CONFUSE_A_CAT_VTT

    # Test duration
    duration = get_transcript_duration(vtt_file)
    assert duration > 0, "Duration should be positive"
    assert duration < 3600, "Duration should be less than an hour for test file"

    # Test speakers (may be empty if no speaker patterns found)
    speakers = get_transcript_speakers(vtt_file)
    assert isinstance(speakers, set), "Speakers should be returned as a set"


@pytest.fixture
def conversation_settings(
    needs_auth: None, embedding_model: IEmbeddingModel
) -> ConversationSettings:
    """Create conversation settings for testing."""
    return ConversationSettings(embedding_model)


@pytest.mark.skipif(
    not has_testdata_file("Confuse-A-Cat.vtt"),
    reason="Test VTT file not found",
)
@pytest.mark.asyncio
async def test_ingest_vtt_transcript(conversation_settings: ConversationSettings):
    """Test importing a VTT file into a Transcript object."""
    vtt_file = CONFUSE_A_CAT_VTT

    # Use in-memory storage to avoid database cleanup issues
    settings = conversation_settings

    # Parse the VTT file
    vtt = webvtt.read(vtt_file)

    # Create messages from captions (parsing multiple speakers per cue)
    messages_list = []
    for caption in vtt:
        if not caption.text.strip():
            continue

        # Parse raw text for voice tags (handles multiple speakers per cue)
        raw_text = getattr(caption, "raw_text", caption.text)
        voice_segments = parse_voice_tags(raw_text)

        for speaker, text in voice_segments:
            if not text.strip():
                continue

            # Calculate timestamp from WebVTT start time
            offset_seconds = webvtt_timestamp_to_seconds(caption.start)
            timestamp = format_timestamp_utc(
                UNIX_EPOCH + timedelta(seconds=offset_seconds)
            )

            metadata = TranscriptMessageMeta(
                speaker=speaker,
                recipients=[],
            )
            message = TranscriptMessage(
                text_chunks=[text], metadata=metadata, timestamp=timestamp
            )
            messages_list.append(message)

    # Create in-memory collections
    msg_coll = MemoryMessageCollection[TranscriptMessage]()
    await msg_coll.extend(messages_list)

    _semref_coll = MemorySemanticRefCollection()
    _semref_index = TermToSemanticRefIndex()

    # Create transcript with in-memory storage
    transcript = await Transcript.create(
        settings,
        name="Test-Confuse-A-Cat",
        tags=["Test-Confuse-A-Cat", "vtt-transcript"],
    )

    # Add messages to the transcript's collections
    await transcript.messages.extend(messages_list)

    # Verify the transcript was created correctly
    assert isinstance(transcript, Transcript)
    assert transcript.name_tag == "Test-Confuse-A-Cat"
    assert "Test-Confuse-A-Cat" in transcript.tags
    assert "vtt-transcript" in transcript.tags

    # Check that messages were created
    message_count = await transcript.messages.size()
    assert message_count > 0, "Should have at least one message"

    # Check message structure
    first_message = None
    async for message in transcript.messages:
        first_message = message
        break

    assert first_message is not None
    assert isinstance(first_message, TranscriptMessage)
    assert isinstance(first_message.metadata, TranscriptMessageMeta)
    assert len(first_message.text_chunks) > 0
    assert first_message.text_chunks[0].strip() != ""

    # Verify message has timestamp
    assert first_message.timestamp is not None
    assert first_message.timestamp.endswith("Z")  # Should be UTC


def test_transcript_message_creation():
    """Test creating transcript messages manually."""
    # Create a transcript message with timestamp
    timestamp = format_timestamp_utc(UNIX_EPOCH + timedelta(seconds=10))
    metadata = TranscriptMessageMeta(speaker="Test Speaker", recipients=[])

    message = TranscriptMessage(
        text_chunks=["This is a test message."],
        metadata=metadata,
        tags=["test"],
        timestamp=timestamp,
    )

    # Test serialization
    serialized = message.serialize()
    assert serialized["textChunks"] == ["This is a test message."]
    assert serialized["metadata"]["speaker"] == "Test Speaker"
    assert serialized["metadata"]["listeners"] == []
    assert serialized["tags"] == ["test"]
    assert serialized["timestamp"] == timestamp

    # Test deserialization
    deserialized = TranscriptMessage.deserialize(serialized)
    assert deserialized.text_chunks == ["This is a test message."]
    assert deserialized.metadata.speaker == "Test Speaker"
    assert deserialized.metadata.recipients == []
    assert deserialized.tags == ["test"]
    assert deserialized.timestamp == timestamp


@pytest.mark.asyncio
async def test_transcript_creation():
    """Test creating an empty transcript."""
    # Create a minimal transcript for testing structure
    embedding_model = create_test_embedding_model()
    settings = ConversationSettings(embedding_model)

    transcript = await Transcript.create(
        settings=settings, name="Test Transcript", tags=["test", "empty"]
    )

    assert transcript.name_tag == "Test Transcript"
    assert "test" in transcript.tags
    assert "empty" in transcript.tags
    assert await transcript.messages.size() == 0


@pytest.mark.asyncio
async def test_transcript_knowledge_extraction_slow(
    really_needs_auth: None, embedding_model: IEmbeddingModel
):
    """
    Test that knowledge extraction works during transcript ingestion.

    This test verifies the complete ingestion pipeline:
    1. Parses first 5 messages from Parrot Sketch VTT file
    2. Creates transcript with in-memory storage (fast)
    3. Runs build_index() with auto_extract_knowledge=True
    4. Verifies both mechanical extraction (entities/actions from metadata)
       and LLM extraction (topics from content) work correctly
    """
    # Use in-memory storage for speed
    settings = ConversationSettings(embedding_model)

    # Parse first 5 captions from Parrot Sketch
    vtt_file = PARROT_SKETCH_VTT
    if not os.path.exists(vtt_file):
        pytest.skip(f"Test file {vtt_file} not found")

    vtt = webvtt.read(vtt_file)

    # Create messages from first 5 captions
    messages_list = []
    # vtt is indexable but not iterable
    for i in range(min(len(vtt), 5)):
        caption = vtt[i]
        if not caption.text.strip():
            continue

        speaker = getattr(caption, "voice", None)
        text = caption.text.strip()

        # Calculate timestamp from WebVTT start time
        offset_seconds = webvtt_timestamp_to_seconds(caption.start)
        timestamp = format_timestamp_utc(UNIX_EPOCH + timedelta(seconds=offset_seconds))

        metadata = TranscriptMessageMeta(
            speaker=speaker,
            recipients=[],
        )
        message = TranscriptMessage(
            text_chunks=[text], metadata=metadata, timestamp=timestamp
        )
        messages_list.append(message)

    # Create in-memory collections
    msg_coll = MemoryMessageCollection[TranscriptMessage]()
    await msg_coll.extend(messages_list)

    _semref_coll = MemorySemanticRefCollection()
    _semref_index = TermToSemanticRefIndex()

    # Create transcript with in-memory storage
    transcript = await Transcript.create(
        settings,
        name="Parrot-Test",
        tags=["test", "parrot"],
    )

    # Verify we have messages
    assert len(messages_list) >= 3, "Need at least 3 messages for testing"

    # Enable knowledge extraction
    settings.semantic_ref_index_settings.auto_extract_knowledge = True
    settings.semantic_ref_index_settings.concurrency = 10

    # Add messages with indexing (this should extract knowledge)
    result = await transcript.add_messages_with_indexing(messages_list)

    # Verify messages and semantic refs were created
    assert await transcript.messages.size() == len(messages_list)
    assert result.messages_added == len(messages_list)
    assert result.semrefs_added > 0, "Should have extracted some semantic references"

    semref_count = await transcript.semantic_refs.size()
    assert semref_count > 0, "Should have semantic refs"

    # Verify we have different types of knowledge
    knowledge_types = set()
    async for semref in transcript.semantic_refs:
        knowledge_types.add(semref.knowledge.knowledge_type)

    # Should have mechanical extraction (entities/actions from speakers)
    assert "entity" in knowledge_types, "Should have extracted entities"
    assert "action" in knowledge_types, "Should have extracted actions"

    # Should have LLM extraction (topics)
    assert "topic" in knowledge_types, "Should have extracted topics from LLM"

    # Verify semantic ref index was populated
    terms = await transcript.semantic_ref_index.get_terms()
    assert len(terms) > 0, "Should have indexed some terms"

    print(
        f"\nExtracted {semref_count} semantic refs from {len(messages_list)} messages"
    )
    print(f"Knowledge types: {knowledge_types}")
    print(f"Indexed terms: {len(terms)}")


# ---------------------------------------------------------------------------
# split_speaker_name
# ---------------------------------------------------------------------------


class TestSplitSpeakerName:
    def test_single_word(self) -> None:
        result = split_speaker_name("alice")
        assert result is not None
        assert result.first_name == "alice"
        assert result.last_name is None
        assert result.middle_name is None

    def test_two_words(self) -> None:
        result = split_speaker_name("john smith")
        assert result is not None
        assert result.first_name == "john"
        assert result.last_name == "smith"
        assert result.middle_name is None

    def test_three_words(self) -> None:
        result = split_speaker_name("john michael smith")
        assert result is not None
        assert result.first_name == "john"
        assert result.middle_name == "michael"
        assert result.last_name == "smith"

    def test_van_prefix_merged_into_last_name(self) -> None:
        result = split_speaker_name("jan van eyck")
        assert result is not None
        assert result.first_name == "jan"
        assert result.last_name == "van eyck"
        assert result.middle_name is None

    def test_empty_string_returns_none(self) -> None:
        result = split_speaker_name("")
        assert result is None


# ---------------------------------------------------------------------------
# Serialize / deserialize roundtrip (in-memory, no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_serialize_deserialize_roundtrip() -> None:
    """Serialize a transcript and deserialize into a fresh one — data is preserved."""
    embedding_model = create_test_embedding_model()
    settings = ConversationSettings(embedding_model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = False

    # Build original transcript — use add_messages_with_indexing so the
    # message text index (and its embeddings) are populated before serializing.
    original = await Transcript.create(settings, name="roundtrip-test", tags=["foo"])
    msg1 = TranscriptMessage(
        text_chunks=["Hello world"],
        metadata=TranscriptMessageMeta(speaker="Alice", recipients=["Bob"]),
        tags=["t1"],
        timestamp="2024-01-01T00:00:00Z",
    )
    msg2 = TranscriptMessage(
        text_chunks=["Goodbye"],
        metadata=TranscriptMessageMeta(speaker="Bob", recipients=[]),
        tags=[],
        timestamp="2024-01-01T00:01:00Z",
    )
    await original.add_messages_with_indexing([msg1, msg2])
    data = await original.serialize()

    # Deserialize into a fresh transcript.
    fresh_settings = ConversationSettings(embedding_model)
    fresh_settings.semantic_ref_index_settings.auto_extract_knowledge = False
    fresh = await Transcript.create(fresh_settings, name="", tags=[])
    await fresh.deserialize(data)

    assert fresh.name_tag == "roundtrip-test"
    assert "foo" in fresh.tags
    assert await fresh.messages.size() == 2

    first = await fresh.messages.get_item(0)
    assert first.text_chunks == ["Hello world"]
    assert first.metadata.speaker == "Alice"
    assert first.metadata.recipients == ["Bob"]
    assert first.timestamp == "2024-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_transcript_deserialize_non_empty_raises() -> None:
    """Deserializing into a non-empty Transcript raises RuntimeError."""
    embedding_model = create_test_embedding_model()
    settings = ConversationSettings(embedding_model)

    transcript = await Transcript.create(settings, name="test", tags=[])
    await transcript.messages.append(
        TranscriptMessage(
            text_chunks=["existing"],
            metadata=TranscriptMessageMeta(speaker=None, recipients=[]),
        )
    )
    data = await transcript.serialize()

    # Trying to deserialize into it again must raise.
    with pytest.raises(RuntimeError):
        await transcript.deserialize(data)


# ---------------------------------------------------------------------------
# write_to_file / read_from_file roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_and_read_from_file(tmp_path: os.PathLike[str]) -> None:
    """write_to_file + read_from_file preserves names, tags, and messages."""
    embedding_model = create_test_embedding_model()
    settings = ConversationSettings(embedding_model)
    settings.semantic_ref_index_settings.auto_extract_knowledge = False

    original = await Transcript.create(settings, name="file-test", tags=["persisted"])
    msg = TranscriptMessage(
        text_chunks=["Persisted message"],
        metadata=TranscriptMessageMeta(speaker="Eve", recipients=[]),
        timestamp="2024-06-01T12:00:00Z",
    )
    # Use add_messages_with_indexing so embeddings are built before writing.
    await original.add_messages_with_indexing([msg])
    prefix = os.path.join(str(tmp_path), "test_transcript")
    await original.write_to_file(prefix)

    # Verify the _data.json file was written.
    assert os.path.exists(prefix + "_data.json")

    # Read it back.
    fresh_settings = ConversationSettings(embedding_model)
    fresh_settings.semantic_ref_index_settings.auto_extract_knowledge = False
    loaded = await Transcript.read_from_file(prefix, fresh_settings)

    assert loaded.name_tag == "file-test"
    assert "persisted" in loaded.tags
    assert await loaded.messages.size() == 1
    first = await loaded.messages.get_item(0)
    assert first.text_chunks == ["Persisted message"]
    assert first.metadata.speaker == "Eve"


# ---------------------------------------------------------------------------
# Speaker alias building
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_speaker_aliases_full_name() -> None:
    """Full-name speakers create first-name ↔ full-name aliases."""
    embedding_model = create_test_embedding_model()
    settings = ConversationSettings(embedding_model)

    transcript = await Transcript.create(settings, name="alias-test", tags=[])
    msg = TranscriptMessage(
        text_chunks=["Hi"],
        metadata=TranscriptMessageMeta(speaker="John Smith", recipients=[]),
    )
    await transcript.messages.append(msg)

    # Rebuild aliases explicitly.
    await transcript._build_speaker_aliases()

    secondary = transcript._get_secondary_indexes()
    assert secondary.term_to_related_terms_index is not None
    aliases = secondary.term_to_related_terms_index.aliases

    # "john" should be aliased to "john smith" and vice-versa.
    john_aliases = await aliases.lookup_term("john")
    assert john_aliases is not None
    alias_texts = [t.text for t in john_aliases]
    assert "john smith" in alias_texts

    full_aliases = await aliases.lookup_term("john smith")
    assert full_aliases is not None
    assert "john" in [t.text for t in full_aliases]


@pytest.mark.asyncio
async def test_build_speaker_aliases_single_name_no_alias() -> None:
    """Single-word speaker names produce no aliases."""
    embedding_model = create_test_embedding_model()
    settings = ConversationSettings(embedding_model)

    transcript = await Transcript.create(settings, name="alias-test2", tags=[])
    msg = TranscriptMessage(
        text_chunks=["Hello"],
        metadata=TranscriptMessageMeta(speaker="Alice", recipients=[]),
    )
    await transcript.messages.append(msg)
    await transcript._build_speaker_aliases()

    secondary = transcript._get_secondary_indexes()
    assert secondary.term_to_related_terms_index is not None
    aliases = secondary.term_to_related_terms_index.aliases

    # Single-name speaker — no alias entry expected.
    result = await aliases.lookup_term("alice")
    assert not result
