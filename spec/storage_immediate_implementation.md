# Storage Provider Design & Testing

## Overview

This document describes the current storage provider architecture and outlines next steps for unified testing across storage implementations.

## Current Architecture

The storage system follows a clean layered design:

```
Index Building Functions (timestampindex.py, propindex.py, etc.)
         ↓ (use conversation.secondary_indexes - this is the intended pattern)
ConversationSecondaryIndexes (secindex.py)
         ↓ (internally gets indexes from storage provider)
Storage Provider (memorystore.py, sqlitestore.py)
         ↓ (manages actual index instances)
Index Implementations (ConversationIndex, PropertyIndex, etc.)
```

## What's Implemented ✅

- **Storage Provider Interface**: `IStorageProvider` with `get_*_index()` methods for all 6 index types
- **Memory Implementation**: `MemoryStorageProvider` fully implements all index getters
- **SQLite Implementation**: `SqliteStorageProvider` implements all index getters  
- **Integration Layer**: `ConversationSecondaryIndexes` uses storage provider internally
- **Correct Pattern**: Index building functions use `conversation.secondary_indexes` (intended, not deprecated)
- **Tests**: Basic storage index tests exist for memory provider

## Index Types

1. **ConversationIndex** - Term → SemanticRef mappings
2. **PropertyIndex** - Property name → SemanticRef mappings  
3. **TimestampToTextRangeIndex** - Timestamp → TextRange mappings
4. **MessageTextIndex** - Message text → MessageOrdinal mappings (embeddings)
5. **RelatedTermsIndex** - Term → Related terms mappings (aliases + embeddings)
6. **ConversationThreads** - Thread description → Thread mappings

## Next Steps

### ✅ COMPLETED: Unified Storage Provider Testing

**Implementation**: Created comprehensive parameterized testing framework in `test_storage_providers_unified.py` that runs all storage provider functionality with both Memory and SQLite implementations.

**Coverage includes:**
- **Index Creation**: All 6 index types created correctly across both providers
- **Message Collections**: CRUD operations work identically  
- **SemanticRef Collections**: CRUD operations work identically (with provider-specific ID handling)
- **Index Interfaces**: All index types expose correct interfaces and handle empty lookups properly
- **Cross-Provider Validation**: Direct comparison tests ensure both providers return equivalent results

**Testing Pattern Used:**
```python
@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def storage_provider_type(request, embedding_model, temp_db_path):
    # Returns both provider types, tests run twice - once per provider
```

**Results**: 19 parameterized tests, all passing. Both storage implementations maintain behavioral parity.

### ✅ COMPLETED: Circular Import Resolution

**Issue**: Circular imports between `sqlitestore.py` and `podcast.py` modules via `get_storage_provider` function.

**Solution**: Moved `get_storage_provider` to new `typeagent/storage/utils.py` module.

**Files Updated:**
- `typeagent/storage/utils.py` - New module with `get_storage_provider`
- `typeagent/podcasts/podcast.py` - Updated import
- `typeagent/podcasts/podcast_import.py` - Updated import

### Current Architecture Status

The storage system is now fully implemented and tested:

1. **Stable APIs**: Both `MemoryStorageProvider` and `SqliteStorageProvider` implement identical interfaces
2. **Behavioral Parity**: Comprehensive test suite ensures both providers work equivalently  
3. **Correct Usage Pattern**: `conversation.secondary_indexes` remains the intended pattern (not deprecated)
4. **Index Building Functions**: Continue to work unchanged across both storage providers
5. **Future-Ready**: Architecture supports additional storage providers with minimal changes

## Development Environment

- **Testing**: `make test` - run all tests  
- **Type checking**: `make check` - type-check all files
- **Formatting**: `make format` - reformat with black
- **Environment**: Activate `.venv`, use `make clean venv` if needed
- **Unified Testing**: `python -m pytest test/test_storage_providers_unified.py` - test both storage providers

## Architectural Clarification

The implementation follows a clean layered architecture:

```
Index Building Functions (timestampindex.py, propindex.py, etc.)
         ↓ (use conversation.secondary_indexes)
ConversationSecondaryIndexes (secindex.py)
         ↓ (internally gets indexes from storage provider)
Storage Provider (memorystore.py)
         ↓ (manages actual index instances)
Index Implementations (ConversationIndex, PropertyIndex, etc.)
```

**What's implemented**:
- ✅ Storage provider has `get_*_index()` methods
- ✅ `ConversationSecondaryIndexes` uses storage provider internally
- ✅ Index building functions work unchanged (no migration needed)
- ✅ Tests validate the integration works

**What remains**:
- ❌ Multi-conversation support in storage provider (when needed)
- ❌ Complete test coverage and documentation

The foundation is solid. The current lazy index creation approach is sufficient - no explicit lifecycle management needed.

## Current State Analysis

### Index Types and Current Implementations

Based on code analysis, we have **7 index implementations** in `IConversationSecondaryIndexes`:

1. **ConversationIndex** (`semrefindex.py`):
   - Type: `ITermToSemanticRefIndex`
   - Storage: `_map: dict[str, list[ScoredSemanticRefOrdinal]]`
   - Creates: Term → SemanticRef mappings for entities, topics, actions

2. **PropertyIndex** (`propindex.py`):
   - Type: `IPropertyToSemanticRefIndex`
   - Storage: `_map: dict[str, list[ScoredSemanticRefOrdinal]]`
   - Creates: Property name → SemanticRef mappings

3. **TimestampToTextRangeIndex** (`timestampindex.py`):
   - Type: `ITimestampToTextRangeIndex`
   - Storage: `_ranges: list[TimestampedTextRange]` (sorted by timestamp)
   - Creates: Timestamp → TextRange mappings
   - **SQLite Note**: In SQLite implementation, this will be replaced by direct queries on Messages table using `start_timestamp`/`end_timestamp` columns

4. **MessageTextIndex** (`messageindex.py`):
   - Type: `IMessageTextEmbeddingIndex`
   - Storage: `TextToTextLocationIndex` (embeddings)
   - Creates: Message text → MessageOrdinal mappings

5. **RelatedTermsIndex** (`reltermsindex.py`):
   - Type: `ITermToRelatedTermsIndex`
   - Storage: `_alias_map` + `_term_index` (embeddings)
   - Creates: Term → Related terms mappings
   - **SQLite Note**: In SQLite implementation, this will be split into two tables - `RelatedTermsAliases` for exact aliases and `RelatedTermsFuzzy` for conversation-derived fuzzy matches

6. **ConversationThreads** (`convthreads.py`):
   - Type: `IConversationThreads`
   - Storage: `threads: list[Thread]` + `vector_base: VectorBase`
   - Creates: Thread description → Thread mappings

7. **EmbeddingIndex** (`fuzzyindex.py`):
   - Type: Custom wrapper around `VectorBase`
   - Storage: NumPy arrays via VectorBase
   - Creates: Text embedding vectors for fuzzy search
   - Note: Used internally by RelatedTermsIndex and MessageTextIndex

### Current Index Creation Pattern

Index creation is currently **scattered** across multiple files:

- `secindex.py`: `ConversationSecondaryIndexes` class coordinates some indexes
- `semrefindex.py`: Functions like `build_conversation_index()`, `build_semantic_ref_index()`
- `timestampindex.py`: `build_timestamp_index()`
- `messageindex.py`: Index creation within `MessageTextIndex.add_messages()`
- Individual index classes have their own `add_*()` methods

This scattered approach makes it difficult to:
- Track which indexes exist for a conversation
- Ensure consistent index creation
- Migrate to SQLite storage cleanly

## Step 0: Prepare Development Environment ✅ DONE

Following the Python coding guidelines:

- **Testing**: Use `pytest`, `assert` statements, minimal mocking
- **Type checking**: Use `pyright` (or Pylance in VS Code)
- **Commands**:
  - `make test` - run all tests
  - `make check` - type-check all files
  - `make format` - reformat with black
- **Environment**: Activate `.venv`, use `make clean venv` if needed

## Step 1: Centralize Index Creation in Storage Provider ✅ DONE

### 1.1 Extend IStorageProvider Interface ✅ DONE

~~Add index management methods to `interfaces.py`:~~ **COMPLETED**

The `IStorageProvider` interface has been successfully extended with all 6 index getter methods:

```python
# ✅ IMPLEMENTED in interfaces.py
class IStorageProvider[TMessage: IMessage](Protocol):
    # ... existing methods ...

    # Index getters - ALL 6 index types for this conversation
    async def get_conversation_index(self) -> ITermToSemanticRefIndex: ...
    async def get_property_index(self) -> IPropertyToSemanticRefIndex: ...
    async def get_timestamp_index(self) -> ITimestampToTextRangeIndex: ...
    async def get_message_text_index(self) -> IMessageTextIndex[TMessage]: ...
    async def get_related_terms_index(self) -> ITermToRelatedTermsIndex: ...
    async def get_conversation_threads(self) -> IConversationThreads: ...

    # ❌ TODO: Multi-conversation support when needed
    # async def create_indexes_for_conversation(
    #     self, conversation_id: str
    # ) -> None: ...
    # async def drop_indexes_for_conversation(
    #     self, conversation_id: str
    # ) -> None: ...
```### 1.2 Update MemoryStorageProvider ✅ DONE

~~Modify `memorystore.py` to implement index management:~~ **COMPLETED**

The `MemoryStorageProvider` has been successfully implemented with:
- All 6 index getter methods implemented
- Proper initialization through a factory method (`create()`)
- Index isolation per conversation (though currently single conversation)

```python
# ✅ IMPLEMENTED in memorystore.py
class MemoryStorageProvider[TMessage: IMessage](IStorageProvider[TMessage]):
    async def get_conversation_index(self) -> ITermToSemanticRefIndex:
        return self._conversation_index

    async def get_property_index(self) -> IPropertyToSemanticRefIndex:
        return self._property_index

    # ... all other index getters implemented
```

### 1.3 Keep Existing Index Creation Methods, Route Through Storage Provider ⚠️ PARTIALLY DONE

**Status**: Some index building functions have been updated to use storage provider, but many still access `conversation.secondary_indexes` directly.

**What's been done**:
- `ConversationSecondaryIndexes` class exists and has integration with storage provider
- Tests show it can get indexes from storage provider

**What still needs updating**:
- Multiple files still use `conversation.secondary_indexes` pattern:
  - `timestampindex.py`: Still accesses `conversation.secondary_indexes.timestamp_index`
  - `propindex.py`: Still uses `conversation.secondary_indexes`
  - `messageindex.py`: Still accesses `conversation.secondary_indexes.message_index`
  - `reltermsindex.py`: Still uses `conversation.secondary_indexes`
  - Several files in search functionality

#### ~~Update ConversationSecondaryIndexes Class~~ **COMPLETED**

~~Modify `secindex.py` to get indexes from storage provider:~~ The integration has been implemented and tested.

    async def get_message_text_index(
        self, conversation_id: str
    ) -> IMessageTextIndex[TMessage]: ...

    async def get_related_terms_index(
        self, conversation_id: str
    ) -> ITermToRelatedTermsIndex: ...

    async def get_conversation_threads(
        self, conversation_id: str
    ) -> IConversationThreads: ...

    # EmbeddingIndex is used internally by other indexes, not exposed directly

    # Index lifecycle
    async def create_indexes_for_conversation(
        self, conversation_id: str
    ) -> None: ...

    async def drop_indexes_for_conversation(
        self, conversation_id: str
    ) -> None: ...
```

### 1.2 Update MemoryStorageProvider

Modify `memorystore.py` to implement index management:

```python
class MemoryStorageProvider[TMessage: IMessage](IStorageProvider[TMessage]):
    def __init__(self):
        # ... existing init ...
        self._conversation_indexes: dict[str, ConversationIndex] = {}
        self._property_indexes: dict[str, PropertyIndex] = {}
        self._timestamp_indexes: dict[str, TimestampToTextRangeIndex] = {}
        self._message_text_indexes: dict[str, MessageTextIndex] = {}
        self._related_terms_indexes: dict[str, RelatedTermsIndex] = {}
        self._conversation_threads: dict[str, ConversationThreads] = {}

    async def get_conversation_index(
        self, conversation_id: str
    ) -> ITermToSemanticRefIndex:
        if conversation_id not in self._conversation_indexes:
            self._conversation_indexes[conversation_id] = ConversationIndex()
        return self._conversation_indexes[conversation_id]

    async def get_related_terms_index(
        self, conversation_id: str
    ) -> ITermToRelatedTermsIndex:
        if conversation_id not in self._related_terms_indexes:
            # Use default settings for now
            from .reltermsindex import RelatedTermsIndex, RelatedTermIndexSettings
            settings = RelatedTermIndexSettings()
            self._related_terms_indexes[conversation_id] = RelatedTermsIndex(settings)
        return self._related_terms_indexes[conversation_id]

    async def get_conversation_threads(
        self, conversation_id: str
    ) -> IConversationThreads:
        if conversation_id not in self._conversation_threads:
            self._conversation_threads[conversation_id] = ConversationThreads()
        return self._conversation_threads[conversation_id]

    # ... similar methods for other index types ...

    async def create_indexes_for_conversation(
        self, conversation_id: str
    ) -> None:
        # Ensure all indexes exist for this conversation
        await self.get_conversation_index(conversation_id)
        await self.get_property_index(conversation_id)
        await self.get_timestamp_index(conversation_id)
        await self.get_message_text_index(conversation_id)
        await self.get_related_terms_index(conversation_id)
        await self.get_conversation_threads(conversation_id)

    async def drop_indexes_for_conversation(
        self, conversation_id: str
    ) -> None:
        self._conversation_indexes.pop(conversation_id, None)
        self._property_indexes.pop(conversation_id, None)
        self._timestamp_indexes.pop(conversation_id, None)
        self._message_text_indexes.pop(conversation_id, None)
        self._related_terms_indexes.pop(conversation_id, None)
        self._conversation_threads.pop(conversation_id, None)
```

### 1.3 Keep Existing Index Creation Methods, Route Through Storage Provider

**Key Change**: Instead of moving the index creation logic, we'll **keep the existing methods** in their current files but **update them to use storage provider indexes**.

#### Update ConversationSecondaryIndexes Class

Modify `secindex.py` to get indexes from storage provider:

```python
class ConversationSecondaryIndexes[TMessage: IMessage](IConversationSecondaryIndexes[TMessage]):
    def __init__(self, storage_provider: IStorageProvider[TMessage], conversation_id: str):
        self._storage_provider = storage_provider
        self._conversation_id = conversation_id
        # Initialize all indexes through storage provider
        self._property_index: IPropertyToSemanticRefIndex | None = None
        self._timestamp_index: ITimestampToTextRangeIndex | None = None
        self._related_terms_index: ITermToRelatedTermsIndex | None = None
        self._threads: IConversationThreads | None = None
        self._message_index: IMessageTextIndex[TMessage] | None = None

    @property
    async def property_to_semantic_ref_index(self) -> IPropertyToSemanticRefIndex | None:
        if self._property_index is None:
            self._property_index = await self._storage_provider.get_property_index(self._conversation_id)
        return self._property_index

    @property
    async def timestamp_index(self) -> ITimestampToTextRangeIndex | None:
        if self._timestamp_index is None:
            self._timestamp_index = await self._storage_provider.get_timestamp_index(self._conversation_id)
        return self._timestamp_index

    # ... similar async properties for other indexes ...
```

#### Update Index Building Functions

Modify existing functions to use storage provider indexes:

```python
# In semrefindex.py
async def build_conversation_index[TMessage: IMessage](
    conversation: IConversation[TMessage, ConversationIndex],
    conversation_settings: importing.ConversationSettings,
    event_handler: IndexingEventHandlers | None = None,
) -> IndexingResults:
    # Get indexes from storage provider instead of conversation properties
    storage_provider = conversation.storage_provider
    conversation_index = await storage_provider.get_conversation_index(conversation.conversation_id)

    # Keep existing building logic, just use storage provider index
    result = IndexingResults()
    result.semantic_refs = await build_semantic_ref_index(
        conversation,
        conversation_settings.semantic_ref_index_settings,
        event_handler,
    )
    # ... rest of building logic stays the same ...

# In timestampindex.py
async def build_timestamp_index(conversation: IConversation) -> ListIndexingResult:
    if conversation.messages:
        # Get timestamp index from storage provider
        storage_provider = conversation.storage_provider
        timestamp_index = await storage_provider.get_timestamp_index(conversation.conversation_id)

        # Use existing logic with storage provider index
        return await add_to_timestamp_index(
            timestamp_index,
            conversation.messages,
            0,
        )
    return ListIndexingResult(0)

# Similar updates for other index building functions...
```

## Step 2: Update Tests ✅ MOSTLY DONE

### 2.1 Test Index Centralization ✅ DONE

~~Create tests for the new storage provider index methods:~~ **COMPLETED**

Tests have been successfully created:
- `test/test_storage_indexes.py` - Tests all 6 index types creation
- `test/test_secindex_storage_integration.py` - Tests ConversationSecondaryIndexes integration

```python
# ✅ IMPLEMENTED in test/test_storage_indexes.py
@pytest.mark.asyncio
async def test_all_index_creation(storage, needs_auth):
    """Test that all 6 index types are created and accessible."""
    conv_index = await storage.get_conversation_index()
    assert conv_index is not None

    prop_index = await storage.get_property_index()
    assert prop_index is not None
    # ... tests for all index types
```

### 2.2 Update Existing Tests ⚠️ PARTIALLY DONE

**Status**: Most tests pass, but some may still need updates to use new storage provider pattern instead of direct `conversation.secondary_indexes` access.

**Completed**:
- Core storage index tests working
- Integration tests working

**Still needed**:
- Review and update any remaining tests that access `conversation.secondary_indexes` directly

## Step 3: Migration Strategy ✅ DONE

### 3.1 Backward Compatibility ✅ DONE

The migration has maintained backward compatibility:
- Old conversation properties still exist and work
- ConversationSecondaryIndexes class still exists but now uses storage provider internally
- Existing code continues to work during transition

### 3.2 Gradual Migration 🔄 IN PROGRESS

1. **Phase 1**: ✅ Add new storage provider methods alongside existing code
2. **Phase 2**: ⚠️ Update index building functions to use new methods (PARTIALLY DONE)
3. **Phase 3**: ❌ Update all tests and calling code (TODO)
4. **Phase 4**: ❌ Remove deprecated conversation properties (TODO)

```python
# In test/test_storage_indexes.py
import pytest
from typeagent.storage.memorystore import MemoryStorageProvider
from typeagent.knowpro.interfaces import (
    ITermToSemanticRefIndex, IPropertyToSemanticRefIndex,
    ITimestampToTextRangeIndex, IMessageTextIndex,
    ITermToRelatedTermsIndex, IConversationThreads
)

@pytest.mark.asyncio
async def test_all_index_creation():
    """Test that all 7 index types are created lazily."""
    storage = MemoryStorageProvider()

    # Test all index types
    conv_index = await storage.get_conversation_index("conv1")
    assert isinstance(conv_index, ITermToSemanticRefIndex)

    prop_index = await storage.get_property_index("conv1")
    assert isinstance(prop_index, IPropertyToSemanticRefIndex)

    time_index = await storage.get_timestamp_index("conv1")
    assert isinstance(time_index, ITimestampToTextRangeIndex)

    msg_index = await storage.get_message_text_index("conv1")
    assert isinstance(msg_index, IMessageTextIndex)

    rel_index = await storage.get_related_terms_index("conv1")
    assert isinstance(rel_index, ITermToRelatedTermsIndex)

    threads = await storage.get_conversation_threads("conv1")
    assert isinstance(threads, IConversationThreads)

@pytest.mark.asyncio
async def test_index_isolation():
    """Test that indexes are isolated per conversation."""
    storage = MemoryStorageProvider()

    index1 = await storage.get_conversation_index("conv1")
    index2 = await storage.get_conversation_index("conv2")

    assert index1 is not index2

    # Add term to one index
    index1.add_term("test", 0)

    # Should not appear in other index
    assert len(index2.lookup_term("test")) == 0

@pytest.mark.asyncio
async def test_index_persistence_per_conversation():
    """Test that same index instance is returned for same conversation."""
    storage = MemoryStorageProvider()

    # All index types should return same instance for same conversation
    conv1_1 = await storage.get_conversation_index("conv1")
    conv1_2 = await storage.get_conversation_index("conv1")
    assert conv1_1 is conv1_2

    prop1_1 = await storage.get_property_index("conv1")
    prop1_2 = await storage.get_property_index("conv1")
    assert prop1_1 is prop1_2

@pytest.mark.asyncio
async def test_drop_indexes():
    """Test that dropping indexes cleans up properly."""
    storage = MemoryStorageProvider()

    # Create all index types
    await storage.get_conversation_index("conv1")
    await storage.get_property_index("conv1")
    await storage.get_timestamp_index("conv1")
    await storage.get_message_text_index("conv1")
    await storage.get_related_terms_index("conv1")
    await storage.get_conversation_threads("conv1")

    # Drop them
    await storage.drop_indexes_for_conversation("conv1")

    # New indexes should be created on next access
    new_conv_index = await storage.get_conversation_index("conv1")
    assert len(new_conv_index) == 0
```

### 2.2 Update Existing Tests

Update tests that currently access `conversation.semantic_ref_index` to use the new methods:

```python
# Before:
# index = conversation.semantic_ref_index

# After:
index = await conversation.get_conversation_index()
```

## Step 3: Migration Strategy

### 3.1 Backward Compatibility

To avoid breaking existing code during transition:

1. Keep old conversation properties but mark them deprecated
2. Implement them as wrappers around storage provider methods
3. Add deprecation warnings
4. Remove in later phase

### 3.2 Gradual Migration

1. **Phase 1**: Add new storage provider methods alongside existing code
2. **Phase 2**: Update index building functions to use new methods
3. **Phase 3**: Update all tests and calling code
4. **Phase 4**: Remove deprecated conversation properties

## UPDATED Implementation Plan - Next Steps

Update (Aug 18, 2025): Team decision — Steps 1–3 are not needed now. Focus shifts to docs and SQLite.

### Decisions

- Add Index Lifecycle Methods (create/drop): Not needed now. Deferred until we truly need explicit lifecycle hooks.
- Multi-conversation support in MemoryStorageProvider: Not needed now. Defer until there’s a concrete multi-conversation use case.
- Additional/edge-case test coverage: Skipped for now; covered by existing suite and separate slow E2E tests.

### Current Focus

1. Documentation updates to reflect the finalized pattern (30–45 minutes)
2. Proceed to SQLite implementation (tracked separately)

### Notes

- By design, index-building functions continue to use `conversation.secondary_indexes`. No migration is required.
- `ConversationSecondaryIndexes` remains the integration layer and pulls real indexes from the storage provider lazily.

## Updated Success Criteria

### ✅ COMPLETED:
- [x] Storage provider interface extended with index methods
- [x] MemoryStorageProvider implements all index getters
- [x] Basic tests for storage indexes created
- [x] Integration tests for ConversationSecondaryIndexes
- [x] Backward compatibility maintained

### ❌ REMAINING:
- [ ] Documentation updated (reflect centralized storage provider usage and current design)

### Deferred (not needed now):
- Index lifecycle methods (create/drop)
- Multi-conversation support in MemoryStorageProvider
- Additional/edge-case tests beyond current suite and separate E2E

## Recommended Next Action Sequence

1. Update documentation (30–45 minutes)
    - Clarify the finalized pattern and developer-facing guidance.
2. Start SQLite storage provider implementation (separate effort)
3. Keep lifecycle/multi-conversation work deferred until a concrete need arises

**Total immediate effort: ~30–45 minutes for docs. SQLite tracked separately.**

**Current Status: READY FOR SQLITE IMPLEMENTATION** ✅

The core architecture is complete and working. Index building functions correctly use the `conversation.secondary_indexes` pattern, and `ConversationSecondaryIndexes` correctly integrates with the storage provider internally. The lazy index creation approach is sufficient for current needs.