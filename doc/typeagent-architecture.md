# TypeAgent Architecture Design

## Overview
TypeAgent implements a conversation-based knowledge processing system with dual storage backends and structured retrieval. The architecture consists of four main components operating in a pipeline.

## Core Components

### 1. Storage Layer
**Dual-implementation pattern** providing identical interfaces with different backing stores:

- **Collections (2 types)**:
  - `MessageCollection` - Stores conversation messages with ordinal indexing
  - `SemanticRefCollection` - Stores semantic references and knowledge artifacts

- **Indexes (6 types)**:
  - `SemanticRefIndex` - Term → SemanticRef mappings for content discovery
  - `PropertyIndex` - Property name → SemanticRef mappings for structured queries  
  - `TimestampToTextRangeIndex` - Temporal navigation and filtering
  - `MessageTextIndex` - Embedding-based semantic similarity search
  - `RelatedTermsIndex` - Term expansion and alias resolution
  - `ConversationThreads` - Thread organization and context grouping

**Storage Providers**:
- `MemoryStorageProvider` - In-memory collections with fast access, no persistence
- `SqliteStorageProvider` - SQLite-backed persistence with identical API surface

### 2. Knowledge Extractor
**Multi-mode content processing** that transforms raw input into structured knowledge:

- **Basic Mode**: Rule-based extraction from titles, headings, metadata
- **AI Mode**: LLM-powered entity, topic, and relationship extraction
- **Hybrid Enhancement**: Combines basic extraction with AI summarization
- **Batch Processing**: Handles multiple content items with progress tracking

**Components**:
- `ContentExtractor` - Core extraction engine with mode switching
- `AIModelManager` - Handles LLM integration and fallback strategies
- `KnowledgeTranslator` - Converts natural language to structured knowledge schemas

### 3. Query Pipeline
**Structured RAG system** that translates natural language queries into multi-stage retrieval:

- **Query Translation**: Natural language → structured search expressions
- **Multi-Index Search**: Parallel queries across semantic and structured indexes
- **Result Fusion**: Combines entity, topic, message, and action matches
- **Fallback Strategy**: Raw text similarity when structured search fails
- **Thread Context**: Applies conversation thread filtering and scoping

**Pipeline Stages**:
1. **Language Processing**: Parse user intent and extract search terms
2. **Index Querying**: Execute parallel searches across relevant indexes  
3. **Score Fusion**: Merge and rank results from multiple sources
4. **Context Building**: Prepare enriched context for answer generation
5. **Answer Generation**: LLM-powered response synthesis with citations

### 4. Integration Layer
**Conversation interface** that coordinates storage, extraction, and querying:

- `ConversationSecondaryIndexes` - Unified index access and coordination
- `SearchProcessor` - Orchestrates multi-stage query execution
- `SearchResponse` - Structured result packaging with metadata
- `AnswerGenerator` - Context-aware response synthesis

## Data Flow

```
Input → Knowledge Extractor → Storage Layer → Query Pipeline → Response
  ↓         ↓                    ↓              ↓              ↓
Text    Entities/Topics    Collections+     Parallel      Generated
Audio   Actions/Facets     Indexes (6)      Index         Answer +
Images  Relations                           Queries       Citations
```

## Key Design Principles

- **Storage Abstraction**: Identical APIs for memory vs persistent storage enable seamless switching
- **Parallel Indexing**: Six specialized indexes support different query patterns and access paths
- **Graceful Degradation**: System operates with basic extraction when AI models unavailable
- **Structured RAG**: Combines semantic similarity with structured knowledge for precision+recall
- **Mode Flexibility**: Extraction modes balance processing speed vs knowledge quality

## Implementation Status
- ✅ Dual storage providers with full API parity
- ✅ Six-index architecture with unified testing
- ✅ Multi-mode knowledge extraction pipeline  
- ✅ Natural language query processing with fallbacks
- ✅ Thread-aware search and context building
