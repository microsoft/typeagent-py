# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite storage provider implementation."""

import json
import sqlite3

from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from .collections import SqliteMessageCollection, SqliteSemanticRefCollection
from .messageindex import SqliteMessageTextIndex
from .propindex import SqlitePropertyIndex
from .reltermsindex import SqliteRelatedTermsIndex
from .semrefindex import SqliteTermToSemanticRefIndex
from .timestampindex import SqliteTimestampToTextRangeIndex
from .schema import (
    CONVERSATIONS_SCHEMA,
    MESSAGE_TEXT_INDEX_SCHEMA,
    MESSAGES_SCHEMA,
    PROPERTY_INDEX_SCHEMA,
    RELATED_TERMS_ALIASES_SCHEMA,
    RELATED_TERMS_FUZZY_SCHEMA,
    SEMANTIC_REF_INDEX_SCHEMA,
    SEMANTIC_REFS_SCHEMA,
    ConversationMetadata,
    get_db_schema_version,
    init_db_schema,
)


class SqliteStorageProvider[TMessage: interfaces.IMessage](
    interfaces.IStorageProvider[TMessage]
):
    """SQLite-backed storage provider implementation."""

    def __init__(
        self,
        db_path: str = ":memory:",
        conversation_id: str = "default",
        message_type: type[TMessage] = None,  # type: ignore
        semantic_ref_type: type[interfaces.SemanticRef] = None,  # type: ignore
        conversation_index_settings=None,
        message_text_index_settings: MessageTextIndexSettings | None = None,
        related_term_index_settings: RelatedTermIndexSettings | None = None,
    ):
        self.db_path = db_path
        self.conversation_id = conversation_id
        self.message_type = message_type
        self.semantic_ref_type = semantic_ref_type

        # Settings with defaults (require embedding settings)
        self.conversation_index_settings = conversation_index_settings or {}
        if message_text_index_settings is None:
            # Create default embedding settings if not provided
            from ...aitools.embeddings import AsyncEmbeddingModel
            from ...aitools.vectorbase import TextEmbeddingIndexSettings

            model = AsyncEmbeddingModel()
            embedding_settings = TextEmbeddingIndexSettings(model)
            message_text_index_settings = MessageTextIndexSettings(embedding_settings)
        self.message_text_index_settings = message_text_index_settings

        if related_term_index_settings is None:
            # Use the same embedding settings
            embedding_settings = message_text_index_settings.embedding_index_settings
            related_term_index_settings = RelatedTermIndexSettings(embedding_settings)
        self.related_term_index_settings = related_term_index_settings

        # Initialize database connection
        self.db = sqlite3.connect(db_path)

        # Configure SQLite for optimal bulk insertion performance
        self.db.execute("PRAGMA foreign_keys = ON")
        # Improve write performance for bulk operations
        self.db.execute("PRAGMA synchronous = NORMAL")  # Faster than FULL, still safe
        self.db.execute(
            "PRAGMA journal_mode = WAL"
        )  # Write-Ahead Logging for better concurrency
        self.db.execute("PRAGMA cache_size = -64000")  # 64MB cache (negative = KB)
        self.db.execute("PRAGMA temp_store = MEMORY")  # Store temp tables in memory
        self.db.execute("PRAGMA mmap_size = 268435456")  # 256MB memory-mapped I/O

        # Initialize schema
        init_db_schema(self.db)

        # Initialize collections
        # Initialize message collection first
        self._message_collection = SqliteMessageCollection(self.db, self.message_type)
        self._semantic_ref_collection = SqliteSemanticRefCollection(self.db)

        # Initialize indexes
        self._term_to_semantic_ref_index = SqliteTermToSemanticRefIndex(self.db)
        self._property_index = SqlitePropertyIndex(self.db)
        self._timestamp_index = SqliteTimestampToTextRangeIndex(self.db)
        self._message_text_index = SqliteMessageTextIndex(
            self.db,
            self.message_text_index_settings,
            self._message_collection,
        )
        # Initialize related terms index
        self._related_terms_index = SqliteRelatedTermsIndex(
            self.db, self.related_term_index_settings.embedding_index_settings
        )

        # Connect message collection to message text index for automatic indexing
        self._message_collection.set_message_text_index(self._message_text_index)

    async def __aenter__(self) -> "SqliteStorageProvider[TMessage]":
        """Enter transaction context."""
        self.db.execute("BEGIN IMMEDIATE")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit transaction context. Commits on success, rolls back on exception."""
        if exc_type is None:
            self.db.commit()
        else:
            self.db.rollback()

    async def close(self) -> None:
        """Close the database connection. COMMITS."""
        if hasattr(self, "db"):
            self.db.commit()
            self.db.close()
            del self.db

    def __del__(self) -> None:
        """Ensure database is closed when object is deleted. ROLLS BACK."""
        # Can't use async in __del__, so close directly
        if hasattr(self, "db"):
            self.db.rollback()
            self.db.close()
            del self.db

    @property
    def messages(self) -> SqliteMessageCollection[TMessage]:
        return self._message_collection

    @property
    def semantic_refs(self) -> SqliteSemanticRefCollection:
        return self._semantic_ref_collection

    @property
    def term_to_semantic_ref_index(self) -> SqliteTermToSemanticRefIndex:
        return self._term_to_semantic_ref_index

    @property
    def property_index(self) -> SqlitePropertyIndex:
        return self._property_index

    @property
    def timestamp_index(self) -> SqliteTimestampToTextRangeIndex:
        return self._timestamp_index

    @property
    def message_text_index(self) -> SqliteMessageTextIndex:
        return self._message_text_index

    @property
    def related_terms_index(self) -> SqliteRelatedTermsIndex:
        return self._related_terms_index

    # Async getters required by base class
    async def get_message_collection(
        self, message_type: type[TMessage] | None = None
    ) -> interfaces.IMessageCollection[TMessage]:
        """Get the message collection."""
        return self._message_collection

    async def get_semantic_ref_collection(self) -> interfaces.ISemanticRefCollection:
        """Get the semantic reference collection."""
        return self._semantic_ref_collection

    async def get_semantic_ref_index(self) -> interfaces.ITermToSemanticRefIndex:
        """Get the semantic reference index."""
        return self._term_to_semantic_ref_index

    async def get_property_index(self) -> interfaces.IPropertyToSemanticRefIndex:
        """Get the property index."""
        return self._property_index

    async def get_timestamp_index(self) -> interfaces.ITimestampToTextRangeIndex:
        """Get the timestamp index."""
        return self._timestamp_index

    async def get_message_text_index(self) -> interfaces.IMessageTextIndex[TMessage]:
        """Get the message text index."""
        return self._message_text_index

    async def get_related_terms_index(self) -> interfaces.ITermToRelatedTermsIndex:
        """Get the related terms index."""
        return self._related_terms_index

    async def get_conversation_threads(self) -> interfaces.IConversationThreads:
        """Get the conversation threads."""
        # For now, return a simple implementation
        # In a full implementation, this would be stored/retrieved from SQLite
        from ...storage.memory.convthreads import ConversationThreads

        return ConversationThreads(
            self.message_text_index_settings.embedding_index_settings
        )

    async def clear(self) -> None:
        """Clear all data from the storage provider."""
        cursor = self.db.cursor()
        # Clear in reverse dependency order
        cursor.execute("DELETE FROM RelatedTermsFuzzy")
        cursor.execute("DELETE FROM RelatedTermsAliases")
        cursor.execute("DELETE FROM MessageTextIndex")
        cursor.execute("DELETE FROM PropertyIndex")
        cursor.execute("DELETE FROM SemanticRefIndex")
        cursor.execute("DELETE FROM SemanticRefs")
        cursor.execute("DELETE FROM Messages")
        cursor.execute("DELETE FROM ConversationMetadata")

        # Clear in-memory indexes
        await self._message_text_index.clear()

    def serialize(self) -> dict:
        """Serialize all storage provider data."""
        return {
            "termToSemanticRefIndexData": self._term_to_semantic_ref_index.serialize(),
            "relatedTermsIndexData": self._related_terms_index.serialize(),
        }

    async def deserialize(self, data: dict) -> None:
        """Deserialize storage provider data."""
        # Deserialize term to semantic ref index
        if data.get("termToSemanticRefIndexData"):
            await self._term_to_semantic_ref_index.deserialize(
                data["termToSemanticRefIndexData"]
            )

        # Deserialize related terms index
        if data.get("relatedTermsIndexData"):
            await self._related_terms_index.deserialize(data["relatedTermsIndexData"])

        # Deserialize message text index
        if data.get("messageIndexData"):
            await self._message_text_index.deserialize(data["messageIndexData"])

    def get_conversation_metadata(self) -> ConversationMetadata | None:
        """Get conversation metadata."""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT name_tag, schema_version, created_at, updated_at, tags, extra FROM ConversationMetadata LIMIT 1"
        )
        row = cursor.fetchone()
        if row:
            return ConversationMetadata(
                name_tag=row[0],
                schema_version=row[1],
                created_at=row[2],
                updated_at=row[3],
                tags=json.loads(row[4]) if row[4] else [],
                extra=json.loads(row[5]) if row[5] else {},
            )
        return None

    def update_conversation_metadata(
        self, created_at: str | None = None, updated_at: str | None = None
    ) -> None:
        """Update conversation metadata."""
        cursor = self.db.cursor()

        # Check if conversation metadata exists
        cursor.execute("SELECT 1 FROM ConversationMetadata LIMIT 1")

        if cursor.fetchone():
            # Update existing
            updates = []
            params = []
            if created_at is not None:
                updates.append("created_at = ?")
                params.append(created_at)
            if updated_at is not None:
                updates.append("updated_at = ?")
                params.append(updated_at)

            if updates:
                cursor.execute(
                    f"UPDATE ConversationMetadata SET {', '.join(updates)}",
                    params,
                )
        else:
            # Insert new with default values
            name_tag = f"conversation_{self.conversation_id}"
            schema_version = "1.0"
            tags = json.dumps([])
            extra = json.dumps({})

            cursor.execute(
                "INSERT INTO ConversationMetadata (name_tag, schema_version, created_at, updated_at, tags, extra) VALUES (?, ?, ?, ?, ?, ?)",
                (name_tag, schema_version, created_at, updated_at, tags, extra),
            )

    def get_db_version(self) -> int:
        """Get the database schema version."""
        version_str = get_db_schema_version(self.db)
        try:
            return int(version_str.split(".")[0])  # Get major version as int
        except (ValueError, AttributeError):
            return 1  # Default version
