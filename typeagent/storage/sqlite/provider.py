# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite storage provider implementation."""

import sqlite3
from datetime import datetime, timezone

from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from ...knowpro.interfaces import ConversationMetadata
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
    get_db_schema_version,
    init_db_schema,
    _set_conversation_metadata,
)


class SqliteStorageProvider[TMessage: interfaces.IMessage](
    interfaces.IStorageProvider[TMessage]
):
    """SQLite-backed storage provider implementation.

    This provider performs consistency checks on database initialization to ensure
    that existing embeddings match the configured embedding_size. If a mismatch is
    detected, a ValueError is raised with a descriptive error message.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        conversation_id: str = "default",
        message_type: type[TMessage] = None,  # type: ignore
        semantic_ref_type: type[interfaces.SemanticRef] = None,  # type: ignore
        message_text_index_settings: MessageTextIndexSettings | None = None,
        related_term_index_settings: RelatedTermIndexSettings | None = None,
    ):
        self.db_path = db_path
        self.conversation_id = conversation_id
        self.message_type = message_type
        self.semantic_ref_type = semantic_ref_type

        # Settings with defaults (require embedding settings)
        if message_text_index_settings is None:
            # Create default embedding settings if not provided
            from ...aitools.vectorbase import TextEmbeddingIndexSettings

            embedding_settings = TextEmbeddingIndexSettings()
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

        # Initialize conversation metadata if this is a new database
        self._init_conversation_metadata_if_needed()

        # Check embedding consistency before initializing indexes
        self._check_embedding_consistency()

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

    def _check_embedding_consistency(self) -> None:
        """Check that existing embeddings in the database match the expected embedding size.

        This method is called during initialization to ensure that embeddings stored in the
        database match the embedding_size specified in ConversationSettings. This prevents
        runtime errors when trying to use embeddings of incompatible sizes.

        Raises:
            ValueError: If embeddings in the database don't match the expected size.
        """
        from .schema import deserialize_embedding

        cursor = self.db.cursor()
        expected_size = (
            self.message_text_index_settings.embedding_index_settings.embedding_size
        )

        # Check message text index embeddings
        cursor.execute("SELECT embedding FROM MessageTextIndex LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            embedding = deserialize_embedding(row[0])
            actual_size = len(embedding)
            if actual_size != expected_size:
                raise ValueError(
                    f"Message text index embedding size mismatch: "
                    f"database contains embeddings of size {actual_size}, "
                    f"but ConversationSettings specifies embedding_size={expected_size}. "
                    f"The database was likely created with a different embedding model. "
                    f"Please use the same embedding model or create a new database."
                )

        # Check related terms fuzzy index embeddings
        cursor.execute("SELECT term_embedding FROM RelatedTermsFuzzy LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            embedding = deserialize_embedding(row[0])
            actual_size = len(embedding)
            if actual_size != expected_size:
                raise ValueError(
                    f"Related terms index embedding size mismatch: "
                    f"database contains embeddings of size {actual_size}, "
                    f"but ConversationSettings specifies embedding_size={expected_size}. "
                    f"The database was likely created with a different embedding model. "
                    f"Please use the same embedding model or create a new database."
                )

    def _init_conversation_metadata_if_needed(self) -> None:
        """Initialize conversation metadata if the database is new (empty metadata table).

        This is called once during provider initialization to set up default metadata
        including name_tag, schema_version, and timestamps. If metadata already exists,
        this does nothing.
        """
        from ...knowpro.universal_message import format_timestamp_utc


        # Initialize with default values in a transaction
        current_time = datetime.now(timezone.utc)
        with self.db:
            cursor = self.db.cursor()

            # Check if metadata already exists (inside transaction to avoid race conditions)
            cursor.execute("SELECT 1 FROM ConversationMetadata LIMIT 1")
            if cursor.fetchone() is not None:
                # Metadata already exists, don't overwrite
                return

            _set_conversation_metadata(
                self.db,
                name_tag=f"conversation_{self.conversation_id}",
                schema_version=str(get_db_schema_version(self.db)),
                created_at=format_timestamp_utc(current_time),
                updated_at=format_timestamp_utc(current_time),
            )

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

        # Get all key-value pairs
        cursor.execute("SELECT key, value FROM ConversationMetadata")
        rows = cursor.fetchall()

        if not rows:
            return None

        # Build metadata structure - always use list for consistency
        metadata_dict: dict[str, list[str]] = {}
        for key, value in rows:
            if key not in metadata_dict:
                metadata_dict[key] = []
            metadata_dict[key].append(value)

        # Helper to get single value from list (for well-known keys)
        def get_single(key: str, default: str = "") -> str:
            values = metadata_dict.get(key, [])
            if len(values) > 1:
                raise ValueError(
                    f"Expected single value for key '{key}', got {len(values)}"
                )
            return values[0] if values else default

        # Helper to parse datetime from ISO string
        def parse_datetime(key: str) -> datetime:
            value_str = get_single(key)
            if not value_str:
                return datetime.now(timezone.utc)
            # Handle both formats: with and without timezone
            if value_str.endswith("Z"):
                value_str = value_str[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(value_str)
            except ValueError:
                # Fallback for other formats
                return datetime.now(timezone.utc)

        # Extract standard fields (single values expected)
        name_tag = get_single("name_tag")
        schema_version_str = get_single("schema_version", "1")
        try:
            schema_version = int(schema_version_str)
        except ValueError:
            schema_version = 1  # Default to version 1 if parsing fails
        created_at = parse_datetime("created_at")
        updated_at = parse_datetime("updated_at")

        # Handle tags (multiple values allowed)
        tags = metadata_dict.get("tag", [])

        # Build extra dict from remaining keys
        standard_keys = {
            "name_tag",
            "schema_version",
            "created_at",
            "updated_at",
            "tag",
        }
        extra = {}
        for key, values in metadata_dict.items():
            if key not in standard_keys:
                # For extra fields, join multiple values
                extra[key] = ", ".join(values)

        return ConversationMetadata(
            name_tag=name_tag,
            schema_version=schema_version,
            created_at=created_at,
            updated_at=updated_at,
            tags=tags,
            extra=extra,
        )

    def set_conversation_metadata(self, **kwds: str | list[str] | None) -> None:
        """Set or update conversation metadata key-value pairs.

        Args:
            **kwds: Metadata keys and values where:
                - str value: Sets a single key-value pair (replaces existing)
                - list[str] value: Sets multiple values for the same key
                - None value: Deletes all rows for the given key

        Example:
            provider.set_conversation_metadata(
                name_tag="my_conversation",
                schema_version="0.1",
                created_at="2024-01-01T00:00:00Z",
                tag=["python", "ai"],  # Multiple tags
                custom_field="value"
            )
        """
        _set_conversation_metadata(self.db, **kwds)

    def update_conversation_timestamps(
        self,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ) -> None:
        """Update conversation timestamps.

        Args:
            created_at: Optional creation timestamp
            updated_at: Optional last updated timestamp
        """
        from ...knowpro.universal_message import format_timestamp_utc

        # Check if any metadata exists
        cursor = self.db.cursor()
        cursor.execute("SELECT 1 FROM ConversationMetadata LIMIT 1")

        if not cursor.fetchone():
            # Insert default values if no metadata exists
            name_tag = f"conversation_{self.conversation_id}"
            schema_version = "1.0"

            metadata_kwds: dict[str, str | None] = {
                "name_tag": name_tag,
                "schema_version": schema_version,
            }
            if created_at is not None:
                metadata_kwds["created_at"] = format_timestamp_utc(created_at)
            if updated_at is not None:
                metadata_kwds["updated_at"] = format_timestamp_utc(updated_at)
            _set_conversation_metadata(self.db, **metadata_kwds)
        else:
            # Update only the specified fields
            metadata_kwds = {}
            if created_at is not None:
                metadata_kwds["created_at"] = format_timestamp_utc(created_at)
            if updated_at is not None:
                metadata_kwds["updated_at"] = format_timestamp_utc(updated_at)
            if metadata_kwds:
                _set_conversation_metadata(self.db, **metadata_kwds)

    def get_db_version(self) -> int:
        """Get the database schema version."""
        return get_db_schema_version(self.db)
