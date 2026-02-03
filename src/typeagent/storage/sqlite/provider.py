# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite storage provider implementation."""

from datetime import datetime, timezone
import sqlite3

from ...aitools.embeddings import AsyncEmbeddingModel
from ...aitools.vectorbase import TextEmbeddingIndexSettings
from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from ...knowpro.interfaces import ConversationMetadata, STATUS_INGESTED
from .collections import SqliteMessageCollection, SqliteSemanticRefCollection
from .messageindex import SqliteMessageTextIndex
from .propindex import SqlitePropertyIndex
from .reltermsindex import SqliteRelatedTermsIndex
from .schema import (
    _set_conversation_metadata,
    CONVERSATION_SCHEMA_VERSION,
    get_db_schema_version,
    init_db_schema,
)
from .semrefindex import SqliteTermToSemanticRefIndex
from .timestampindex import SqliteTimestampToTextRangeIndex


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
        message_type: type[TMessage] = None,  # type: ignore
        semantic_ref_type: type[interfaces.SemanticRef] = None,  # type: ignore
        message_text_index_settings: MessageTextIndexSettings | None = None,
        related_term_index_settings: RelatedTermIndexSettings | None = None,
        metadata: ConversationMetadata | None = None,
    ):
        self.db_path = db_path
        self.message_type = message_type
        self.semantic_ref_type = semantic_ref_type
        self._metadata = metadata

        provided_message_settings = message_text_index_settings
        provided_related_settings = related_term_index_settings

        # Initialize database connection with autocommit mode
        # isolation_level=None enables manual transaction control via BEGIN/COMMIT
        self.db = sqlite3.connect(db_path, isolation_level=None)

        # Configure SQLite for optimal bulk insertion performance
        # TODO: Move into init_db_schema()
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

        self.message_text_index_settings, self.related_term_index_settings = (
            self._resolve_embedding_settings(
                provided_message_settings, provided_related_settings
            )
        )

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

    def _conversation_metadata_exists(self) -> bool:
        cursor = self.db.cursor()
        cursor.execute("SELECT 1 FROM ConversationMetadata LIMIT 1")
        return cursor.fetchone() is not None

    def _get_single_metadata_value(self, key: str) -> str | None:
        cursor = self.db.cursor()
        cursor.execute("SELECT value FROM ConversationMetadata WHERE key = ?", (key,))
        rows = cursor.fetchall()
        return rows[0][0] if rows else None

    def _resolve_embedding_settings(
        self,
        provided_message_settings: MessageTextIndexSettings | None,
        provided_related_settings: RelatedTermIndexSettings | None,
    ) -> tuple[MessageTextIndexSettings, RelatedTermIndexSettings]:
        metadata_exists = self._conversation_metadata_exists()
        stored_size_str = self._get_single_metadata_value("embedding_size")
        stored_name = self._get_single_metadata_value("embedding_name")
        stored_size = int(stored_size_str) if stored_size_str else None

        if provided_message_settings is None:
            if stored_size is not None or stored_name is not None:
                embedding_model = AsyncEmbeddingModel(
                    embedding_size=stored_size,
                    model_name=stored_name,
                )
                base_embedding_settings = TextEmbeddingIndexSettings(
                    embedding_model=embedding_model,
                    embedding_size=stored_size,
                )
            else:
                base_embedding_settings = TextEmbeddingIndexSettings()
            message_settings = MessageTextIndexSettings(base_embedding_settings)
        else:
            message_settings = provided_message_settings
            base_embedding_settings = message_settings.embedding_index_settings
            provided_size = base_embedding_settings.embedding_size
            provided_name = base_embedding_settings.embedding_model.model_name
            if stored_size is not None and stored_size != provided_size:
                raise ValueError(
                    f"Conversation metadata embedding_size "
                    f"({stored_size}) does not match provided embedding size ({provided_size})."
                )
            if stored_name is not None and stored_name != provided_name:
                raise ValueError(
                    f"Conversation metadata embedding_model "
                    f"({stored_name}) does not match provided embedding model ({provided_name})."
                )

        if provided_related_settings is None:
            related_settings = RelatedTermIndexSettings(base_embedding_settings)
        else:
            related_settings = provided_related_settings
            related_embedding_settings = related_settings.embedding_index_settings
            related_size = related_embedding_settings.embedding_size
            related_name = related_embedding_settings.embedding_model.model_name
            if related_size != base_embedding_settings.embedding_size:
                raise ValueError(
                    "Related term index embedding_size does not match message text index embedding_size"
                )
            if related_name != base_embedding_settings.embedding_model.model_name:
                raise ValueError(
                    "Related term index embedding_model does not match message text index embedding_model"
                )
            if related_settings.embedding_index_settings is not base_embedding_settings:
                related_settings.embedding_index_settings = base_embedding_settings

        actual_size = base_embedding_settings.embedding_size
        actual_name = base_embedding_settings.embedding_model.model_name

        if self._metadata is not None:
            if self._metadata.embedding_size is None:
                self._metadata.embedding_size = actual_size
            elif self._metadata.embedding_size != actual_size:
                raise ValueError(
                    "Conversation metadata embedding_size does not match provider settings"
                )

            if self._metadata.embedding_model is None:
                self._metadata.embedding_model = actual_name
            elif self._metadata.embedding_model != actual_name:
                raise ValueError(
                    "Conversation metadata embedding_model does not match provider settings"
                )

        if metadata_exists:
            metadata_updates: dict[str, str] = {}
            if stored_size is None:
                metadata_updates["embedding_size"] = str(actual_size)
            if stored_name is None:
                metadata_updates["embedding_name"] = actual_name
            if metadata_updates:
                _set_conversation_metadata(self.db, **metadata_updates)

        return message_settings, related_settings

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

        This does NOT start a transaction - the metadata will be committed
        when the first actual write operation (e.g., adding messages) commits.
        This ensures we don't create empty databases with only metadata.
        """
        from ...knowpro.universal_message import format_timestamp_utc

        current_time = datetime.now(timezone.utc)
        cursor = self.db.cursor()

        # Check if metadata already exists
        cursor.execute("SELECT 1 FROM ConversationMetadata LIMIT 1")
        if cursor.fetchone() is not None:
            return

        # Use provided metadata values, or generate defaults for None
        if self._metadata:
            name_tag = self._metadata.name_tag or "conversation"
            tags = self._metadata.tags
            extras = self._metadata.extra or {}
        else:
            name_tag = "conversation"
            tags = None
            extras = {}

        actual_embedding_size = (
            self.message_text_index_settings.embedding_index_settings.embedding_size
        )
        actual_embedding_name = (
            self.message_text_index_settings.embedding_index_settings.embedding_model.model_name
        )

        metadata_embedding_size = (
            self._metadata.embedding_size
            if self._metadata and self._metadata.embedding_size is not None
            else actual_embedding_size
        )
        metadata_embedding_name = (
            self._metadata.embedding_model
            if self._metadata and self._metadata.embedding_model is not None
            else actual_embedding_name
        )

        extras = {
            key: value
            for key, value in extras.items()
            if key not in {"embedding_size", "embedding_name"}
        }

        # Always auto-generate schema_version and timestamps
        # Don't use 'with self.db:' - let first write operation commit
        _set_conversation_metadata(
            self.db,
            name_tag=name_tag,
            schema_version=str(get_db_schema_version(self.db)),
            created_at=format_timestamp_utc(current_time),
            updated_at=format_timestamp_utc(current_time),
            tag=tags,  # None or list of tags
            embedding_size=str(metadata_embedding_size),
            embedding_name=metadata_embedding_name,
            **extras,
        )

    async def __aenter__(self) -> "SqliteStorageProvider[TMessage]":
        """Enter transaction context."""
        if self.db.in_transaction:
            raise RuntimeError(
                "Cannot start a new transaction: a transaction is already in progress. "
                "This may happen if: (1) you're nesting 'async with storage:' blocks, "
                "(2) a previous transaction was not properly committed/rolled back, or "
                "(3) the database file was left in an inconsistent state from a crash."
            )
        self.db.execute("BEGIN IMMEDIATE")
        # Initialize metadata on first write transaction
        self._init_conversation_metadata_if_needed()
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

    def get_conversation_metadata(self) -> ConversationMetadata:
        """Get conversation metadata."""
        cursor = self.db.cursor()

        # Get all key-value pairs
        cursor.execute("SELECT key, value FROM ConversationMetadata")
        rows = cursor.fetchall()

        # If no rows at all, return empty instance (all fields None)
        if not rows:
            return ConversationMetadata()

        # Build metadata structure - always use list for consistency
        metadata_dict: dict[str, list[str]] = {}
        for key, value in rows:
            if key not in metadata_dict:
                metadata_dict[key] = []
            metadata_dict[key].append(value)

        # Helper to get single value from list (returns None if key missing)
        def get_single(key: str) -> str | None:
            values = metadata_dict.get(key)
            if values is None:
                return None
            if len(values) > 1:
                raise ValueError(
                    f"Expected single value for key '{key}', got {len(values)}"
                )
            return values[0]

        # Helper to parse datetime from ISO string
        def parse_datetime(value_str: str) -> datetime:
            # Handle both formats: with and without timezone
            if value_str.endswith("Z"):
                value_str = value_str[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(value_str)
            except ValueError:
                # Fallback for other formats
                return datetime.now(timezone.utc)

        # Extract standard fields (None if not found)
        name_tag = get_single("name_tag")

        schema_version_str = get_single("schema_version")
        schema_version = int(schema_version_str) if schema_version_str else None

        created_at_str = get_single("created_at")
        created_at = parse_datetime(created_at_str) if created_at_str else None

        updated_at_str = get_single("updated_at")
        updated_at = parse_datetime(updated_at_str) if updated_at_str else None

        embedding_size_str = get_single("embedding_size")
        embedding_size = int(embedding_size_str) if embedding_size_str else None

        embedding_model = get_single("embedding_name")

        # Handle tags (multiple values allowed, None if key doesn't exist)
        tags = metadata_dict.get("tag")

        # Build extra dict from remaining keys
        standard_keys = {
            "name_tag",
            "schema_version",
            "created_at",
            "updated_at",
            "tag",
            "embedding_size",
            "embedding_name",
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
            embedding_size=embedding_size,
            embedding_model=embedding_model,
            tags=tags,
            extra=extra if extra else None,
        )

    def set_conversation_metadata(self, **kwds: str | list[str] | None) -> None:
        """Set or update conversation metadata key-value pairs.

        Args:
            **kwds: Metadata keys and values where:
                - str | int value: Sets a single key-value pair (replaces existing)
                - list[str | int] value: Sets multiple values for the same key
                - None value: Deletes all rows for the given key

        Example:
            provider.set_conversation_metadata(
                name_tag="my_conversation",
                schema_version="1",
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
            name_tag = self._metadata.name_tag if self._metadata else "conversation"
            schema_version = str(CONVERSATION_SCHEMA_VERSION)
            actual_embedding_size = (
                self.message_text_index_settings.embedding_index_settings.embedding_size
            )
            actual_embedding_name = (
                self.message_text_index_settings.embedding_index_settings.embedding_model.model_name
            )

            metadata_kwds: dict[str, str | None] = {
                "name_tag": name_tag or "conversation",
                "schema_version": schema_version,
                "embedding_size": str(actual_embedding_size),
                "embedding_name": actual_embedding_name,
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

    def is_source_ingested(self, source_id: str) -> bool:
        """Check if a source has already been ingested.

        This is a read-only operation that can be called outside of a transaction.

        Args:
            source_id: External source identifier (email ID, file path, etc.)

        Returns:
            True if the source has been ingested, False otherwise.
        """
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT status FROM IngestedSources WHERE source_id = ?", (source_id,)
        )
        row = cursor.fetchone()
        return row is not None and row[0] == STATUS_INGESTED

    def get_source_status(self, source_id: str) -> str | None:
        """Get the ingestion status of a source.

        Args:
            source_id: External source identifier (email ID, file path, etc.)

        Returns:
            The status string if the source exists, or None if it hasn't been ingested.
        """
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT status FROM IngestedSources WHERE source_id = ?", (source_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def mark_source_ingested(
        self, source_id: str, status: str = STATUS_INGESTED
    ) -> None:
        """Mark a source as ingested.

        This performs an INSERT but does NOT commit. It should be called within
        a transaction context (e.g., inside `async with storage_provider:`).
        The commit happens when the transaction context exits successfully.

        Args:
            source_id: External source identifier (email ID, file path, etc.)
        """
        cursor = self.db.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO IngestedSources (source_id, status) VALUES (?, ?)",
            (source_id, status),
        )
