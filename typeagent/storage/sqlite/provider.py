# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SQLite storage provider implementation."""

import sqlite3
import typing

from ...knowpro import interfaces
from ...knowpro.convsettings import MessageTextIndexSettings, RelatedTermIndexSettings
from .collections import SqliteMessageCollection, SqliteSemanticRefCollection
from .indexes import (
    SqliteMessageTextIndex,
    SqlitePropertyIndex,
    SqliteRelatedTermsIndex,
    SqliteTermToSemanticRefIndex,
    SqliteTimestampToTextRangeIndex,
)
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
            from ...aitools.embeddings import AsyncEmbeddingModel, TEST_MODEL_NAME
            from ...aitools.vectorbase import TextEmbeddingIndexSettings

            model = AsyncEmbeddingModel(model_name=TEST_MODEL_NAME)
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
        self.db.execute("PRAGMA foreign_keys = ON")

        # Initialize schema
        init_db_schema(self.db)

        # Initialize collections
        self._message_collection = SqliteMessageCollection(self.db, self.message_type)
        self._semantic_ref_collection = SqliteSemanticRefCollection(self.db)

        # Initialize indexes
        self._term_to_semantic_ref_index = SqliteTermToSemanticRefIndex(self.db)
        self._property_index = SqlitePropertyIndex(self.db)
        self._timestamp_index = SqliteTimestampToTextRangeIndex(self.db)
        self._message_text_index = SqliteMessageTextIndex(
            self.db, self.message_text_index_settings
        )
        self._related_terms_index = SqliteRelatedTermsIndex(self.db)

    async def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, "db"):
            self.db.close()

    def __del__(self) -> None:
        """Ensure database is closed when object is deleted."""
        # Can't use async in __del__, so close directly
        if hasattr(self, "db"):
            self.db.close()

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
        with self.db:
            cursor = self.db.cursor()
            # Clear in reverse dependency order
            cursor.execute("DELETE FROM RelatedTermsFuzzy")
            cursor.execute("DELETE FROM RelatedTermsAliases")
            cursor.execute("DELETE FROM MessageTextIndex")
            cursor.execute("DELETE FROM PropertyIndex")
            cursor.execute("DELETE FROM SemanticRefIndex")
            cursor.execute("DELETE FROM SemanticRefs")
            cursor.execute("DELETE FROM Messages")
            cursor.execute("DELETE FROM Conversations")

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
            self._term_to_semantic_ref_index.deserialize(
                data["termToSemanticRefIndexData"]
            )

        # Deserialize related terms index
        if data.get("relatedTermsIndexData"):
            await self._related_terms_index.deserialize(data["relatedTermsIndexData"])

    def get_conversation_metadata(self) -> ConversationMetadata | None:
        """Get conversation metadata."""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT created_at, updated_at FROM Conversations WHERE conversation_id = ?",
            (self.conversation_id,),
        )
        row = cursor.fetchone()
        if row:
            return ConversationMetadata(
                name_tag=f"conversation_{self.conversation_id}",
                schema_version="1.0",
                created_at=row[0],
                updated_at=row[1],
                tags=[],
                extra={},
            )
        return None

    def update_conversation_metadata(
        self, created_at: str | None = None, updated_at: str | None = None
    ) -> None:
        """Update conversation metadata."""
        with self.db:
            cursor = self.db.cursor()

            # Check if conversation exists
            cursor.execute(
                "SELECT 1 FROM Conversations WHERE conversation_id = ?",
                (self.conversation_id,),
            )

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
                    params.append(self.conversation_id)
                    cursor.execute(
                        f"UPDATE Conversations SET {', '.join(updates)} WHERE conversation_id = ?",
                        params,
                    )
            else:
                # Insert new
                cursor.execute(
                    "INSERT INTO Conversations (conversation_id, created_at, updated_at) VALUES (?, ?, ?)",
                    (self.conversation_id, created_at, updated_at),
                )

    def get_db_version(self) -> int:
        """Get the database schema version."""
        version_str = get_db_schema_version(self.db)
        try:
            return int(version_str.split(".")[0])  # Get major version as int
        except (ValueError, AttributeError):
            return 1  # Default version
