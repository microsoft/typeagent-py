#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test to verify property index population in storage providers."""

import asyncio
import tempfile
import os
import pytest

import numpy as np

from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.utils import load_dotenv
from typeagent.knowpro.interfaces import Tag, SemanticRef, TextRange, TextLocation
from typeagent.knowpro import kplib
from typeagent.knowpro.convsettings import MessageTextIndexSettings
from typeagent.knowpro.convsettings import RelatedTermIndexSettings
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.podcasts.podcast import PodcastMessage
from typeagent.storage import SqliteStorageProvider

from fixtures import really_needs_auth


class MockEmbeddingModel(AsyncEmbeddingModel):
    def __init__(self):
        super().__init__(embedding_size=3, model_name="test")

    async def get_embeddings(self, keys: list[str]) -> np.ndarray:
        result = np.random.rand(len(keys), 3).astype(np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / norms


@pytest.mark.asyncio
async def test_property_index_population_from_database(really_needs_auth):
    """Test that property index is correctly populated when reopening a database."""
    load_dotenv()
    temp_db_file = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    temp_db_path = temp_db_file.name
    temp_db_file.close()

    try:
        embedding_model = MockEmbeddingModel()
        embedding_settings = TextEmbeddingIndexSettings(embedding_model)
        message_text_settings = MessageTextIndexSettings(embedding_settings)
        related_terms_settings = RelatedTermIndexSettings(embedding_settings)

        # Create and populate database
        storage1 = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=PodcastMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )

        # Add test semantic refs with all knowledge types
        location = TextLocation(message_ordinal=0)
        text_range = TextRange(start=location)

        test_data = [
            # Entity with facets
            SemanticRef(
                semantic_ref_ordinal=0,
                range=text_range,
                knowledge=kplib.ConcreteEntity(
                    name="John Doe",
                    type=["person", "speaker"],
                    facets=[kplib.Facet(name="role", value="host")],
                ),
            ),
            # Action
            SemanticRef(
                semantic_ref_ordinal=1,
                range=text_range,
                knowledge=kplib.Action(
                    verbs=["discuss", "explain"],
                    verb_tense="present",
                    subject_entity_name="John Doe",
                    object_entity_name="technology",
                    indirect_object_entity_name="audience",
                ),
            ),
            # Tag
            SemanticRef(
                semantic_ref_ordinal=2,
                range=text_range,
                knowledge=Tag(text="interview"),
            ),
        ]

        sem_ref_collection = await storage1.get_semantic_ref_collection()
        for sem_ref in test_data:
            await sem_ref_collection.append(sem_ref)

        await storage1.close()

        # Reopen database and verify property index
        # Use the same embedding settings to avoid dimension mismatch
        embedding_model2 = MockEmbeddingModel()
        embedding_settings2 = TextEmbeddingIndexSettings(embedding_model2)
        message_text_settings2 = MessageTextIndexSettings(embedding_settings2)
        related_terms_settings2 = RelatedTermIndexSettings(embedding_settings2)

        storage2 = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=PodcastMessage,
            message_text_index_settings=message_text_settings2,
            related_term_index_settings=related_terms_settings2,
        )

        # Create a test conversation and build property index
        from typeagent.podcasts.podcast import Podcast
        from typeagent.knowpro.convsettings import ConversationSettings
        from typeagent.storage.memory.propindex import build_property_index

        settings2 = ConversationSettings()
        settings2.storage_provider = storage2
        conversation = await Podcast.create(settings2)

        # Build property index from the semantic refs
        await build_property_index(conversation)

        prop_index = await storage2.get_property_index()
        from typeagent.knowpro.interfaces import IPropertyToSemanticRefIndex

        assert isinstance(prop_index, IPropertyToSemanticRefIndex)

        # Verify property index is populated
        prop_size = await prop_index.size()
        assert prop_size > 0, "Property index should not be empty"

        # Test entity properties
        name_lookup = await prop_index.lookup_property("name", "john doe")
        assert (
            name_lookup is not None and len(name_lookup) > 0
        ), "Entity name should be indexed"

        type_lookup = await prop_index.lookup_property("type", "person")
        assert (
            type_lookup is not None and len(type_lookup) > 0
        ), "Entity type should be indexed"

        facet_name_lookup = await prop_index.lookup_property("facet.name", "role")
        assert (
            facet_name_lookup is not None and len(facet_name_lookup) > 0
        ), "Facet name should be indexed"

        facet_value_lookup = await prop_index.lookup_property("facet.value", "host")
        assert (
            facet_value_lookup is not None and len(facet_value_lookup) > 0
        ), "Facet value should be indexed"

        # Test action properties
        verb_lookup = await prop_index.lookup_property("verb", "discuss explain")
        assert (
            verb_lookup is not None and len(verb_lookup) > 0
        ), "Action verbs should be indexed"

        subject_lookup = await prop_index.lookup_property("subject", "john doe")
        assert (
            subject_lookup is not None and len(subject_lookup) > 0
        ), "Action subject should be indexed"

        object_lookup = await prop_index.lookup_property("object", "technology")
        assert (
            object_lookup is not None and len(object_lookup) > 0
        ), "Action object should be indexed"

        indirect_object_lookup = await prop_index.lookup_property(
            "indirectobject", "audience"
        )
        assert (
            indirect_object_lookup is not None and len(indirect_object_lookup) > 0
        ), "Action indirect object should be indexed"

        # Test tag properties
        tag_lookup = await prop_index.lookup_property("tag", "interview")
        assert tag_lookup is not None and len(tag_lookup) > 0, "Tag should be indexed"

        await storage2.close()

        print("✅ All property index population tests passed!")

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
