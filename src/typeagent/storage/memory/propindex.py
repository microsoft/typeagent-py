# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import enum
from typing import assert_never

from ...knowpro.collections import TextRangesInScope
from ...knowpro.interfaces import (
    IConversation,
    IPropertyToSemanticRefIndex,
    ISemanticRefCollection,
    ScoredSemanticRefOrdinal,
    SemanticRefOrdinal,
    Tag,
    Topic,
)
from ...knowpro import kplib


class PropertyNames(enum.Enum):
    EntityName = "name"
    EntityType = "type"
    FacetName = "facet.name"
    FacetValue = "facet.value"
    Verb = "verb"
    Subject = "subject"
    Object = "object"
    IndirectObject = "indirectObject"
    Tag = "tag"
    Topic = "topic"


async def add_facet(
    facet: kplib.Facet | None,
    property_index: IPropertyToSemanticRefIndex,
    semantic_ref_ordinal: SemanticRefOrdinal,
) -> None:
    if facet is not None:
        await property_index.add_property(
            PropertyNames.FacetName.value,
            facet.name,
            semantic_ref_ordinal,
        )
        value = facet.value
        if value is not None:
            # If the value is a float, we use .g format store it as a string.
            if isinstance(value, float) and value:
                value = f"{value:g}"
            await property_index.add_property(
                PropertyNames.FacetValue.value,
                str(value),
                semantic_ref_ordinal,
            )


async def add_entity_properties_to_index(
    entity: kplib.ConcreteEntity,
    property_index: IPropertyToSemanticRefIndex,
    semantic_ref_ordinal: SemanticRefOrdinal,
) -> None:
    await property_index.add_property(
        PropertyNames.EntityName.value,
        entity.name,
        semantic_ref_ordinal,
    )
    for type in entity.type:
        await property_index.add_property(
            PropertyNames.EntityType.value,
            type,
            semantic_ref_ordinal,
        )
    # Add every facet name as a separate term.
    if entity.facets:
        for facet in entity.facets:
            await add_facet(facet, property_index, semantic_ref_ordinal)


async def add_action_properties_to_index(
    action: kplib.Action,
    property_index: IPropertyToSemanticRefIndex,
    semantic_ref_ordinal: SemanticRefOrdinal,
) -> None:
    await property_index.add_property(
        PropertyNames.Verb.value,
        " ".join(action.verbs),
        semantic_ref_ordinal,
    )
    if action.subject_entity_name != "none":
        await property_index.add_property(
            PropertyNames.Subject.value,
            action.subject_entity_name,
            semantic_ref_ordinal,
        )
    if action.object_entity_name != "none":
        await property_index.add_property(
            PropertyNames.Object.value,
            action.object_entity_name,
            semantic_ref_ordinal,
        )
    if action.indirect_object_entity_name != "none":
        await property_index.add_property(
            PropertyNames.IndirectObject.value,
            action.indirect_object_entity_name,
            semantic_ref_ordinal,
        )


async def build_property_index(conversation: IConversation) -> None:
    await add_to_property_index(conversation, 0)


async def add_to_property_index(
    conversation: IConversation,
    start_at_ordinal: SemanticRefOrdinal,
) -> None:
    """Add semantic references from a conversation to the property index starting at a specific ordinal."""
    if (
        csi := conversation.secondary_indexes
    ) and conversation.semantic_refs is not None:
        # Check if semantic_refs collection is not empty
        if await conversation.semantic_refs.size() == 0:
            return

        if (property_index := csi.property_to_semantic_ref_index) is None:
            property_index = csi.property_to_semantic_ref_index = PropertyIndex()

        semantic_refs = conversation.semantic_refs
        size = await semantic_refs.size()

        for semantic_ref_ordinal, semantic_ref in enumerate(
            await semantic_refs.get_slice(start_at_ordinal, size),
            start_at_ordinal,
        ):
            assert semantic_ref.semantic_ref_ordinal == semantic_ref_ordinal
            if isinstance(semantic_ref.knowledge, kplib.Action):
                await add_action_properties_to_index(
                    semantic_ref.knowledge, property_index, semantic_ref_ordinal
                )
            elif isinstance(semantic_ref.knowledge, kplib.ConcreteEntity):
                await add_entity_properties_to_index(
                    semantic_ref.knowledge, property_index, semantic_ref_ordinal
                )
            elif isinstance(semantic_ref.knowledge, Tag):
                tag = semantic_ref.knowledge
                await property_index.add_property(
                    PropertyNames.Tag.value, tag.text, semantic_ref_ordinal
                )
            elif isinstance(semantic_ref.knowledge, Topic):
                pass
            else:
                assert_never(semantic_ref.knowledge)


class PropertyIndex(IPropertyToSemanticRefIndex):
    def __init__(self):
        self._map: dict[str, list[ScoredSemanticRefOrdinal]] = {}

    async def size(self) -> int:
        return len(self._map)

    async def get_values(self) -> list[str]:
        terms: list[str] = []
        for key in self._map.keys():
            nv = split_property_term_text(key)
            terms.append(nv[1])
        return terms

    async def add_property(
        self,
        property_name: str,
        value: str,
        semantic_ref_ordinal: SemanticRefOrdinal | ScoredSemanticRefOrdinal,
    ) -> None:
        term_text = make_property_term_text(property_name, value)
        if isinstance(semantic_ref_ordinal, int):
            semantic_ref_ordinal = ScoredSemanticRefOrdinal(
                semantic_ref_ordinal,
                1.0,
            )
        term_text = self._prepare_term_text(term_text)
        if term_text in self._map:
            self._map[term_text].append(semantic_ref_ordinal)
        else:
            self._map[term_text] = [semantic_ref_ordinal]

    async def clear(self) -> None:
        self._map = {}

    async def lookup_property(
        self,
        property_name: str,
        value: str,
    ) -> list[ScoredSemanticRefOrdinal] | None:
        term_text = make_property_term_text(property_name, value)
        return self._map.get(self._prepare_term_text(term_text))

    async def remove_property(self, prop_name: str, semref_id: int) -> None:
        """Remove all properties for a specific property name and semantic ref."""
        # Find and remove entries matching both property name and semref_id
        keys_to_remove = []
        for term_text, scored_refs in self._map.items():
            prop_name_from_term, _ = split_property_term_text(term_text)
            # Remove "prop." prefix
            if prop_name_from_term.startswith("prop."):
                prop_name_from_term = prop_name_from_term[5:]

            if prop_name_from_term == prop_name:
                # Filter out entries with matching semref_id
                filtered_refs = [
                    ref for ref in scored_refs if ref.semantic_ref_ordinal != semref_id
                ]
                if filtered_refs:
                    self._map[term_text] = filtered_refs
                else:
                    keys_to_remove.append(term_text)

        # Remove empty entries
        for key in keys_to_remove:
            del self._map[key]

    async def remove_all_for_semref(self, semref_id: int) -> None:
        """Remove all properties for a specific semantic ref."""
        keys_to_remove = []
        for term_text, scored_refs in self._map.items():
            # Filter out entries with matching semref_id
            filtered_refs = [
                ref for ref in scored_refs if ref.semantic_ref_ordinal != semref_id
            ]
            if filtered_refs:
                self._map[term_text] = filtered_refs
            else:
                keys_to_remove.append(term_text)

        # Remove empty entries
        for key in keys_to_remove:
            del self._map[key]

    def _prepare_term_text(self, term_text: str) -> str:
        """Do any pre-processing of the term."""
        return term_text.lower()


async def lookup_property_in_property_index(
    property_index: IPropertyToSemanticRefIndex,
    property_name: str,
    property_value: str,
    semantic_refs: ISemanticRefCollection,
    ranges_in_scope: TextRangesInScope | None = None,
) -> list[ScoredSemanticRefOrdinal] | None:
    scored_refs = await property_index.lookup_property(
        property_name,
        property_value,
    )
    if ranges_in_scope is not None and scored_refs:
        filtered_refs = []
        for sr in scored_refs:
            semantic_ref = await semantic_refs.get_item(sr.semantic_ref_ordinal)
            if ranges_in_scope.is_range_in_scope(semantic_ref.range):
                filtered_refs.append(sr)
        scored_refs = filtered_refs

    return scored_refs or None  # Return None if no results


async def is_known_property(
    property_index: IPropertyToSemanticRefIndex | None,
    property_name: PropertyNames,
    property_value: str,
) -> bool:
    if property_index is not None:
        semantic_refs_with_name = await property_index.lookup_property(
            property_name.value,
            property_value,
        )
        return semantic_refs_with_name is not None and len(semantic_refs_with_name) > 0
    else:
        return False


PROPERTY_DELIMITER = "@@"


def make_property_term_text(name: str, value: str) -> str:
    return f"prop.{name}{PROPERTY_DELIMITER}{value}"


def split_property_term_text(term_text: str) -> tuple[str, str]:
    parts = term_text.split(PROPERTY_DELIMITER, 1)
    return parts[0], parts[1]
