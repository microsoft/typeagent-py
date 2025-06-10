# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import enum

from .collections import TextRangesInScope
from .interfaces import (
    IConversation,
    IPropertyToSemanticRefIndex,
    ISemanticRefCollection,
    ListIndexingResult,
    ScoredSemanticRefOrdinal,
    SemanticRefOrdinal,
    Tag,
)
from . import kplib


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


def add_facet(
    facet: kplib.Facet | None,
    property_index: IPropertyToSemanticRefIndex,
    semantic_ref_ordinal: SemanticRefOrdinal,
) -> None:
    if facet is not None:
        property_index.add_property(
            PropertyNames.FacetName.value,
            facet.name,
            semantic_ref_ordinal,
        )
        if facet.value is not None:
            property_index.add_property(
                PropertyNames.FacetValue.value,
                str(facet.value),
                semantic_ref_ordinal,
            )


def add_entity_properties_to_index(
    entity: kplib.ConcreteEntity,
    property_index: IPropertyToSemanticRefIndex,
    semantic_ref_ordinal: SemanticRefOrdinal,
) -> None:
    property_index.add_property(
        PropertyNames.EntityName.value,
        entity.name,
        semantic_ref_ordinal,
    )
    for type in entity.type:
        property_index.add_property(
            PropertyNames.EntityType.value,
            type,
            semantic_ref_ordinal,
        )
    # Add every facet name as a separate term.
    if entity.facets:
        for facet in entity.facets:
            add_facet(facet, property_index, semantic_ref_ordinal)


def add_action_properties_to_index(
    action: kplib.Action,
    property_index: IPropertyToSemanticRefIndex,
    semantic_ref_ordinal: SemanticRefOrdinal,
) -> None:
    property_index.add_property(
        PropertyNames.Verb.value,
        " ".join(action.verbs),
        semantic_ref_ordinal,
    )
    if action.subject_entity_name != "none":
        property_index.add_property(
            PropertyNames.Subject.value,
            action.subject_entity_name,
            semantic_ref_ordinal,
        )
    if action.object_entity_name != "none":
        property_index.add_property(
            PropertyNames.Object.value,
            action.object_entity_name,
            semantic_ref_ordinal,
        )
    if action.indirect_object_entity_name != "none":
        property_index.add_property(
            PropertyNames.IndirectObject.value,
            action.indirect_object_entity_name,
            semantic_ref_ordinal,
        )


def build_property_index(conversation: IConversation) -> ListIndexingResult:
    return add_to_property_index(
        conversation,
        0,
    )


def add_to_property_index(
    conversation: IConversation,
    start_at_ordinal: SemanticRefOrdinal,
) -> ListIndexingResult:
    """Add semantic references from a conversation to the property index starting at a specific ordinal."""
    if conversation.secondary_indexes and conversation.semantic_refs:
        if conversation.secondary_indexes.property_to_semantic_ref_index is None:
            conversation.secondary_indexes.property_to_semantic_ref_index = (
                PropertyIndex()
            )

        property_index = conversation.secondary_indexes.property_to_semantic_ref_index
        semantic_refs = conversation.semantic_refs

        for semantic_ref_ordinal, semantic_ref in enumerate(
            semantic_refs[start_at_ordinal : len(semantic_refs)],
            start_at_ordinal,
        ):
            assert semantic_ref.semantic_ref_ordinal == semantic_ref_ordinal
            match semantic_ref.knowledge_type:
                case "action":
                    assert isinstance(semantic_ref.knowledge, kplib.Action)
                    add_action_properties_to_index(
                        semantic_ref.knowledge, property_index, semantic_ref_ordinal
                    )
                case "entity":
                    assert isinstance(semantic_ref.knowledge, kplib.ConcreteEntity)
                    add_entity_properties_to_index(
                        semantic_ref.knowledge, property_index, semantic_ref_ordinal
                    )
                case "tag":
                    tag = semantic_ref.knowledge
                    assert isinstance(tag, Tag)
                    property_index.add_property(
                        PropertyNames.Tag.value, tag.text, semantic_ref_ordinal
                    )
                case _:
                    pass

        return ListIndexingResult(
            number_completed=len(semantic_refs) - start_at_ordinal
        )

    return ListIndexingResult(number_completed=0)


class PropertyIndex(IPropertyToSemanticRefIndex):
    def __init__(self):
        self._map: dict[str, list[ScoredSemanticRefOrdinal]] = {}

    def __len__(self) -> int:
        return len(self._map)

    # Needed because otherwise an empty index would be falsy.
    def __bool__(self) -> bool:
        return True

    def get_values(self) -> list[str]:
        terms: list[str] = []
        for key in self._map.keys():
            nv = split_property_term_text(key)
            terms.append(nv[1])
        return terms

    def add_property(
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

    def clear(self) -> None:
        self._map = {}

    def lookup_property(
        self,
        property_name: str,
        value: str,
    ) -> list[ScoredSemanticRefOrdinal] | None:
        term_text = make_property_term_text(property_name, value)
        return self._map.get(self._prepare_term_text(term_text))

    def _prepare_term_text(self, term_text: str) -> str:
        """Do any pre-processing of the term."""
        return term_text.lower()


def lookup_property_in_property_index(
    property_index: IPropertyToSemanticRefIndex,
    property_name: str,
    property_value: str,
    semantic_refs: ISemanticRefCollection,
    ranges_in_scope: TextRangesInScope | None = None,
) -> list[ScoredSemanticRefOrdinal] | None:
    scored_refs = property_index.lookup_property(
        property_name,
        property_value,
    )
    if ranges_in_scope is not None and scored_refs:
        scored_refs = [
            sr
            for sr in scored_refs
            if ranges_in_scope.is_range_in_scope(
                semantic_refs[sr.semantic_ref_ordinal].range,
            )
        ]

    return scored_refs or None  # Return None if no results


def is_known_property(
    property_index: IPropertyToSemanticRefIndex | None,
    property_name: PropertyNames,
    property_value: str,
) -> bool:
    if property_index is not None:
        semantic_refs_with_name = property_index.lookup_property(
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
