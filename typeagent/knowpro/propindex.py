# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import enum
from typing import cast

from . import kplib

from .interfaces import (
    IConversation,
    IPropertyToSemanticRefIndex,
    ListIndexingResult,
    ScoredSemanticRefOrdinal,
    SemanticRef,
    SemanticRefOrdinal,
    Tag,
)


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
                str(facet),
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
    csi = conversation.secondary_indexes
    if csi is not None and conversation.semantic_refs is not None:
        if csi.property_to_semantic_ref_index is None:
            csi.property_to_semantic_ref_index = PropertyIndex()
        return add_to_property_index(
            csi.property_to_semantic_ref_index,
            conversation.semantic_refs,
            0,
        )
    else:
        return ListIndexingResult(0)


def add_to_property_index(
    property_index: IPropertyToSemanticRefIndex,
    semantic_refs: list[SemanticRef],
    base_semantic_ref_ordinal: SemanticRefOrdinal,
) -> ListIndexingResult:
    for i, semantic_ref in enumerate(semantic_refs):
        semantic_ref_ordinal: SemanticRefOrdinal = base_semantic_ref_ordinal + i
        match semantic_ref.knowledge_type:
            case "action":
                add_action_properties_to_index(
                    cast(kplib.Action, semantic_ref.knowledge),
                    property_index,
                    semantic_ref_ordinal,
                )
            case "entity":
                add_entity_properties_to_index(
                    cast(kplib.ConcreteEntity, semantic_ref.knowledge),
                    property_index,
                    semantic_ref_ordinal,
                )
            case "tag":
                tag = cast(Tag, semantic_ref.knowledge)
                property_index.add_property(
                    PropertyNames.Tag.value,
                    tag.text,
                    semantic_ref_ordinal,
                )
            case _:
                pass
    return ListIndexingResult(len(semantic_refs))


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
        result = self._map.get(self._prepare_term_text(term_text))
        if result is None:
            return []
        else:
            return list(result)  # TODO: Do we need to make a copy?

    def _prepare_term_text(self, term_text: str) -> str:
        # Do any pre-processing of the term.
        return term_text.lower()


def lookup_property_in_property_index(
    property_index: IPropertyToSemanticRefIndex,
    property_name: str,
    property_value: str,
    semantic_refs: list[SemanticRef],
    ranges_in_scope: None = None,  # TODO: TextRangesInScope | None
) -> list[ScoredSemanticRefOrdinal] | None:
    # TODO: See lookupPropertyInPropertyIndex.
    raise NotImplementedError


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
