# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import NotRequired, TypedDict

from ..aitools.vectorbase import NormalizedEmbeddings, VectorBase
from .importing import ConversationSettings, RelatedTermIndexSettings
from .interfaces import (
    IConversation,
    ITermToRelatedTerms,
    TermToRelatedTermsData,
    ITermToRelatedTermsIndex,
    TermsToRelatedTermsDataItem,
    TermsToRelatedTermsIndexData,
    TextEmbeddingIndexData,
    IndexingEventHandlers,
    ListIndexingResult,
    Term,
    TermData,
)


class TermToRelatedTermsMap(ITermToRelatedTerms):
    def __init__(self):
        self.map: dict[str, set[Term]] = {}

    def add_related_term(self, text: str, related_terms: Term | list[Term]) -> None:
        if not isinstance(related_terms, list):
            related_terms = [related_terms]
        for related in related_terms:
            terms = self.map.setdefault(text, set())
            terms.add(related)

    def lookup_term(self, text: str) -> list[Term] | None:
        result = self.map.get(text)
        if result:
            return list(result)
        else:
            return None

    def remove_term(self, text: str) -> None:
        self.map.pop(text, None)

    def clear(self) -> None:
        self.map.clear()

    def serialize(self) -> TermToRelatedTermsData:
        related_terms: list[TermsToRelatedTermsDataItem] = []
        for key, value in self.map.items():
            related_terms.append(
                TermsToRelatedTermsDataItem(
                    termText=key,
                    relatedTerms=[term.serialize() for term in value],
                )
            )
        return TermToRelatedTermsData(relatedTerms=related_terms)

    def deserialize(self, data: TermToRelatedTermsData | None) -> None:
        self.clear()
        if data is None:
            return
        related_terms_data = data.get("relatedTerms")
        if related_terms_data is None:
            return
        for item in related_terms_data:
            term_text = item["termText"]
            related_terms_data = item["relatedTerms"]
            related_terms: list[Term] = [
                Term(term_data["text"], weight=term_data.get("weight"))
                for term_data in related_terms_data
            ]
            self.add_related_term(term_text, related_terms)


async def build_related_terms_index(
    conversation: IConversation,
    settings: ConversationSettings,
    event_handler: IndexingEventHandlers | None = None,
) -> ListIndexingResult:
    csr = conversation.semantic_ref_index
    csi = conversation.secondary_indexes
    if csr and csi:
        if csi.term_to_related_terms_index is None:
            csi.term_to_related_terms_index = RelatedTermsIndex(
                settings.related_term_index_settings
            )
        fuzzy_index = csi.term_to_related_terms_index.fuzzy_index
        all_terms = csr.get_terms()
        if fuzzy_index and all_terms:
            await fuzzy_index.add_keys(all_terms)
        return ListIndexingResult(len(all_terms))
    else:
        return ListIndexingResult(0)


class RelatedTermsIndex(ITermToRelatedTermsIndex):
    def __init__(self, settings: RelatedTermIndexSettings):
        self.settings = settings
        self._alias_map = TermToRelatedTermsMap()
        self._vector_base = VectorBase(settings.embedding_index_settings)

    @property
    def aliases(self) -> TermToRelatedTermsMap:
        return self._alias_map

    @property
    def fuzzy_index(self) -> VectorBase:
        return self._vector_base

    def serialize(self) -> TermsToRelatedTermsIndexData:
        return TermsToRelatedTermsIndexData(
            aliasData=self._alias_map.serialize(),
            textEmbeddingData=TextEmbeddingIndexData(
                textItems=[],  # TODO: Put values here!
                embeddings=self._vector_base.serialize(),
            ),
        )

    def deserialize(self, data: TermsToRelatedTermsIndexData) -> None:
        self._alias_map.clear()
        self._vector_base.clear()
        self._alias_map.deserialize(data.get("aliasData"))
        text_embedding_data = data.get("textEmbeddingData")
        if text_embedding_data is not None:
            embeddings = text_embedding_data.get("embeddings")
            if embeddings is not None:
                self._vector_base.deserialize(embeddings)
