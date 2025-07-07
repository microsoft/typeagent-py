# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections.abc import Sequence
from typing import Callable, Protocol

from ..aitools.vectorbase import ScoredInt, TextEmbeddingIndexSettings, VectorBase

from .collections import TermSet
from .common import is_search_term_wildcard
from .importing import ConversationSettings, RelatedTermIndexSettings
from .interfaces import (
    IConversation,
    ITermToRelatedTerms,
    ITermToRelatedTermsFuzzy,
    SearchTerm,
    TermToRelatedTermsData,
    ITermToRelatedTermsIndex,
    TermsToRelatedTermsDataItem,
    TermsToRelatedTermsIndexData,
    TextEmbeddingIndexData,
    IndexingEventHandlers,
    ListIndexingResult,
    Term,
)
from .query import CompiledTermGroup


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
            await fuzzy_index.add_terms(all_terms)
        return ListIndexingResult(len(all_terms))
    else:
        return ListIndexingResult(0)


class RelatedTermsIndex(ITermToRelatedTermsIndex):
    def __init__(self, settings: RelatedTermIndexSettings):
        self.settings = settings
        self._alias_map = TermToRelatedTermsMap()
        self._term_index = TermEmbeddingIndex(settings.embedding_index_settings)

    @property
    def aliases(self) -> TermToRelatedTermsMap:
        return self._alias_map

    @property
    def fuzzy_index(self) -> ITermToRelatedTermsFuzzy | None:
        return self._term_index

    def serialize(self) -> TermsToRelatedTermsIndexData:
        return TermsToRelatedTermsIndexData(
            aliasData=self._alias_map.serialize(),
            textEmbeddingData=self._term_index.serialize(),
        )

    def deserialize(self, data: TermsToRelatedTermsIndexData) -> None:
        self._alias_map.clear()
        self._term_index.clear()
        self._alias_map.deserialize(data.get("aliasData"))
        text_embedding_data = data.get("textEmbeddingData")
        if text_embedding_data is not None:
            self._term_index.deserialize(text_embedding_data)


async def resolve_related_terms(
    related_terms_index: ITermToRelatedTermsIndex,
    compiled_terms: list[CompiledTermGroup],
    ensure_single_occurrence: bool = True,
    should_resolve_fuzzy: Callable[[SearchTerm], bool] | None = None,
) -> None:
    all_search_terms = [term for ct in compiled_terms for term in ct.terms]
    searchable_terms = TermSet()
    search_terms_needing_related: list[SearchTerm] = []

    for search_term in all_search_terms:
        if is_search_term_wildcard(search_term):
            continue
        searchable_terms.add_or_union(search_term.term)
        term_text = search_term.term.text
        # Resolve any specific term to related term mappings
        if (
            related_terms_index.aliases is not None
            and search_term.related_terms is None
        ):
            search_term.related_terms = related_terms_index.aliases.lookup_term(
                term_text
            )
        # If no mappings to aliases, add to fuzzy retrieval list
        if search_term.related_terms is None:
            if should_resolve_fuzzy is None or should_resolve_fuzzy(search_term):
                search_terms_needing_related.append(search_term)

    if related_terms_index.fuzzy_index is not None and search_terms_needing_related:
        related_terms_for_search_terms = (
            await related_terms_index.fuzzy_index.lookup_terms(
                [st.term.text for st in search_terms_needing_related]
            )
        )
        for i, search_term in enumerate(search_terms_needing_related):
            search_term.related_terms = related_terms_for_search_terms[i]

    # Due to fuzzy matching, a search term may end with related terms that overlap with those of other search terms.
    # This causes scoring problems... duplicate/redundant scoring that can cause items to seem more relevant than they are
    # - The same related term can show up for different search terms but with different weights
    # - related terms may also already be present as search terms
    for ct in compiled_terms:
        dedupe_related_terms(
            ct.terms,
            (
                ct.boolean_op != "and"
                if ensure_single_occurrence
                else ensure_single_occurrence
            ),
        )


def dedupe_related_terms(
    search_terms: list[SearchTerm],
    ensure_single_occurrence: bool,
) -> None:
    all_search_terms = TermSet()
    all_related_terms: TermSet | None = None

    # Collect all unique search and related terms.
    # We end up with (term, maximum weight for term) pairs.
    for st in search_terms:
        all_search_terms.add(st.term)
    if ensure_single_occurrence:
        all_related_terms = TermSet()
        for st in search_terms:
            all_related_terms.add_or_union(st.related_terms)

    for search_term in search_terms:
        if search_term.related_terms is not None and len(search_term.related_terms) > 0:
            unique_related_for_search_term: list[Term] = []
            for candidate_related_term in search_term.related_terms:
                if candidate_related_term in all_search_terms:
                    # This related term is already a search term
                    continue
                if ensure_single_occurrence and all_related_terms is not None:
                    # Each unique related term should be searched for only once,
                    # and (if there were duplicates) assigned the maximum weight assigned to that term
                    term_with_max_weight = all_related_terms.get(candidate_related_term)
                    if (
                        term_with_max_weight is not None
                        and term_with_max_weight.weight == candidate_related_term.weight
                    ):
                        # Associate this related term with the current search term
                        unique_related_for_search_term.append(term_with_max_weight)
                        all_related_terms.remove(candidate_related_term)
                else:
                    unique_related_for_search_term.append(candidate_related_term)
            search_term.related_terms = unique_related_for_search_term


class ITermEmbeddingIndex(ITermToRelatedTermsFuzzy, Protocol):
    def serialize(self) -> TextEmbeddingIndexData:
        raise NotImplementedError

    def deserialize(self, data: TextEmbeddingIndexData) -> None:
        raise NotImplementedError


# TODO: Inherit from TextEmbeddingCache too.
class TermEmbeddingIndex(ITermEmbeddingIndex):
    # The Python version wraps a VectorBase

    settings: TextEmbeddingIndexSettings
    _vectorbase: VectorBase
    _texts: list[str]

    def __init__(
        self,
        settings: TextEmbeddingIndexSettings,
        data: TextEmbeddingIndexData | None = None,
    ):
        self.settings = settings
        self._vectorbase = VectorBase(settings)
        self._texts: list[str] = []
        if data:
            self.deserialize(data)

    def clear(self) -> None:
        self._vectorbase.clear()
        self._texts.clear()

    def serialize(self) -> TextEmbeddingIndexData:
        return TextEmbeddingIndexData(
            textItems=self._texts,
            embeddings=self._vectorbase.serialize(),
        )

    def deserialize(self, data: TextEmbeddingIndexData | None) -> None:
        self.clear()
        if data is not None:
            self._texts = data.get("textItems", [])
            self._vectorbase.deserialize(data.get("embeddings"))

    async def add_terms(
        self, texts: list[str], event_handler: IndexingEventHandlers | None = None
    ) -> ListIndexingResult:
        await self._vectorbase.add_keys(texts)
        self._texts.extend(texts)
        return ListIndexingResult(len(texts))

    async def lookup_term(
        self, text: str, max_hits: int | None = None, min_score: float | None = None
    ) -> list[Term]:
        matches = await self._vectorbase.fuzzy_lookup(
            text, max_hits=max_hits, min_score=min_score
        )
        return self.matches_to_terms(matches)

    async def lookup_terms(
        self,
        texts: list[str],
        max_hits: int | None = None,
        min_score: float | None = None,
    ) -> list[list[Term]]:
        matches = [
            await self._vectorbase.fuzzy_lookup(
                text, max_hits=max_hits, min_score=min_score
            )
            for text in texts
        ]
        return [self.matches_to_terms(m) for m in matches]

    def matches_to_terms(self, matches: list[ScoredInt]) -> list[Term]:
        return [
            Term(text=self._texts[match.item], weight=match.score) for match in matches
        ]
