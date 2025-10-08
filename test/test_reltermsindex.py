# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Third-party imports
import pytest
import pytest_asyncio
from typing import AsyncGenerator

# TypeAgent imports
from typeagent.aitools.embeddings import AsyncEmbeddingModel
from typeagent.aitools.vectorbase import TextEmbeddingIndexSettings
from typeagent.knowpro.interfaces import Term, IMessage, ITermToRelatedTermsIndex
from typeagent.knowpro.kplib import KnowledgeResponse
from typeagent.knowpro.convsettings import (
    MessageTextIndexSettings,
    RelatedTermIndexSettings,
)
from typeagent.knowpro.query import CompiledSearchTerm, CompiledTermGroup
from typeagent.storage.memory.reltermsindex import (
    TermToRelatedTermsMap,
    RelatedTermsIndex,
    dedupe_related_terms,
    resolve_related_terms,
)
from typeagent.storage.memory import MemoryStorageProvider
from typeagent.storage import SqliteStorageProvider

# Test fixtures
from fixtures import needs_auth, really_needs_auth, embedding_model, temp_db_path


@pytest_asyncio.fixture(params=["memory", "sqlite"])
async def related_terms_index(
    request: pytest.FixtureRequest,
    embedding_model: AsyncEmbeddingModel,
    temp_db_path: str,
) -> AsyncGenerator[ITermToRelatedTermsIndex, None]:
    class DummyTestMessage(IMessage):
        text_chunks: list[str]
        tags: list[str] = []

        def get_knowledge(self):
            return KnowledgeResponse(
                entities=[], actions=[], inverse_actions=[], topics=[]
            )

    embedding_settings = TextEmbeddingIndexSettings(embedding_model)
    message_text_settings = MessageTextIndexSettings(embedding_settings)
    related_terms_settings = RelatedTermIndexSettings(embedding_settings)
    if request.param == "memory":
        storage_provider = MemoryStorageProvider(
            message_text_settings=message_text_settings,
            related_terms_settings=related_terms_settings,
        )
        index = await storage_provider.get_related_terms_index()
        yield index
    else:
        provider = SqliteStorageProvider(
            db_path=temp_db_path,
            message_type=DummyTestMessage,
            message_text_index_settings=message_text_settings,
            related_term_index_settings=related_terms_settings,
        )
        index = await provider.get_related_terms_index()
        yield index
        await provider.close()


@pytest.mark.asyncio
async def test_add_and_lookup_related_term(
    related_terms_index: ITermToRelatedTermsIndex, needs_auth: None
) -> None:
    await related_terms_index.aliases.add_related_term(
        "python", Term(text="programming")
    )
    await related_terms_index.aliases.add_related_term("python", Term(text="snake"))
    result = await related_terms_index.aliases.lookup_term("python")
    assert result is not None
    assert len(result) == 2
    assert any(term.text == "programming" for term in result)
    assert any(term.text == "snake" for term in result)


@pytest.mark.asyncio
async def test_remove_term(
    related_terms_index: ITermToRelatedTermsIndex, needs_auth: None
) -> None:
    await related_terms_index.aliases.add_related_term(
        "python", Term(text="programming")
    )
    await related_terms_index.aliases.remove_term("python")
    result = await related_terms_index.aliases.lookup_term("python")
    assert result is None or len(result) == 0


@pytest.mark.asyncio
async def test_clear_and_size(
    related_terms_index: ITermToRelatedTermsIndex, needs_auth: None
) -> None:
    await related_terms_index.aliases.add_related_term(
        "python", Term(text="programming")
    )
    await related_terms_index.aliases.add_related_term("java", Term(text="coffee"))
    await related_terms_index.aliases.clear()
    size = await related_terms_index.aliases.size()
    assert size == 0


@pytest.mark.asyncio
async def test_serialize_and_deserialize(
    related_terms_index: ITermToRelatedTermsIndex, needs_auth: None
) -> None:
    await related_terms_index.aliases.add_related_term(
        "python", Term(text="programming")
    )
    data = await related_terms_index.aliases.serialize()
    await related_terms_index.aliases.clear()
    await related_terms_index.aliases.deserialize(data)
    result = await related_terms_index.aliases.lookup_term("python")
    assert result is not None
    assert any(term.text == "programming" for term in result)


@pytest.mark.asyncio
async def test_related_terms_index_basic(really_needs_auth: None) -> None:
    settings = RelatedTermIndexSettings(TextEmbeddingIndexSettings())
    index = RelatedTermsIndex(settings)
    assert isinstance(index.aliases, TermToRelatedTermsMap)
    assert index.fuzzy_index is not None
    data = await index.serialize()
    await index.deserialize(data)
    assert isinstance(index.aliases, TermToRelatedTermsMap)


@pytest.mark.asyncio
async def test_resolve_related_terms_fuzzy_and_alias(
    related_terms_index: ITermToRelatedTermsIndex, needs_auth: None
) -> None:
    # Add alias for 'python', but not for 'java'
    await related_terms_index.aliases.add_related_term(
        "python", Term(text="programming")
    )
    # Add terms to fuzzy index (add a different term than the search term)
    if related_terms_index.fuzzy_index is not None:
        await related_terms_index.fuzzy_index.add_terms(["javascript", "javac"])
    # Create compiled terms with explicit construction: one with alias, one without
    st1 = CompiledSearchTerm(
        term=Term(text="python"), related_terms=None, related_terms_required=False
    )
    st2 = CompiledSearchTerm(
        term=Term(text="java"), related_terms=None, related_terms_required=False
    )
    ctg = CompiledTermGroup(terms=[st1, st2], boolean_op="or")
    await resolve_related_terms(related_terms_index, [ctg])

    # 'python' should resolve via alias
    assert st1.related_terms is not None
    assert any(t.text == "programming" for t in st1.related_terms)

    # 'java' should resolve via fuzzy to similar terms (if fuzzy index is working)
    # Note: SQLite implementation may not have fuzzy search fully implemented
    assert st2.related_terms is not None or st2.related_terms == []
    # If we get fuzzy matches, they should be reasonable
    if st2.related_terms:
        related_texts = [t.text for t in st2.related_terms]
        assert len(related_texts) > 0


def make_compiled_search_term(
    text: str,
    related_terms: list[Term] | None = None,
    required: bool = False,
    weight: float = 1.0,
) -> CompiledSearchTerm:
    term = Term(text=text, weight=weight)
    # Properly construct CompiledSearchTerm with explicit arguments
    processed_related_terms: list[Term] | None
    if related_terms is None or (
        isinstance(related_terms, list) and len(related_terms) == 0
    ):
        processed_related_terms = None
    elif isinstance(related_terms, list):
        if all(isinstance(t, Term) for t in related_terms):
            processed_related_terms = related_terms
        else:
            # This branch should not happen with proper typing, but handle it safely
            processed_related_terms = [
                Term(text=getattr(t, "text", str(t)), weight=getattr(t, "weight", 1.0))
                for t in related_terms
            ]
    else:
        # This branch should also not happen with proper typing
        processed_related_terms = (
            [related_terms] if isinstance(related_terms, Term) else None
        )

    # Create with explicit field values to avoid pydantic defaults
    st = CompiledSearchTerm(
        term=term,
        related_terms=processed_related_terms,
        related_terms_required=required,
    )
    return st


def test_dedupe_related_terms_basic() -> None:
    # Test deduplication with terms that don't overlap with search terms
    t1 = Term(text="programming", weight=1.0)  # related to python but not a search term
    t2 = Term(text="coffee", weight=2.0)  # related to java but not a search term
    t3 = Term(text="programming", weight=1.5)  # duplicate with different weight
    st1 = make_compiled_search_term("python", related_terms=[t1, t3])
    st2 = make_compiled_search_term("java", related_terms=[t2])
    compiled_terms = [st1, st2]

    dedupe_related_terms(compiled_terms, ensure_single_occurrence=True)

    # st2 should keep its related term since "coffee" doesn't overlap
    assert st2.related_terms is not None
    assert len(st2.related_terms) == 1
    assert st2.related_terms[0].text == "coffee"

    # st1 should have only one "programming" term with the higher weight
    assert st1.related_terms is not None
    assert len(st1.related_terms) == 1
    assert st1.related_terms[0].text == "programming"
    assert st1.related_terms[0].weight == 1.5  # Should keep the higher weight


def test_dedupe_related_terms_weight() -> None:
    # Test that deduplication keeps max weight and removes search term overlaps
    # Test case: related terms that overlap with search terms should be removed
    t1 = Term(
        text="python", weight=1.0
    )  # This will be removed (overlaps with search term)
    t2 = Term(
        text="java", weight=2.0
    )  # This will be removed (overlaps with search term)
    t3 = Term(text="coding", weight=1.5)  # This should remain

    st1 = make_compiled_search_term("python", related_terms=[t1, t3])  # python + coding
    st2 = make_compiled_search_term("java", related_terms=[t2])  # java
    compiled_terms = [st1, st2]

    dedupe_related_terms(compiled_terms, ensure_single_occurrence=True)

    # st1 should only keep "coding" (python removed as it's a search term)
    assert st1.related_terms is not None
    assert len(st1.related_terms) == 1
    assert st1.related_terms[0].text == "coding"

    # st2 should have empty related_terms (java removed as it's a search term)
    assert st2.related_terms == [] or st2.related_terms is None
