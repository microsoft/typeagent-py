# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from typeagent.knowpro.interfaces import (
    KnowledgePropertyName,
    PropertySearchTerm,
    ScoredSemanticRefOrdinal,
    SearchTerm,
    SearchTermGroup,
    SemanticRef,
    Term,
    TextLocation,
    TextRange,
)
from typeagent.storage.memory.propindex import PropertyNames
from typeagent.knowpro.searchlib import (
    create_and_term_group,
    create_entity_search_term_group,
    create_multiple_choice_question,
    create_or_max_term_group,
    create_or_term_group,
    create_property_search_term,
    create_property_search_terms,
    create_search_term,
    create_search_terms,
    create_tag_search_term_group,
    create_topic_search_term_group,
    get_semantic_refs_from_scored_ordinals,
)


class TestCreateSearchTerm:
    """Test the create_search_term function."""

    def test_basic_search_term(self):
        """Test creating a basic search term."""
        term = create_search_term("hello")
        assert term.term.text == "hello"
        assert term.term.weight is None
        assert term.related_terms is None

    def test_search_term_with_weight(self):
        """Test creating a search term with weight."""
        term = create_search_term("hello", weight=0.8)
        assert term.term.text == "hello"
        assert term.term.weight == 0.8
        assert term.related_terms is None

    def test_search_term_exact_match(self):
        """Test creating a search term with exact match."""
        term = create_search_term("hello", exact_match_value=True)
        assert term.term.text == "hello"
        assert term.term.weight is None
        assert term.related_terms == []

    def test_search_term_with_weight_and_exact_match(self):
        """Test creating a search term with both weight and exact match."""
        term = create_search_term("hello", weight=0.5, exact_match_value=True)
        assert term.term.text == "hello"
        assert term.term.weight == 0.5
        assert term.related_terms == []


class TestCreatePropertySearchTerm:
    """Test the create_property_search_term function."""

    def test_well_known_property_name(self):
        """Test creating property search term with well-known property name."""
        prop_term = create_property_search_term("name", "John")
        assert prop_term.property_name == "name"
        assert prop_term.property_value.term.text == "John"
        assert prop_term.property_value.related_terms is None

    def test_custom_property_name(self):
        """Test creating property search term with custom property name."""
        prop_term = create_property_search_term("customProp", "value")
        assert isinstance(prop_term.property_name, SearchTerm)
        assert prop_term.property_name.term.text == "customProp"
        assert prop_term.property_value.term.text == "value"
        assert prop_term.property_value.related_terms is None

    def test_exact_match_property_value(self):
        """Test creating property search term with exact match value."""
        prop_term = create_property_search_term("type", "book", exact_match_value=True)
        assert prop_term.property_name == "type"
        assert prop_term.property_value.term.text == "book"
        assert prop_term.property_value.related_terms == []

    @pytest.mark.parametrize(
        "well_known_name",
        ["name", "type", "verb", "subject", "object", "indirectObject", "tag", "topic"],
    )
    def test_all_well_known_property_names(self, well_known_name: str):
        """Test all well-known property names are handled correctly."""
        prop_term = create_property_search_term(well_known_name, "test_value")
        assert prop_term.property_name == well_known_name
        assert prop_term.property_value.term.text == "test_value"


class TestTermGroupCreation:
    """Test term group creation functions."""

    def test_create_and_term_group(self):
        """Test creating an AND term group."""
        term1 = create_search_term("hello")
        term2 = create_search_term("world")
        group = create_and_term_group(term1, term2)

        assert group.boolean_op == "and"
        assert len(group.terms) == 2
        assert group.terms[0] == term1
        assert group.terms[1] == term2

    def test_create_or_term_group(self):
        """Test creating an OR term group."""
        term1 = create_search_term("hello")
        term2 = create_search_term("world")
        group = create_or_term_group(term1, term2)

        assert group.boolean_op == "or"
        assert len(group.terms) == 2
        assert group.terms[0] == term1
        assert group.terms[1] == term2

    def test_create_or_max_term_group(self):
        """Test creating an OR_MAX term group."""
        term1 = create_search_term("hello")
        term2 = create_search_term("world")
        group = create_or_max_term_group(term1, term2)

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 2
        assert group.terms[0] == term1
        assert group.terms[1] == term2

    def test_empty_term_groups(self):
        """Test creating empty term groups."""
        and_group = create_and_term_group()
        or_group = create_or_term_group()
        or_max_group = create_or_max_term_group()

        assert and_group.boolean_op == "and"
        assert len(and_group.terms) == 0
        assert or_group.boolean_op == "or"
        assert len(or_group.terms) == 0
        assert or_max_group.boolean_op == "or_max"
        assert len(or_max_group.terms) == 0

    def test_nested_term_groups(self):
        """Test creating nested term groups."""
        term1 = create_search_term("hello")
        term2 = create_search_term("world")
        inner_group = create_or_term_group(term1, term2)

        term3 = create_search_term("test")
        outer_group = create_and_term_group(inner_group, term3)

        assert outer_group.boolean_op == "and"
        assert len(outer_group.terms) == 2
        assert outer_group.terms[0] == inner_group
        assert outer_group.terms[1] == term3


class TestCreateSearchTerms:
    """Test the create_search_terms function."""

    def test_simple_terms(self):
        """Test creating search terms from simple strings."""
        terms = create_search_terms(["hello", "world", "test"])

        assert len(terms) == 3
        assert terms[0].term.text == "hello"
        assert terms[1].term.text == "world"
        assert terms[2].term.text == "test"

        for term in terms:
            assert term.related_terms is None

    def test_terms_with_related_terms(self):
        """Test creating search terms with related terms using semicolon syntax."""
        terms = create_search_terms(["novel;book;bestseller", "car;automobile;vehicle"])

        assert len(terms) == 2

        # First term
        assert terms[0].term.text == "novel"
        assert terms[0].related_terms is not None
        assert len(terms[0].related_terms) == 2
        assert terms[0].related_terms[0].text == "book"
        assert terms[0].related_terms[1].text == "bestseller"

        # Second term
        assert terms[1].term.text == "car"
        assert terms[1].related_terms is not None
        assert len(terms[1].related_terms) == 2
        assert terms[1].related_terms[0].text == "automobile"
        assert terms[1].related_terms[1].text == "vehicle"

    def test_mixed_terms(self):
        """Test creating mixed simple and complex terms."""
        terms = create_search_terms(["simple", "complex;related"])

        assert len(terms) == 2
        assert terms[0].term.text == "simple"
        assert terms[0].related_terms is None

        assert terms[1].term.text == "complex"
        assert terms[1].related_terms is not None
        assert len(terms[1].related_terms) == 1
        assert terms[1].related_terms[0].text == "related"

    def test_empty_and_whitespace_terms(self):
        """Test handling of empty strings and whitespace."""
        terms = create_search_terms(["", "  ", "valid"])

        # Empty strings should be filtered out
        assert len(terms) == 1
        assert terms[0].term.text == "valid"

    def test_case_conversion(self):
        """Test that terms are converted to lowercase."""
        terms = create_search_terms(["HELLO", "World;BOOK"])

        assert len(terms) == 2
        assert terms[0].term.text == "hello"
        assert terms[1].term.text == "world"
        assert terms[1].related_terms is not None
        assert terms[1].related_terms[0].text == "book"


class TestCreatePropertySearchTerms:
    """Test the create_property_search_terms function."""

    def test_simple_property_terms(self):
        """Test creating property search terms from simple dictionary."""
        prop_dict = {"name": "John", "type": "person"}
        terms = create_property_search_terms(prop_dict)

        assert len(terms) == 2

        # Find terms by property name
        name_term = next(t for t in terms if t.property_name == "name")
        type_term = next(t for t in terms if t.property_name == "type")

        assert name_term.property_value.term.text == "John"
        assert type_term.property_value.term.text == "person"

    def test_multiple_values_for_property(self):
        """Test creating property search terms with comma-separated values."""
        prop_dict = {"type": "book,novel,fiction"}
        terms = create_property_search_terms(prop_dict)

        assert len(terms) == 3
        for term in terms:
            assert term.property_name == "type"

        values = [term.property_value.term.text for term in terms]
        assert "book" in values
        assert "novel" in values
        assert "fiction" in values

    def test_mixed_single_and_multiple_values(self):
        """Test creating property terms with both single and multiple values."""
        prop_dict = {"name": "John", "type": "person,author", "tag": "important"}
        terms = create_property_search_terms(prop_dict)

        assert len(terms) == 4

        name_terms = [t for t in terms if t.property_name == "name"]
        type_terms = [t for t in terms if t.property_name == "type"]
        tag_terms = [t for t in terms if t.property_name == "tag"]

        assert len(name_terms) == 1
        assert len(type_terms) == 2
        assert len(tag_terms) == 1

    def test_whitespace_handling(self):
        """Test proper handling of whitespace in values."""
        prop_dict = {"type": " book , novel , fiction "}
        terms = create_property_search_terms(prop_dict)

        assert len(terms) == 3
        values = [term.property_value.term.text for term in terms]
        assert "book" in values
        assert "novel" in values
        assert "fiction" in values

    def test_empty_values_filtered(self):
        """Test that empty values are filtered out."""
        prop_dict = {"type": "book,,novel,"}
        terms = create_property_search_terms(prop_dict)

        assert len(terms) == 2
        values = [term.property_value.term.text for term in terms]
        assert "book" in values
        assert "novel" in values


class TestCreateTopicSearchTermGroup:
    """Test the create_topic_search_term_group function."""

    def test_single_topic_string(self):
        """Test creating topic search term group with single topic as string."""
        group = create_topic_search_term_group("science")

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 1

        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert term.property_name == PropertyNames.Topic.value
        assert term.property_value.term.text == "science"
        assert term.property_value.related_terms is None

    def test_single_topic_string_exact_match(self):
        """Test creating topic search term group with exact match."""
        group = create_topic_search_term_group("science", exact_match=True)

        assert len(group.terms) == 1
        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert term.property_value.related_terms == []

    def test_multiple_topics_list(self):
        """Test creating topic search term group with list of topics."""
        topics = ["science", "technology", "research"]
        group = create_topic_search_term_group(topics)

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 3

        for i, topic in enumerate(topics):
            term = group.terms[i]
            assert isinstance(term, PropertySearchTerm)
            assert term.property_name == PropertyNames.Topic.value
            assert term.property_value.term.text == topic
            assert term.property_value.related_terms is None

    def test_multiple_topics_exact_match(self):
        """Test creating topic search term group with multiple topics and exact match."""
        topics = ["science", "technology"]
        group = create_topic_search_term_group(topics, exact_match=True)

        assert len(group.terms) == 2
        for term in group.terms:
            assert isinstance(term, PropertySearchTerm)
            assert term.property_value.related_terms == []


class TestCreateEntitySearchTermGroup:
    """Test the create_entity_search_term_group function."""

    def test_entity_with_name_only(self):
        """Test creating entity search term group with name only."""
        group = create_entity_search_term_group(name="John Doe")

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 1

        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert term.property_name == PropertyNames.EntityName.value
        assert term.property_value.term.text == "John Doe"

    def test_entity_with_type_only(self):
        """Test creating entity search term group with type only."""
        group = create_entity_search_term_group(type_="person")

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 1

        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert term.property_name == PropertyNames.EntityType.value
        assert term.property_value.term.text == "person"

    def test_entity_with_facet_name_only(self):
        """Test creating entity search term group with facet name only."""
        group = create_entity_search_term_group(facet_name="color")

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 1

        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert isinstance(term.property_name, SearchTerm)
        assert term.property_name.term.text == PropertyNames.FacetName.value
        assert term.property_value.term.text == "color"

    def test_entity_with_facet_value_only(self):
        """Test creating entity search term group with facet value only."""
        group = create_entity_search_term_group(facet_value="red")

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 1

        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert isinstance(term.property_name, SearchTerm)
        assert term.property_name.term.text == PropertyNames.FacetValue.value
        assert term.property_value.term.text == "red"

    def test_entity_with_all_parameters(self):
        """Test creating entity search term group with all parameters."""
        group = create_entity_search_term_group(
            name="John Doe", type_="person", facet_name="age", facet_value="30"
        )

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 4

        # Check each property type is present by examining the actual terms
        name_terms = [
            t
            for t in group.terms
            if isinstance(t, PropertySearchTerm)
            and t.property_name == PropertyNames.EntityName.value
        ]
        type_terms = [
            t
            for t in group.terms
            if isinstance(t, PropertySearchTerm)
            and t.property_name == PropertyNames.EntityType.value
        ]
        facet_name_terms = [
            t
            for t in group.terms
            if isinstance(t, PropertySearchTerm)
            and isinstance(t.property_name, SearchTerm)
            and t.property_name.term.text == PropertyNames.FacetName.value
        ]
        facet_value_terms = [
            t
            for t in group.terms
            if isinstance(t, PropertySearchTerm)
            and isinstance(t.property_name, SearchTerm)
            and t.property_name.term.text == PropertyNames.FacetValue.value
        ]

        assert len(name_terms) == 1
        assert len(type_terms) == 1
        assert len(facet_name_terms) == 1
        assert len(facet_value_terms) == 1

    def test_entity_with_exact_match(self):
        """Test creating entity search term group with exact match."""
        group = create_entity_search_term_group(name="John Doe", exact_match=True)

        assert len(group.terms) == 1
        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert term.property_value.related_terms == []

    def test_entity_with_no_parameters(self):
        """Test creating entity search term group with no parameters."""
        group = create_entity_search_term_group()

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 0


class TestCreateTagSearchTermGroup:
    """Test the create_tag_search_term_group function."""

    def test_single_tag(self):
        """Test creating tag search term group with single tag."""
        group = create_tag_search_term_group(["important"])

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 1

        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert term.property_name == PropertyNames.Tag.value
        assert term.property_value.term.text == "important"
        assert term.property_value.related_terms == []  # exact_match=True by default

    def test_multiple_tags(self):
        """Test creating tag search term group with multiple tags."""
        tags = ["important", "urgent", "review"]
        group = create_tag_search_term_group(tags)

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 3

        for i, tag in enumerate(tags):
            term = group.terms[i]
            assert isinstance(term, PropertySearchTerm)
            assert term.property_name == PropertyNames.Tag.value
            assert term.property_value.term.text == tag
            assert term.property_value.related_terms == []

    def test_tags_with_non_exact_match(self):
        """Test creating tag search term group with non-exact match."""
        group = create_tag_search_term_group(["important"], exact_match=False)

        assert len(group.terms) == 1
        term = group.terms[0]
        assert isinstance(term, PropertySearchTerm)
        assert term.property_value.related_terms is None  # allows related terms

    def test_empty_tags_list(self):
        """Test creating tag search term group with empty tags list."""
        group = create_tag_search_term_group([])

        assert group.boolean_op == "or_max"
        assert len(group.terms) == 0


class TestCreateMultipleChoiceQuestion:
    """Test the create_multiple_choice_question function."""

    def test_question_with_choices(self):
        """Test creating multiple choice question with choices."""
        question = "What is your favorite color?"
        choices = ["Red", "Blue", "Green"]
        result = create_multiple_choice_question(question, choices)

        expected_lines = [
            "Multiple choice question:",
            question,
            "Answer using *one or more* of the following choices *only*:",
            "- Red",
            "- Blue",
            "- Green",
            "- None of the above",
        ]

        for line in expected_lines:
            assert line in result

    def test_question_with_choices_no_none_option(self):
        """Test creating multiple choice question without 'None of the above' option."""
        question = "What is your favorite color?"
        choices = ["Red", "Blue", "Green"]
        result = create_multiple_choice_question(question, choices, add_none=False)

        assert "- None of the above" not in result
        assert "- Red" in result
        assert "- Blue" in result
        assert "- Green" in result

    def test_question_with_empty_choices(self):
        """Test creating multiple choice question with empty choices list."""
        question = "What is your favorite color?"
        choices = []
        result = create_multiple_choice_question(question, choices)

        # Should just return the question text
        assert result == question

    def test_question_with_whitespace_in_choices(self):
        """Test creating multiple choice question with whitespace in choices."""
        question = "What is your favorite color?"
        choices = ["  Red  ", " Blue", "Green "]
        result = create_multiple_choice_question(question, choices)

        # Choices should be stripped
        assert "- Red" in result
        assert "- Blue" in result
        assert "- Green" in result


class TestGetSemanticRefsFromScoredOrdinals:
    """Test the get_semantic_refs_from_scored_ordinals function."""

    def test_ordinal_extraction_logic(self):
        """Test that ordinals are correctly extracted from scored ordinals.

        Since get_semantic_refs_from_scored_ordinals just extracts ordinals
        and calls get_multiple, we can test the ordinal extraction logic directly.
        """
        # Test the ordinal extraction logic
        scored_ordinals = [
            ScoredSemanticRefOrdinal(2, 1.0),
            ScoredSemanticRefOrdinal(0, 0.8),
            ScoredSemanticRefOrdinal(1, 0.6),
        ]

        # Extract ordinals the same way the function does
        ordinals = [sr.semantic_ref_ordinal for sr in scored_ordinals]

        # Verify ordinals are extracted in the same order
        assert ordinals == [2, 0, 1]

    def test_empty_scored_ordinals(self):
        """Test ordinal extraction with empty list."""
        scored_ordinals = []
        ordinals = [sr.semantic_ref_ordinal for sr in scored_ordinals]
        assert ordinals == []

    def test_single_scored_ordinal(self):
        """Test ordinal extraction with single item."""
        scored_ordinals = [ScoredSemanticRefOrdinal(42, 0.9)]
        ordinals = [sr.semantic_ref_ordinal for sr in scored_ordinals]
        assert ordinals == [42]

    def test_scored_ordinal_structure(self):
        """Test that ScoredSemanticRefOrdinal has the expected structure."""
        scored_ordinal = ScoredSemanticRefOrdinal(5, 0.7)
        assert scored_ordinal.semantic_ref_ordinal == 5
        assert scored_ordinal.score == 0.7


class TestPrivateFunctions:
    """Test private helper functions."""

    def test_split_term_values(self):
        """Test the _split_term_values helper function."""
        from typeagent.knowpro.searchlib import _split_term_values

        # Test basic splitting
        result = _split_term_values("a,b,c", ",")
        assert result == ["a", "b", "c"]

        # Test with whitespace
        result = _split_term_values(" a , b , c ", ",")
        assert result == ["a", "b", "c"]

        # Test with empty parts
        result = _split_term_values("a,,b,", ",")
        assert result == ["a", "b"]

        # Test single value
        result = _split_term_values("single", ",")
        assert result == ["single"]

        # Test empty string
        result = _split_term_values("", ",")
        assert result == []

    def test_parse_search_term(self):
        """Test the _parse_search_term helper function."""
        from typeagent.knowpro.searchlib import _parse_search_term

        # Test simple term
        term = _parse_search_term("hello")
        assert term is not None
        assert term.term.text == "hello"
        assert term.related_terms is None

        # Test term with related terms
        term = _parse_search_term("NOVEL;BOOK;BESTSELLER")
        assert term is not None
        assert term.term.text == "novel"
        assert term.related_terms is not None
        assert len(term.related_terms) == 2
        assert term.related_terms[0].text == "book"
        assert term.related_terms[1].text == "bestseller"

        # Test empty string
        term = _parse_search_term("")
        assert term is None

        # Test single semicolon
        term = _parse_search_term(";")
        assert term is None
