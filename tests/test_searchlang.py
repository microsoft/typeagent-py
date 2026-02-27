# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typeagent.knowpro.search import SearchOptions
from typeagent.knowpro.searchlang import (
    LanguageQueryCompileOptions,
    LanguageSearchOptions,
)


class TestSearchOptionsRepr:
    """Tests for the custom __repr__ on SearchOptions and LanguageSearchOptions."""

    def test_all_defaults_shows_non_none_fields(self) -> None:
        """Default fields that are not None (like exact_match=False) appear."""
        opts = SearchOptions()
        r = repr(opts)
        assert r.startswith("SearchOptions(")
        # exact_match defaults to False, which is not None, so it shows up:
        assert "exact_match=False" in r
        # None-valued fields are omitted:
        assert "max_knowledge_matches" not in r

    def test_non_none_fields_shown(self) -> None:
        opts = SearchOptions(max_knowledge_matches=10, threshold_score=0.5)
        r = repr(opts)
        assert "max_knowledge_matches=10" in r
        assert "threshold_score=0.5" in r
        # Fields left at None are omitted:
        assert "max_message_matches" not in r
        assert "max_chars_in_budget" not in r

    def test_false_field_shown(self) -> None:
        """False is not None, so it should appear."""
        opts = SearchOptions(exact_match=False)
        assert "exact_match=False" in repr(opts)

    def test_true_field_shown(self) -> None:
        opts = SearchOptions(exact_match=True)
        assert "exact_match=True" in repr(opts)

    def test_all_fields_set(self) -> None:
        """When every field is non-None, all appear in repr."""
        opts = SearchOptions(
            max_knowledge_matches=10,
            exact_match=True,
            max_message_matches=20,
            max_chars_in_budget=5000,
            threshold_score=0.75,
        )
        r = repr(opts)
        assert "max_knowledge_matches=10" in r
        assert "exact_match=True" in r
        assert "max_message_matches=20" in r
        assert "max_chars_in_budget=5000" in r
        assert "threshold_score=0.75" in r

    def test_zero_values_shown(self) -> None:
        """Zero is not None, so numeric zeros should appear."""
        opts = SearchOptions(max_knowledge_matches=0, threshold_score=0.0)
        r = repr(opts)
        assert "max_knowledge_matches=0" in r
        assert "threshold_score=0.0" in r

    def test_no_dunder_or_method_names(self) -> None:
        """The repr must not contain dunder names or method objects."""
        opts = SearchOptions(max_knowledge_matches=5)
        r = repr(opts)
        assert "__init__" not in r
        assert "__eq__" not in r
        assert "bound method" not in r


class TestLanguageSearchOptionsRepr:
    """Tests for LanguageSearchOptions.__repr__ (subclass of SearchOptions)."""

    def test_all_defaults_shows_class_name(self) -> None:
        opts = LanguageSearchOptions()
        r = repr(opts)
        # Subclass name, not parent name:
        assert r.startswith("LanguageSearchOptions(")

    def test_inherited_and_own_fields(self) -> None:
        opts = LanguageSearchOptions(
            max_knowledge_matches=5,
            compile_options=LanguageQueryCompileOptions(exact_scope=True),
        )
        r = repr(opts)
        assert "LanguageSearchOptions(" in r
        assert "max_knowledge_matches=5" in r
        assert "compile_options=" in r
        assert "exact_scope=True" in r

    def test_none_fields_omitted(self) -> None:
        opts = LanguageSearchOptions()
        r = repr(opts)
        assert "compile_options" not in r
        assert "model_instructions" not in r
        assert "max_knowledge_matches" not in r

    def test_no_private_fields(self) -> None:
        """Fields starting with _ should never appear in repr."""
        opts = LanguageSearchOptions(max_knowledge_matches=3)
        r = repr(opts)
        # No key=value pair where the key starts with underscore:
        inside = r.split("(", 1)[1].rstrip(")")
        for part in inside.split(", "):
            if "=" in part:
                key = part.split("=", 1)[0]
                assert not key.startswith("_"), f"private field {key!r} in repr"
