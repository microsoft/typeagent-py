# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the TypeScript schema text that TypeChat renders from our schema classes.

That text is what the model actually sees, so it must carry the comments we
intend to send and agree with the validator about which fields are required.
"""

import dataclasses
import re

import pytest

import pydantic
from pydantic.fields import FieldInfo
import typechat

from typeagent.knowpro import (
    answer_response_schema,
    date_time_schema,
    knowledge_schema,
    search_query_schema,
)
from typeagent.knowpro.knowledge_schema import Action

SCHEMA_MODULES = (
    answer_response_schema,
    date_time_schema,
    knowledge_schema,
    search_query_schema,
)


def schema_classes() -> list[type]:
    """All dataclasses defined in the modules that double as TypeChat schemas."""
    return [
        obj
        for module in SCHEMA_MODULES
        for obj in vars(module).values()
        if isinstance(obj, type)
        and dataclasses.is_dataclass(obj)
        and obj.__module__ == module.__name__
    ]


def render_interface_body(cls: type) -> str:
    """Return the body of the TypeScript interface that TypeChat renders for cls."""
    result = typechat.python_type_to_typescript_schema(cls)
    assert not result.errors
    match = re.search(
        rf"^interface {cls.__name__} \{{\n(.*?)^\}}",
        result.typescript_schema_str,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"No interface rendered for {cls.__name__}"
    return match.group(1)


def test_action_subject_entity_facet_comment_is_rendered() -> None:
    lines = [line.strip() for line in render_interface_body(Action).splitlines()]
    index = lines.index("subject_entity_facet?: Facet | null;")
    assert lines[index - 1].startswith(
        "// If the action implies this additional facet or property"
    )


def test_action_verb_tense_is_rendered_required() -> None:
    lines = [line.strip() for line in render_interface_body(Action).splitlines()]
    assert "verb_tense: VerbTense;" in lines


def test_action_verb_tense_is_required_by_validator() -> None:
    validator = typechat.TypeChatValidator[Action](Action)
    assert isinstance(validator.validate_object({"verbs": ["go"]}), typechat.Failure)
    assert isinstance(
        validator.validate_object({"verbs": ["go"], "verb_tense": "past"}),
        typechat.Success,
    )


def test_action_field_aliases() -> None:
    adapter = pydantic.TypeAdapter(Action)
    snake = adapter.validate_python(
        {"verbs": ["go"], "verb_tense": "past", "subject_entity_name": "Alice"}
    )
    camel = adapter.validate_python(
        {"verbs": ["go"], "verbTense": "past", "subjectEntityName": "Alice"}
    )
    assert snake == camel
    assert adapter.dump_python(snake, by_alias=True) == {
        "verbs": ["go"],
        "verbTense": "past",
        "subjectEntityName": "Alice",
        "objectEntityName": "none",
        "indirectObjectEntityName": "none",
        "params": None,
        "subjectEntityFacet": None,
    }


@pytest.mark.parametrize("cls", schema_classes(), ids=lambda cls: cls.__name__)
def test_rendered_optionality_matches_validation(cls: type) -> None:
    """A field rendered with '?' must be optional for the validator, and vice versa."""
    body = render_interface_body(cls)
    fields: dict[str, FieldInfo] = getattr(cls, "__pydantic_fields__")
    assert fields
    for name, field in fields.items():
        match = re.search(rf"^\s*{name}(\??):", body, re.MULTILINE)
        assert match is not None, f"{cls.__name__}.{name} is not rendered"
        rendered_required = match.group(1) == ""
        assert (
            rendered_required == field.is_required()
        ), f"{cls.__name__}.{name}: schema and validator disagree on optionality"
