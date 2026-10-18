# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pydantic dataclass decorator that adds camelCase aliases to every field.

Fields are declared in snake_case. By default, both the snake_case name and
its generated camelCase form are accepted on input; output uses camelCase
when ``by_alias=True``. Explicit field aliases override the corresponding
generated aliases. A custom serialization alias needs a matching validation
alias if its output must round-trip; setting it alone does not change the
accepted input names.
"""

from collections.abc import Callable
import dataclasses
from typing import Any, cast, overload

from typing_extensions import dataclass_transform

from pydantic import AliasChoices, AliasGenerator, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic.dataclasses import dataclass as _pydantic_dataclass

# AliasChoices rather than validate_by_name, which needs pydantic 2.11.
_ALIAS_GENERATOR = AliasGenerator(
    validation_alias=lambda name: AliasChoices(name, to_camel(name)),
    serialization_alias=to_camel,
)


@overload
def dataclass[T](__cls: type[T], /, **kwargs: Any) -> type[T]: ...


@overload
def dataclass[T](**kwargs: Any) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(field_specifiers=(dataclasses.field, Field))
def dataclass[T](
    __cls: type[T] | None = None, /, **kwargs: Any
) -> Callable[[type[T]], type[T]] | type[T]:
    """Like ``pydantic.dataclasses.dataclass``, plus camelCase aliases."""
    config = ConfigDict(alias_generator=_ALIAS_GENERATOR)
    config.update(kwargs.pop("config", None) or {})

    def wrap(cls: type[T]) -> type[T]:
        return cast(type[T], _pydantic_dataclass(cls, config=config, **kwargs))

    if __cls is None:
        return wrap

    return wrap(__cls)
