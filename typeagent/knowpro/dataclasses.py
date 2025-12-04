# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Compatibility helpers for pydantic dataclasses."""

from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

from typing_extensions import dataclass_transform

from pydantic.dataclasses import dataclass as _pydantic_dataclass

from .field_helpers import CamelCaseField

T = TypeVar("T")


@overload
def dataclass(__cls: type[T], /, **kwargs: Any) -> type[T]: ...


@overload
def dataclass(**kwargs: Any) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(field_specifiers=(CamelCaseField,))
def dataclass(
    __cls: type[T] | None = None, /, **kwargs: Any
) -> Callable[[type[T]], type[T]] | type[T]:
    """Wrapper that preserves pydantic behavior while informing type-checkers."""

    def wrap(cls: type[T]) -> type[T]:
        return cast(type[T], _pydantic_dataclass(cls, **kwargs))

    if __cls is None:
        return wrap

    return wrap(__cls)
