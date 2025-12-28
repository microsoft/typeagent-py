# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Compatibility helpers for pydantic dataclasses."""

from collections.abc import Callable
from typing import Any, cast, overload

from typing_extensions import dataclass_transform

from pydantic.dataclasses import dataclass as _pydantic_dataclass

from .field_helpers import CamelCaseField


@overload
def dataclass[T](__cls: type[T], /, **kwargs: Any) -> type[T]: ...


@overload
def dataclass[T](**kwargs: Any) -> Callable[[type[T]], type[T]]: ...


@dataclass_transform(field_specifiers=(CamelCaseField,))
def dataclass[T](
    __cls: type[T] | None = None, /, **kwargs: Any
) -> Callable[[type[T]], type[T]] | type[T]:
    """Wrapper that preserves pydantic behavior while informing type-checkers."""

    def wrap(cls: type[T]) -> type[T]:
        return cast(type[T], _pydantic_dataclass(cls, **kwargs))

    if __cls is None:
        return wrap

    return wrap(__cls)
