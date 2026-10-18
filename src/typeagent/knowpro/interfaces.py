# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Aggregated knowpro interfaces for backwards compatibility."""

from __future__ import annotations

from .core_types import *
from .core_types import __all__ as _core_all
from .index_protocols import *
from .index_protocols import __all__ as _indexes_all
from .search_terms import *
from .search_terms import __all__ as _search_all
from .serialization_data import *
from .serialization_data import __all__ as _serialization_all
from .storage_protocols import *
from .storage_protocols import __all__ as _storage_all

# pyright: reportUnsupportedDunderAll=false
__all__ = _core_all + _indexes_all + _search_all + _serialization_all + _storage_all
