# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Aggregated knowpro interfaces for backwards compatibility."""

from __future__ import annotations

from .interfaces_core import *
from .interfaces_core import __all__ as _core_all
from .interfaces_indexes import *
from .interfaces_indexes import __all__ as _indexes_all
from .interfaces_storage import *
from .interfaces_storage import __all__ as _storage_all
from .search_terms import *
from .search_terms import __all__ as _search_all
from .serialization_data import *
from .serialization_data import __all__ as _serialization_all

# pyright: reportUnsupportedDunderAll=false
__all__ = _core_all + _indexes_all + _search_all + _serialization_all + _storage_all
