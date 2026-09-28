# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import unicodedata

from .interfaces import SearchTerm


def normalize_term(term: str) -> str:
    """Canonical form of an index term, shared by all storage backends.

    Strips surrounding whitespace, applies NFC normalization, collapses runs
    of whitespace to a single space, and lowercases. Every term index must
    apply this on both write and lookup so backends agree on term identity.
    """
    term = unicodedata.normalize("NFC", term.strip())
    return re.sub(r"\s+", " ", term).lower()


def is_search_term_wildcard(search_term: SearchTerm) -> bool:
    """Check if a search term is a wildcard."""
    return search_term.term.text == "*"
