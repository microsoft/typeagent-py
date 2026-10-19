# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import unicodedata

from .interfaces import SearchTerm


def normalize_term(term: str) -> str:
    """Canonical form of an index term, shared by all storage backends.

    Strips surrounding whitespace, lowercases, applies NFC normalization, and
    collapses runs of whitespace to a single space. Every term index must
    apply this on both write and lookup so backends agree on term identity.

    Lowercasing comes before NFC because lowercasing can produce a sequence
    that composes differently (e.g. "T\u0308" -> "t\u0308" -> "\u1e97"); in the
    reverse order the result isn't idempotent, so re-normalizing a stored term
    (as deserialization does) would change it.
    """
    term = unicodedata.normalize("NFC", term.strip().lower())
    return re.sub(r"\s+", " ", term)


def is_search_term_wildcard(search_term: SearchTerm) -> bool:
    """Check if a search term is a wildcard."""
    return search_term.term.text == "*"
