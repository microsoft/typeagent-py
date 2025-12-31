# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .interfaces import SearchTerm


def is_search_term_wildcard(search_term: SearchTerm) -> bool:
    """Check if a search term is a wildcard."""
    return search_term.term.text == "*"
