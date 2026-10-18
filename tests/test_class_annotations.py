# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Check that every class's annotations in the typeagent package evaluate.

On Python 3.14+ annotations are evaluated lazily in the class namespace, so a
class attribute or property with the same name as an imported module (e.g. a
`semantic_ref_index` property next to `semantic_ref_index.TermToSemanticRefIndex`
annotations) shadows the module and makes `__annotations__` raise.
"""

import importlib
import inspect
from pathlib import Path

import typeagent


def _all_module_names() -> list[str]:
    # pkgutil.walk_packages() skips implicit namespace packages (directories
    # without __init__.py, e.g. typeagent/knowpro), so walk the files instead.
    names: list[str] = []
    for root in typeagent.__path__:
        root_path = Path(root)
        for path in sorted(root_path.rglob("*.py")):
            parts = path.relative_to(root_path).with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            names.append(".".join(("typeagent", *parts)))
    return names


def test_module_discovery_includes_namespace_packages() -> None:
    names = _all_module_names()
    assert "typeagent.knowpro.conversation_base" in names
    assert "typeagent.storage.memory.provider" in names


def test_all_class_annotations_evaluate() -> None:
    failures: list[str] = []
    for module_name in _all_module_names():
        module = importlib.import_module(module_name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            try:
                cls.__annotations__
            except Exception as e:
                failures.append(f"{module.__name__}.{name}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)
