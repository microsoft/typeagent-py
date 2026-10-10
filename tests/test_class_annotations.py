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
import pkgutil

import typeagent


def test_all_class_annotations_evaluate() -> None:
    failures: list[str] = []
    for module_info in pkgutil.walk_packages(typeagent.__path__, "typeagent."):
        module = importlib.import_module(module_info.name)
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            try:
                cls.__annotations__
            except Exception as e:
                failures.append(f"{module.__name__}.{name}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)
