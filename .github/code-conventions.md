When generating Python code (e.g. when translating TypeScript to Python),
follow these guidelines:

* When creating a new file, add a copyright header to the top:
```
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
```

* Assume Python 3.12

* Always strip trailing spaces

* Keep class and type names in `PascalCase`
* Use `python_case` for variable/field and function/method names

* Use `Literal` for unions of string literals
* Keep union notation (`X | Y`) for other unions
* Use `Protocol` for interfaces whose name starts with `I` followed by a capital letter
* Use `dataclass` for other classes and structured types
* Use `type` for type aliases (`PascalCase` again)
* Use `list`, `tuple`, `dict`, `set` etc., not `List` etc.

* Translate `foo?: string` to `foo: str | None = None`

* When writing tests:
  - don't mock; use the regular implementation (maybe introduce a fixture to create it)
  - assume `pytest`; use `assert` statements
  - match the type annotations of the tested functions
  - read the code of the tested functions to understand their behavior
  - When using fixtures:
    - Fully type-annotate the fixture definitions (including return type)
    - Fully type-annotate fixture usages

* Don't put imports inside functions.
  Put them at the top of the file with the other imports.
  Exception: imports in a `if __name__ == "__main__":` block or a `main()` function.
  Another exception: pydantic and logfire.
  Final exception: to avoid circular import errors.

* **Import Architecture Rules**:
  - **Never import a symbol from a module that just re-exports it**
  - **Always import directly from the module that defines the symbol**
  - **Exception**: Package `__init__.py` files that explicitly re-export with `__all__`
  - **Exception**: Explicit re-export patterns like `from ... import X as X` or marked with "# For export"
  - This prevents circular imports and makes dependencies clear

* Order imports alphabetically after lowercasing; group them as follows
  (with a blank line between groups):
  1. standard library imports
  2. established third-party libraries
  3. experimental third-party libraries (e.g. `typechat`)
  4. local imports (e.g. `from typeagent.knowpro import ...`)

* **Error Handling**: Don't use `try/except Exception` to catch errors broadly.
  Let errors bubble up naturally for proper error handling and debugging at higher levels.

* **Code Validation**: Don't use `py_compile` for syntax checking.
  Use `pyright` or `make check` instead for proper type checking and validation.

* **Deprecations**: Don't deprecate things -- just delete them and fix the usage sites.
  Don't create backward compatibility APIs or exports or whatever. Fix the usage sites.
