---
applyTo: '**/*.py'
---

**DO NOT BE OBSEQUIOUS**

# For Agent mode

**NEVER use TEST_MODEL_NAME or "test" embedding model outside of test files**

Never run git commands that many any changes. (`git status` and `git diff` are fine)

**NEVER COMMIT CODE. Do not run `git commit` or any other git commands that make changes to the repository. Not even `git add`**

When moving, copying or deleting files, use the git commands: `git mv`, `git cp`, `git rm`

When the working directory is ~/TypeAgent/python/ta/:

- Don't use '!' on the command line, it's some bash magic (even inside single quotes)
- Activate `.venv`: make venv; source .venv/bin/activate
- To get API keys in ad-hoc code, run `typeagent.aitools.utils.load_dotenv()`
- Use pytest to run tests in test/
- Use pyright to check type annotations in tools/, test/, typeagent/
- Ignore build/, dist/
- You can also use the pylance extension for type checking in VS Code
- Use `make check` to type-check all files
- Use `make test` to run all tests
- Use `make check test` to run `make check` and if it passes also run `make test`

## Package Management with uv

- Use `uv add <package>` to add new dependencies
- Use `uv add <package> --upgrade` to upgrade existing packages
- **Important**: uv automatically updates `pyproject.toml` when adding/upgrading packages
- **Do NOT** manually edit `pyproject.toml` dependency versions after running uv commands
- uv maintains consistency between `pyproject.toml`, `uv.lock`, and installed packages
- Trust uv's automatic version resolution and file management

**IMPORTANT! YOU ARE NOT DONE UNTIL `make check test format` PASSES**

# Code generation

When generating Python code (e.g. when translating TypeScript to Python),
please follow these guidelines:

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
* Use `Protocol` for interfaces whose name starts with `I`
  followed by a capital letter
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
