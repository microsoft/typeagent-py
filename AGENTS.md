**DO NOT BE OBSEQUIOUS**

**NEVER use TEST_MODEL_NAME or "test" embedding model outside of test files**

Never run git commands that make any changes. (`git status` and `git diff` are fine)
Exceptions: `git push`, `git worktree`, `git branch` (for tracking setup), as instructed below.

**NEVER COMMIT CODE.** Do not run `git commit` or any other git commands
that make changes to the repository. Exception: Worktrees/Branches below.
`git add` is fine.

When moving, copying or deleting files, use the git commands: `git mv`, `git cp`, `git rm`

## Worktrees and Branches

- Each session uses its own worktree with a feature branch
- Create worktrees with: `git worktree add ../<repo>-<branch-name> -b <branch-name>`
- Push the branch to the `me` remote: `git push me <branch-name>`
- Set upstream to `me/<branch-name>`: `git branch --set-upstream-to me/<branch-name>`
- **Never** upstream to `me/main` — that must stay identical to `origin/main`
- The worktree directory name should be `<repo>-<branch-name>` (sibling of the main checkout)
- **Work in the worktree directory**, not the main checkout — edit files there, run tests there
- VS Code may show buffers from the main checkout; ignore those when working in a worktree.
  When in doubt, verify edits landed on disk with `cat` or `grep` in the terminal.

## Debugging discipline

- When a bug seems impossible, suspect stale files or wrong working directory — not exotic causes.
- If you're tempted to blame installed package versions, `__pycache__`, or similar,
  **stop and ask the user** before investigating further. You're probably on the wrong track.

**Whenever the user tells you how to do something, states a preference, or corrects you,
extract a general rule and add it to AGENTS.md** (unless it's already covered -- maybe
reformulate since it apparently didn't work). This applies even without being asked.
In all cases show what you added to AGENTS.md.

- Don't use '!' on the command line, it's some bash magic (even inside single quotes)
- When running 'make' commands, do not use the venv (the Makefile uses 'uv run')
- To get API keys in ad-hoc code, call `load_dotenv()`
- Use `pytest test` to run tests in test/
- Use `pyright` to check type annotations in src/, tools/,  tests/, examples/
- Ignore build/, dist/
- You can also use the pylance extension for type checking in VS Code
- Use `make check` to type-check all files
- Use `make test` to run all tests
- Use `make check test` to run `make check` and if it passes also run `make test`
- Use `make format` to format all files using `black`. Do this before reporting success.
- When validating changes, first run `pytest` only on new/modified test files, then run `make format check test` once at the end.
- While building `add_messages.py` before dedicated tests exist, skip running the full test suite; run full tests after those tests are added.
- Keep ad-hoc and performance benchmarks under `tools/`, not `tests/`, so `make test` does not run them.
- In add-messages pipeline chunk processing, compute chunk-text embeddings with uncached model calls and related-term embeddings with cached model calls.
- In add-messages pipeline flow, lower stop_at_message_id to min(existing, failing_message_id), and always enqueue queue-1 sentinels even when the input iterator fails so workers can drain and exit cleanly.
- In add-messages pipeline data structures, use `TextLocation` as the chunk identifier instead of a formatted string chunk ID.
- In add-messages reassembler validation, prefer explicit guard checks over wrapping validation-only logic in `try/except` blocks.
- In add-messages reassembler validation, prefer a single `validation_error` variable with consistent `if/elif` checks over helper functions for simple message-only validation.
- When adding precomputed-embedding write paths, expose explicit `*_with_embeddings` methods and have existing methods compute embeddings then delegate to those methods.
- In asyncio code, avoid locks for in-memory state updates that do not `await` between read/modify/write; use locks only when a critical section spans `await` points.
- Name returned summary/value objects as `*Result`; reserve `*State` for mutable shared/internal state.
- Keep internal helper type naming consistent within a module; avoid mixing underscored and non-underscored helper class names without a clear API-boundary reason.
- Prefer variable names that reflect role rather than lifecycle; for accumulators like message assemblies, use neutral names (e.g., `assembly`) instead of state-qualified names (e.g., `existing`).
- Avoid potential import cycles between conversation orchestration and pipeline modules by using neutral payload protocols/arguments instead of importing concrete pipeline result classes across modules.
- Prefer ordinal type aliases (e.g., `MessageOrdinal`, `ChunkOrdinal`) over raw `int` in pipeline code for readability.
- When the user asks to "fix the test only", update tests/mocks first and avoid adding production compatibility fallbacks unless explicitly requested.

## Package Management with uv

- Use `uv add <package>` to add new dependencies
- Use `uv add <package> --upgrade` to upgrade existing packages
- **Important**: uv automatically updates `pyproject.toml` when adding/upgrading packages
- **Do NOT** manually edit `pyproject.toml` dependency versions after running uv commands
- uv maintains consistency between `pyproject.toml`, `uv.lock`, and installed packages
- Trust uv's automatic version resolution and file management

**IMPORTANT! YOU ARE NOT DONE UNTIL `make format check test` PASSES**

# Code generation

When generating Python code (e.g. when translating TypeScript to Python),
please follow these guidelines:

* When creating a new file, add a copyright header to the top:
```
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
```

* Assume Python 3.12

* `from __future__ import annotations` is not allowed.

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
