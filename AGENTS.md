**DO NOT BE OBSEQUIOUS**

**NEVER use TEST_MODEL_NAME or "test" embedding model outside of test files**

Git mutations are blocked by a PreToolUse hook (`scripts/block_git_mutations.py`).

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
- Keep ad-hoc and performance benchmarks under `tools/`, not `tests/`, so `make test` does not run them.

## Package Management with uv

- Use `uv add <package>` to add new dependencies
- Use `uv add <package> --upgrade` to upgrade existing packages
- **Important**: uv automatically updates `pyproject.toml` when adding/upgrading packages
- **Do NOT** manually edit `pyproject.toml` dependency versions after running uv commands
- uv maintains consistency between `pyproject.toml`, `uv.lock`, and installed packages
- Trust uv's automatic version resolution and file management

**IMPORTANT! YOU ARE NOT DONE UNTIL `make format check test` PASSES**

For code generation and style conventions, see [.github/code-conventions.md](.github/code-conventions.md).
