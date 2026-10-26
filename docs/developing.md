# Developing and Contributing

**Always follow the code of conduct, see [Code of Conduct](../CODE_OF_CONDUCT.md).**

To contribute, submit issues or PRs to
[our repo](https://github.com/microsoft/typeagent-py).

To develop, you're mostly on your own.
We use [uv](https://docs.astral.sh/uv/) for some things.
Check out the [Makefile](../Makefile) for some recipes.

## Node.js for type checking

Pyright runs on Node.js. Install **Node.js 24.22.1 or a newer 24.x release**
and ensure `node` is on `PATH` before running `make check` (or
`.\make.bat check` on Windows). CI uses Node.js 24.22.1.

This release includes the
[Brotli decoder security fix for CVE-2026-111497](https://nodejs.org/en/blog/release/v24.22.1).
Updating the Python Pyright package alone does not upgrade an existing Node.js
installation.

## A note about OpenAI plans

As you can see in [Environment Variables](./env-vars.md) there are some
env vars you have to set before you can **use** typeagent.

This applies even more so to developing. In particualar, `make test`
with a Free OpenAI account requires that you upgrade to at least a
Tier 1 account ($5 paid) -- the fully free tier just doesn't have enough
quota to support running the tests, and you would see all tests that
make actual OpenAI requests fail with 429 errors.
