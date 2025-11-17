# Developing and Contributing

**Always follow the code of conduct, see [Code of Conduct](../CODE_OF_CONDUCT.md).**

To contribute, submit issues or PRs to
[our repo](https://github.com/microsoft/typeagent-py).

To develop, you're mostly on your own.
We use [uv](https://docs.astral.sh/uv/) for some things.
Check out the [Makefile](../Makefile) for some recipes.

## A note about OpenAI plans

As you can see in [Environment Variables](./env-vars.md) there are some
env vars you have to set before you can **use** typeagent.

This applies even more so to developing. In particualar, `make test`
with a Free OpenAI account requires that you upgrade to at least a
Tier 1 account ($5 paid) -- the fully free tier just doesn't have enough
quota to support running the tests, and you would see all tests that
make actual OpenAI requests fail with 429 errors.
