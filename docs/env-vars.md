# Environment Variables

No LLM-using application today works without API tokens and/or other
authentication secrets. These are almost always passed via environment
variables.

Typeagent currently supports two families of environment variables:

- Those for (public) OpenAI servers.
- Those for the Azure OpenAI service.

## OPENAI environment variables

The (public) OpenAI environment variables include the following:

### Required:

- `OPENAI_API_KEY`: Your secret API key that you get from the
  [OpenAI dashboard](https://platform.openai.com/api-keys).
- `OPENAI_MODEL`: An environment variable introduced by
  [TypeChat](https://microsoft.github.io/TypeChat/docs/examples/)
  indicating the model to use (e.g.`gpt-4o`).

### Optional:

- `OPENAI_BASE_URL`: The URL for an OpenAI-compatible embedding server,
  e.g. [Infinity](https://github.com/michaelfeil/infinity). With this
  option `OPENAI_API_KEY` also needs to be set, but can be any value.
- `OPENAI_ENDPOINT`: The URL for an server compatible with the OpenAI
  Chat Completions API. Make sure the `OPENAI_MODEL` variable matches
  with the deployed model name, e.g. 'llama:3.2:1b'

## Azure OpenAI environment variables

If you are using the OpenAI service hosted by Azure, you need different
environment variables, starting with:

- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key (or in some cases `identity`).
- `AZURE_OPENAI_ENDPOINT`: The full URL of the Azure OpenAI REST API
  (e.g. `https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=2023-05-15`).
- `AZURE_OPENAI_ENDPOINT_EMBEDDING`:
  The full URL of the Azure OpenAI REST API for embeddings
  (e.g. `https://YOUR_RESOURCE_NAME.openai.azure.com/openai/deployments/YOUR_EMBEDDING_DEPLOYMENT_NAME/embeddings?api-version=2024-08-01-preview`).

If you use Azure OpenAI you will know where to get these
(or ask your sysadmin).

## Conflicts

If you set both `OPENAI_API_KEY` and `AZURE_OPENAI_API_KEY`,
`OPENAI_API_KEY` will win.

## Other ways to specify environment variables

It is recommended to put your environment variables in a file named
`.env` in the current or parent directory.
To pick up these variables, call `typeagent.aitools.utils.load_dotenv()`
at the start of your program (before calling any typeagent functions).
(For simplicity this is not shown in
[Getting Started](getting-started.md).)
