# OpenAI plugin

Continuum 0.4.0 packages the memory library as an OpenAI-compatible plugin with a bundled skill and MCP server.

## Tool surface

| Tool | Mutation | Purpose |
|---|---|---|
| `search_memory` | No | Route and search relevant memory books |
| `read_memory` | No | Expand one record and its provenance |
| `list_memory_books` | No | Inspect available books and record counts |
| `remember_memory` | Yes | Store durable future-useful information |
| `supersede_memory` | Yes | Correct a record while preserving history |

Caller namespaces are not model arguments. Local stdio derives the namespace from `CONTINUUM_NAMESPACE`; authenticated HTTP derives an opaque namespace from the validated OAuth issuer and subject.

## Local development

Install the plugin extra and initialize the standard books:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[plugin]'
continuum-memory library-init
continuum-memory-mcp --transport stdio
```

The repository root is a plugin package. `.codex-plugin/plugin.json` declares the skill and bundled `.mcp.json` server. The repository marketplace is at `.agents/plugins/marketplace.json`.

The bundled MCP command expects the Python package to be installed in the environment that launches it. `PLUGIN_DATA` keeps installed-plugin databases outside the immutable plugin package.

## Local Streamable HTTP

Unauthenticated HTTP is restricted to explicit loopback development:

```bash
continuum-memory-mcp --transport streamable-http --insecure-local-http
```

The endpoint is `http://127.0.0.1:8766/mcp` by default.

## Authenticated ChatGPT deployment

Production HTTP mode is an OAuth 2.1 resource server. Configure an external authorization server and an RFC 7662 introspection endpoint:

```bash
export CONTINUUM_MCP_ISSUER_URL=https://auth.example.com/
export CONTINUUM_MCP_RESOURCE_URL=https://memory.example.com/mcp
export CONTINUUM_MCP_INTROSPECTION_URL=https://auth.example.com/oauth/introspect
export CONTINUUM_MCP_INTROSPECTION_CLIENT_ID=continuum-resource-server
export CONTINUUM_MCP_INTROSPECTION_CLIENT_SECRET=replace-me

continuum-memory-mcp \
  --transport streamable-http \
  --host 0.0.0.0 \
  --port 8000 \
  --path /mcp
```

The authorization server must issue tokens whose audience contains the exact `CONTINUUM_MCP_RESOURCE_URL`, whose subject uniquely identifies the user, and whose scopes contain `memory:read`. Write tools additionally require `memory:write`.

Terminate TLS at a trusted reverse proxy or application platform. Do not expose the dependency-free JSON API as the public plugin endpoint.

The production image persists its library at `/var/lib/continuum`, serves a public `GET /health`, and exposes the portal's domain-verification response at `GET /.well-known/openai-apps-challenge` when `CONTINUUM_OPENAI_CHALLENGE_TOKEN` is set:

```bash
docker build -t continuum-memory:0.4.0 .
docker run --rm -p 127.0.0.1:8000:8000 \
  --env-file .env.local \
  -v continuum-data:/var/lib/continuum \
  continuum-memory:0.4.0
```

## ChatGPT registration

Deploy the authenticated Streamable HTTP endpoint first. In ChatGPT developer mode, register its public `/mcp` URL, then package the returned registered connection identifier in `.app.json` for ChatGPT-hosted distribution. The bundled `.mcp.json` remains useful for local Codex and repository installs.

Public-directory submission additionally requires operator-owned privacy-policy and terms URLs, a deployed service, and review of the MCP server and its data-handling behavior. Use the prepared [submission packet](SUBMISSION.md) for listing copy, test cases, reviewer checks, and release notes.
