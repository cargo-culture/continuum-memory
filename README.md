# Continuum Memory

Continuum Memory is a local-first persistent memory substrate for language-model systems. It provides durable recall, provenance-aware context packets, hierarchical memory pages, and a controlled mechanism for behavioral policies to develop from repeated outcomes.

It expands a model's **effective** context. It does not alter the model's native context window or weights.

## What is implemented

- Append-audited SQLite persistence with WAL mode
- Full-text retrieval through SQLite FTS5
- Dense embeddings through either a deterministic offline provider or OpenAI
- Bounded locality-sensitive hashing for vector candidate generation
- Hybrid ranking across lexical, semantic, entity, temporal, salience, confidence, and continuity features
- Weighted reciprocal-rank candidate fusion that prevents one retrieval source from consuming the cap
- Adaptive multi-probe LSH plus exact-vector fallback for semantic-only queries
- Conservative inflection expansion, registered entity aliases, and linked-evidence graph expansion
- Maximum-marginal-relevance selection under a hard token budget
- Namespaces, temporal validity, deduplication, supersession, archival, and tombstoning
- Expandable memory pages with provenance links
- Deterministic and schema-constrained OpenAI consolidation adapters
- Behavioral policy states: candidate, testing, promoted, and retired
- CLI and dependency-free local JSON HTTP API
- OpenAI-compatible plugin package with a bundled skill and MCP server
- OAuth-isolated Streamable HTTP deployment for ChatGPT
- Federated specialized memory books with persisted routing metadata
- Concurrent per-book retrieval, cross-book fusion, deduplication, and global budgeting
- Reproducible functional tests and synthetic recall benchmark

## Quick start

Continuum Memory requires Python 3.10 or newer and has no mandatory third-party dependencies.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .

continuum-memory init
continuum-memory --namespace demo remember \
  "The cobalt scheduler coordinates Zephyr deployment windows." \
  --kind semantic --entity Zephyr --tag architecture

continuum-memory --namespace demo recall "cobalt scheduler"
continuum-memory --namespace demo context "What controls Zephyr deployments?"
continuum-memory --namespace demo alias IBM "International Business Machines"
```

Create the standard federated library and store project memory in its own book:

```bash
continuum-memory library-init
continuum-memory --namespace demo book-remember projects \
  "The cobalt scheduler coordinates Zephyr deployment windows." \
  --kind semantic --entity Zephyr
continuum-memory --namespace demo library-recall \
  "What coordinates Zephyr deployments?"
```

The standard router searches `working` and `identity` for continuity plus the strongest topical book, skipping any book that holds nothing for the caller. Selected books use separate SQLite connections and run concurrently. Results are deduplicated and fused into one globally budgeted context packet. See [Federated memory books](docs/FEDERATED_BOOKS.md).

## ChatGPT and Codex plugin

Continuum 0.5.0 includes a plugin manifest, a memory-use skill, and an MCP server:

```bash
pip install -e '.[plugin]'
continuum-memory-mcp --transport stdio
```

The MCP surface exposes focused tools for search, provenance expansion, book listing, durable writes, and supersession. The model never supplies a namespace: local installs use `CONTINUUM_NAMESPACE`, while authenticated HTTP deployments derive an opaque namespace from the verified OAuth identity.

Tool responses are bounded and report their own size. Search returns budgeted excerpts, `read_memory` expands one record with a capped provenance fan-out, and every record carries `is_excerpt` and `full_token_count`.

For ChatGPT, deploy the server through authenticated Streamable HTTP and register the public `/mcp` endpoint in developer mode. See [OpenAI plugin integration](docs/PLUGIN.md).

A production Dockerfile, persistent-volume Compose service, GitHub Actions verification, tagged GHCR publishing, policy pages, domain-verification route, and a review-ready [submission packet](docs/SUBMISSION.md) are included.

The default hashing embedder is fast, deterministic, and offline. It provides lexical-neighborhood vectors for development and testing.

For semantic embeddings and model-assisted consolidation:

```bash
pip install -e '.[openai]'
export CONTINUUM_EMBEDDING_PROVIDER=openai
export CONTINUUM_EMBEDDING_MODEL=text-embedding-3-small
export CONTINUUM_EMBEDDING_DIMENSIONS=512

continuum-memory --namespace demo reindex
continuum-memory --namespace demo consolidate --provider openai
```

Books are independent stores, so a library consolidates one book at a time:

```bash
continuum-memory --namespace demo book-consolidate episodic --provider openai
```

Keep `OPENAI_API_KEY` in a private `.env.local` file. The application searches the current directory and its nearest parents without logging secret values.

## Context paging

The resident context packet contains only the highest-utility memories that fit the supplied token budget. Each item retains a stable memory id, confidence, timestamp, kind, score, and source. Summary and overflow ids form a page index that the calling model can expand through `page` or `GET /v1/pages/{id}`.

```bash
continuum-memory --namespace demo context \
  "What did we decide about deployment scheduling?" \
  --token-budget 1200 --limit 8
```

For small namespaces where recall matters more than latency, force an exact dense-vector scan:

```bash
continuum-memory --namespace demo recall \
  "Describe the idea even if none of my wording matches" \
  --exact-scan-mode always
```

`adaptive` is the default: exact scanning activates for semantic-only queries with no lexical or entity candidates. `always` is bounded to 15,000 active records in the reference implementation. Larger deployments should replace this fallback with an HNSW vector adapter.

Rendered packets explicitly mark memory as untrusted contextual data. Retrieved text is never treated as an executable instruction by the memory system.

## Behavioral development

A behavioral statement is not adopted after one observation. Outcomes move it through a validated lifecycle:

```bash
continuum-memory --namespace agent policy-evidence \
  "Verify provenance before presenting a recalled claim as current." success
```

Three decisive successes at a success rate of at least 0.75 promote a policy. Promotion materializes a procedural memory so the policy participates in future retrieval. Sufficient contradictory outcomes retire the policy and archive that procedural memory. All evidence and transitions remain auditable.

This is experience-driven behavioral adaptation in external state. It is deliberately separate from weight updates, unconstrained self-modification, and claims of sentience.

## Local HTTP API

```bash
continuum-memory serve --host 127.0.0.1 --port 8765
```

Write a memory:

```bash
curl -s http://127.0.0.1:8765/v1/memories \
  -H 'content-type: application/json' \
  -d '{
    "namespace": "demo",
    "kind": "episodic",
    "content": "The second retrieval experiment used an amber routing table.",
    "importance": 0.7,
    "entities": ["Retrieval Experiment"]
  }'
```

Build a context packet:

```bash
curl -s http://127.0.0.1:8765/v1/context \
  -H 'content-type: application/json' \
  -d '{
    "namespace": "demo",
    "query": "Which routing table did the experiment use?",
    "limit": 8,
    "token_budget": 1200
  }'
```

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Storage and provider status |
| `POST` | `/v1/memories` | Persist a memory |
| `GET` | `/v1/memories/{id}` | Read one materialized memory |
| `POST` | `/v1/memories/{id}/supersede` | Replace a claim while preserving history |
| `POST` | `/v1/recall` | Retrieve ranked memories |
| `POST` | `/v1/context` | Build a token-budgeted context packet |
| `GET` | `/v1/pages/{id}` | Expand content and provenance |
| `POST` | `/v1/consolidate` | Create hierarchical summaries and derived claims |
| `POST` | `/v1/policies/evidence` | Record a behavioral outcome |
| `GET` | `/v1/policies` | Inspect policy lifecycle state |
| `POST` | `/v1/entity-aliases` | Register an alias-to-entity mapping |
| `GET` | `/v1/events` | Read the append-only audit stream |
| `POST` | `/v1/reindex` | Recompute active-memory embeddings |

The included server has no authentication or encryption layer and defaults to loopback. Do not expose it directly to an untrusted network.

The federated API runs separately:

```bash
continuum-memory serve-library --host 127.0.0.1 --port 8765
```

It exposes `/v1/books`, `/v1/books/{book}/memories`, `/v1/library/recall`, `/v1/library/context`, and book-qualified page expansion.

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python benchmarks/synthetic_recall.py \
  --records 2000 --queries 100 --dimensions 192
PYTHONPATH=src python benchmarks/synthetic_recall.py \
  --records 15000 --queries 100 --dimensions 192 --exact-scan-queries 20
PYTHONPATH=src python benchmarks/retrieval_miss_suite.py
PYTHONPATH=src python benchmarks/federated_recall.py \
  --books 4 --records-per-book 1000 --queries 100 --dimensions 192
```

On the reference container, the 5,000-memory offline benchmark produced:

- hit@5: `1.000`
- recall p50: `16.9 ms`
- recall p95: `29.1 ms`
- recall p99: `32.9 ms`
- write p50: `1.4 ms`
- write p95: `2.9 ms`

Synthetic exact-marker retrieval is a latency regression test, not evidence of semantic quality. Semantic evaluation requires paraphrases, temporal contradictions, noisy episodes, and human-scored relevance judgments.

The retrieval-miss suite contains six deterministic adversarial cases. The legacy candidate path misses all six; version 0.2 recovers all six. See [Retrieval hardening](docs/RETRIEVAL_HARDENING.md) for the failure modes and tradeoffs.

At the selected 15,000-memory ceiling, the same reference environment retained `1.000` hit@5. Indexed recall measured `32.6 ms` p50 and `70.2 ms` p95. Forced exact-vector scans measured `280.7 ms` p50 and `332.6 ms` p95. Compared with the 25,000-memory experiment, p95 improved by `31.2%` on the indexed path and `34.4%` on the forced scan.

The four-book benchmark retained `1.000` hit@5. Router-selected three-book retrieval measured `95.8 ms` p95. Searching all four concurrently measured `137.4 ms` p95 versus `213.2 ms` sequentially, a `35.5%` reduction.

## Design documents

- [Architecture and invariants](docs/ARCHITECTURE.md)
- [Research and evaluation plan](docs/RESEARCH_PLAN.md)
- [Retrieval hardening](docs/RETRIEVAL_HARDENING.md)
- [Federated memory books](docs/FEDERATED_BOOKS.md)
- [OpenAI plugin integration](docs/PLUGIN.md)

The OpenAI adapter follows the official [embeddings](https://developers.openai.com/api/docs/guides/embeddings) and [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs) patterns.
