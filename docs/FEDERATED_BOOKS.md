# Federated memory books

Continuum 0.3 adds a first-class library of specialized memory books. Each book owns an independent SQLite/WAL database, FTS index, vector buckets, provenance graph, event stream, and policy state. Independent connections allow selected books to perform reads concurrently.

## Standard library

`library-init` creates eight books:

| Book | Purpose | Routing behavior |
|---|---|---|
| `working` | Current task and unresolved execution state | Always searched |
| `identity` | Stable user preferences and continuity | Always searched |
| `projects` | Requirements, decisions, files, and milestones | Routed |
| `episodic` | Conversations, events, actions, and outcomes | Routed |
| `semantic` | Facts, concepts, entities, and relationships | Routed |
| `procedural` | Workflows, methods, and reusable solutions | Routed |
| `sources` | Documents, citations, provenance, and evidence | Routed |
| `behavioral` | Strategies, failures, successes, and policies | Routed |

The default router selects three books: the two always-on continuity books and the highest-scoring topical book. Explicit `book_ids` bypass automatic routing.

## Retrieval path

1. Embed the query once.
2. Score book descriptors using keyword matches, lexical overlap, embedding similarity, priority, and always-search status.
3. Search selected books concurrently through independent database connections.
4. Combine per-book rankings through weighted reciprocal-rank fusion.
5. Remove exact cross-book content duplicates.
6. Apply a soft per-book quota and one global token budget.
7. Render book-qualified memory and page references into one context packet.

The global budget remains 1,600 tokens by default. Adding books expands searchable storage; it does not expand simultaneous model attention or multiply the packet budget.

## CLI

```bash
continuum-memory library-init

continuum-memory --namespace demo book-remember projects \
  "Continuum uses independent databases for concurrent book retrieval." \
  --kind semantic --entity Continuum

continuum-memory --namespace demo library-recall \
  "How does the project search memory books?"

continuum-memory --namespace demo library-context \
  "Continue the active project using my established preferences."
```

Create a custom book:

```bash
continuum-memory book-create worldbuilding \
  --title "Worldbuilding canon" \
  --description "Characters, settings, chronology, and accepted fictional canon" \
  --keyword character --keyword setting --keyword canon --priority 1.2
```

Use `--book projects --book sources` on `library-recall` or `library-context` to bypass routing and search an explicit set.

## HTTP API

Run the federated server with `continuum-memory serve-library`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Library health and aggregate record count |
| `GET` | `/v1/books` | List books and per-namespace counts |
| `POST` | `/v1/books` | Create a book |
| `POST` | `/v1/books/{book}/memories` | Store a memory in a book |
| `GET` | `/v1/books/{book}/pages/{id}` | Expand book-qualified provenance |
| `POST` | `/v1/library/recall` | Retrieve across selected books |
| `POST` | `/v1/library/context` | Build a federated context packet |

## Measured concurrency

The reference benchmark used four books with 1,000 records each, 192-dimensional hashing embeddings, 100 queries, and exact scanning disabled.

| Search mode | hit@5 | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| Router-selected three books | 1.000 | 78.8 ms | 95.8 ms | 123.3 ms |
| All four, concurrent | 1.000 | 109.1 ms | 137.4 ms | 156.8 ms |
| All four, sequential | 1.000 | 174.3 ms | 213.2 ms | 242.4 ms |

Concurrent fan-out reduced all-book p95 by 35.5% relative to sequential search. Routing reduced work further by avoiding the fourth book. These synthetic exact-marker results measure regression behavior and latency, not general semantic quality.

## Consistency boundaries

- Writes are atomic within a book, not across several books.
- Exact duplicates are removed during retrieval, but contradictory claims in different books remain separate evidence.
- Each book retains the 15,000-record exact-scan ceiling. A routed query can trigger scans in several selected books.
- Book routing can miss a relevant collection. Explicit book selection is available for high-stakes or cross-domain requests.
- Cross-book multi-hop reasoning requires retrieved links or a later library-level graph index; provenance graphs currently remain local to each book.
