# Retrieval hardening

Version 0.2 targets failures where a relevant memory exists and is valid but never reaches final reranking.

| Miss mode | Earlier behavior | Mitigation |
|---|---|---|
| Candidate-source saturation | Lexical ids were appended first and could fill the cap | Weighted reciprocal-rank fusion across every source |
| Inflection mismatch | FTS required the query's surface form | Conservative plural, gerund, and past-tense variants |
| Entity alias | Entity matching required identical canonical text | Namespace-scoped alias equivalence closure |
| LSH boundary | A one-bit bucket change removed the vector candidate | Adaptive Hamming-distance-one multi-probe |
| Missing or degraded vector bucket | Approximate index failure could hide an otherwise exact semantic match | Bounded exact-vector fallback |
| Detached source evidence | Matching a summary did not surface its leaves | One-hop provenance graph expansion and page indexing |

## Candidate fusion

Every retrieval source produces an independently ordered list. Continuum combines them with weighted reciprocal-rank fusion before applying the global candidate cap. Vector and entity sources receive slightly higher fusion weights than lexical candidates; resident and graph sources provide coverage without dominating.

Fusion changes candidate admission, not final relevance. Admitted memories still receive the complete feature score and token-budgeted diversity selection.

## Adaptive vector recovery

The ordinary path remains:

1. Exact FTS and entity lookup
2. Exact LSH buckets
3. Dense reranking of the fused candidate set

If a query has no lexical or entity evidence and exact LSH coverage is weak, Continuum queries buckets one Hamming bit away. For namespaces up to 15,000 active records, it then computes exact cosine similarity over stored vectors. This scan is deliberately limited to semantic-only queries under `adaptive` mode so normal lexical latency remains bounded.

Per request:

- `off`: approximate vector candidates only
- `adaptive`: exact scan only for semantic-only queries
- `always`: exact scan whenever the namespace is within the configured bound

An HNSW or equivalent vector adapter remains the correct solution above the local scan bound.

## Scale-bound comparison

The reference-container runs used 192-dimensional hashing embeddings, 100 indexed queries, and 20 forced exact scans. Both scales retained `1.000` hit@5.

| Scale and path | p50 | p95 | p99 |
|---|---:|---:|---:|
| 15,000 indexed hybrid | 32.6 ms | 70.2 ms | 97.7 ms |
| 15,000 forced exact scan | 280.7 ms | 332.6 ms | 333.2 ms |
| 25,000 indexed hybrid | 44.5 ms | 102.1 ms | 104.3 ms |
| 25,000 forced exact scan | 427.7 ms | 507.3 ms | 559.9 ms |

The 15,000 limit keeps indexed p95 below 100 ms on the reference host and reduces exact-scan p95 by 34.4% relative to 25,000. Exact recovery remains linear, so the default stays `adaptive` and lexical or entity-supported queries do not pay the scan cost.

## Alias semantics

Aliases are namespace-scoped and form a bounded equivalence closure. Registering `IBM` as an alias of `International Business Machines` allows queries using either representation to find memories stored under the other. Alias registration is audited as an event.

Aliases improve recall but can introduce false positives when a short form is ambiguous. Separate namespaces or more specific aliases should be used for overloaded acronyms.

## Graph expansion

Top fused candidates seed a one-hop traversal over `summarizes`, `derived_from`, and `supported_by` links. Linked records enter reranking with a graph-proximity feature. Direct source ids also appear first in the page index even when the resident token budget cannot include them.

The traversal is limited to one hop and a fixed candidate count to prevent high-degree provenance nodes from flooding retrieval.

## Regression evidence

`benchmarks/retrieval_miss_suite.py` reconstructs the legacy candidate behavior alongside the hardened path. Its six intentionally adversarial cases produce 0/6 legacy hits and 6/6 hardened hits. These cases are narrow regression controls, not a substitute for judged semantic retrieval datasets.
