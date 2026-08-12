# Research and evaluation plan

## Primary question

Can structured, persistent, recursively consolidated experience produce coherent longitudinal behavioral development in a largely fixed model while maintaining low recall latency, provenance, stability, and bounded prompt usage?

## Hypotheses

1. Hybrid retrieval outperforms vector-only and lexical-only baselines on longitudinal recall.
2. Hierarchical paging preserves answer accuracy while reducing resident prompt tokens.
3. Explicit temporal validity and supersession reduce stale-fact answers.
4. Outcome-gated procedural memory improves repeated-task performance without rapid behavioral drift.
5. Provenance links reduce unsupported claims when the calling model is instructed to cite memory ids.
6. Periodic source-grounded consolidation limits recursive summary distortion.

## Evaluation suites

### Recall quality

- Exact fact recall at increasing corpus sizes
- Paraphrase and synonym recall
- Entity aliases and relationship queries
- Distractor episodes sharing vocabulary
- Very old but highly salient memories
- Relevant memories with low lexical overlap

Metrics: recall@k, mean reciprocal rank, nDCG, false-positive rate, and token-adjusted utility.

### Temporal and epistemic correctness

- Facts superseded several times
- Simultaneously valid claims scoped to different dates
- Conflicting sources with unequal confidence
- Inferences whose supporting evidence is later retired
- Questions asking what was believed at a historical time

Metrics: current-fact accuracy, historical-fact accuracy, contradiction surfacing, and provenance completeness.

### Context efficiency

Compare full transcript, flat retrieval, summary-only retrieval, and paged retrieval at identical model and output budgets.

Metrics: answer accuracy, resident tokens, page fetches, end-to-end latency, and cost per correct answer.

### Behavioral development

Use repeated tasks in which some strategies are stable, some are context-dependent, and some become invalid later.

Metrics:

- trials to policy promotion
- improvement after promotion
- inappropriate transfer to unrelated tasks
- trials to retirement after environment change
- persistence across process restarts
- recovery after misleading evidence
- divergence between separate namespaces

### Memory integrity and attacks

- Prompt injection stored inside source documents
- Deliberately false high-salience memories
- Duplicate floods
- Cross-namespace retrieval attempts
- Provenance cycles
- Malformed timestamps and extreme token counts
- Model-generated policy hypotheses based on a single event

Metrics: instruction-following from memory, contamination rate, isolation failures, and rejected invalid transitions.

## Ablations

Run each corpus with the following features removed independently:

- semantic score
- lexical score
- entity score
- recency
- confidence
- continuity prior
- token-efficiency adjustment
- maximum marginal relevance
- consolidation
- behavioral evidence gate

This identifies whether apparent improvement comes from actual memory architecture or simply from adding more text to the prompt.

## Performance budgets

Initial local targets, excluding model inference and remote embedding latency:

| Operation | Target |
|---|---:|
| Cached exact/entity lookup p50 | 10 ms |
| Hybrid local recall p50 | 50 ms |
| Hybrid local recall p95 | 100 ms |
| Durable local write p95 | 20 ms |
| Context selection overhead | 20 ms |
| Provenance page expansion p95 | 25 ms |

Remote embeddings require precomputation and query-vector caching to preserve these budgets.

## Development sequence

1. Establish reproducible retrieval and temporal benchmarks.
2. Add judged paraphrase and contradiction corpora.
3. Replace LSH with an HNSW-backed adapter and compare latency/recall curves.
4. Add asynchronous consolidation checkpoints and recursive source grounding.
5. Connect a fixed model and run longitudinal behavioral trials.
6. Train or calibrate retrieval weights only after collecting counterfactual judgments.
7. Add multi-tenant security and deletion guarantees before external deployment.

## Retrieval-miss regression gate

Version 0.2 adds deterministic cases for candidate-source saturation, morphological variation, entity aliases, absent vector buckets, linked source evidence, and one-bit LSH drift. Every release must preserve 6/6 recovery while keeping the ordinary 5,000-record lexical benchmark under 100 ms p95 on the reference container.

## Federated-library regression gate

Book routing, genuinely concurrent execution, cross-book deduplication, global token budgeting, persistence, and partial-book failure behavior require deterministic coverage. Router recall must be reported separately from within-book recall so topical routing misses are not hidden inside aggregate hit@k. Performance runs should compare one-book, routed three-book, all-book concurrent, and all-book sequential fan-out under fixed CPU allocation.
