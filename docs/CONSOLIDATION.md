# Merge-on-Consolidate

*Design spec. Turning the memory store from an append-only list into something
that forms priors.*

Status: proposed, unimplemented. Reconciled against the schema in
`src/continuum_memory/store.py`, the adapter protocol in
`src/continuum_memory/consolidation.py`, and the ranking features in
`src/continuum_memory/retrieval.py` as of schema version 2. Where an existing
mechanism already provides what this design needs, the spec now names it rather
than proposing a parallel one.

---

## 1. Problem

The store currently appends. Three hundred structurally similar episodes produce
three hundred rows, each competing for the same retrieval budget. Retrieval then
selects the top-k most similar, which for a saturated topic means returning
several near-duplicates and spending the token budget on redundancy.

A parametric memory does not behave this way. Three hundred similar experiences
produce one strengthened generalization, and the episodes are discarded. The
useful properties that follows from are:

- **Interference as a feature.** Similar traces blend into structure rather than
  accumulating side by side.
- **Graded magnitude.** A memory has strength, not mere presence.
- **Density under pressure.** Adding experience makes the representation better,
  not longer.

The goal of this spec is to obtain those three properties without gradients,
and without discarding the audit and correction properties the store currently
has.

Explicit non-goals: weight modification, unconstrained self-modification, and any
claim that the resulting system is more than experience-conditioned behaviour in
external state. This spec follows the framing already stated in the README.

---

## 2. Where merging happens

**Decision: in `consolidate`, not on the write path.**

Merge-on-write would require, for every `remember` call, a nearest-neighbour
lookup plus a probable model call to generalize. That converts a ~1.4 ms write
into a network round trip and makes ingestion failure-prone at exactly the moment
the caller is least able to handle it. `MemoryService.remember` is currently a
single `insert_memory` with no neighbour lookup, and should stay that way.

Deferred merging keeps writes append-only and fast, and puts the expensive
generalization step in a batch pass that already exists, already has model
adapters wired in, and can be scheduled, retried, and audited as a unit.

The cost is latency to prior formation: a cluster of episodes does not become an
abstraction until the next consolidation run. This is acceptable and arguably
correct — a disposition that forms from a single afternoon's episodes is a
disposition formed on too little evidence.

---

## 3. Merge semantics

### 3.1 Candidate selection

Within a namespace, and within a single `kind`, cluster active memories by
embedding similarity above a threshold `MERGE_TAU`. Clusters below
`MERGE_MIN_CLUSTER` members are left alone.

Candidate generation should reuse the existing vector index rather than scanning
pairwise: `store.search_vector_buckets` for approximate neighbours, with
`store.exact_vector_candidates` as the bounded fallback under the same
`exact_scan_max_records` limit retrieval already respects. Embeddings are
normalized at the service boundary, so cosine similarity is a dot product.

Constraints on clustering:

- **Never merge across `kind`.** An episodic trace and a promoted procedural
  policy are different objects with different lifecycles. Note that
  `list_for_consolidation` selects only `working`, `episodic`, and `source`
  memories; merge candidate selection needs its own query, since abstractions
  are themselves merge candidates (§4.3) and are not of those kinds.
- **Never merge across namespace.** Federation boundaries are semantic, not just
  partitioning. Across *books* this is already impossible by construction —
  `FederatedMemoryService` gives each book its own SQLite file — so only the
  within-store namespace check needs enforcing.
- **Never merge a memory that is tombstoned, or one referenced by
  `policy_evidence.source_memory_id` for a policy in `testing` state.** There is
  no "under supersession review" status in the schema; the status enum is
  `active`, `superseded`, `archived`, `tombstoned`, and clustering already
  restricts itself to `active`.
- **Do not merge across a temporal-validity contradiction.** If two memories
  assert incompatible facts about the same entity at different times, that is a
  supersession case, not a merge case. Route it to the existing path. The
  `contradiction_group` column is the key: treat a shared non-null
  `contradiction_group` as a hard merge exclusion. Retrieval already assumes
  these semantics — `_memory_similarity` caps same-group pairs at `0.45` so MMR
  does not treat contradictory memories as redundant.

**`MERGE_TAU` is per embedding model, not a global constant.** The default
embedding provider is `hashing`, and `HashingEmbedder` supplies
lexical-neighborhood vectors, not semantic ones — its own docstring says so. A
cosine of `0.88` under that embedder measures token and trigram overlap, so
clustering would group near-duplicate *wordings* and miss exactly the
paraphrase-spread cluster this design exists to collapse. Two consequences:

- Store `MERGE_TAU` per `embedding_model` and refuse to apply a threshold tuned
  for one model to vectors produced by another. The store already records
  `embedding_model` per row and already refuses to compare across models in
  retrieval.
- Under the `hashing` provider, restrict merge to the deterministic no-rewrite
  path (§3.2). Content-rewriting generalization on lexical vectors produces
  abstractions over clusters that were never semantically coherent.

Suggested starting values, for a real semantic embedder: `MERGE_TAU = 0.88`,
`MERGE_MIN_CLUSTER = 3`. Both need tuning against real data; see §7.

### 3.2 Producing the abstraction

The generalization step is a schema-constrained model call. It does **not** fit
the existing `Consolidator` protocol, which is
`consolidate(memories) -> ConsolidationBundle`. Add a second protocol rather than
overloading that one:

```python
class Merger(Protocol):
    def generalize(self, cluster: list[MemoryRecord]) -> Abstraction: ...
```

Output, constrained:

```
{
  "content":        str,    # the generalization
  "confidence":     float,  # not inherited; assessed over the cluster
  "entities":       [str],  # union, deduplicated
  "covers":         [str],  # member ids the abstraction actually accounts for
  "outliers":       [str]   # members it does not — these stay active
}
```

Member ids are strings, not integers: ids are `f"{prefix}_{uuid4().hex}"`. As in
`OpenAIConsolidator`, every returned id must be validated against the input
cluster before use, and an abstraction whose `covers` set contains an unknown id
must be discarded rather than partially applied.

The `outliers` field matters. A clustering threshold will sometimes group a
member that does not belong. The adapter must be able to decline to absorb it,
and declined members remain active and unmerged rather than being silently
folded into a generalization that misrepresents them.

The deterministic path cannot generalize, and cannot reuse
`DeterministicConsolidator` — that class returns `claims=[]` unconditionally and
produces an extractive synopsis, which is a different operation. The
deterministic merger is new code: strength accumulation and field union with no
content rewrite, where the cluster's highest-confidence member becomes the
representative. This is weaker but not wrong, and it keeps offline installs
functional.

### 3.3 Strength

The abstraction carries:

- `observation_count` — number of absorbed episodes
- `strength` — a decaying scalar, see §5
- `first_observed`, `last_observed` — bounding timestamps of the cluster

`observation_count` is the closest analogue to weight magnitude the store can
hold. It should influence retrieval ranking as a distinct feature, not be folded
into `confidence` — confidence is *how sure we are the claim is true*, count is
*how much experience it rests on*. These come apart routinely.

Adding it as a distinct feature is not free. `RecallFeatures` gains a ninth
field, and `RetrievalConfig`'s eight weights currently sum to exactly `1.00`, so
introducing a strength weight means redistributing the existing ones. That
invalidates the measured numbers in `docs/RETRIEVAL_HARDENING.md`, which must be
re-run rather than carried forward.

---

## 4. Provenance under merge

This is the part that constrains the rest of the design.

Merging is lossy by nature, and lossiness is precisely the property being sought.
But the store's most valuable safety features — supersession, tombstoning, the
append-only audit stream — all depend on there being a row to point at. Once
three hundred episodes have become one abstraction, a false belief among them can
no longer be individually retracted.

**Decision: archive, do not delete. Preserve back-pointers. Re-derive on
contradiction.**

### 4.1 Archival

Most of what this section originally proposed already exists. `MemoryStatus.ARCHIVED`
gives exactly the cold semantics required: every candidate query in the store
filters `status = 'active'`, and `set_status` removes the row from the FTS index
while retaining content, embedding, and metadata. **Do not add a new status
value.** Note that `ARCHIVED` is already used for the procedural memory of a
retired policy, so `absorbed_into` becomes the discriminator between the two
kinds of cold row.

Absorbed episodes are therefore moved to `ARCHIVED` — not deleted, not
tombstoned — and carry `absorbed_into = <abstraction_id>`.

The back-pointer in the other direction should reuse `memory_links` with the
existing `derived_from` link type rather than adding a `derived_from` column.
That buys three things already built:

- `add_link` rejects cross-namespace links, enforcing §3.1's namespace rule by
  construction.
- `linked_candidates` filters `status = 'active'`, so archived members cannot
  re-enter retrieval through one-hop graph expansion.
- `derived_from` is already one of the three link types graph expansion and
  page indexing traverse, so provenance surfacing needs no new plumbing.

One correction is required before this is safe. `MemoryService.page` collects
`derived_from` targets and calls `store.get_memories`, which has **no status
filter** — so the MCP "Read memory" tool on an abstraction over three hundred
episodes would return three hundred full records. Either cap the number of
source memories a page returns, or exclude archived members from `page` and
expose them only through the re-derivation read path.

The relationship is recorded in the audit stream as a single merge event
containing the cluster membership, the threshold used, the embedding model the
threshold was tuned for, the adapter and model that produced the generalization,
and the outlier decisions.

Storage cost is real and should be stated plainly: this design trades disk for
correctability. Cold storage can be compacted (embeddings dropped, content
compressed) since cold rows are only read during re-derivation. Dropping an
embedding must also delete the row's `vector_buckets` entries, as
`update_embedding` does.

### 4.2 Re-derivation

When a contradiction arrives — a supersession event, an explicit retraction, or
a policy retirement — targeting an abstraction:

1. Wake the `derived_from` set from cold storage.
2. Apply the retraction to the specific member(s) it names. If the retraction
   cannot be attributed to specific members, mark the whole abstraction
   `confidence_degraded` and stop; do not guess.
3. Re-run generalization over the surviving members.
4. Write the new abstraction, supersede the old one through the existing
   `supersedes_id` path. The new abstraction must be written with
   `allow_duplicate=True`, since a re-derivation that produces identical content
   would otherwise be swallowed by the content-hash dedupe in `remember`.
5. Emit a re-derivation event carrying both abstraction ids and the retracted
   member ids.

Re-derivation is expensive. That is correct — it should be rare, and its cost
should be visible rather than amortized silently into normal operation.

### 4.3 Merge depth

Abstractions may themselves be merged. `derived_from` chains must be traversable
transitively during re-derivation, which means:

- cascade depth must be bounded (`MAX_MERGE_DEPTH`, suggest 3)
- cycles must be impossible — nothing in `memory_links` prevents them, so this
  must be enforced at merge time: an abstraction may only absorb members
  strictly older than itself
- re-derivation at depth *n* triggers re-derivation of all dependents

Without the depth bound, a single retraction against a deeply nested abstraction
can trigger an unbounded cascade. Bound it and accept that very old memory
becomes effectively immutable — which is, not incidentally, also true of weights.

---

## 5. Decay and reconsolidation

Two mechanisms, both cheap, both increasing weight-likeness.

**Decay.** `strength` declines on a time constant since `last_accessed`. Decay
does not delete; it lowers retrieval priority. A memory decayed below
`FLOOR_STRENGTH` becomes a merge candidate at a lower threshold, since weak
traces should blend more readily than strong ones.

This is complementary to, not a duplicate of, the existing `recency` feature,
which is keyed on `updated_at` rather than `last_accessed` — age since write, not
age since use.

That distinction creates a problem merging must handle. A freshly written
abstraction over year-old episodes gets a full recency score, because `recency`
is `exp(-ln2 · (now - updated_at) / half_life)` and the merge just set
`updated_at`. Every re-derivation refreshes it again, so each retraction makes
the memory look newer. Either the abstraction must carry an effective age
derived from `last_observed` for the purpose of the recency feature, or `strength`
must be keyed on something merge does not touch. Do not leave both keyed on write
time.

**Reconsolidation.** Retrieval-and-use restores strength. This requires the
caller to report use, which is the same attestation problem described in §6 —
if the model reports its own use, the signal is unreliable. Interim position:
count retrieval-into-context as weak evidence of use and weight it low.

The instrumentation for that interim position already exists. `touch_memories`
sets `last_accessed` and increments `access_count` for every selected hit on
every retrieval, and it is called with the ids that entered the packet — not
the ones the model demonstrably used. That is precisely the weak signal
described above, already recorded. Only the strength function consuming it is
missing.

The stronger and more interesting version — recall *rewrites* the trace, as in
human reconsolidation — is deliberately deferred. It makes every read a
potential write, which is a substantial change to the concurrency and audit
model, and it should not be bundled with this work.

---

## 6. Outcome attestation

Flagged here because it is the highest-risk open problem in the system and it
interacts directly with merging.

The behavioural policy lifecycle promotes a statement after three decisive
successes at a rate ≥ 0.75 (`promotion_evidence = 3`, `promotion_rate = 0.75` in
`add_policy_evidence`). Two concerns:

**The threshold is thin.** Three observations is very little evidence for a
disposition that will then materialize as a procedural memory, participate in
retrieval, and thereby influence the outcomes that confirm it. That is a
positive feedback loop with an early-accident failure mode. Recommend raising
the count, adding a minimum time-span requirement so three successes in one
session cannot promote, and adding decay so an unreinforced promoted policy
drifts back toward `testing`. No such decay exists today: state only ever
advances on new evidence.

**Self-scoring is unsound — and is a door this system has not yet opened.** The
MCP surface exposes five tools: search, read, list books, remember, and
supersede. `record_policy_evidence` is reachable only from the HTTP API and the
CLI, so today the developmental record is written by the operator or harness,
not by the model scoring itself.

That is the current state, and it is the right one. If the model recorded its own
`policy-evidence success`, the developmental record would be written by a party
with no reliable access to its own performance and a systematic bias toward
judging its own outputs favourably. Merging compounds this: bad evidence does not
just sit in a row, it gets generalized into a prior.

So the provenance distinction below should be implemented *before* anything
exposes policy scoring to the model, not retrofitted afterwards. Options, roughly
in order of preference:

1. External or delayed scoring — a separate pass, ideally a different model or a
   human, judging outcomes after the fact.
2. Outcome signals grounded in something non-self-reported — task completion,
   test results, user correction events, explicit user feedback.
3. If self-scoring is ever exposed, mark such evidence with a distinct provenance
   flag, weight it lower, and require a higher promotion threshold for policies
   supported only by self-attested evidence.

Do not let self-attested evidence and externally grounded evidence flow into the
same counter undifferentiated.

---

## 7. Evaluation

The existing synthetic recall benchmark cannot evaluate this work. It measures
exact-marker retrieval latency, and the README already says so. Merging changes
*what is stored*, so it must be evaluated on semantic outcomes.

The paraphrase tests below cannot be run under the default `hashing` embedder for
reasons unrelated to merge quality (§3.1). Evaluation requires a real semantic
embedder, and results must be reported per embedding model.

Minimum viable evaluation:

- **Paraphrase recall.** Query a merged abstraction using wording that matches
  none of the absorbed episodes. Does it surface?
- **Compression ratio.** Rows before and after consolidation, against retrieval
  quality on a held-out query set. Merging that improves compression while
  degrading recall is a loss.
- **Outlier fidelity.** Inject a semantically distinct member into a tight
  cluster. Does the adapter route it to `outliers` rather than absorbing it?
- **Re-derivation correctness.** Merge, retract a member, re-derive. Does the
  resulting abstraction no longer entail the retracted claim?
- **Contradiction routing.** Two temporally incompatible memories about one
  entity must reach supersession, never merge.
- **Cascade bound.** Retract at depth `MAX_MERGE_DEPTH`. Does the cascade
  terminate?
- **Ranking regression.** Re-run the `docs/RETRIEVAL_HARDENING.md` suite after
  the weight redistribution in §3.3. Hit@5 must not regress.

`MERGE_TAU` should be swept rather than assumed, per embedding model. The failure
mode of a low threshold is silent semantic corruption — distinct facts blended
into a plausible-sounding abstraction that is true of nothing. It will not show
up as an error. It shows up as retrieval that feels vaguely right and is wrong in
specifics, which is the hardest class of bug to notice and the most damaging to
trust in the store.

---

## 8. Implementation order

1. Schema: `absorbed_into`, `observation_count`, `strength`, `first_observed`,
   `last_observed`. No new status value — `ARCHIVED` is the cold state — and no
   `derived_from` column, since `memory_links` already carries that link type.
   Fix the unfiltered `get_memories` call in `MemoryService.page` (§4.1) in this
   step, before anything can create a three-hundred-member provenance set.
2. Clustering pass with all §3.1 exclusions, over the existing vector-bucket
   index. No merging yet — emit proposed clusters for inspection and eyeball them
   against real data, per embedding model.
3. Deterministic merge only: strength accumulation, field union, archival, audit
   events. Verifiable without model calls, and the only merge path permitted
   under the `hashing` embedder.
4. Adapter-backed generalization behind the new `Merger` protocol, including
   outlier handling and id validation.
5. Re-derivation path, with the cascade bound.
6. Decay and strength-weighted ranking. This redistributes `RetrievalConfig`
   weights and invalidates the published retrieval benchmarks; re-run them as
   part of this step rather than deferring to §7.
7. Evaluation suite per §7, then sweep `MERGE_TAU` per embedding model.

Steps 1–3 are safe and reversible: nothing is lost, and archival is undoable.
Step 4 is where information starts being destroyed. Do not proceed past step 3
without step 2's clusters having been inspected by hand.

---

## 9. Open questions

- Should abstractions be retrievable *alongside* their outliers, or does the
  abstraction shadow them? Shadowing is denser; alongside is safer.
- Does `identity` participate in merging at all, or is it exempt? An always-
  resident identity book that self-modifies through generalization is a
  persistent prompt-injection surface, and the existing untrusted-data framing
  is harder to enforce for content that is injected every session regardless of
  query. Recommend exempting `identity` from merge until this is thought
  through properly.

  Note that the exposure is already wider than merge. `OpenAIConsolidator`
  allows `identity` among its claim kinds, and the identity book is
  `always_search=True`, so the existing consolidation path can already mint
  identity-kind claims that are injected every session. That deserves scrutiny
  on its own schedule, independent of this work.
- Is there a floor below which a namespace is too small to merge? Merging over
  a handful of memories generalizes from noise.
- Should `observation_count` be visible to the model in rendered packets? It is
  useful calibration information and also a thing the model may reason about
  incorrectly.
