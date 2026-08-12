from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .embeddings import EmbeddingProvider
from .models import ContextPacket, MemoryKind, MemoryRecord, RecallFeatures, RecallHit
from .store import SQLiteMemoryStore
from .utils import (
    approximate_token_count,
    canonical_entity,
    expand_query_tokens,
    infer_entities,
    normalize_vector,
    QUERY_STOPWORDS,
    tokenize,
    utc_now,
)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    lexical_weight: float = 0.20
    semantic_weight: float = 0.30
    entity_weight: float = 0.15
    graph_weight: float = 0.08
    recency_weight: float = 0.09
    importance_weight: float = 0.07
    confidence_weight: float = 0.05
    continuity_weight: float = 0.06
    half_life_days: float = 45.0
    candidate_limit: int = 256
    lexical_candidates: int = 96
    vector_candidates: int = 160
    entity_candidates: int = 64
    resident_candidates: int = 32
    diversity_lambda: float = 0.82
    diversity_pool: int = 96
    page_index_size: int = 16
    vector_probe_radius: int = 1
    vector_probe_candidate_trigger: int = 32
    graph_seed_count: int = 24
    graph_candidates: int = 64
    reciprocal_rank_constant: int = 60
    exact_scan_mode: str = "adaptive"
    exact_scan_max_records: int = 15_000
    exact_scan_result_limit: int = 128


class HybridRetriever:
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedder: EmbeddingProvider,
        *,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 8,
        token_budget: int = 1600,
        context_entities: Sequence[str] | None = None,
        include_expired: bool = False,
        exact_scan_mode: str | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> ContextPacket:
        if not namespace.strip():
            raise ValueError("namespace cannot be empty")
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        if token_budget < 32:
            raise ValueError("token_budget must be at least 32")

        started = time.perf_counter()
        now = utc_now()
        raw_query_tokens = tokenize(query)
        content_query_tokens = [
            token for token in raw_query_tokens if token not in QUERY_STOPWORDS
        ]
        query_tokens = expand_query_tokens(content_query_tokens or raw_query_tokens)
        inferred = infer_entities(query)
        context_entities = list(context_entities or [])
        entities = list(dict.fromkeys([*context_entities, *inferred]))

        semantic_available = True
        if query_vector is not None:
            query_vector = normalize_vector(query_vector)
            semantic_available = bool(query_vector)
        else:
            try:
                query_vector = normalize_vector(
                    self.embedder.embed(query if query.strip() else "active persistent context")
                )
            except Exception:
                semantic_available = False
                query_vector = []

        lexical = self.store.search_fts(
            namespace, query_tokens, min(self.config.lexical_candidates, self.config.candidate_limit)
        )
        entity_scores = self.store.search_entities(
            namespace, entities, min(self.config.entity_candidates, self.config.candidate_limit)
        )
        vector_bucket_scores: dict[str, float] = {}
        multiprobe_used = False
        if query_vector:
            vector_bucket_scores = self.store.search_vector_buckets(
                namespace,
                self.embedder.model_name,
                query_vector,
                min(self.config.vector_candidates, self.config.candidate_limit),
                probe_radius=0,
            )
            if (
                self.config.vector_probe_radius == 1
                and len(vector_bucket_scores) < self.config.vector_probe_candidate_trigger
                and not lexical
                and not entity_scores
            ):
                vector_bucket_scores = self.store.search_vector_buckets(
                    namespace,
                    self.embedder.model_name,
                    query_vector,
                    min(self.config.vector_candidates, self.config.candidate_limit),
                    probe_radius=1,
                )
                multiprobe_used = True
        resident = self.store.recent_candidates(namespace, self.config.resident_candidates)
        resident_scores = {
            memory_id: 1.0 / (1.0 + rank) for rank, memory_id in enumerate(resident)
        }

        exact_scan_used = False
        exact_scan_skipped_limit = False
        if query_vector and query.strip():
            mode = (exact_scan_mode or self.config.exact_scan_mode).casefold()
            if mode not in {"off", "adaptive", "always"}:
                raise ValueError("exact_scan_mode must be off, adaptive, or always")
            active_count = self.store.memory_count(namespace)
            # Adaptive exact scan is reserved for genuinely semantic-only queries.
            # Lexical/entity evidence already supplies a bounded, indexed path.
            weak_candidates = not lexical and not entity_scores
            should_scan = mode == "always" or (mode == "adaptive" and weak_candidates)
            if should_scan and active_count <= self.config.exact_scan_max_records:
                exact_scores = self.store.exact_vector_candidates(
                    namespace,
                    self.embedder.model_name,
                    query_vector,
                    scan_limit=self.config.exact_scan_max_records,
                    result_limit=self.config.exact_scan_result_limit,
                )
                for memory_id, score in exact_scores.items():
                    vector_bucket_scores[memory_id] = max(
                        vector_bucket_scores.get(memory_id, 0.0), score
                    )
                vector_bucket_scores = dict(
                    sorted(vector_bucket_scores.items(), key=lambda item: item[1], reverse=True)
                )
                exact_scan_used = True
            elif should_scan:
                exact_scan_skipped_limit = True

        initial_sources = {
            "lexical": lexical,
            "vector": vector_bucket_scores,
            "entity": entity_scores,
            "resident": resident_scores,
        }
        initial_ids = self._fuse_candidate_sources(
            initial_sources, limit=self.config.candidate_limit
        )
        graph_scores = self.store.linked_candidates(
            namespace,
            initial_ids[: self.config.graph_seed_count],
            limit=self.config.graph_candidates,
        )
        candidate_ids = self._fuse_candidate_sources(
            {**initial_sources, "graph": graph_scores}, limit=self.config.candidate_limit
        )
        records = self.store.get_memories(candidate_ids)
        scored: list[RecallHit] = []
        half_life_seconds = max(1.0, self.config.half_life_days * 86400.0)
        context_entity_set = {canonical_entity(entity) for entity in context_entities}

        for memory in records:
            if not include_expired:
                if memory.valid_from is not None and memory.valid_from > now:
                    continue
                if memory.valid_until is not None and memory.valid_until < now:
                    continue
            semantic = 0.0
            if (
                query_vector
                and memory.embedding
                and memory.embedding_model == self.embedder.model_name
                and len(query_vector) == len(memory.embedding)
            ):
                semantic = max(0.0, min(1.0, sum(a * b for a, b in zip(query_vector, memory.embedding))))
            age = max(0.0, now - memory.updated_at)
            recency = math.exp(-math.log(2.0) * age / half_life_seconds)
            memory_entities = {canonical_entity(entity) for entity in memory.entities}
            explicit_overlap = (
                len(context_entity_set & memory_entities) / max(1, len(context_entity_set))
                if context_entity_set
                else 0.0
            )
            continuity = self._continuity(memory, explicit_overlap)
            packet_tokens = self._packet_cost(memory)
            token_efficiency = 1.0 / math.sqrt(1.0 + packet_tokens / 128.0)
            features = RecallFeatures(
                lexical=lexical.get(memory.id, 0.0),
                semantic=semantic,
                entity=max(entity_scores.get(memory.id, 0.0), explicit_overlap),
                graph=graph_scores.get(memory.id, 0.0),
                recency=recency,
                importance=memory.importance,
                confidence=memory.confidence,
                continuity=continuity,
                token_efficiency=token_efficiency,
            )
            score = self._score(features)
            scored.append(RecallHit(memory=memory, score=score, features=features))

        scored.sort(key=lambda item: (item.score, item.memory.updated_at), reverse=True)
        selected, used_tokens = self._select_diverse(scored, limit=limit, token_budget=token_budget)
        for rank, hit in enumerate(selected, start=1):
            hit.rank = rank

        selected_ids = [hit.memory.id for hit in selected]
        selected_set = set(selected_ids)
        linked_pages = list(
            self.store.linked_candidates(
                namespace, selected_ids, limit=self.config.page_index_size
            ).keys()
        )
        detailed_pages = [
            hit.memory.id
            for hit in selected
            if hit.memory.kind in {MemoryKind.SUMMARY, MemoryKind.REFLECTIVE, MemoryKind.SOURCE}
        ]
        overflow_pages = [hit.memory.id for hit in scored if hit.memory.id not in selected_set]
        page_ids = list(
            dict.fromkeys([*linked_pages, *detailed_pages, *overflow_pages])
        )[: self.config.page_index_size]
        retrieval_id = self.store.new_id("ret")
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.store.touch_memories(selected_ids, now)
        self.store.record_retrieval(
            retrieval_id=retrieval_id,
            namespace=namespace,
            query=query,
            selected_ids=selected_ids,
            candidates=len(scored),
            token_budget=token_budget,
            used_tokens=used_tokens,
            latency_ms=latency_ms,
            features={
                "semantic_available": semantic_available,
                "embedding_model": self.embedder.model_name,
                "lexical_candidates": len(lexical),
                "vector_candidates": len(vector_bucket_scores),
                "entity_candidates": len(entity_scores),
                "graph_candidates": len(graph_scores),
                "exact_scan_used": exact_scan_used,
                "exact_scan_skipped_limit": exact_scan_skipped_limit,
                "multiprobe_used": multiprobe_used,
            },
        )
        return ContextPacket(
            namespace=namespace,
            query=query,
            generated_at=now,
            token_budget=token_budget,
            used_tokens=used_tokens,
            hits=selected,
            page_ids=page_ids,
            retrieval_id=retrieval_id,
        )

    def _score(self, features: RecallFeatures) -> float:
        config = self.config
        base = (
            config.lexical_weight * features.lexical
            + config.semantic_weight * features.semantic
            + config.entity_weight * features.entity
            + config.graph_weight * features.graph
            + config.recency_weight * features.recency
            + config.importance_weight * features.importance
            + config.confidence_weight * features.confidence
            + config.continuity_weight * features.continuity
        )
        return base * (0.86 + 0.14 * features.token_efficiency)

    def _fuse_candidate_sources(
        self, sources: dict[str, dict[str, float]], *, limit: int
    ) -> list[str]:
        source_weights = {
            "lexical": 1.00,
            "vector": 1.20,
            "entity": 1.10,
            "resident": 0.35,
            "graph": 0.80,
        }
        fused: dict[str, float] = {}
        for source_name, candidates in sources.items():
            weight = source_weights.get(source_name, 1.0)
            for rank, (memory_id, source_score) in enumerate(candidates.items(), start=1):
                reciprocal_rank = weight / (self.config.reciprocal_rank_constant + rank)
                score_tiebreak = weight * max(0.0, source_score) * 1e-4
                fused[memory_id] = fused.get(memory_id, 0.0) + reciprocal_rank + score_tiebreak
        return [
            memory_id
            for memory_id, _ in sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
        ]

    @staticmethod
    def _continuity(memory: MemoryRecord, explicit_overlap: float) -> float:
        kind_base = {
            MemoryKind.WORKING: 1.0,
            MemoryKind.IDENTITY: 0.9,
            MemoryKind.PROCEDURAL: 0.8,
            MemoryKind.REFLECTIVE: 0.6,
            MemoryKind.SUMMARY: 0.5,
        }.get(memory.kind, 0.2)
        return min(1.0, max(kind_base, explicit_overlap))

    @staticmethod
    def _packet_cost(memory: MemoryRecord) -> int:
        return approximate_token_count(memory.summary or memory.content) + 28

    def _select_diverse(
        self, scored: list[RecallHit], *, limit: int, token_budget: int
    ) -> tuple[list[RecallHit], int]:
        # MMR is quadratic in the selected count. Relevance outside this bounded
        # pool is still exposed through page_ids, but does not consume hot-path CPU.
        remaining = list(scored[: self.config.diversity_pool])
        token_sets = {
            hit.memory.id: set(tokenize(hit.memory.summary or hit.memory.content)) for hit in remaining
        }
        max_redundancy = {hit.memory.id: 0.0 for hit in remaining}
        selected: list[RecallHit] = []
        used_tokens = 0
        while remaining and len(selected) < limit:
            best_index = -1
            best_adjusted = float("-inf")
            for index, hit in enumerate(remaining):
                cost = self._packet_cost(hit.memory)
                if used_tokens + cost > token_budget:
                    continue
                adjusted = (
                    self.config.diversity_lambda * hit.score
                    - (1.0 - self.config.diversity_lambda) * max_redundancy[hit.memory.id]
                )
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_index = index
            if best_index < 0:
                break
            chosen = remaining.pop(best_index)
            selected.append(chosen)
            used_tokens += self._packet_cost(chosen.memory)
            for candidate in remaining:
                similarity = self._memory_similarity(
                    candidate.memory,
                    chosen.memory,
                    token_sets[candidate.memory.id],
                    token_sets[chosen.memory.id],
                )
                max_redundancy[candidate.memory.id] = max(
                    max_redundancy[candidate.memory.id], similarity
                )
        return selected, used_tokens

    @staticmethod
    def _memory_similarity(
        left: MemoryRecord,
        right: MemoryRecord,
        left_tokens: set[str],
        right_tokens: set[str],
    ) -> float:
        if (
            left.embedding
            and right.embedding
            and left.embedding_model == right.embedding_model
            and len(left.embedding) == len(right.embedding)
        ):
            # All embeddings are normalized at the service boundary.
            vector_similarity = max(
                0.0, min(1.0, sum(a * b for a, b in zip(left.embedding, right.embedding)))
            )
        else:
            vector_similarity = 0.0
        union = left_tokens | right_tokens
        lexical_similarity = len(left_tokens & right_tokens) / len(union) if union else 0.0
        if left.contradiction_group and left.contradiction_group == right.contradiction_group:
            return min(0.45, max(vector_similarity, lexical_similarity))
        return max(vector_similarity, lexical_similarity)
