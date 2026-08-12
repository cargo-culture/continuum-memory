from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .consolidation import Consolidator, DeterministicConsolidator
from .embeddings import EmbeddingProvider, HashingEmbedder
from .models import (
    ConsolidationResult,
    ContextPacket,
    MemoryInput,
    MemoryKind,
    MemoryPage,
    MemoryRecord,
    MemoryStatus,
    PolicyRecord,
    PolicyState,
    RecallHit,
)
from .retrieval import HybridRetriever, RetrievalConfig
from .store import SQLiteMemoryStore
from .utils import (
    approximate_token_count,
    clamp,
    infer_entities,
    normalize_vector,
    stable_content_hash,
    unique_preserving_order,
    utc_now,
)


class MemoryService:
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedder: EmbeddingProvider | None = None,
        *,
        retrieval_config: RetrievalConfig | None = None,
        strict_embeddings: bool = False,
    ) -> None:
        self.store = store
        self.embedder = embedder or HashingEmbedder()
        self.retriever = HybridRetriever(store, self.embedder, config=retrieval_config)
        self.strict_embeddings = strict_embeddings

    def remember(self, value: MemoryInput) -> MemoryRecord:
        value = self._normalized_input(value)
        content_hash = stable_content_hash(value.namespace, value.kind.value, value.content)
        if not value.allow_duplicate and not value.supersedes_id:
            duplicate = self.store.find_duplicate(value.namespace, content_hash)
            if duplicate is not None:
                return duplicate

        embed_text = self._embedding_text(value)
        embedding: list[float] | None
        embedding_model: str | None
        metadata = dict(value.metadata)
        try:
            embedding = normalize_vector(self.embedder.embed(embed_text))
            embedding_model = self.embedder.model_name
        except Exception as exc:
            if self.strict_embeddings:
                raise
            embedding = None
            embedding_model = None
            metadata["embedding_pending"] = True
            metadata["embedding_error"] = type(exc).__name__

        created_at = utc_now()
        return self.store.insert_memory(
            memory_id=self.store.new_id("mem"),
            namespace=value.namespace,
            kind=value.kind,
            content=value.content,
            summary=value.summary,
            source_uri=value.source_uri,
            source_type=value.source_type,
            importance=value.importance,
            confidence=value.confidence,
            created_at=created_at,
            valid_from=value.valid_from,
            valid_until=value.valid_until,
            token_count=approximate_token_count(value.content),
            supersedes_id=value.supersedes_id,
            contradiction_group=value.contradiction_group,
            metadata=metadata,
            tags=value.tags,
            entities=value.entities,
            content_hash=content_hash,
            embedding=embedding,
            embedding_model=embedding_model,
        )

    def _normalized_input(self, value: MemoryInput) -> MemoryInput:
        content = value.content.strip()
        namespace = value.namespace.strip()
        if not content:
            raise ValueError("memory content cannot be empty")
        if len(content.encode("utf-8")) > 2_000_000:
            raise ValueError("a single memory cannot exceed 2 MB")
        if not namespace or len(namespace) > 200:
            raise ValueError("namespace must contain between 1 and 200 characters")
        kind = value.kind if isinstance(value.kind, MemoryKind) else MemoryKind(value.kind)
        if value.valid_from is not None and value.valid_until is not None:
            if value.valid_until < value.valid_from:
                raise ValueError("valid_until cannot precede valid_from")
        supplied_entities = [entity.strip() for entity in value.entities if entity.strip()]
        inferred = infer_entities(content)
        entities = unique_preserving_order([*supplied_entities, *inferred])[:64]
        tags = unique_preserving_order(tag.strip().casefold() for tag in value.tags if tag.strip())[:64]
        return MemoryInput(
            content=content,
            namespace=namespace,
            kind=kind,
            summary=value.summary.strip() if value.summary and value.summary.strip() else None,
            source_uri=value.source_uri,
            source_type=value.source_type,
            importance=clamp(value.importance),
            confidence=clamp(value.confidence),
            entities=entities,
            tags=tags,
            metadata=dict(value.metadata),
            valid_from=value.valid_from,
            valid_until=value.valid_until,
            contradiction_group=value.contradiction_group,
            supersedes_id=value.supersedes_id,
            allow_duplicate=value.allow_duplicate,
        )

    @staticmethod
    def _embedding_text(value: MemoryInput) -> str:
        parts = [value.kind.value, value.summary or "", value.content]
        if value.entities:
            parts.append("Entities: " + ", ".join(value.entities))
        if value.tags:
            parts.append("Tags: " + ", ".join(value.tags))
        return "\n".join(part for part in parts if part)

    def supersede(
        self,
        memory_id: str,
        new_content: str,
        *,
        confidence: float | None = None,
        source_uri: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        old = self.store.get_memory(memory_id)
        if old is None:
            raise KeyError(f"memory not found: {memory_id}")
        return self.remember(
            MemoryInput(
                namespace=old.namespace,
                kind=old.kind,
                content=new_content,
                source_uri=source_uri or old.source_uri,
                source_type=old.source_type,
                importance=old.importance,
                confidence=old.confidence if confidence is None else confidence,
                entities=old.entities,
                tags=old.tags,
                metadata={**old.metadata, **(metadata or {})},
                valid_from=utc_now(),
                valid_until=old.valid_until,
                contradiction_group=old.contradiction_group,
                supersedes_id=old.id,
                allow_duplicate=True,
            )
        )

    def recall(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 8,
        token_budget: int = 1600,
        context_entities: Sequence[str] | None = None,
        exact_scan_mode: str | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> list[RecallHit]:
        return self.context(
            query,
            namespace=namespace,
            limit=limit,
            token_budget=token_budget,
            context_entities=context_entities,
            exact_scan_mode=exact_scan_mode,
            query_vector=query_vector,
        ).hits

    def context(
        self,
        query: str,
        *,
        namespace: str = "default",
        limit: int = 8,
        token_budget: int = 1600,
        context_entities: Sequence[str] | None = None,
        exact_scan_mode: str | None = None,
        query_vector: Sequence[float] | None = None,
    ) -> ContextPacket:
        return self.retriever.retrieve(
            query,
            namespace=namespace,
            limit=limit,
            token_budget=token_budget,
            context_entities=context_entities,
            exact_scan_mode=exact_scan_mode,
            query_vector=query_vector,
        )

    def page(self, memory_id: str) -> MemoryPage:
        memory = self.store.get_memory(memory_id)
        if memory is None:
            raise KeyError(f"memory not found: {memory_id}")
        outgoing = self.store.get_links(memory_id, "outgoing")
        incoming = self.store.get_links(memory_id, "incoming")
        source_ids = [
            link["target_id"]
            for link in outgoing
            if link["link_type"] in {"summarizes", "derived_from", "supported_by"}
        ]
        return MemoryPage(
            memory=memory,
            outgoing_links=outgoing,
            incoming_links=incoming,
            source_memories=self.store.get_memories(source_ids),
        )

    def consolidate(
        self,
        *,
        namespace: str = "default",
        consolidator: Consolidator | None = None,
        limit: int = 50,
        since: float | None = None,
    ) -> ConsolidationResult:
        source_memories = self.store.list_for_consolidation(namespace, limit=limit, since=since)
        if not source_memories:
            raise ValueError("no active episodic, working, or source memories to consolidate")
        engine = consolidator or DeterministicConsolidator()
        bundle = engine.consolidate(source_memories)
        source_ids = [memory.id for memory in source_memories]
        source_id_set = set(source_ids)
        all_entities = unique_preserving_order(
            entity for memory in source_memories for entity in memory.entities
        )[:64]
        summary = self.remember(
            MemoryInput(
                namespace=namespace,
                kind=MemoryKind.SUMMARY,
                content=bundle.synopsis,
                summary=bundle.synopsis[:500],
                source_type="memory_consolidation",
                importance=0.72,
                confidence=min(memory.confidence for memory in source_memories),
                entities=all_entities,
                tags=["consolidated", "memory-map"],
                metadata={
                    "source_memory_ids": source_ids,
                    "contradiction_notes": bundle.contradiction_notes,
                    "consolidator": type(engine).__name__,
                },
            )
        )
        for source_id in source_ids:
            if source_id != summary.id:
                self.store.add_link(summary.id, source_id, "summarizes")

        claim_ids: list[str] = []
        for claim in bundle.claims:
            claim_sources = [source_id for source_id in claim.source_memory_ids if source_id in source_id_set]
            if not claim_sources:
                continue
            source_confidence = min(
                memory.confidence for memory in source_memories if memory.id in set(claim_sources)
            )
            claim_memory = self.remember(
                MemoryInput(
                    namespace=namespace,
                    kind=claim.kind,
                    content=claim.content,
                    source_type="derived_consolidation",
                    importance=claim.importance,
                    confidence=min(claim.confidence, source_confidence),
                    entities=claim.entities,
                    tags=[*claim.tags, "derived"],
                    metadata={"source_memory_ids": claim_sources, "summary_memory_id": summary.id},
                )
            )
            claim_ids.append(claim_memory.id)
            for source_id in claim_sources:
                if source_id != claim_memory.id:
                    self.store.add_link(claim_memory.id, source_id, "derived_from")

        policy_ids: list[str] = []
        for hypothesis in bundle.policy_hypotheses:
            hypothesis_sources = [
                source_id for source_id in hypothesis.source_memory_ids if source_id in source_id_set
            ]
            if len(hypothesis_sources) < 2:
                continue
            policy = self.store.ensure_policy(
                namespace,
                hypothesis.statement,
                confidence=hypothesis.confidence,
                metadata={"source_memory_ids": hypothesis_sources, "status": "unvalidated_hypothesis"},
            )
            policy_ids.append(policy.id)
        return ConsolidationResult(
            summary_memory_id=summary.id,
            claim_memory_ids=list(dict.fromkeys(claim_ids)),
            policy_ids=list(dict.fromkeys(policy_ids)),
            source_memory_ids=source_ids,
        )

    def record_policy_evidence(
        self,
        statement: str,
        *,
        outcome: int,
        namespace: str = "default",
        source_memory_id: str | None = None,
        note: str | None = None,
    ) -> PolicyRecord:
        policy = self.store.add_policy_evidence(
            namespace,
            statement,
            outcome=outcome,
            source_memory_id=source_memory_id,
            note=note,
        )
        procedural_content = f"Validated behavioral policy: {policy.statement}"
        policy_hash = stable_content_hash(namespace, MemoryKind.PROCEDURAL.value, procedural_content)
        if policy.state == PolicyState.PROMOTED:
            self.remember(
                MemoryInput(
                    namespace=namespace,
                    kind=MemoryKind.PROCEDURAL,
                    content=procedural_content,
                    summary=policy.statement,
                    source_type="behavioral_validation",
                    importance=0.82,
                    confidence=policy.confidence,
                    tags=["promoted-policy", "validated"],
                    metadata={"policy_id": policy.id, "policy_state": policy.state.value},
                )
            )
        elif policy.state == PolicyState.RETIRED:
            memory = self.store.find_duplicate(namespace, policy_hash)
            if memory is not None:
                self.store.set_status(memory.id, MemoryStatus.ARCHIVED)
        return policy

    def list_policies(
        self, namespace: str = "default", *, states: Sequence[PolicyState] | None = None
    ) -> list[PolicyRecord]:
        return self.store.list_policies(namespace, states=states)

    def register_entity_alias(
        self, alias: str, canonical: str, *, namespace: str = "default"
    ) -> None:
        self.store.register_entity_alias(namespace, alias, canonical)

    def reindex_embeddings(self, namespace: str = "default", *, batch_size: int = 64) -> int:
        total = self.store.memory_count(namespace)
        updated = 0
        for offset in range(0, total, batch_size):
            ids = self.store.list_memory_ids(namespace, limit=batch_size, offset=offset)
            memories = self.store.get_memories(ids)
            texts = [
                self._embedding_text(
                    MemoryInput(
                        content=memory.content,
                        namespace=memory.namespace,
                        kind=memory.kind,
                        summary=memory.summary,
                        entities=memory.entities,
                        tags=memory.tags,
                    )
                )
                for memory in memories
            ]
            vectors = [normalize_vector(vector) for vector in self.embedder.embed_many(texts)]
            for memory, vector in zip(memories, vectors):
                self.store.update_embedding(memory.id, vector, self.embedder.model_name)
                updated += 1
        return updated

    def health(self, namespace: str = "default") -> dict[str, Any]:
        return {
            "status": "ok",
            "namespace": namespace,
            "active_memories": self.store.memory_count(namespace),
            "embedding_model": self.embedder.model_name,
            "fts_available": self.store.fts_available,
        }
