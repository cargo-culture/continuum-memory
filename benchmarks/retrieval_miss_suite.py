#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import statistics
import tempfile
import time
from pathlib import Path

from continuum_memory import MemoryInput, MemoryKind, MemoryService, SQLiteMemoryStore
from continuum_memory.retrieval import RetrievalConfig
from continuum_memory.utils import canonical_entity, infer_entities, lsh_buckets, normalize_vector, tokenize


class BrokenEmbedder:
    model_name = "broken"
    dimension = 64

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding unavailable")

    def embed_many(self, texts):
        raise RuntimeError("embedding unavailable")


class MappingEmbedder:
    model_name = "mapping-v1"
    dimension = 64

    def embed(self, text: str) -> list[float]:
        if "TARGET-SEMANTIC" in text or text == "meaning preserving paraphrase":
            return [1.0] + [0.0] * 63
        return [0.0, 1.0] + [0.0] * 62

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


class FixedEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.model_name = "probe-test"
        self.dimension = len(vector)

    def embed(self, text: str) -> list[float]:
        return list(self.vector)

    def embed_many(self, texts):
        return [list(self.vector) for _ in texts]


def legacy_entity_candidates(
    store: SQLiteMemoryStore, namespace: str, entities: list[str], limit: int
) -> dict[str, float]:
    canonicals = list(dict.fromkeys(canonical_entity(entity) for entity in entities if entity.strip()))
    if not canonicals:
        return {}
    placeholders = ",".join("?" for _ in canonicals)
    with store._lock:
        rows = store._connection.execute(
            f"""
            SELECT memory_entities.memory_id, count(*) AS matches
            FROM memory_entities
            JOIN memories ON memories.id = memory_entities.memory_id
            WHERE memory_entities.namespace = ?
              AND memory_entities.canonical IN ({placeholders})
              AND memories.status = 'active'
            GROUP BY memory_entities.memory_id
            ORDER BY matches DESC, memories.updated_at DESC LIMIT ?
            """,
            [namespace, *canonicals, limit],
        ).fetchall()
    return {row["memory_id"]: row["matches"] / max(1, len(canonicals)) for row in rows}


def legacy_candidates(
    service: MemoryService,
    query: str,
    namespace: str,
    config: RetrievalConfig,
    context_entities: list[str] | None = None,
) -> list[str]:
    lexical = service.store.search_fts(namespace, tokenize(query), config.lexical_candidates)
    entities = [*(context_entities or []), *infer_entities(query)]
    entity_scores = legacy_entity_candidates(
        service.store, namespace, entities, config.entity_candidates
    )
    try:
        query_vector = normalize_vector(service.embedder.embed(query or "active persistent context"))
        vectors = service.store.search_vector_buckets(
            namespace,
            service.embedder.model_name,
            query_vector,
            config.vector_candidates,
            probe_radius=0,
        )
    except Exception:
        vectors = {}
    resident = service.store.recent_candidates(namespace, config.resident_candidates)
    return list(dict.fromkeys([*lexical, *vectors, *entity_scores, *resident]))[
        : config.candidate_limit
    ]


def timed_context(service: MemoryService, *args, **kwargs):
    started = time.perf_counter()
    packet = service.context(*args, **kwargs)
    return packet, (time.perf_counter() - started) * 1000.0


def candidate_saturation_case() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteMemoryStore(Path(directory) / "fusion.db")
        config = RetrievalConfig(
            candidate_limit=4,
            lexical_candidates=4,
            vector_candidates=4,
            entity_candidates=4,
            resident_candidates=0,
            exact_scan_mode="off",
            graph_seed_count=0,
            graph_candidates=0,
        )
        service = MemoryService(store, BrokenEmbedder(), retrieval_config=config)
        for index in range(12):
            service.remember(MemoryInput(namespace="n", content=f"Alpha common distractor {index}."))
        target = service.remember(
            MemoryInput(
                namespace="n",
                kind=MemoryKind.IDENTITY,
                content="Alpha identity record for the concealed subject.",
                entities=["Needle Entity"],
                importance=1.0,
            )
        )
        legacy = target.id in legacy_candidates(service, "alpha", "n", config, ["Needle Entity"])
        packet, latency = timed_context(
            service, "alpha", namespace="n", limit=1, context_entities=["Needle Entity"]
        )
        improved = packet.hits[0].memory.id == target.id
        store.close()
    return {"case": "candidate_source_saturation", "legacy_hit": legacy, "improved_hit": improved, "latency_ms": latency}


def morphology_case() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteMemoryStore(Path(directory) / "morph.db")
        config = RetrievalConfig(resident_candidates=1, exact_scan_mode="off", graph_seed_count=0)
        service = MemoryService(store, BrokenEmbedder(), retrieval_config=config)
        target = service.remember(MemoryInput(namespace="n", content="The deployment schedule is locked."))
        for index in range(5):
            service.remember(MemoryInput(namespace="n", content=f"Unrelated note {index}."))
        legacy = target.id in legacy_candidates(service, "deployments", "n", config)
        packet, latency = timed_context(service, "deployments", namespace="n", limit=3)
        improved = target.id in {hit.memory.id for hit in packet.hits}
        store.close()
    return {"case": "morphological_variation", "legacy_hit": legacy, "improved_hit": improved, "latency_ms": latency}


def alias_case() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteMemoryStore(Path(directory) / "alias.db")
        config = RetrievalConfig(resident_candidates=1, exact_scan_mode="off")
        service = MemoryService(store, BrokenEmbedder(), retrieval_config=config)
        target = service.remember(
            MemoryInput(
                namespace="n",
                content="The Zeta contract was renewed for another year.",
                entities=["International Business Machines"],
            )
        )
        for index in range(8):
            service.remember(MemoryInput(namespace="n", content=f"Unrelated record {index}."))
        service.register_entity_alias("IBM", "International Business Machines", namespace="n")
        legacy = target.id in legacy_candidates(service, "", "n", config, ["IBM"])
        packet, latency = timed_context(
            service, "", namespace="n", limit=1, context_entities=["IBM"]
        )
        improved = packet.hits[0].memory.id == target.id
        store.close()
    return {"case": "entity_alias", "legacy_hit": legacy, "improved_hit": improved, "latency_ms": latency}


def exact_fallback_case() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteMemoryStore(Path(directory) / "fallback.db")
        config = RetrievalConfig(
            resident_candidates=4,
            exact_scan_mode="adaptive",
            exact_scan_max_records=100,
            exact_scan_result_limit=16,
            graph_seed_count=0,
        )
        service = MemoryService(store, MappingEmbedder(), retrieval_config=config)
        target = service.remember(
            MemoryInput(namespace="n", content="TARGET-SEMANTIC: an unguessable proposition.")
        )
        with store.transaction() as connection:
            connection.execute("DELETE FROM vector_buckets WHERE memory_id = ?", (target.id,))
        for index in range(20):
            service.remember(MemoryInput(namespace="n", content=f"Unrelated archive item {index}."))
        legacy = target.id in legacy_candidates(
            service, "meaning preserving paraphrase", "n", config
        )
        packet, latency = timed_context(
            service, "meaning preserving paraphrase", namespace="n", limit=3
        )
        improved = packet.hits[0].memory.id == target.id
        store.close()
    return {"case": "adaptive_exact_vector_fallback", "legacy_hit": legacy, "improved_hit": improved, "latency_ms": latency}


def graph_case() -> dict:
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteMemoryStore(Path(directory) / "graph.db")
        config = RetrievalConfig(resident_candidates=1, exact_scan_mode="off", graph_seed_count=8)
        service = MemoryService(store, BrokenEmbedder(), retrieval_config=config)
        source = service.remember(
            MemoryInput(namespace="n", content="The selected cryptographic key was amber.")
        )
        for index in range(20):
            service.remember(MemoryInput(namespace="n", content=f"Unrelated note {index}."))
        summary = service.remember(
            MemoryInput(
                namespace="n",
                kind=MemoryKind.SUMMARY,
                content="Routing outcome and key-selection decision.",
            )
        )
        store.add_link(summary.id, source.id, "summarizes")
        legacy = source.id in legacy_candidates(service, "routing outcome", "n", config)
        packet, latency = timed_context(service, "routing outcome", namespace="n", limit=2)
        improved = source.id in {hit.memory.id for hit in packet.hits} or source.id in packet.page_ids
        store.close()
    return {"case": "linked_source_evidence", "legacy_hit": legacy, "improved_hit": improved, "latency_ms": latency}


def multiprobe_case() -> dict:
    randomizer = random.Random(8192)
    query = normalize_vector([randomizer.uniform(-1.0, 1.0) for _ in range(64)])
    query_buckets = dict(lsh_buckets(query))
    neighbor = None
    for _ in range(20_000):
        candidate = normalize_vector([randomizer.uniform(-1.0, 1.0) for _ in range(64)])
        candidate_buckets = dict(lsh_buckets(candidate))
        if (
            all(candidate_buckets[band] != bucket for band, bucket in query_buckets.items())
            and any(
                (candidate_buckets[band] ^ bucket).bit_count() == 1
                for band, bucket in query_buckets.items()
            )
        ):
            neighbor = candidate
            break
    if neighbor is None:
        raise RuntimeError("could not construct one-bit LSH neighbor")
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteMemoryStore(Path(directory) / "probe.db")
        service = MemoryService(store, FixedEmbedder(neighbor))
        target = service.remember(MemoryInput(namespace="n", content="Probe target."))
        started = time.perf_counter()
        legacy = target.id in store.search_vector_buckets(
            "n", "probe-test", query, 10, probe_radius=0
        )
        improved = target.id in store.search_vector_buckets(
            "n", "probe-test", query, 10, probe_radius=1
        )
        latency = (time.perf_counter() - started) * 1000.0
        store.close()
    return {"case": "one_bit_lsh_neighbor", "legacy_hit": legacy, "improved_hit": improved, "latency_ms": latency}


def main() -> None:
    cases = [
        candidate_saturation_case(),
        morphology_case(),
        alias_case(),
        exact_fallback_case(),
        graph_case(),
        multiprobe_case(),
    ]
    latencies = [case["latency_ms"] for case in cases]
    output = {
        "cases": cases,
        "legacy_hits": sum(case["legacy_hit"] for case in cases),
        "improved_hits": sum(case["improved_hit"] for case in cases),
        "total_cases": len(cases),
        "improved_recall": sum(case["improved_hit"] for case in cases) / len(cases),
        "suite_latency_mean_ms": statistics.fmean(latencies),
        "suite_latency_max_ms": max(latencies),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
