from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from continuum_memory import MemoryInput, MemoryKind, MemoryService, SQLiteMemoryStore
from continuum_memory.retrieval import RetrievalConfig
from continuum_memory.utils import lsh_buckets, normalize_vector


class BrokenEmbedder:
    model_name = "broken"
    dimension = 64

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding unavailable")

    def embed_many(self, texts):
        raise RuntimeError("embedding unavailable")


class FixedEmbedder:
    def __init__(self, vector: list[float], model_name: str = "fixed") -> None:
        self.vector = normalize_vector(vector)
        self.model_name = model_name
        self.dimension = len(vector)

    def embed(self, text: str) -> list[float]:
        return list(self.vector)

    def embed_many(self, texts):
        return [list(self.vector) for _ in texts]


class MappingEmbedder:
    model_name = "mapping-v1"
    dimension = 64

    def __init__(self) -> None:
        self.target = [1.0] + [0.0] * 63
        self.other = [0.0, 1.0] + [0.0] * 62

    def embed(self, text: str) -> list[float]:
        if "TARGET-SEMANTIC" in text or text == "meaning preserving paraphrase":
            return list(self.target)
        return list(self.other)

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


class RetrievalMissTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "misses.db"
        self.store = SQLiteMemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tempdir.cleanup()

    def test_rank_fusion_prevents_entity_candidate_starvation(self) -> None:
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
        service = MemoryService(
            self.store, BrokenEmbedder(), retrieval_config=config, strict_embeddings=False
        )
        for index in range(12):
            service.remember(
                MemoryInput(namespace="fusion", content=f"Alpha common distractor {index}.")
            )
        target = service.remember(
            MemoryInput(
                namespace="fusion",
                kind=MemoryKind.IDENTITY,
                content="Alpha identity record for the concealed subject.",
                entities=["Needle Entity"],
                importance=1.0,
            )
        )
        packet = service.context(
            "alpha", namespace="fusion", limit=1, token_budget=300, context_entities=["Needle Entity"]
        )
        self.assertEqual(target.id, packet.hits[0].memory.id)

    def test_morphological_expansion_recalls_plural_query(self) -> None:
        config = RetrievalConfig(
            resident_candidates=1,
            exact_scan_mode="off",
            graph_seed_count=0,
            graph_candidates=0,
        )
        service = MemoryService(self.store, BrokenEmbedder(), retrieval_config=config)
        target = service.remember(
            MemoryInput(namespace="morph", content="The deployment schedule is locked.")
        )
        for index in range(5):
            service.remember(MemoryInput(namespace="morph", content=f"Unrelated note {index}."))
        hits = service.recall("deployments", namespace="morph", limit=3)
        self.assertIn(target.id, [hit.memory.id for hit in hits])

    def test_conjunctive_fts_avoids_common_term_candidate_flood(self) -> None:
        service = MemoryService(self.store, BrokenEmbedder())
        target = service.remember(
            MemoryInput(namespace="strict-fts", content="Amber scheduler markerZX91.")
        )
        for index in range(30):
            service.remember(
                MemoryInput(namespace="strict-fts", content=f"Amber distractor {index}.")
            )
        candidates = self.store.search_fts(
            "strict-fts", ["amber", "markerzx91"], limit=20
        )
        self.assertEqual([target.id], list(candidates))

    def test_registered_alias_resolves_entity_without_lexical_overlap(self) -> None:
        config = RetrievalConfig(resident_candidates=1, exact_scan_mode="off")
        service = MemoryService(self.store, BrokenEmbedder(), retrieval_config=config)
        target = service.remember(
            MemoryInput(
                namespace="aliases",
                content="The Zeta contract was renewed for another year.",
                entities=["International Business Machines"],
            )
        )
        for index in range(8):
            service.remember(
                MemoryInput(namespace="aliases", content=f"Unrelated supplier record {index}.")
            )
        service.register_entity_alias(
            "IBM", "International Business Machines", namespace="aliases"
        )
        packet = service.context(
            "", namespace="aliases", limit=1, token_budget=300, context_entities=["IBM"]
        )
        self.assertEqual(target.id, packet.hits[0].memory.id)

    def test_multi_probe_lsh_recovers_one_bit_neighbor(self) -> None:
        randomizer = random.Random(8192)
        dimension = 64
        query = normalize_vector([randomizer.uniform(-1.0, 1.0) for _ in range(dimension)])
        query_buckets = dict(lsh_buckets(query))
        neighbor = None
        for _ in range(20_000):
            candidate = normalize_vector(
                [randomizer.uniform(-1.0, 1.0) for _ in range(dimension)]
            )
            candidate_buckets = dict(lsh_buckets(candidate))
            exact_matches = sum(
                candidate_buckets[band] == bucket for band, bucket in query_buckets.items()
            )
            one_bit_matches = sum(
                (candidate_buckets[band] ^ bucket).bit_count() == 1
                for band, bucket in query_buckets.items()
            )
            if exact_matches == 0 and one_bit_matches > 0:
                neighbor = candidate
                break
        self.assertIsNotNone(neighbor, "failed to construct a one-bit LSH neighbor")
        service = MemoryService(self.store, FixedEmbedder(neighbor, "probe-test"))
        target = service.remember(MemoryInput(namespace="probe", content="Probe target."))
        exact = self.store.search_vector_buckets(
            "probe", "probe-test", query, 10, probe_radius=0
        )
        multiprobe = self.store.search_vector_buckets(
            "probe", "probe-test", query, 10, probe_radius=1
        )
        self.assertNotIn(target.id, exact)
        self.assertIn(target.id, multiprobe)

    def test_adaptive_exact_scan_recovers_missing_vector_bucket(self) -> None:
        config = RetrievalConfig(
            resident_candidates=4,
            exact_scan_mode="adaptive",
            exact_scan_max_records=100,
            exact_scan_result_limit=16,
            graph_seed_count=0,
            graph_candidates=0,
        )
        service = MemoryService(self.store, MappingEmbedder(), retrieval_config=config)
        target = service.remember(
            MemoryInput(
                namespace="fallback",
                content="TARGET-SEMANTIC: an otherwise unguessable stored proposition.",
            )
        )
        with self.store.transaction() as connection:
            connection.execute("DELETE FROM vector_buckets WHERE memory_id = ?", (target.id,))
        for index in range(20):
            service.remember(
                MemoryInput(namespace="fallback", content=f"Unrelated archive item {index}.")
            )
        packet = service.context(
            "meaning preserving paraphrase", namespace="fallback", limit=3, token_budget=500
        )
        self.assertEqual(target.id, packet.hits[0].memory.id)

    def test_graph_expansion_surfaces_linked_source_evidence(self) -> None:
        config = RetrievalConfig(
            resident_candidates=1,
            exact_scan_mode="off",
            graph_seed_count=8,
            graph_candidates=8,
        )
        service = MemoryService(self.store, BrokenEmbedder(), retrieval_config=config)
        source = service.remember(
            MemoryInput(namespace="graph", content="The selected cryptographic key was amber.")
        )
        for index in range(20):
            service.remember(MemoryInput(namespace="graph", content=f"Unrelated note {index}."))
        summary = service.remember(
            MemoryInput(
                namespace="graph",
                kind=MemoryKind.SUMMARY,
                content="Routing outcome and key-selection decision.",
            )
        )
        self.store.add_link(summary.id, source.id, "summarizes")
        packet = service.context("routing outcome", namespace="graph", limit=2, token_budget=500)
        visible_ids = {hit.memory.id for hit in packet.hits}
        self.assertIn(summary.id, visible_ids)
        self.assertTrue(source.id in visible_ids or source.id in packet.page_ids)


if __name__ == "__main__":
    unittest.main()
