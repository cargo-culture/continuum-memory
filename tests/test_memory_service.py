from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from continuum_memory import HashingEmbedder, MemoryInput, MemoryKind, MemoryService, SQLiteMemoryStore
from continuum_memory.models import MemoryStatus, PolicyState


class MemoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "memory.db"
        self.store = SQLiteMemoryStore(self.db_path)
        self.service = MemoryService(self.store, HashingEmbedder(192))

    def tearDown(self) -> None:
        self.store.close()
        self.tempdir.cleanup()

    def test_persistence_and_namespace_isolation(self) -> None:
        target = self.service.remember(
            MemoryInput(
                namespace="project-a",
                content="The cobalt scheduler coordinates Zephyr deployment windows.",
                entities=["Zephyr"],
                tags=["architecture"],
            )
        )
        self.service.remember(
            MemoryInput(namespace="project-b", content="Cobalt paint is stored in the workshop.")
        )
        self.store.close()

        self.store = SQLiteMemoryStore(self.db_path)
        self.service = MemoryService(self.store, HashingEmbedder(192))
        hits = self.service.recall("cobalt scheduler Zephyr", namespace="project-a")
        self.assertTrue(hits)
        self.assertEqual(target.id, hits[0].memory.id)
        self.assertTrue(all(hit.memory.namespace == "project-a" for hit in hits))

    def test_deduplication_supersession_and_event_history(self) -> None:
        first = self.service.remember(
            MemoryInput(namespace="n", kind=MemoryKind.SEMANTIC, content="The launch date is October 4.")
        )
        duplicate = self.service.remember(
            MemoryInput(namespace="n", kind=MemoryKind.SEMANTIC, content="  The launch date is October 4.  ")
        )
        self.assertEqual(first.id, duplicate.id)

        replacement = self.service.supersede(first.id, "The launch date is October 11.")
        old = self.store.get_memory(first.id)
        self.assertIsNotNone(old)
        self.assertEqual(MemoryStatus.SUPERSEDED, old.status)
        self.assertEqual(first.id, replacement.supersedes_id)
        hits = self.service.recall("launch date October", namespace="n")
        self.assertTrue(hits)
        self.assertEqual(replacement.id, hits[0].memory.id)
        self.assertNotIn(first.id, [hit.memory.id for hit in hits])
        event_types = [event["event_type"] for event in self.store.list_events("n")]
        self.assertEqual(
            ["memory.created", "memory.superseded", "memory.created"],
            event_types,
        )

    def test_context_respects_token_budget_and_marks_memory_untrusted(self) -> None:
        for index in range(8):
            self.service.remember(
                MemoryInput(
                    namespace="budget",
                    content=f"Orchid protocol note {index}: " + ("calibration " * 25),
                )
            )
        packet = self.service.context(
            "orchid calibration", namespace="budget", limit=8, token_budget=150
        )
        self.assertLessEqual(packet.used_tokens, 150)
        self.assertLess(len(packet.hits), 8)
        self.assertIn("untrusted contextual data", packet.render())
        self.assertIn("<memory_context", packet.render())

    def test_deterministic_consolidation_builds_expandable_provenance_page(self) -> None:
        sources = [
            self.service.remember(
                MemoryInput(
                    namespace="history",
                    kind=MemoryKind.EPISODIC,
                    content=f"Experiment {index} used the amber routing table.",
                    importance=0.6 + index * 0.1,
                )
            )
            for index in range(3)
        ]
        result = self.service.consolidate(namespace="history")
        page = self.service.page(result.summary_memory_id)
        self.assertEqual(MemoryKind.SUMMARY, page.memory.kind)
        self.assertEqual({source.id for source in sources}, {source.id for source in page.source_memories})
        self.assertEqual(3, len(page.outgoing_links))
        self.assertTrue(all(link["link_type"] == "summarizes" for link in page.outgoing_links))

    def test_policy_requires_repeated_evidence_and_retires_after_failures(self) -> None:
        statement = "Check provenance before treating a recalled claim as current."
        first = self.service.record_policy_evidence(statement, outcome=1, namespace="agent")
        second = self.service.record_policy_evidence(statement, outcome=1, namespace="agent")
        third = self.service.record_policy_evidence(statement, outcome=1, namespace="agent")
        self.assertEqual(PolicyState.CANDIDATE, first.state)
        self.assertEqual(PolicyState.TESTING, second.state)
        self.assertEqual(PolicyState.PROMOTED, third.state)
        policy_memories = self.service.recall("provenance recalled claim", namespace="agent")
        promoted = [hit.memory for hit in policy_memories if "promoted-policy" in hit.memory.tags]
        self.assertEqual(1, len(promoted))

        current = third
        for _ in range(6):
            current = self.service.record_policy_evidence(statement, outcome=-1, namespace="agent")
        self.assertEqual(PolicyState.RETIRED, current.state)
        retired_memory = self.store.get_memory(promoted[0].id)
        self.assertEqual(MemoryStatus.ARCHIVED, retired_memory.status)

    def test_context_packet_is_json_serializable(self) -> None:
        self.service.remember(MemoryInput(content="A serializable memory."))
        packet = self.service.context("serializable")
        encoded = json.dumps(packet.to_dict())
        self.assertIn(packet.retrieval_id, encoded)


if __name__ == "__main__":
    unittest.main()
