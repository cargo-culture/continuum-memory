from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from continuum_memory import FederatedMemoryService, HashingEmbedder, __version__
from continuum_memory.plugin import (
    MAX_TOKEN_BUDGET,
    PAGE_LINK_LIMIT,
    PAGE_SOURCE_LIMIT,
    RECORD_EXCERPT_TOKENS,
    ContinuumPluginAdapter,
)
from continuum_memory.utils import approximate_token_count


class PluginAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.library = FederatedMemoryService(
            Path(self.tempdir.name) / "books",
            HashingEmbedder(128),
        )
        self.adapter = ContinuumPluginAdapter(self.library)
        self.adapter.initialize()

    def tearDown(self) -> None:
        self.library.close()
        self.tempdir.cleanup()

    def test_write_search_read_and_supersede(self) -> None:
        stored = self.adapter.remember_memory(
            "projects",
            "The launch window is October 10.",
            namespace="alice",
            kind="semantic",
            entities=["Launch"],
        )
        memory_id = stored["memory"]["memory_id"]

        result = self.adapter.search_memory(
            "launch window",
            namespace="alice",
            book_ids=["projects"],
        )
        self.assertEqual(memory_id, result["memories"][0]["memory"]["memory_id"])
        self.assertIn("untrusted", result["trust_notice"].casefold())

        page = self.adapter.read_memory("projects", memory_id, namespace="alice")
        self.assertEqual("The launch window is October 10.", page["memory"]["content"])
        self.assertNotIn("namespace", page["memory"])

        replacement = self.adapter.supersede_memory(
            "projects",
            memory_id,
            "The launch window is October 11.",
            namespace="alice",
        )
        self.assertEqual(memory_id, replacement["superseded"])
        self.assertEqual(memory_id, replacement["replacement"]["supersedes_id"])

    def test_search_payload_stays_within_the_reported_budget(self) -> None:
        body = (
            "The deployment checklist requires the operations group to confirm "
            "the rollback plan before release. "
        ) * 60
        for index in range(12):
            self.adapter.remember_memory(
                "projects",
                f"Decision {index}. {body}",
                namespace="alice",
                summary=f"Decision {index} about the rollback plan.",
            )

        result = self.adapter.search_memory(
            "rollback plan deployment checklist",
            namespace="alice",
            book_ids=["projects"],
        )

        self.assertTrue(result["memories"])
        # Retrieval budgets each hit on `summary or content`. Returning the full
        # content instead would spend context the packet never accounted for.
        for hit in result["memories"]:
            memory = hit["memory"]
            self.assertEqual(memory["summary"], memory["content"])
            self.assertTrue(memory["is_excerpt"])
            self.assertGreater(
                memory["full_token_count"], approximate_token_count(memory["content"])
            )
        serialized = approximate_token_count(json.dumps(result, ensure_ascii=False))
        self.assertLessEqual(serialized, result["token_budget"])
        # used_tokens charges memory text only; response_tokens is what the
        # caller actually pays, and must describe the response that was sent.
        self.assertGreater(result["response_tokens"], result["used_tokens"])
        self.assertAlmostEqual(serialized, result["response_tokens"], delta=16)

    def test_search_token_budget_is_capped(self) -> None:
        self.adapter.remember_memory(
            "projects", "The launch window is October 10.", namespace="alice"
        )
        result = self.adapter.search_memory(
            "launch window", namespace="alice", token_budget=10_000_000
        )
        self.assertEqual(MAX_TOKEN_BUDGET, result["token_budget"])

    def test_read_memory_caps_provenance_expansion(self) -> None:
        for index in range(30):
            self.adapter.remember_memory(
                "projects",
                f"Episode {index}: the team confirmed the rollback plan.",
                namespace="alice",
                kind="episodic",
            )
        consolidated = self.library.consolidate("projects", namespace="alice", limit=30)

        page = self.adapter.read_memory(
            "projects", consolidated.summary_memory_id, namespace="alice"
        )

        self.assertEqual(30, page["source_memory_count"])
        self.assertLessEqual(len(page["source_memories"]), PAGE_SOURCE_LIMIT)
        self.assertLessEqual(len(page["outgoing_links"]), PAGE_LINK_LIMIT)
        # A source that was not expanded is still reachable as a page of its own.
        self.assertEqual(30, len(page["source_memory_ids"]))
        expanded = {item["memory_id"] for item in page["source_memories"]}
        self.assertTrue(expanded.issubset(set(page["source_memory_ids"])))

    def test_oversized_record_is_excerpted(self) -> None:
        stored = self.adapter.remember_memory(
            "sources",
            "Paragraph. " * 40_000,
            namespace="alice",
        )
        memory = stored["memory"]
        self.assertTrue(memory["is_excerpt"])
        self.assertLessEqual(
            approximate_token_count(memory["content"]), RECORD_EXCERPT_TOKENS
        )

    def test_page_and_supersede_are_namespace_isolated(self) -> None:
        stored = self.adapter.remember_memory(
            "identity",
            "Alice prefers concise status reports.",
            namespace="alice",
            kind="identity",
        )
        memory_id = stored["memory"]["memory_id"]
        with self.assertRaises(KeyError):
            self.adapter.read_memory("identity", memory_id, namespace="bob")
        with self.assertRaises(KeyError):
            self.adapter.supersede_memory(
                "identity",
                memory_id,
                "Bob prefers long reports.",
                namespace="bob",
            )

    def test_standard_books_are_idempotently_initialized(self) -> None:
        self.adapter.initialize()
        books = self.adapter.list_memory_books(namespace="alice")["books"]
        self.assertEqual(8, len(books))


class PluginPackageTests(unittest.TestCase):
    def test_manifest_and_server_map_are_consistent(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / ".codex-plugin" / "plugin.json").read_text())
        server_map = json.loads((root / ".mcp.json").read_text())
        self.assertEqual("continuum-memory", manifest["name"])
        # The manifest carries its own copy of the version; drift between it and
        # the package is invisible until an install reports the wrong one.
        self.assertEqual(__version__, manifest["version"])
        self.assertEqual("./skills/", manifest["skills"])
        self.assertEqual("./.mcp.json", manifest["mcpServers"])
        self.assertIn("continuum-memory", server_map)
        self.assertTrue((root / "skills" / "continuum-memory" / "SKILL.md").is_file())


if __name__ == "__main__":
    unittest.main()
