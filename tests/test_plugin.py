from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from continuum_memory import FederatedMemoryService, HashingEmbedder
from continuum_memory.plugin import ContinuumPluginAdapter


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
        self.assertEqual("0.4.0", manifest["version"])
        self.assertEqual("./skills/", manifest["skills"])
        self.assertEqual("./.mcp.json", manifest["mcpServers"])
        self.assertIn("continuum-memory", server_map)
        self.assertTrue((root / "skills" / "continuum-memory" / "SKILL.md").is_file())


if __name__ == "__main__":
    unittest.main()
