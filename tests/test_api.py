from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from continuum_memory import HashingEmbedder, MemoryService, SQLiteMemoryStore
from continuum_memory.api import create_server


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = SQLiteMemoryStore(Path(self.tempdir.name) / "api.db")
        self.service = MemoryService(self.store, HashingEmbedder(128))
        self.server = create_server(self.service, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.store.close()
        self.tempdir.cleanup()

    def request(self, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_write_recall_context_and_page(self) -> None:
        status, memory = self.request(
            "POST",
            "/v1/memories",
            {
                "namespace": "api-test",
                "kind": "semantic",
                "content": "Violet keys open the telemetry archive.",
                "entities": ["Telemetry Archive"],
            },
        )
        self.assertEqual(201, status)
        status, alias = self.request(
            "POST",
            "/v1/entity-aliases",
            {
                "namespace": "api-test",
                "alias": "TA",
                "canonical": "Telemetry Archive",
            },
        )
        self.assertEqual(201, status)
        self.assertEqual("TA", alias["alias"])
        status, recalled = self.request(
            "POST",
            "/v1/recall",
            {
                "namespace": "api-test",
                "query": "TA",
                "limit": 3,
                "exact_scan_mode": "off",
            },
        )
        self.assertEqual(200, status)
        self.assertEqual(memory["id"], recalled["hits"][0]["memory"]["id"])
        status, context = self.request(
            "POST",
            "/v1/context",
            {"namespace": "api-test", "query": "telemetry archive"},
        )
        self.assertEqual(200, status)
        self.assertIn("<memory_context", context["rendered"])
        status, page = self.request("GET", f"/v1/pages/{memory['id']}")
        self.assertEqual(200, status)
        self.assertEqual(memory["id"], page["memory"]["id"])

    def test_health(self) -> None:
        status, payload = self.request("GET", "/health?namespace=api-test")
        self.assertEqual(200, status)
        self.assertEqual("ok", payload["status"])


if __name__ == "__main__":
    unittest.main()
