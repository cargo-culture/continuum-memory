from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from continuum_memory import FederatedMemoryService, HashingEmbedder
from continuum_memory.federation_api import create_federation_server


class FederationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.library = FederatedMemoryService(
            Path(self.tempdir.name) / "books",
            HashingEmbedder(128),
        )
        self.server = create_federation_server(self.library, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.library.close()
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

    def test_book_create_write_recall_context_and_page(self) -> None:
        status, book = self.request(
            "POST",
            "/v1/books",
            {
                "book_id": "projects",
                "title": "Projects",
                "description": "Project requirements and decisions",
                "keywords": ["project", "decision"],
            },
        )
        self.assertEqual(201, status)
        self.assertEqual("projects", book["book_id"])

        status, memory = self.request(
            "POST",
            "/v1/books/projects/memories",
            {
                "namespace": "api-test",
                "content": "The cobalt scheduler controls deployment windows.",
            },
        )
        self.assertEqual(201, status)

        status, recalled = self.request(
            "POST",
            "/v1/library/recall",
            {
                "namespace": "api-test",
                "query": "cobalt scheduler",
                "book_ids": ["projects"],
            },
        )
        self.assertEqual(200, status)
        self.assertEqual(["projects"], recalled["searched_book_ids"])
        self.assertEqual(memory["id"], recalled["hits"][0]["memory"]["id"])

        status, context = self.request(
            "POST",
            "/v1/library/context",
            {
                "namespace": "api-test",
                "query": "deployment windows",
                "book_ids": ["projects"],
            },
        )
        self.assertEqual(200, status)
        self.assertIn('<memory book="projects"', context["rendered"])

        status, page = self.request(
            "GET",
            f"/v1/books/projects/pages/{memory['id']}",
        )
        self.assertEqual(200, status)
        self.assertEqual(memory["id"], page["memory"]["id"])

        status, listed = self.request("GET", "/v1/books?namespace=api-test")
        self.assertEqual(200, status)
        self.assertEqual(1, listed["books"][0]["active_memories"])


if __name__ == "__main__":
    unittest.main()
