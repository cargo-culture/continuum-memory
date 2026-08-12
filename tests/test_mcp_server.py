from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from mcp import Client
from starlette.requests import Request

from continuum_memory import FederatedMemoryService, HashingEmbedder
from continuum_memory.mcp_server import create_mcp_server
from continuum_memory.plugin import ContinuumPluginAdapter


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.library = FederatedMemoryService(
            Path(self.tempdir.name) / "books",
            HashingEmbedder(128),
        )
        self.server = create_mcp_server(
            ContinuumPluginAdapter(self.library),
            namespace_provider=lambda: "mcp-test",
        )

    async def asyncTearDown(self) -> None:
        self.library.close()
        self.tempdir.cleanup()

    async def test_tools_are_focused_and_annotated(self) -> None:
        async with Client(self.server) as client:
            tools = await client.list_tools()
        by_name = {tool.name: tool for tool in tools.tools}
        self.assertEqual(
            {
                "list_memory_books",
                "read_memory",
                "remember_memory",
                "search_memory",
                "supersede_memory",
            },
            set(by_name),
        )
        self.assertTrue(by_name["search_memory"].annotations.read_only_hint)
        self.assertFalse(by_name["remember_memory"].annotations.read_only_hint)
        self.assertFalse(by_name["supersede_memory"].annotations.destructive_hint)

    async def test_end_to_end_tool_calls(self) -> None:
        async with Client(self.server) as client:
            stored = await client.call_tool(
                "remember_memory",
                {
                    "book_id": "projects",
                    "content": "The cobalt scheduler owns deployment windows.",
                    "kind": "semantic",
                },
            )
            self.assertFalse(stored.is_error)
            searched = await client.call_tool(
                "search_memory",
                {"query": "cobalt scheduler", "book_ids": ["projects"]},
            )
            self.assertFalse(searched.is_error)
            self.assertEqual(
                "projects",
                searched.structured_content["memories"][0]["book_id"],
            )

    async def test_public_health_and_domain_challenge_routes(self) -> None:
        routes = {route.path: route for route in self.server._custom_starlette_routes}
        request = Request({"type": "http", "method": "GET", "path": "/"})

        health = await routes["/health"].endpoint(request)
        self.assertEqual(200, health.status_code)
        self.assertIn(b'"status":"ok"', health.body)

        with patch.dict(
            "os.environ",
            {"CONTINUUM_OPENAI_CHALLENGE_TOKEN": "review-token"},
        ):
            challenge = await routes[
                "/.well-known/openai-apps-challenge"
            ].endpoint(request)
        self.assertEqual(200, challenge.status_code)
        self.assertEqual(b"review-token", challenge.body)


if __name__ == "__main__":
    unittest.main()
