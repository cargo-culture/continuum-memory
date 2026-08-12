from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from continuum_memory import (
    ContextPacket,
    FederatedMemoryService,
    HashingEmbedder,
    MemoryInput,
)


class CountingHashingEmbedder(HashingEmbedder):
    def __init__(self, dimension: int) -> None:
        super().__init__(dimension)
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        return super().embed(text)


class FederationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "books"
        self.library = FederatedMemoryService(
            self.root,
            HashingEmbedder(128),
            max_books=3,
            max_workers=4,
        )

    def tearDown(self) -> None:
        self.library.close()
        self.tempdir.cleanup()

    def test_standard_books_route_continuity_plus_topic(self) -> None:
        self.library.create_standard_books()
        routes = self.library.route("Continue the project design and unresolved requirements")
        self.assertEqual(3, len(routes))
        self.assertIn("working", [route.book_id for route in routes])
        self.assertIn("identity", [route.book_id for route in routes])
        self.assertIn("projects", [route.book_id for route in routes])

    def test_catalog_and_memories_persist_by_book(self) -> None:
        self.library.create_book(
            "projects",
            title="Projects",
            description="Project requirements and decisions",
            keywords=["project", "decision"],
        )
        memory = self.library.remember(
            "projects",
            MemoryInput(namespace="demo", content="The project uses a cobalt scheduler."),
        )
        self.library.close()
        self.library = FederatedMemoryService(self.root, HashingEmbedder(128))
        packet = self.library.context(
            "cobalt scheduler",
            namespace="demo",
            book_ids=["projects"],
            limit=3,
        )
        self.assertEqual(memory.id, packet.hits[0].memory.id)
        self.assertEqual("projects", packet.hits[0].book_id)

    def test_concurrent_searches_overlap(self) -> None:
        for book_id in ("alpha", "beta"):
            self.library.create_book(
                book_id,
                title=book_id.title(),
                description=f"{book_id} test records",
            )
        barrier = threading.Barrier(2, timeout=2)
        thread_ids: set[int] = set()
        thread_lock = threading.Lock()

        def fake_context(*_args, **kwargs):
            with thread_lock:
                thread_ids.add(threading.get_ident())
            barrier.wait()
            return ContextPacket(
                namespace=kwargs["namespace"],
                query="test",
                generated_at=time.time(),
                token_budget=kwargs["token_budget"],
                used_tokens=0,
                hits=[],
                page_ids=[],
                retrieval_id="ret_test",
            )

        self.library._books["alpha"].service.context = fake_context  # type: ignore[method-assign]
        self.library._books["beta"].service.context = fake_context  # type: ignore[method-assign]
        packet = self.library.context("test", book_ids=["alpha", "beta"])
        self.assertEqual({"alpha", "beta"}, set(packet.searched_book_ids))
        self.assertEqual(2, len(thread_ids))

    def test_global_budget_and_cross_book_deduplication(self) -> None:
        for book_id in ("semantic", "sources"):
            self.library.create_book(
                book_id,
                title=book_id.title(),
                description=f"{book_id} knowledge",
                keywords=[book_id],
            )
            self.library.remember(
                book_id,
                MemoryInput(
                    namespace="demo",
                    content="Shared proposition: the amber routing table is authoritative.",
                ),
            )
        packet = self.library.context(
            "amber routing table",
            namespace="demo",
            book_ids=["semantic", "sources"],
            limit=8,
            token_budget=100,
        )
        self.assertLessEqual(packet.used_tokens, 100)
        self.assertEqual(1, len(packet.hits))

    def test_explicit_books_bypass_automatic_router(self) -> None:
        for book_id in ("one", "two", "three"):
            self.library.create_book(
                book_id,
                title=book_id,
                description=book_id,
            )
        routes = self.library.route("anything", book_ids=["three", "one"])
        self.assertEqual(["three", "one"], [route.book_id for route in routes])

    def test_query_embedding_is_shared_across_selected_books(self) -> None:
        root = Path(self.tempdir.name) / "counting-books"
        embedder = CountingHashingEmbedder(128)
        library = FederatedMemoryService(root, embedder, max_workers=2)
        try:
            for book_id in ("alpha", "beta"):
                library.create_book(book_id, title=book_id, description=book_id)
                library.remember(book_id, MemoryInput(content=f"{book_id} memory"))
            embedder.calls = 0
            library.context("memory", book_ids=["alpha", "beta"])
            self.assertEqual(1, embedder.calls)
        finally:
            library.close()

    def test_one_book_failure_returns_other_book_results(self) -> None:
        for book_id in ("failing", "healthy"):
            self.library.create_book(book_id, title=book_id, description=book_id)
        target = self.library.remember(
            "healthy", MemoryInput(content="The healthy book contains amber evidence.")
        )

        def fail(*_args, **_kwargs):
            raise RuntimeError("simulated book failure")

        self.library._books["failing"].service.context = fail  # type: ignore[method-assign]
        packet = self.library.context(
            "amber evidence", book_ids=["failing", "healthy"], limit=3
        )
        self.assertEqual(target.id, packet.hits[0].memory.id)
        self.assertEqual("RuntimeError", packet.failures["failing"])


if __name__ == "__main__":
    unittest.main()
