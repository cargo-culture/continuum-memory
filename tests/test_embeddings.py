from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from continuum_memory.embeddings import HashingEmbedder, OpenAIEmbedder
from continuum_memory.utils import cosine_similarity, lsh_buckets


class EmbeddingTests(unittest.TestCase):
    def test_hashing_embeddings_are_deterministic_and_normalized(self) -> None:
        embedder = HashingEmbedder(128)
        left = embedder.embed("persistent memory retrieval")
        right = embedder.embed("persistent memory retrieval")
        self.assertEqual(left, right)
        self.assertAlmostEqual(1.0, math.sqrt(sum(value * value for value in left)), places=6)
        self.assertAlmostEqual(1.0, cosine_similarity(left, right), places=6)

    def test_identical_vectors_share_all_lsh_buckets(self) -> None:
        vector = HashingEmbedder(128).embed("hierarchical context paging")
        self.assertEqual(lsh_buckets(vector), lsh_buckets(list(vector)))
        self.assertEqual(8, len(lsh_buckets(vector)))

    def test_openai_adapter_batches_orders_and_normalizes_results(self) -> None:
        class FakeEmbeddings:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    data=[
                        SimpleNamespace(index=1, embedding=[0.0, 3.0, 4.0]),
                        SimpleNamespace(index=0, embedding=[2.0, 0.0, 0.0]),
                    ]
                )

        fake_embeddings = FakeEmbeddings()
        embedder = OpenAIEmbedder(model="test-embedding", dimensions=64)
        embedder._dimension = 3
        embedder._client = SimpleNamespace(embeddings=fake_embeddings)
        vectors = embedder.embed_many(["first", "second"])
        self.assertEqual([1.0, 0.0, 0.0], vectors[0])
        self.assertEqual([0.0, 0.6, 0.8], vectors[1])
        self.assertEqual(["first", "second"], fake_embeddings.kwargs["input"])
        self.assertEqual("test-embedding", fake_embeddings.kwargs["model"])


if __name__ == "__main__":
    unittest.main()
