from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from continuum_memory import HashingEmbedder, MemoryInput, MemoryKind, MemoryService, SQLiteMemoryStore
from continuum_memory.consolidation import OpenAIConsolidator


class ConsolidationAdapterTests(unittest.TestCase):
    def test_schema_constrained_adapter_validates_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "consolidation.db")
            service = MemoryService(store, HashingEmbedder(128))
            first = service.remember(MemoryInput(content="First routing experiment succeeded."))
            second = service.remember(MemoryInput(content="Second routing experiment succeeded."))

            class FakeResponses:
                def parse(self, **kwargs):
                    schema = kwargs["text_format"]
                    return SimpleNamespace(
                        output_parsed=schema(
                            synopsis="Two routing experiments succeeded.",
                            claims=[
                                {
                                    "content": "The routing approach succeeded twice.",
                                    "kind": "semantic",
                                    "confidence": 0.8,
                                    "importance": 0.7,
                                    "source_memory_ids": [first.id, second.id],
                                    "entities": [],
                                    "tags": ["routing"],
                                },
                                {
                                    "content": "Unsupported claim.",
                                    "kind": "semantic",
                                    "confidence": 1.0,
                                    "importance": 1.0,
                                    "source_memory_ids": ["mem_invented"],
                                    "entities": [],
                                    "tags": [],
                                },
                            ],
                            policy_hypotheses=[
                                {
                                    "statement": "Reuse the validated routing approach.",
                                    "confidence": 0.65,
                                    "source_memory_ids": [first.id, second.id],
                                }
                            ],
                            contradiction_notes=[],
                        )
                    )

            engine = OpenAIConsolidator(model="test-model")
            engine._client = SimpleNamespace(responses=FakeResponses())
            bundle = engine.consolidate([first, second])
            self.assertEqual(1, len(bundle.claims))
            self.assertEqual(MemoryKind.SEMANTIC, bundle.claims[0].kind)
            self.assertEqual({first.id, second.id}, set(bundle.claims[0].source_memory_ids))
            self.assertEqual(1, len(bundle.policy_hypotheses))
            store.close()


if __name__ == "__main__":
    unittest.main()
