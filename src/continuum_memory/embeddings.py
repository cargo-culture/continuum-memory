from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from .utils import normalize_vector, tokenize


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

class HashingEmbedder:
    """Fast deterministic feature hashing for offline use and tests.

    It supplies lexical-neighborhood vectors, not true semantic embeddings.
    The interface is deliberately identical to the remote embedding adapter.
    """

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 64:
            raise ValueError("dimension must be at least 64")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return f"hashing-v1-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _hash_feature(feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if value & 1 else -1.0
        return value, sign

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        words = tokenize(text)
        features: list[tuple[str, float]] = []
        features.extend((f"w:{word}", 1.0) for word in words)
        features.extend((f"b:{left}_{right}", 1.35) for left, right in zip(words, words[1:]))
        compact = " ".join(words)
        if len(compact) <= 2048:
            features.extend((f"c:{compact[i:i + 3]}", 0.18) for i in range(max(0, len(compact) - 2)))
        for feature, weight in features:
            hashed, sign = self._hash_feature(feature)
            vector[hashed % self._dimension] += sign * weight
        return normalize_vector(vector)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed(self, text: str) -> list[float]:
        return self._embed_one(text)


class OpenAIEmbedder:
    """Batched OpenAI embeddings adapter with lazy client initialization."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int = 512,
        api_key: str | None = None,
        batch_size: int = 128,
    ) -> None:
        if dimensions < 64:
            raise ValueError("dimensions must be at least 64")
        self.model = model
        self._dimension = dimensions
        self.api_key = api_key
        self.batch_size = max(1, min(batch_size, 2048))
        self._client = None

    @property
    def model_name(self) -> str:
        return f"openai:{self.model}:{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI support is not installed. Install continuum-memory[openai]."
                ) from exc
            self._client = OpenAI(api_key=self.api_key) if self.api_key else OpenAI()
        return self._client

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("embedding inputs cannot be empty")
        client = self._get_client()
        output: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self._dimension,
                encoding_format="float",
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            output.extend(normalize_vector(item.embedding) for item in ordered)
        if len(output) != len(texts):
            raise RuntimeError("embedding provider returned an unexpected result count")
        return output

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]
