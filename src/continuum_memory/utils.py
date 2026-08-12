from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence


TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
CAPITALIZED_RE = re.compile(r"\b(?:[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,3})\b")
ENTITY_STOPWORDS = {
    "a",
    "an",
    "and",
    "find",
    "i",
    "it",
    "memory",
    "record",
    "that",
    "the",
    "these",
    "this",
    "those",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "you",
}
QUERY_STOPWORDS = ENTITY_STOPWORDS | {
    "about",
    "are",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "of",
    "on",
    "please",
    "tell",
    "to",
    "was",
    "were",
    "with",
}
LSH_BANDS = 8
LSH_BITS_PER_BAND = 12
LSH_SAMPLES_PER_BIT = 8


def utc_now() -> float:
    return time.time()


def timestamp_to_iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def tokenize(value: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(value) if len(token) > 1]


def expand_query_tokens(tokens: Sequence[str], limit: int = 32) -> list[str]:
    """Add conservative inflection variants without replacing original tokens."""
    result = list(dict.fromkeys(token.casefold() for token in tokens if len(token) > 1))
    variants: list[str] = []
    for token in result:
        if len(token) > 4 and token.endswith("ies"):
            variants.append(token[:-3] + "y")
        elif len(token) > 4 and token.endswith("es"):
            variants.extend((token[:-1], token[:-2]))
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            variants.append(token[:-1])
        if len(token) > 5 and token.endswith("ing"):
            stem = token[:-3]
            variants.extend((stem, stem + "e"))
            if len(stem) > 2 and stem[-1] == stem[-2]:
                variants.append(stem[:-1])
        if len(token) > 4 and token.endswith("ed"):
            stem = token[:-2]
            variants.extend((stem, stem + "e"))
            if len(stem) > 2 and stem[-1] == stem[-2]:
                variants.append(stem[:-1])
    for variant in variants:
        if len(variant) > 1 and variant not in result:
            result.append(variant)
        if len(result) >= limit:
            break
    return result[:limit]


def approximate_token_count(value: str) -> int:
    if not value:
        return 0
    words = len(TOKEN_RE.findall(value))
    character_estimate = math.ceil(len(value) / 4)
    return max(1, math.ceil((words + character_estimate) / 2))


def stable_content_hash(namespace: str, kind: str, content: str) -> str:
    raw = f"{namespace}\0{kind}\0{normalize_text(content)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonical_entity(value: str) -> str:
    return normalize_text(value).strip(".,:;!?()[]{}\"'")


def infer_entities(value: str, limit: int = 24) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in CAPITALIZED_RE.findall(value):
        entity = match.strip()
        canonical = canonical_entity(entity)
        if len(canonical) < 2 or canonical in ENTITY_STOPWORDS or canonical in seen:
            continue
        seen.add(canonical)
        result.append(entity)
        if len(result) >= limit:
            break
    return result


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def vector_to_blob(vector: Sequence[float] | None) -> bytes | None:
    if vector is None:
        return None
    if not vector:
        return b""
    return struct.pack(f"<{len(vector)}f", *vector)


def blob_to_vector(blob: bytes | None, dimension: int | None) -> list[float] | None:
    if blob is None or dimension is None:
        return None
    if dimension == 0:
        return []
    expected = dimension * 4
    if len(blob) != expected:
        return None
    return list(struct.unpack(f"<{dimension}f", blob))


def normalize_vector(vector: Sequence[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return [0.0 for _ in vector]
    return [float(value / magnitude) for value in vector]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    denominator = left_norm * right_norm
    if denominator == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / denominator))


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value & 0xFFFFFFFF


def lsh_buckets(
    vector: Sequence[float],
    *,
    bands: int = LSH_BANDS,
    bits_per_band: int = LSH_BITS_PER_BAND,
    samples_per_bit: int = LSH_SAMPLES_PER_BIT,
) -> list[tuple[int, int]]:
    """Return deterministic random-hyperplane LSH buckets for a dense vector."""
    if not vector:
        return []
    dimension = len(vector)
    buckets: list[tuple[int, int]] = []
    for band in range(bands):
        bucket = 0
        for bit in range(bits_per_band):
            projection = 0.0
            seed = _mix32((band + 1) * 0x9E3779B1 ^ (bit + 1) * 0x85EBCA77)
            for sample in range(samples_per_bit):
                seed = _mix32(seed + sample * 0x27D4EB2D)
                index = seed % dimension
                sign = 1.0 if seed & 0x80000000 else -1.0
                projection += vector[index] * sign
            if projection >= 0:
                bucket |= 1 << bit
        buckets.append((band, bucket))
    return buckets


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
