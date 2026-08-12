from __future__ import annotations

import json
import heapq
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import MemoryKind, MemoryRecord, MemoryStatus, PolicyRecord, PolicyState
from .utils import (
    blob_to_vector,
    canonical_entity,
    clamp,
    json_dumps,
    json_loads,
    LSH_BITS_PER_BAND,
    lsh_buckets,
    utc_now,
    vector_to_blob,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    source_uri TEXT,
    source_type TEXT,
    importance REAL NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    valid_from REAL,
    valid_until REAL,
    last_accessed REAL,
    access_count INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    supersedes_id TEXT,
    contradiction_group TEXT,
    metadata_json TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding BLOB,
    embedding_dim INTEGER,
    embedding_model TEXT,
    FOREIGN KEY (supersedes_id) REFERENCES memories(id)
);

CREATE INDEX IF NOT EXISTS idx_memories_namespace_status
    ON memories(namespace, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_hash
    ON memories(namespace, content_hash, status);
CREATE INDEX IF NOT EXISTS idx_memories_kind
    ON memories(namespace, kind, status);
CREATE INDEX IF NOT EXISTS idx_memories_resident
    ON memories(namespace, status, importance DESC, confidence DESC, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_contradiction
    ON memories(namespace, contradiction_group) WHERE contradiction_group IS NOT NULL;

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    entity TEXT NOT NULL,
    canonical TEXT NOT NULL,
    PRIMARY KEY (memory_id, canonical),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_entities_namespace_canonical
    ON memory_entities(namespace, canonical, memory_id);

CREATE TABLE IF NOT EXISTS entity_aliases (
    namespace TEXT NOT NULL,
    alias TEXT NOT NULL,
    canonical TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (namespace, alias)
);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_canonical
    ON entity_aliases(namespace, canonical, alias);

CREATE TABLE IF NOT EXISTS vector_buckets (
    memory_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    band INTEGER NOT NULL,
    bucket INTEGER NOT NULL,
    PRIMARY KEY (memory_id, embedding_model, band),
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_vector_bucket_lookup
    ON vector_buckets(namespace, embedding_model, band, bucket);

CREATE TABLE IF NOT EXISTS memory_links (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    link_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (source_id, target_id, link_type),
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_links_target ON memory_links(target_id, link_type);

CREATE TABLE IF NOT EXISTS memory_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    memory_id TEXT,
    namespace TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    occurred_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_namespace_sequence
    ON memory_events(namespace, sequence);

CREATE TABLE IF NOT EXISTS retrieval_log (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    query TEXT NOT NULL,
    selected_ids_json TEXT NOT NULL,
    candidates INTEGER NOT NULL,
    token_budget INTEGER NOT NULL,
    used_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    features_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_retrieval_namespace_created
    ON retrieval_log(namespace, created_at DESC);

CREATE TABLE IF NOT EXISTS policies (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    statement TEXT NOT NULL,
    statement_hash TEXT NOT NULL,
    state TEXT NOT NULL,
    confidence REAL NOT NULL,
    successes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    neutral INTEGER NOT NULL DEFAULT 0,
    evidence_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    promoted_at REAL,
    retired_at REAL,
    metadata_json TEXT NOT NULL,
    UNIQUE(namespace, statement_hash)
);
CREATE INDEX IF NOT EXISTS idx_policies_namespace_state
    ON policies(namespace, state, updated_at DESC);

CREATE TABLE IF NOT EXISTS policy_evidence (
    id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL,
    source_memory_id TEXT,
    outcome INTEGER NOT NULL,
    note TEXT,
    created_at REAL NOT NULL,
    FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE,
    FOREIGN KEY (source_memory_id) REFERENCES memories(id)
);
CREATE INDEX IF NOT EXISTS idx_policy_evidence_policy
    ON policy_evidence(policy_id, created_at);
"""


class SQLiteMemoryStore:
    """Append-audited SQLite store with FTS5 and bounded LSH vector candidates."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), timeout=30.0, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._configure()
        self._initialize()

    def _configure(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._connection.execute("PRAGMA temp_store = MEMORY")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            self._connection.execute("PRAGMA mmap_size = 268435456")

    def _initialize(self) -> None:
        # sqlite3.executescript manages its own transaction boundary.
        with self._lock:
            self._connection.executescript(SCHEMA)
            self._connection.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', '2')"
            )
        self.fts_available = self._initialize_fts()

    def _initialize_fts(self) -> bool:
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        memory_id UNINDEXED,
                        namespace UNINDEXED,
                        content,
                        summary,
                        entities,
                        tokenize='unicode61 remove_diacritics 2'
                    )
                    """
                )
            return True
        except sqlite3.OperationalError:
            return False

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex}"

    def find_duplicate(self, namespace: str, content_hash: str) -> MemoryRecord | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT * FROM memories
                WHERE namespace = ? AND content_hash = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
                """,
                (namespace, content_hash),
            ).fetchone()
        return self._record_from_single_row(row) if row else None

    def insert_memory(
        self,
        *,
        memory_id: str,
        namespace: str,
        kind: MemoryKind,
        content: str,
        summary: str | None,
        source_uri: str | None,
        source_type: str | None,
        importance: float,
        confidence: float,
        created_at: float,
        valid_from: float | None,
        valid_until: float | None,
        token_count: int,
        supersedes_id: str | None,
        contradiction_group: str | None,
        metadata: dict[str, Any],
        tags: Sequence[str],
        entities: Sequence[str],
        content_hash: str,
        embedding: Sequence[float] | None,
        embedding_model: str | None,
    ) -> MemoryRecord:
        entity_pairs = []
        seen_entities: set[str] = set()
        for entity in entities:
            canonical = canonical_entity(entity)
            if not canonical or canonical in seen_entities:
                continue
            seen_entities.add(canonical)
            entity_pairs.append((str(entity).strip(), canonical))
        clean_tags = list(dict.fromkeys(tag.strip().casefold() for tag in tags if tag.strip()))
        embedding_dim = len(embedding) if embedding is not None else None
        with self.transaction() as connection:
            if supersedes_id:
                old = connection.execute(
                    "SELECT namespace, status, version FROM memories WHERE id = ?", (supersedes_id,)
                ).fetchone()
                if old is None:
                    raise KeyError(f"memory not found: {supersedes_id}")
                if old["namespace"] != namespace:
                    raise ValueError("cannot supersede a memory in another namespace")
                if old["status"] == MemoryStatus.TOMBSTONED.value:
                    raise ValueError("cannot supersede a tombstoned memory")
                connection.execute(
                    "UPDATE memories SET status = 'superseded', updated_at = ? WHERE id = ?",
                    (created_at, supersedes_id),
                )
                self._upsert_fts(connection, supersedes_id, remove=True)
                self._write_event(
                    connection,
                    namespace=namespace,
                    memory_id=supersedes_id,
                    event_type="memory.superseded",
                    payload={"superseded_by": memory_id},
                    occurred_at=created_at,
                )
                version = int(old["version"]) + 1
            else:
                version = 1
            connection.execute(
                """
                INSERT INTO memories(
                    id, namespace, kind, content, summary, source_uri, source_type,
                    importance, confidence, status, created_at, updated_at, valid_from,
                    valid_until, token_count, version, supersedes_id, contradiction_group,
                    metadata_json, tags_json, content_hash, embedding, embedding_dim,
                    embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    namespace,
                    kind.value,
                    content,
                    summary,
                    source_uri,
                    source_type,
                    clamp(importance),
                    clamp(confidence),
                    created_at,
                    created_at,
                    valid_from,
                    valid_until,
                    token_count,
                    version,
                    supersedes_id,
                    contradiction_group,
                    json_dumps(metadata),
                    json_dumps(clean_tags),
                    content_hash,
                    vector_to_blob(embedding),
                    embedding_dim,
                    embedding_model,
                ),
            )
            connection.executemany(
                "INSERT INTO memory_entities(memory_id, namespace, entity, canonical) VALUES (?, ?, ?, ?)",
                [(memory_id, namespace, entity, canonical) for entity, canonical in entity_pairs],
            )
            if embedding is not None and embedding_model:
                connection.executemany(
                    """
                    INSERT INTO vector_buckets(memory_id, namespace, embedding_model, band, bucket)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (memory_id, namespace, embedding_model, band, bucket)
                        for band, bucket in lsh_buckets(embedding)
                    ],
                )
            self._upsert_fts(
                connection,
                memory_id,
                namespace=namespace,
                content=content,
                summary=summary or "",
                entities=" ".join(entity for entity, _ in entity_pairs),
            )
            self._write_event(
                connection,
                namespace=namespace,
                memory_id=memory_id,
                event_type="memory.created",
                payload={
                    "id": memory_id,
                    "kind": kind.value,
                    "content": content,
                    "summary": summary,
                    "source_uri": source_uri,
                    "source_type": source_type,
                    "importance": clamp(importance),
                    "confidence": clamp(confidence),
                    "valid_from": valid_from,
                    "valid_until": valid_until,
                    "version": version,
                    "supersedes_id": supersedes_id,
                    "contradiction_group": contradiction_group,
                    "entities": [entity for entity, _ in entity_pairs],
                    "tags": clean_tags,
                    "metadata": metadata,
                    "content_hash": content_hash,
                    "embedding_model": embedding_model,
                    "embedding_dim": embedding_dim,
                },
                occurred_at=created_at,
            )
        record = self.get_memory(memory_id)
        if record is None:
            raise RuntimeError("memory insert did not materialize")
        return record

    def _write_event(
        self,
        connection: sqlite3.Connection,
        *,
        namespace: str,
        memory_id: str | None,
        event_type: str,
        payload: dict[str, Any],
        occurred_at: float | None = None,
    ) -> str:
        event_id = self.new_id("evt")
        connection.execute(
            """
            INSERT INTO memory_events(event_id, memory_id, namespace, event_type, payload_json, occurred_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (event_id, memory_id, namespace, event_type, json_dumps(payload), occurred_at or utc_now()),
        )
        return event_id

    def _upsert_fts(
        self,
        connection: sqlite3.Connection,
        memory_id: str,
        *,
        namespace: str | None = None,
        content: str = "",
        summary: str = "",
        entities: str = "",
        remove: bool = False,
    ) -> None:
        if not getattr(self, "fts_available", True):
            return
        try:
            connection.execute("DELETE FROM memory_fts WHERE memory_id = ?", (memory_id,))
            if not remove and namespace is not None:
                connection.execute(
                    """
                    INSERT INTO memory_fts(memory_id, namespace, content, summary, entities)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory_id, namespace, content, summary, entities),
                )
        except sqlite3.OperationalError:
            self.fts_available = False

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._record_from_single_row(row) if row else None

    def _record_from_single_row(self, row: sqlite3.Row) -> MemoryRecord:
        memory_id = row["id"]
        with self._lock:
            entity_rows = self._connection.execute(
                "SELECT entity FROM memory_entities WHERE memory_id = ? ORDER BY entity", (memory_id,)
            ).fetchall()
        entities = [item["entity"] for item in entity_rows]
        return self._row_to_record(row, entities)

    def get_memories(self, memory_ids: Iterable[str]) -> list[MemoryRecord]:
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})", ids
            ).fetchall()
            entity_rows = self._connection.execute(
                f"SELECT memory_id, entity FROM memory_entities WHERE memory_id IN ({placeholders}) ORDER BY entity",
                ids,
            ).fetchall()
        facets: dict[str, list[str]] = {memory_id: [] for memory_id in ids}
        for row in entity_rows:
            facets.setdefault(row["memory_id"], []).append(row["entity"])
        by_id = {row["id"]: self._row_to_record(row, facets.get(row["id"], [])) for row in rows}
        return [by_id[memory_id] for memory_id in ids if memory_id in by_id]

    @staticmethod
    def _row_to_record(row: sqlite3.Row, entities: list[str]) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            namespace=row["namespace"],
            kind=MemoryKind(row["kind"]),
            content=row["content"],
            summary=row["summary"],
            source_uri=row["source_uri"],
            source_type=row["source_type"],
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            status=MemoryStatus(row["status"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            last_accessed=row["last_accessed"],
            access_count=int(row["access_count"]),
            token_count=int(row["token_count"]),
            version=int(row["version"]),
            supersedes_id=row["supersedes_id"],
            contradiction_group=row["contradiction_group"],
            entities=entities,
            tags=list(json_loads(row["tags_json"], [])),
            metadata=dict(json_loads(row["metadata_json"], {})),
            content_hash=row["content_hash"],
            embedding_model=row["embedding_model"],
            embedding_dim=row["embedding_dim"],
            embedding=blob_to_vector(row["embedding"], row["embedding_dim"]),
        )

    def search_fts(self, namespace: str, query_tokens: Sequence[str], limit: int) -> dict[str, float]:
        tokens = list(dict.fromkeys(token for token in query_tokens if token.isalnum()))[:16]
        if not tokens:
            return {}
        if not self.fts_available:
            return self._search_like(namespace, tokens, limit)
        strict_expression = " AND ".join(f'"{token}"*' for token in tokens)
        broad_expression = " OR ".join(f'"{token}"*' for token in tokens)
        try:
            with self._lock:
                def run(expression: str) -> list[sqlite3.Row]:
                    return self._connection.execute(
                        """
                        SELECT memory_fts.memory_id,
                               bm25(memory_fts, 0.0, 0.0, 1.0, 1.4, 1.8) AS rank_value
                        FROM memory_fts
                        JOIN memories ON memories.id = memory_fts.memory_id
                        WHERE memory_fts MATCH ? AND memory_fts.namespace = ?
                          AND memories.status = 'active'
                        ORDER BY rank_value ASC LIMIT ?
                        """,
                        (expression, namespace, limit),
                    ).fetchall()

                # A conjunctive pass keeps specific queries from sorting every row
                # that contains one common term. Broad OR retrieval is retained as
                # a recall fallback when the strict pass finds nothing.
                rows = run(strict_expression)
                if not rows and broad_expression != strict_expression:
                    rows = run(broad_expression)
        except sqlite3.OperationalError:
            return self._search_like(namespace, tokens, limit)
        return {row["memory_id"]: 1.0 / (1.0 + rank) for rank, row in enumerate(rows)}

    def _search_like(self, namespace: str, tokens: Sequence[str], limit: int) -> dict[str, float]:
        clauses = " OR ".join("lower(content || ' ' || coalesce(summary, '')) LIKE ?" for _ in tokens)
        params: list[Any] = [namespace, *[f"%{token.casefold()}%" for token in tokens], limit]
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT id FROM memories
                WHERE namespace = ? AND status = 'active' AND ({clauses})
                ORDER BY updated_at DESC LIMIT ?
                """,
                params,
            ).fetchall()
        return {row["id"]: 1.0 / (1.0 + rank) for rank, row in enumerate(rows)}

    def search_entities(self, namespace: str, entities: Sequence[str], limit: int) -> dict[str, float]:
        original_canonicals = list(
            dict.fromkeys(canonical_entity(entity) for entity in entities if entity.strip())
        )
        if not original_canonicals:
            return {}
        canonicals = self.expand_entity_canonicals(namespace, original_canonicals)
        placeholders = ",".join("?" for _ in canonicals)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT memory_entities.memory_id, count(*) AS matches
                FROM memory_entities
                JOIN memories ON memories.id = memory_entities.memory_id
                WHERE memory_entities.namespace = ?
                  AND memory_entities.canonical IN ({placeholders})
                  AND memories.status = 'active'
                GROUP BY memory_entities.memory_id
                ORDER BY matches DESC, memories.updated_at DESC LIMIT ?
                """,
                [namespace, *canonicals, limit],
            ).fetchall()
        denominator = max(1, len(original_canonicals))
        return {row["memory_id"]: min(1.0, row["matches"] / denominator) for row in rows}

    def register_entity_alias(self, namespace: str, alias: str, canonical: str) -> None:
        clean_namespace = namespace.strip()
        clean_alias = canonical_entity(alias)
        clean_canonical = canonical_entity(canonical)
        if not clean_namespace or not clean_alias or not clean_canonical:
            raise ValueError("namespace, alias, and canonical entity must be non-empty")
        if clean_alias == clean_canonical:
            raise ValueError("alias and canonical entity must differ")
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO entity_aliases(namespace, alias, canonical, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, alias)
                DO UPDATE SET canonical = excluded.canonical, created_at = excluded.created_at
                """,
                (clean_namespace, clean_alias, clean_canonical, now),
            )
            self._write_event(
                connection,
                namespace=clean_namespace,
                memory_id=None,
                event_type="entity.alias_registered",
                payload={"alias": clean_alias, "canonical": clean_canonical},
                occurred_at=now,
            )

    def expand_entity_canonicals(self, namespace: str, canonicals: Sequence[str]) -> list[str]:
        expanded = set(canonical_entity(value) for value in canonicals if value)
        frontier = set(expanded)
        # A short closure supports alias chains while preventing unbounded graph walks.
        with self._lock:
            for _ in range(3):
                if not frontier:
                    break
                placeholders = ",".join("?" for _ in frontier)
                rows = self._connection.execute(
                    f"""
                    SELECT alias, canonical FROM entity_aliases
                    WHERE namespace = ? AND (alias IN ({placeholders}) OR canonical IN ({placeholders}))
                    """,
                    [namespace, *frontier, *frontier],
                ).fetchall()
                discovered = {
                    value
                    for row in rows
                    for value in (row["alias"], row["canonical"])
                    if value not in expanded
                }
                expanded.update(discovered)
                frontier = discovered
        return sorted(expanded)

    def search_vector_buckets(
        self,
        namespace: str,
        embedding_model: str,
        query_vector: Sequence[float],
        limit: int,
        probe_radius: int = 1,
    ) -> dict[str, float]:
        buckets = lsh_buckets(query_vector)
        if not buckets:
            return {}
        if probe_radius not in {0, 1}:
            raise ValueError("probe_radius must be 0 or 1")
        if probe_radius == 0:
            clauses = " OR ".join(
                "(vector_buckets.band = ? AND vector_buckets.bucket = ?)" for _ in buckets
            )
            params: list[Any] = [namespace, embedding_model]
            for band, bucket in buckets:
                params.extend([band, bucket])
            params.append(limit)
            with self._lock:
                rows = self._connection.execute(
                    f"""
                    SELECT vector_buckets.memory_id, count(*) AS weighted_matches
                    FROM vector_buckets
                    JOIN memories ON memories.id = vector_buckets.memory_id
                    WHERE vector_buckets.namespace = ? AND vector_buckets.embedding_model = ?
                      AND ({clauses}) AND memories.status = 'active'
                    GROUP BY vector_buckets.memory_id
                    ORDER BY weighted_matches DESC, memories.importance DESC LIMIT ?
                    """,
                    params,
                ).fetchall()
            denominator = max(1, len(buckets))
            return {
                row["memory_id"]: min(1.0, row["weighted_matches"] / denominator)
                for row in rows
            }
        probes: list[tuple[int, int, float]] = []
        for band, bucket in buckets:
            probes.append((band, bucket, 1.0))
            probes.extend(
                (band, bucket ^ (1 << bit), 0.45) for bit in range(LSH_BITS_PER_BAND)
            )
        values_clause = ",".join("(?, ?, ?)" for _ in probes)
        params: list[Any] = []
        for band, bucket, weight in probes:
            params.extend([band, bucket, weight])
        params.extend([namespace, embedding_model, limit])
        with self._lock:
            rows = self._connection.execute(
                f"""
                WITH query_buckets(band, bucket, probe_weight) AS (
                    VALUES {values_clause}
                )
                SELECT vector_buckets.memory_id, sum(query_buckets.probe_weight) AS weighted_matches
                FROM query_buckets
                JOIN vector_buckets
                  ON vector_buckets.band = query_buckets.band
                 AND vector_buckets.bucket = query_buckets.bucket
                JOIN memories ON memories.id = vector_buckets.memory_id
                WHERE vector_buckets.namespace = ? AND vector_buckets.embedding_model = ?
                  AND memories.status = 'active'
                GROUP BY vector_buckets.memory_id
                ORDER BY weighted_matches DESC, memories.importance DESC LIMIT ?
                """,
                params,
            ).fetchall()
        denominator = max(1, len(buckets))
        return {
            row["memory_id"]: min(1.0, row["weighted_matches"] / denominator) for row in rows
        }

    def exact_vector_candidates(
        self,
        namespace: str,
        embedding_model: str,
        query_vector: Sequence[float],
        *,
        scan_limit: int,
        result_limit: int,
    ) -> dict[str, float]:
        if not query_vector or scan_limit < 1 or result_limit < 1:
            return {}
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id, embedding, embedding_dim FROM memories
                WHERE namespace = ? AND status = 'active' AND embedding_model = ?
                  AND embedding IS NOT NULL
                ORDER BY id LIMIT ?
                """,
                (namespace, embedding_model, scan_limit),
            ).fetchall()
        scored: list[tuple[float, str]] = []
        for row in rows:
            vector = blob_to_vector(row["embedding"], row["embedding_dim"])
            if vector is None or len(vector) != len(query_vector):
                continue
            score = max(0.0, min(1.0, sum(a * b for a, b in zip(query_vector, vector))))
            scored.append((score, row["id"]))
        return {
            memory_id: score
            for score, memory_id in heapq.nlargest(result_limit, scored, key=lambda item: item[0])
        }

    def linked_candidates(
        self,
        namespace: str,
        seed_ids: Sequence[str],
        *,
        limit: int = 64,
        link_types: Sequence[str] = ("summarizes", "derived_from", "supported_by"),
    ) -> dict[str, float]:
        seeds = list(dict.fromkeys(seed_ids))
        types = list(dict.fromkeys(link_types))
        if not seeds or not types:
            return {}
        seed_placeholders = ",".join("?" for _ in seeds)
        type_placeholders = ",".join("?" for _ in types)
        with self._lock:
            if self._connection.execute("SELECT 1 FROM memory_links LIMIT 1").fetchone() is None:
                return {}
            outgoing = self._connection.execute(
                f"""
                SELECT memory_links.target_id AS neighbor_id, max(memory_links.weight) AS weight
                FROM memory_links
                JOIN memories ON memories.id = memory_links.target_id
                WHERE memory_links.source_id IN ({seed_placeholders})
                  AND memory_links.link_type IN ({type_placeholders})
                  AND memories.namespace = ? AND memories.status = 'active'
                GROUP BY memory_links.target_id ORDER BY weight DESC LIMIT ?
                """,
                [*seeds, *types, namespace, limit],
            ).fetchall()
            incoming = self._connection.execute(
                f"""
                SELECT memory_links.source_id AS neighbor_id, max(memory_links.weight) AS weight
                FROM memory_links
                JOIN memories ON memories.id = memory_links.source_id
                WHERE memory_links.target_id IN ({seed_placeholders})
                  AND memory_links.link_type IN ({type_placeholders})
                  AND memories.namespace = ? AND memories.status = 'active'
                GROUP BY memory_links.source_id ORDER BY weight DESC LIMIT ?
                """,
                [*seeds, *types, namespace, limit],
            ).fetchall()
        seed_set = set(seeds)
        result: dict[str, float] = {}
        for row in [*outgoing, *incoming]:
            if row["neighbor_id"] not in seed_set:
                result[row["neighbor_id"]] = max(
                    result.get(row["neighbor_id"], 0.0), min(1.0, float(row["weight"]))
                )
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True)[:limit])

    def recent_candidates(self, namespace: str, limit: int) -> list[str]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT id FROM memories
                WHERE namespace = ? AND status = 'active'
                ORDER BY importance DESC, confidence DESC, updated_at DESC LIMIT ?
                """,
                (namespace, limit),
            ).fetchall()
        return [row["id"] for row in rows]

    def list_memory_ids(
        self,
        namespace: str,
        *,
        limit: int = 1000,
        offset: int = 0,
        statuses: Sequence[MemoryStatus] = (MemoryStatus.ACTIVE,),
    ) -> list[str]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT id FROM memories
                WHERE namespace = ? AND status IN ({placeholders})
                ORDER BY created_at, id LIMIT ? OFFSET ?
                """,
                [namespace, *[status.value for status in statuses], limit, offset],
            ).fetchall()
        return [row["id"] for row in rows]

    def memory_count(
        self,
        namespace: str,
        *,
        statuses: Sequence[MemoryStatus] = (MemoryStatus.ACTIVE,),
    ) -> int:
        if not statuses:
            return 0
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            row = self._connection.execute(
                f"SELECT count(*) AS total FROM memories WHERE namespace = ? AND status IN ({placeholders})",
                [namespace, *[status.value for status in statuses]],
            ).fetchone()
        return int(row["total"])

    def list_for_consolidation(
        self,
        namespace: str,
        *,
        limit: int = 50,
        since: float | None = None,
    ) -> list[MemoryRecord]:
        params: list[Any] = [namespace]
        since_clause = ""
        if since is not None:
            since_clause = "AND created_at >= ?"
            params.append(since)
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT id FROM memories
                WHERE namespace = ? AND status = 'active'
                  AND kind IN ('working', 'episodic', 'source') {since_clause}
                ORDER BY created_at DESC LIMIT ?
                """,
                params,
            ).fetchall()
        return self.get_memories(reversed([row["id"] for row in rows]))

    def touch_memories(self, memory_ids: Sequence[str], at: float | None = None) -> None:
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        now = at or utc_now()
        with self.transaction() as connection:
            connection.execute(
                f"""
                UPDATE memories SET last_accessed = ?, access_count = access_count + 1
                WHERE id IN ({placeholders})
                """,
                [now, *ids],
            )

    def set_status(self, memory_id: str, status: MemoryStatus) -> None:
        now = utc_now()
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT namespace, status FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"memory not found: {memory_id}")
            connection.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, memory_id),
            )
            if status == MemoryStatus.ACTIVE:
                full = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
                entity_rows = connection.execute(
                    "SELECT entity FROM memory_entities WHERE memory_id = ?", (memory_id,)
                ).fetchall()
                self._upsert_fts(
                    connection,
                    memory_id,
                    namespace=full["namespace"],
                    content=full["content"],
                    summary=full["summary"] or "",
                    entities=" ".join(item["entity"] for item in entity_rows),
                )
            else:
                self._upsert_fts(connection, memory_id, remove=True)
            self._write_event(
                connection,
                namespace=row["namespace"],
                memory_id=memory_id,
                event_type=f"memory.{status.value}",
                payload={"previous_status": row["status"], "status": status.value},
                occurred_at=now,
            )

    def update_embedding(
        self, memory_id: str, embedding: Sequence[float], embedding_model: str
    ) -> None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT namespace FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"memory not found: {memory_id}")
            connection.execute(
                "UPDATE memories SET embedding = ?, embedding_dim = ?, embedding_model = ?, updated_at = ? WHERE id = ?",
                (vector_to_blob(embedding), len(embedding), embedding_model, utc_now(), memory_id),
            )
            connection.execute("DELETE FROM vector_buckets WHERE memory_id = ?", (memory_id,))
            connection.executemany(
                """
                INSERT INTO vector_buckets(memory_id, namespace, embedding_model, band, bucket)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (memory_id, row["namespace"], embedding_model, band, bucket)
                    for band, bucket in lsh_buckets(embedding)
                ],
            )

    def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        *,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if source_id == target_id:
            raise ValueError("a memory cannot link to itself")
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT id, namespace FROM memories WHERE id IN (?, ?)", (source_id, target_id)
            ).fetchall()
            if len(rows) != 2:
                raise KeyError("source or target memory does not exist")
            if len({row["namespace"] for row in rows}) != 1:
                raise ValueError("cross-namespace links are not allowed")
            connection.execute(
                """
                INSERT INTO memory_links(source_id, target_id, link_type, weight, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, link_type)
                DO UPDATE SET weight = excluded.weight, metadata_json = excluded.metadata_json
                """,
                (source_id, target_id, link_type, float(weight), json_dumps(metadata or {}), utc_now()),
            )

    def get_links(self, memory_id: str, direction: str) -> list[dict[str, Any]]:
        if direction not in {"outgoing", "incoming"}:
            raise ValueError("direction must be outgoing or incoming")
        column = "source_id" if direction == "outgoing" else "target_id"
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM memory_links WHERE {column} = ? ORDER BY created_at", (memory_id,)
            ).fetchall()
        return [
            {
                "source_id": row["source_id"],
                "target_id": row["target_id"],
                "link_type": row["link_type"],
                "weight": row["weight"],
                "metadata": json_loads(row["metadata_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_retrieval(
        self,
        *,
        retrieval_id: str,
        namespace: str,
        query: str,
        selected_ids: Sequence[str],
        candidates: int,
        token_budget: int,
        used_tokens: int,
        latency_ms: float,
        features: dict[str, Any],
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO retrieval_log(
                    id, namespace, query, selected_ids_json, candidates, token_budget,
                    used_tokens, latency_ms, features_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    retrieval_id,
                    namespace,
                    query,
                    json_dumps(list(selected_ids)),
                    candidates,
                    token_budget,
                    used_tokens,
                    latency_ms,
                    json_dumps(features),
                    utc_now(),
                ),
            )

    def list_events(self, namespace: str, *, after_sequence: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM memory_events
                WHERE namespace = ? AND sequence > ?
                ORDER BY sequence LIMIT ?
                """,
                (namespace, after_sequence, limit),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "event_id": row["event_id"],
                "memory_id": row["memory_id"],
                "namespace": row["namespace"],
                "event_type": row["event_type"],
                "payload": json_loads(row["payload_json"], {}),
                "occurred_at": row["occurred_at"],
            }
            for row in rows
        ]

    @staticmethod
    def _statement_hash(namespace: str, statement: str) -> str:
        import hashlib

        return hashlib.sha256(f"{namespace}\0{' '.join(statement.casefold().split())}".encode()).hexdigest()

    def ensure_policy(
        self,
        namespace: str,
        statement: str,
        *,
        confidence: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> PolicyRecord:
        clean_statement = " ".join(statement.split())
        if not clean_statement:
            raise ValueError("policy statement cannot be empty")
        statement_hash = self._statement_hash(namespace, clean_statement)
        now = utc_now()
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO policies(
                    id, namespace, statement, statement_hash, state, confidence,
                    created_at, updated_at, metadata_json
                ) VALUES (?, ?, ?, ?, 'candidate', ?, ?, ?, ?)
                ON CONFLICT(namespace, statement_hash) DO UPDATE SET
                    confidence = max(policies.confidence, excluded.confidence),
                    updated_at = excluded.updated_at
                """,
                (
                    self.new_id("pol"),
                    namespace,
                    clean_statement,
                    statement_hash,
                    clamp(confidence),
                    now,
                    now,
                    json_dumps(metadata or {}),
                ),
            )
            row = connection.execute(
                "SELECT * FROM policies WHERE namespace = ? AND statement_hash = ?",
                (namespace, statement_hash),
            ).fetchone()
        return self._row_to_policy(row)

    def add_policy_evidence(
        self,
        namespace: str,
        statement: str,
        *,
        outcome: int,
        source_memory_id: str | None = None,
        note: str | None = None,
        confidence: float = 0.5,
        promotion_evidence: int = 3,
        promotion_rate: float = 0.75,
        retirement_evidence: int = 5,
        retirement_rate: float = 0.35,
    ) -> PolicyRecord:
        if outcome not in {-1, 0, 1}:
            raise ValueError("outcome must be -1, 0, or 1")
        policy = self.ensure_policy(namespace, statement, confidence=confidence)
        now = utc_now()
        with self.transaction() as connection:
            if source_memory_id is not None:
                memory = connection.execute(
                    "SELECT namespace FROM memories WHERE id = ?", (source_memory_id,)
                ).fetchone()
                if memory is None:
                    raise KeyError(f"memory not found: {source_memory_id}")
                if memory["namespace"] != namespace:
                    raise ValueError("policy evidence must share the policy namespace")
            connection.execute(
                """
                INSERT INTO policy_evidence(id, policy_id, source_memory_id, outcome, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.new_id("pev"), policy.id, source_memory_id, outcome, note, now),
            )
            row = connection.execute("SELECT * FROM policies WHERE id = ?", (policy.id,)).fetchone()
            successes = int(row["successes"]) + (1 if outcome == 1 else 0)
            failures = int(row["failures"]) + (1 if outcome == -1 else 0)
            neutral = int(row["neutral"]) + (1 if outcome == 0 else 0)
            evidence_count = int(row["evidence_count"]) + 1
            decisive = successes + failures
            success_rate = successes / decisive if decisive else 0.0
            old_state = PolicyState(row["state"])
            state = old_state
            promoted_at = row["promoted_at"]
            retired_at = row["retired_at"]
            if old_state != PolicyState.RETIRED:
                if decisive >= promotion_evidence and success_rate >= promotion_rate:
                    state = PolicyState.PROMOTED
                    promoted_at = promoted_at or now
                elif decisive >= retirement_evidence and success_rate < retirement_rate:
                    state = PolicyState.RETIRED
                    retired_at = now
                elif evidence_count >= 2:
                    state = PolicyState.TESTING
            empirical = (successes + 1) / (decisive + 2) if decisive else float(row["confidence"])
            new_confidence = clamp(0.7 * empirical + 0.3 * float(row["confidence"]))
            connection.execute(
                """
                UPDATE policies SET state = ?, confidence = ?, successes = ?, failures = ?,
                    neutral = ?, evidence_count = ?, updated_at = ?, promoted_at = ?, retired_at = ?
                WHERE id = ?
                """,
                (
                    state.value,
                    new_confidence,
                    successes,
                    failures,
                    neutral,
                    evidence_count,
                    now,
                    promoted_at,
                    retired_at,
                    policy.id,
                ),
            )
            self._write_event(
                connection,
                namespace=namespace,
                memory_id=source_memory_id,
                event_type="policy.evidence_recorded",
                payload={
                    "policy_id": policy.id,
                    "outcome": outcome,
                    "previous_state": old_state.value,
                    "state": state.value,
                },
                occurred_at=now,
            )
            updated = connection.execute("SELECT * FROM policies WHERE id = ?", (policy.id,)).fetchone()
        return self._row_to_policy(updated)

    def get_policy(self, policy_id: str) -> PolicyRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM policies WHERE id = ?", (policy_id,)).fetchone()
        return self._row_to_policy(row) if row else None

    def list_policies(
        self, namespace: str, *, states: Sequence[PolicyState] | None = None
    ) -> list[PolicyRecord]:
        params: list[Any] = [namespace]
        state_clause = ""
        if states:
            placeholders = ",".join("?" for _ in states)
            state_clause = f"AND state IN ({placeholders})"
            params.extend(state.value for state in states)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM policies WHERE namespace = ? {state_clause} ORDER BY updated_at DESC",
                params,
            ).fetchall()
        return [self._row_to_policy(row) for row in rows]

    @staticmethod
    def _row_to_policy(row: sqlite3.Row) -> PolicyRecord:
        return PolicyRecord(
            id=row["id"],
            namespace=row["namespace"],
            statement=row["statement"],
            state=PolicyState(row["state"]),
            confidence=float(row["confidence"]),
            successes=int(row["successes"]),
            failures=int(row["failures"]),
            neutral=int(row["neutral"]),
            evidence_count=int(row["evidence_count"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            promoted_at=row["promoted_at"],
            retired_at=row["retired_at"],
            metadata=dict(json_loads(row["metadata_json"], {})),
        )
