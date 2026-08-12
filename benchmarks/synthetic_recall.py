#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from continuum_memory import HashingEmbedder, MemoryInput, MemoryKind, MemoryService, SQLiteMemoryStore
from continuum_memory.retrieval import RetrievalConfig


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def run(
    records: int,
    queries: int,
    dimension: int,
    exact_scan_queries: int = 0,
) -> dict[str, float | int]:
    topics = [
        "scheduler",
        "telemetry",
        "archive",
        "routing",
        "calibration",
        "governance",
        "provenance",
        "deployment",
    ]
    with tempfile.TemporaryDirectory() as directory:
        store = SQLiteMemoryStore(Path(directory) / "benchmark.db")
        service = MemoryService(store, HashingEmbedder(dimension))
        ids: list[str] = []
        write_latencies: list[float] = []
        write_started = time.perf_counter()
        for index in range(records):
            marker = f"marker{index:06d}"
            topic = topics[index % len(topics)]
            started = time.perf_counter()
            memory = service.remember(
                MemoryInput(
                    namespace="benchmark",
                    kind=MemoryKind.EPISODIC,
                    content=(
                        f"Record {marker} describes the {topic} subsystem. "
                        f"Its validation sequence is batch {index % 97} and its owner is Unit {index % 31}."
                    ),
                    importance=0.4 + (index % 5) * 0.1,
                    entities=[f"Unit {index % 31}"],
                )
            )
            write_latencies.append((time.perf_counter() - started) * 1000.0)
            ids.append(memory.id)
        total_write_seconds = time.perf_counter() - write_started

        query_latencies: list[float] = []
        hits = 0
        query_count = min(queries, records)
        for query_index in range(query_count):
            record_index = (query_index * 7919) % records
            marker = f"marker{record_index:06d}"
            topic = topics[record_index % len(topics)]
            started = time.perf_counter()
            recalled = service.recall(
                f"Find {marker} in the {topic} subsystem",
                namespace="benchmark",
                limit=5,
                token_budget=600,
            )
            query_latencies.append((time.perf_counter() - started) * 1000.0)
            if ids[record_index] in {hit.memory.id for hit in recalled}:
                hits += 1

        exact_scan_latencies: list[float] = []
        exact_scan_hits = 0
        exact_query_count = min(exact_scan_queries, records)
        for query_index in range(exact_query_count):
            record_index = (query_index * 7919) % records
            marker = f"marker{record_index:06d}"
            topic = topics[record_index % len(topics)]
            started = time.perf_counter()
            recalled = service.recall(
                f"Find {marker} in the {topic} subsystem",
                namespace="benchmark",
                limit=5,
                token_budget=600,
                exact_scan_mode="always",
            )
            exact_scan_latencies.append((time.perf_counter() - started) * 1000.0)
            if ids[record_index] in {hit.memory.id for hit in recalled}:
                exact_scan_hits += 1
        store.close()
    result: dict[str, float | int] = {
        "records": records,
        "queries": query_count,
        "embedding_dimensions": dimension,
        "hit_at_5": hits / max(1, query_count),
        "writes_per_second": records / max(total_write_seconds, 1e-9),
        "write_p50_ms": percentile(write_latencies, 0.50),
        "write_p95_ms": percentile(write_latencies, 0.95),
        "recall_mean_ms": statistics.fmean(query_latencies) if query_latencies else 0.0,
        "recall_p50_ms": percentile(query_latencies, 0.50),
        "recall_p95_ms": percentile(query_latencies, 0.95),
        "recall_p99_ms": percentile(query_latencies, 0.99),
    }
    if exact_scan_queries:
        result.update(
            {
                "exact_scan_queries": exact_query_count,
                "exact_scan_hit_at_5": exact_scan_hits / max(1, exact_query_count),
                "exact_scan_mean_ms": (
                    statistics.fmean(exact_scan_latencies) if exact_scan_latencies else 0.0
                ),
                "exact_scan_p50_ms": percentile(exact_scan_latencies, 0.50),
                "exact_scan_p95_ms": percentile(exact_scan_latencies, 0.95),
                "exact_scan_p99_ms": percentile(exact_scan_latencies, 0.99),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Continuum Memory recall benchmark")
    parser.add_argument("--records", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dimensions", type=int, default=192)
    parser.add_argument(
        "--exact-scan-queries",
        type=int,
        default=0,
        help="Also force full exact-vector scans for this many queries",
    )
    args = parser.parse_args()
    if args.records < 1 or args.queries < 1 or args.exact_scan_queries < 0:
        parser.error("records and queries must be positive; exact scan queries cannot be negative")
    if args.exact_scan_queries and args.records > RetrievalConfig().exact_scan_max_records:
        parser.error(
            "forced exact-scan benchmark exceeds RetrievalConfig.exact_scan_max_records"
        )
    print(
        json.dumps(
            run(args.records, args.queries, args.dimensions, args.exact_scan_queries),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
