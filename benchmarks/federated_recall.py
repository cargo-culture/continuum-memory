#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from continuum_memory import FederatedMemoryService, HashingEmbedder, MemoryInput
from continuum_memory.utils import approximate_token_count


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def metrics(latencies: list[float], hits: int) -> dict[str, float | int]:
    return {
        "queries": len(latencies),
        "hit_at_5": hits / max(1, len(latencies)),
        "mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
    }


def run(books: int, records_per_book: int, queries: int, dimensions: int) -> dict:
    if books < 3:
        raise ValueError("benchmark requires at least three books")
    with tempfile.TemporaryDirectory() as directory:
        library = FederatedMemoryService(
            Path(directory) / "books",
            HashingEmbedder(dimensions),
            max_books=3,
            max_workers=books,
        )
        book_ids = ["working", "identity"] + [f"topic{index}" for index in range(books - 2)]
        for index, book_id in enumerate(book_ids):
            library.create_book(
                book_id,
                title=f"{book_id.title()} memory",
                description=f"Records and evidence about {book_id}.",
                keywords=[book_id, f"domain{index}"],
                priority=1.2 if book_id in {"working", "identity"} else 1.0,
                always_search=book_id in {"working", "identity"},
            )

        target_ids: dict[tuple[str, int], str] = {}
        sample_tokens: list[int] = []
        write_started = time.perf_counter()
        for book_index, book_id in enumerate(book_ids):
            for record_index in range(records_per_book):
                marker = f"marker{book_index:02d}{record_index:06d}"
                content = (
                    f"{book_id} record {marker} describes domain{book_index} evidence. "
                    f"Its sequence is batch {record_index % 97} and owner is Unit {record_index % 31}."
                )
                memory = library.remember(
                    book_id,
                    MemoryInput(
                        namespace="benchmark",
                        content=content,
                        entities=[f"Unit {record_index % 31}"],
                    ),
                )
                target_ids[(book_id, record_index)] = memory.id
                if len(sample_tokens) < 100:
                    sample_tokens.append(approximate_token_count(content))
        write_seconds = time.perf_counter() - write_started

        query_specs: list[tuple[str, str, str]] = []
        query_count = min(queries, records_per_book * books)
        for query_index in range(query_count):
            book_id = book_ids[query_index % books]
            book_index = book_ids.index(book_id)
            record_index = (query_index * 7919) % records_per_book
            marker = f"marker{book_index:02d}{record_index:06d}"
            query = f"Find {marker} in the {book_id} domain{book_index} memory"
            query_specs.append((book_id, target_ids[(book_id, record_index)], query))

        def query_run(*, explicit_books: list[str] | None, workers: int) -> dict[str, float | int]:
            library.max_workers = workers
            latencies: list[float] = []
            hits = 0
            for _book_id, target_id, query in query_specs:
                started = time.perf_counter()
                recalled = library.recall(
                    query,
                    namespace="benchmark",
                    limit=5,
                    token_budget=600,
                    book_ids=explicit_books,
                    exact_scan_mode="off",
                )
                latencies.append((time.perf_counter() - started) * 1000.0)
                if target_id in {hit.memory.id for hit in recalled}:
                    hits += 1
            return metrics(latencies, hits)

        routed = query_run(explicit_books=None, workers=books)
        all_concurrent = query_run(explicit_books=book_ids, workers=books)
        all_sequential = query_run(explicit_books=book_ids, workers=1)
        library.close()

    average_record_tokens = statistics.fmean(sample_tokens) if sample_tokens else 0.0
    return {
        "books": books,
        "records_per_book": records_per_book,
        "total_records": books * records_per_book,
        "embedding_dimensions": dimensions,
        "average_record_tokens": average_record_tokens,
        "addressable_content_tokens": books * records_per_book * average_record_tokens,
        "writes_per_second": books * records_per_book / max(write_seconds, 1e-9),
        "routed_three_books": routed,
        "all_books_concurrent": all_concurrent,
        "all_books_sequential": all_sequential,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Federated Continuum Memory benchmark")
    parser.add_argument("--books", type=int, default=4)
    parser.add_argument("--records-per-book", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dimensions", type=int, default=192)
    args = parser.parse_args()
    if min(args.books, args.records_per_book, args.queries) < 1:
        parser.error("books, records per book, and queries must be positive")
    print(
        json.dumps(
            run(args.books, args.records_per_book, args.queries, args.dimensions),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
