from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .api import create_server
from .consolidation import DeterministicConsolidator, OpenAIConsolidator
from .factory import create_federated_service, create_service
from .federation_api import create_federation_server
from .models import MemoryInput, MemoryKind, PolicyState


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _add_memory_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("content")
    parser.add_argument("--kind", choices=[kind.value for kind in MemoryKind], default="episodic")
    parser.add_argument("--summary")
    parser.add_argument("--source-uri")
    parser.add_argument("--source-type")
    parser.add_argument("--importance", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=1.0)
    parser.add_argument("--entity", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--valid-from", type=float)
    parser.add_argument("--valid-until", type=float)


def _memory_from_args(args: argparse.Namespace) -> MemoryInput:
    return MemoryInput(
        content=args.content,
        namespace=args.namespace,
        kind=MemoryKind(args.kind),
        summary=args.summary,
        source_uri=args.source_uri,
        source_type=args.source_type,
        importance=args.importance,
        confidence=args.confidence,
        entities=args.entity,
        tags=args.tag,
        valid_from=args.valid_from,
        valid_until=args.valid_until,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="continuum-memory",
        description="Persistent, provenance-aware memory for language-model systems.",
    )
    parser.add_argument("--db", type=Path, help="SQLite database path")
    parser.add_argument("--library-dir", type=Path, help="memory-book library directory")
    parser.add_argument("--max-books", type=int, help="maximum automatically routed books")
    parser.add_argument("--workers", type=int, help="concurrent memory-book workers")
    parser.add_argument("--namespace", default="default", help="memory namespace")
    parser.add_argument(
        "--embedding-provider", choices=("hashing", "openai"), help="embedding provider"
    )
    parser.add_argument("--embedding-model", help="remote embedding model")
    parser.add_argument("--dimensions", type=int, help="embedding dimensions")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="initialize the database")
    subparsers.add_parser("health", help="show storage health")

    remember = subparsers.add_parser("remember", help="store one memory")
    _add_memory_arguments(remember)

    recall = subparsers.add_parser("recall", help="retrieve relevant memories")
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=8)
    recall.add_argument("--token-budget", type=int, default=1600)
    recall.add_argument("--entity", action="append", default=[])
    recall.add_argument("--json", action="store_true")
    recall.add_argument("--exact-scan-mode", choices=("off", "adaptive", "always"))

    context = subparsers.add_parser("context", help="render an LLM-ready context packet")
    context.add_argument("query")
    context.add_argument("--limit", type=int, default=8)
    context.add_argument("--token-budget", type=int, default=1600)
    context.add_argument("--entity", action="append", default=[])
    context.add_argument("--json", action="store_true")
    context.add_argument("--exact-scan-mode", choices=("off", "adaptive", "always"))

    page = subparsers.add_parser("page", help="expand a memory and its provenance links")
    page.add_argument("memory_id")

    supersede = subparsers.add_parser("supersede", help="replace a claim without erasing history")
    supersede.add_argument("memory_id")
    supersede.add_argument("content")
    supersede.add_argument("--confidence", type=float)
    supersede.add_argument("--source-uri")

    consolidate = subparsers.add_parser("consolidate", help="build hierarchical summaries")
    consolidate.add_argument("--provider", choices=("deterministic", "openai"), default="deterministic")
    consolidate.add_argument("--limit", type=int, default=50)
    consolidate.add_argument("--since", type=float)

    evidence = subparsers.add_parser("policy-evidence", help="record a behavioral outcome")
    evidence.add_argument("statement")
    evidence.add_argument("outcome", choices=("success", "neutral", "failure"))
    evidence.add_argument("--source-memory-id")
    evidence.add_argument("--note")

    policies = subparsers.add_parser("policies", help="list behavioral policy state")
    policies.add_argument("--state", action="append", choices=[state.value for state in PolicyState])

    alias = subparsers.add_parser("alias", help="register an entity alias")
    alias.add_argument("alias")
    alias.add_argument("canonical")

    events = subparsers.add_parser("events", help="read the append-only event stream")
    events.add_argument("--after", type=int, default=0)
    events.add_argument("--limit", type=int, default=100)

    reindex = subparsers.add_parser("reindex", help="recompute embeddings for active memories")
    reindex.add_argument("--batch-size", type=int, default=64)

    serve = subparsers.add_parser("serve", help="run the local JSON API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    subparsers.add_parser("library-init", help="create the standard specialized memory books")
    subparsers.add_parser("books", help="list memory books and active record counts")

    book_create = subparsers.add_parser("book-create", help="create a specialized memory book")
    book_create.add_argument("book_id")
    book_create.add_argument("--title", required=True)
    book_create.add_argument("--description", default="")
    book_create.add_argument("--keyword", action="append", default=[])
    book_create.add_argument("--priority", type=float, default=1.0)
    book_create.add_argument("--always-search", action="store_true")

    book_remember = subparsers.add_parser("book-remember", help="store a memory in one book")
    book_remember.add_argument("book_id")
    _add_memory_arguments(book_remember)

    for name, help_text in (
        ("library-recall", "retrieve across routed memory books"),
        ("library-context", "render a federated LLM context packet"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("query")
        command.add_argument("--book", action="append", default=None)
        command.add_argument("--limit", type=int, default=8)
        command.add_argument("--token-budget", type=int, default=1600)
        command.add_argument("--entity", action="append", default=[])
        command.add_argument("--json", action="store_true")
        command.add_argument("--exact-scan-mode", choices=("off", "adaptive", "always"))

    book_page = subparsers.add_parser("book-page", help="expand a page from one memory book")
    book_page.add_argument("book_id")
    book_page.add_argument("memory_id")

    book_consolidate = subparsers.add_parser(
        "book-consolidate", help="build hierarchical summaries inside one memory book"
    )
    book_consolidate.add_argument("book_id")
    book_consolidate.add_argument(
        "--provider", choices=("deterministic", "openai"), default="deterministic"
    )
    book_consolidate.add_argument("--limit", type=int, default=50)
    book_consolidate.add_argument("--since", type=float)

    serve_library = subparsers.add_parser("serve-library", help="run the federated JSON API")
    serve_library.add_argument("--host", default="127.0.0.1")
    serve_library.add_argument("--port", type=int, default=8765)
    return parser


FEDERATED_COMMANDS = {
    "library-init",
    "books",
    "book-create",
    "book-remember",
    "library-recall",
    "library-context",
    "book-page",
    "book-consolidate",
    "serve-library",
}


def _run_federated(args: argparse.Namespace) -> int:
    library, settings = create_federated_service(
        library_dir=args.library_dir,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.dimensions,
        max_books=args.max_books,
        max_workers=args.workers,
    )
    try:
        if args.command == "library-init":
            created = library.create_standard_books()
            _json(
                {
                    "created": [book.to_dict() for book in created],
                    "books": library.list_books(args.namespace),
                }
            )
        elif args.command == "books":
            _json(library.list_books(args.namespace))
        elif args.command == "book-create":
            _json(
                library.create_book(
                    args.book_id,
                    title=args.title,
                    description=args.description,
                    keywords=args.keyword,
                    priority=args.priority,
                    always_search=args.always_search,
                ).to_dict()
            )
        elif args.command == "book-remember":
            _json(library.remember(args.book_id, _memory_from_args(args)).to_dict())
        elif args.command in {"library-recall", "library-context"}:
            packet = library.context(
                args.query,
                namespace=args.namespace,
                limit=args.limit,
                token_budget=args.token_budget,
                context_entities=args.entity,
                book_ids=args.book,
                max_books=args.max_books,
                exact_scan_mode=args.exact_scan_mode,
            )
            if args.command == "library-context":
                _json(packet.to_dict()) if args.json else print(packet.render())
            elif args.json:
                _json(
                    {
                        "searched_book_ids": packet.searched_book_ids,
                        "latency_ms": packet.latency_ms,
                        "hits": [hit.to_dict() for hit in packet.hits],
                    }
                )
            else:
                for hit in packet.hits:
                    text = hit.memory.summary or hit.memory.content
                    print(
                        f"{hit.rank:>2} {hit.score:.4f} {hit.book_id}/{hit.memory.id} "
                        f"{hit.memory.kind.value}: {text}"
                    )
        elif args.command == "book-page":
            _json(library.page(args.book_id, args.memory_id).to_dict())
        elif args.command == "book-consolidate":
            consolidator = (
                OpenAIConsolidator(model=settings.consolidation_model)
                if args.provider == "openai"
                else DeterministicConsolidator()
            )
            _json(
                library.consolidate(
                    args.book_id,
                    namespace=args.namespace,
                    consolidator=consolidator,
                    limit=args.limit,
                    since=args.since,
                ).to_dict()
            )
        elif args.command == "serve-library":
            server = create_federation_server(library, host=args.host, port=args.port)
            print(f"Continuum Memory library listening on http://{args.host}:{args.port}")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        return 0
    finally:
        library.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in FEDERATED_COMMANDS:
        try:
            return _run_federated(args)
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    service, settings = create_service(
        db_path=args.db,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        embedding_dimensions=args.dimensions,
    )
    try:
        if args.command in {"init", "health"}:
            _json(service.health(args.namespace))
        elif args.command == "remember":
            memory = service.remember(_memory_from_args(args))
            _json(memory.to_dict())
        elif args.command == "recall":
            hits = service.recall(
                args.query,
                namespace=args.namespace,
                limit=args.limit,
                token_budget=args.token_budget,
                context_entities=args.entity,
                exact_scan_mode=args.exact_scan_mode,
            )
            if args.json:
                _json([hit.to_dict() for hit in hits])
            else:
                for hit in hits:
                    text = hit.memory.summary or hit.memory.content
                    print(f"{hit.rank:>2} {hit.score:.4f} {hit.memory.id} {hit.memory.kind.value}: {text}")
        elif args.command == "context":
            packet = service.context(
                args.query,
                namespace=args.namespace,
                limit=args.limit,
                token_budget=args.token_budget,
                context_entities=args.entity,
                exact_scan_mode=args.exact_scan_mode,
            )
            _json(packet.to_dict()) if args.json else print(packet.render())
        elif args.command == "page":
            _json(service.page(args.memory_id).to_dict())
        elif args.command == "supersede":
            _json(
                service.supersede(
                    args.memory_id,
                    args.content,
                    confidence=args.confidence,
                    source_uri=args.source_uri,
                ).to_dict()
            )
        elif args.command == "consolidate":
            consolidator = (
                OpenAIConsolidator(model=settings.consolidation_model)
                if args.provider == "openai"
                else DeterministicConsolidator()
            )
            _json(
                service.consolidate(
                    namespace=args.namespace,
                    consolidator=consolidator,
                    limit=args.limit,
                    since=args.since,
                ).to_dict()
            )
        elif args.command == "policy-evidence":
            outcome = {"success": 1, "neutral": 0, "failure": -1}[args.outcome]
            _json(
                service.record_policy_evidence(
                    args.statement,
                    outcome=outcome,
                    namespace=args.namespace,
                    source_memory_id=args.source_memory_id,
                    note=args.note,
                ).to_dict()
            )
        elif args.command == "policies":
            states = [PolicyState(state) for state in args.state] if args.state else None
            _json([policy.to_dict() for policy in service.list_policies(args.namespace, states=states)])
        elif args.command == "alias":
            service.register_entity_alias(args.alias, args.canonical, namespace=args.namespace)
            _json(
                {
                    "namespace": args.namespace,
                    "alias": args.alias,
                    "canonical": args.canonical,
                }
            )
        elif args.command == "events":
            _json(service.store.list_events(args.namespace, after_sequence=args.after, limit=args.limit))
        elif args.command == "reindex":
            _json({"reindexed": service.reindex_embeddings(args.namespace, batch_size=args.batch_size)})
        elif args.command == "serve":
            server = create_server(
                service,
                host=args.host,
                port=args.port,
                consolidation_model=settings.consolidation_model,
            )
            print(f"Continuum Memory listening on http://{args.host}:{args.port}")
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
        return 0
    except (ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        service.store.close()


if __name__ == "__main__":
    raise SystemExit(main())
