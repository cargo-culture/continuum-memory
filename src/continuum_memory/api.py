from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .consolidation import DeterministicConsolidator, OpenAIConsolidator
from .models import MemoryInput, MemoryKind, MemoryStatus, PolicyState
from .service import MemoryService


MEMORY_PATH = re.compile(r"^/v1/memories/([^/]+)$")
PAGE_PATH = re.compile(r"^/v1/pages/([^/]+)$")
SUPERSEDE_PATH = re.compile(r"^/v1/memories/([^/]+)/supersede$")


def _memory_input(body: dict[str, Any]) -> MemoryInput:
    content = body.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    metadata = body.get("metadata", {})
    entities = body.get("entities", [])
    tags = body.get("tags", [])
    if not isinstance(metadata, dict) or not isinstance(entities, list) or not isinstance(tags, list):
        raise ValueError("metadata must be an object; entities and tags must be arrays")
    return MemoryInput(
        content=content,
        namespace=str(body.get("namespace", "default")),
        kind=MemoryKind(body.get("kind", MemoryKind.EPISODIC.value)),
        summary=body.get("summary"),
        source_uri=body.get("source_uri"),
        source_type=body.get("source_type"),
        importance=float(body.get("importance", 0.5)),
        confidence=float(body.get("confidence", 1.0)),
        entities=[str(entity) for entity in entities],
        tags=[str(tag) for tag in tags],
        metadata=metadata,
        valid_from=body.get("valid_from"),
        valid_until=body.get("valid_until"),
        contradiction_group=body.get("contradiction_group"),
        supersedes_id=body.get("supersedes_id"),
        allow_duplicate=bool(body.get("allow_duplicate", False)),
    )


def make_handler(service: MemoryService, *, consolidation_model: str = "gpt-5.6"):
    class MemoryRequestHandler(BaseHTTPRequestHandler):
        server_version = "ContinuumMemory/0.3.0"
        max_body_bytes = 2_500_000

        def _send(self, status: int, payload: dict[str, Any] | list[Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                return {}
            if content_length > self.max_body_bytes:
                raise ValueError("request body exceeds 2.5 MB")
            raw = self.rfile.read(content_length)
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        @staticmethod
        def _query(path: str) -> tuple[str, dict[str, list[str]]]:
            parsed = urlparse(path)
            return parsed.path, parse_qs(parsed.query)

        def do_GET(self) -> None:  # noqa: N802
            path, query = self._query(self.path)
            try:
                namespace = query.get("namespace", ["default"])[0]
                if path == "/health":
                    self._send(HTTPStatus.OK, service.health(namespace))
                    return
                memory_match = MEMORY_PATH.match(path)
                if memory_match:
                    memory = service.store.get_memory(memory_match.group(1))
                    if memory is None:
                        raise KeyError("memory not found")
                    self._send(HTTPStatus.OK, memory.to_dict())
                    return
                page_match = PAGE_PATH.match(path)
                if page_match:
                    self._send(HTTPStatus.OK, service.page(page_match.group(1)).to_dict())
                    return
                if path == "/v1/policies":
                    raw_states = query.get("state", [])
                    states = [PolicyState(state) for state in raw_states] if raw_states else None
                    self._send(
                        HTTPStatus.OK,
                        {"policies": [policy.to_dict() for policy in service.list_policies(namespace, states=states)]},
                    )
                    return
                if path == "/v1/events":
                    after = int(query.get("after", ["0"])[0])
                    limit = min(1000, int(query.get("limit", ["100"])[0]))
                    self._send(
                        HTTPStatus.OK,
                        {"events": service.store.list_events(namespace, after_sequence=after, limit=limit)},
                    )
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except Exception as exc:
                self._handle_exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            path, _ = self._query(self.path)
            try:
                body = self._read_json()
                if path == "/v1/memories":
                    memory = service.remember(_memory_input(body))
                    self._send(HTTPStatus.CREATED, memory.to_dict())
                    return
                supersede_match = SUPERSEDE_PATH.match(path)
                if supersede_match:
                    content = body.get("content")
                    if not isinstance(content, str):
                        raise ValueError("content must be a string")
                    memory = service.supersede(
                        supersede_match.group(1),
                        content,
                        confidence=body.get("confidence"),
                        source_uri=body.get("source_uri"),
                        metadata=body.get("metadata"),
                    )
                    self._send(HTTPStatus.CREATED, memory.to_dict())
                    return
                if path in {"/v1/recall", "/v1/context"}:
                    packet = service.context(
                        str(body.get("query", "")),
                        namespace=str(body.get("namespace", "default")),
                        limit=int(body.get("limit", 8)),
                        token_budget=int(body.get("token_budget", 1600)),
                        context_entities=[str(value) for value in body.get("context_entities", [])],
                        exact_scan_mode=body.get("exact_scan_mode"),
                    )
                    if path == "/v1/recall":
                        self._send(
                            HTTPStatus.OK,
                            {
                                "retrieval_id": packet.retrieval_id,
                                "used_tokens": packet.used_tokens,
                                "hits": [hit.to_dict() for hit in packet.hits],
                            },
                        )
                    else:
                        self._send(HTTPStatus.OK, packet.to_dict())
                    return
                if path == "/v1/policies/evidence":
                    outcome_value = body.get("outcome")
                    outcome_map = {"success": 1, "neutral": 0, "failure": -1}
                    outcome = outcome_map.get(outcome_value, outcome_value)
                    policy = service.record_policy_evidence(
                        str(body.get("statement", "")),
                        outcome=int(outcome),
                        namespace=str(body.get("namespace", "default")),
                        source_memory_id=body.get("source_memory_id"),
                        note=body.get("note"),
                    )
                    self._send(HTTPStatus.OK, policy.to_dict())
                    return
                if path == "/v1/entity-aliases":
                    alias = body.get("alias")
                    canonical = body.get("canonical")
                    if not isinstance(alias, str) or not isinstance(canonical, str):
                        raise ValueError("alias and canonical must be strings")
                    namespace = str(body.get("namespace", "default"))
                    service.register_entity_alias(alias, canonical, namespace=namespace)
                    self._send(
                        HTTPStatus.CREATED,
                        {"namespace": namespace, "alias": alias, "canonical": canonical},
                    )
                    return
                if path == "/v1/consolidate":
                    provider = str(body.get("provider", "deterministic")).casefold()
                    if provider == "openai":
                        consolidator = OpenAIConsolidator(model=consolidation_model)
                    elif provider == "deterministic":
                        consolidator = DeterministicConsolidator()
                    else:
                        raise ValueError("provider must be deterministic or openai")
                    result = service.consolidate(
                        namespace=str(body.get("namespace", "default")),
                        consolidator=consolidator,
                        limit=int(body.get("limit", 50)),
                        since=body.get("since"),
                    )
                    self._send(HTTPStatus.CREATED, result.to_dict())
                    return
                if path == "/v1/reindex":
                    count = service.reindex_embeddings(
                        namespace=str(body.get("namespace", "default")),
                        batch_size=int(body.get("batch_size", 64)),
                    )
                    self._send(HTTPStatus.OK, {"reindexed": count})
                    return
                if path == "/v1/memories/status":
                    memory_id = body.get("memory_id")
                    status = body.get("status")
                    if not isinstance(memory_id, str) or not isinstance(status, str):
                        raise ValueError("memory_id and status must be strings")
                    service.store.set_status(memory_id, MemoryStatus(status))
                    self._send(HTTPStatus.OK, {"status": "updated"})
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except Exception as exc:
                self._handle_exception(exc)

        def _handle_exception(self, exc: Exception) -> None:
            if isinstance(exc, KeyError):
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found", "message": str(exc)})
            elif isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
                self._send(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(exc)})
            else:
                self.log_error("request failed: %s", type(exc).__name__)
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})

        def log_message(self, format: str, *args: object) -> None:
            super().log_message(format, *args)

    return MemoryRequestHandler


def create_server(
    service: MemoryService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    consolidation_model: str = "gpt-5.6",
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        (host, port), make_handler(service, consolidation_model=consolidation_model)
    )
