from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .api import _memory_input
from .federation import FederatedMemoryService


BOOK_MEMORY_PATH = re.compile(r"^/v1/books/([^/]+)/memories$")
BOOK_PAGE_PATH = re.compile(r"^/v1/books/([^/]+)/pages/([^/]+)$")


def make_federation_handler(service: FederatedMemoryService):
    class FederationRequestHandler(BaseHTTPRequestHandler):
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
            value = json.loads(self.rfile.read(content_length).decode("utf-8"))
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
                if path == "/v1/books":
                    self._send(
                        HTTPStatus.OK,
                        {"books": service.list_books(namespace)},
                    )
                    return
                page_match = BOOK_PAGE_PATH.match(path)
                if page_match:
                    page = service.page(page_match.group(1), page_match.group(2))
                    self._send(HTTPStatus.OK, page.to_dict())
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except Exception as exc:
                self._handle_exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            path, _ = self._query(self.path)
            try:
                body = self._read_json()
                if path == "/v1/books":
                    book_id = body.get("book_id")
                    title = body.get("title")
                    description = body.get("description", "")
                    keywords = body.get("keywords", [])
                    if not isinstance(book_id, str) or not isinstance(title, str):
                        raise ValueError("book_id and title must be strings")
                    if not isinstance(description, str) or not isinstance(keywords, list):
                        raise ValueError("description must be a string and keywords an array")
                    book = service.create_book(
                        book_id,
                        title=title,
                        description=description,
                        keywords=[str(item) for item in keywords],
                        priority=float(body.get("priority", 1.0)),
                        always_search=bool(body.get("always_search", False)),
                    )
                    self._send(HTTPStatus.CREATED, book.to_dict())
                    return
                memory_match = BOOK_MEMORY_PATH.match(path)
                if memory_match:
                    memory = service.remember(memory_match.group(1), _memory_input(body))
                    self._send(HTTPStatus.CREATED, memory.to_dict())
                    return
                if path in {"/v1/library/recall", "/v1/library/context"}:
                    raw_books = body.get("book_ids")
                    if raw_books is not None and not isinstance(raw_books, list):
                        raise ValueError("book_ids must be an array")
                    packet = service.context(
                        str(body.get("query", "")),
                        namespace=str(body.get("namespace", "default")),
                        limit=int(body.get("limit", 8)),
                        token_budget=int(body.get("token_budget", 1600)),
                        context_entities=[str(item) for item in body.get("context_entities", [])],
                        book_ids=(
                            [str(item) for item in raw_books]
                            if raw_books is not None
                            else None
                        ),
                        max_books=(
                            int(body["max_books"])
                            if body.get("max_books") is not None
                            else None
                        ),
                        exact_scan_mode=body.get("exact_scan_mode"),
                    )
                    if path == "/v1/library/recall":
                        self._send(
                            HTTPStatus.OK,
                            {
                                "retrieval_id": packet.retrieval_id,
                                "used_tokens": packet.used_tokens,
                                "searched_book_ids": packet.searched_book_ids,
                                "routes": [route.to_dict() for route in packet.routes],
                                "latency_ms": packet.latency_ms,
                                "hits": [hit.to_dict() for hit in packet.hits],
                            },
                        )
                    else:
                        self._send(HTTPStatus.OK, packet.to_dict())
                    return
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            except Exception as exc:
                self._handle_exception(exc)

        def _handle_exception(self, exc: Exception) -> None:
            if isinstance(exc, KeyError):
                self._send(HTTPStatus.NOT_FOUND, {"error": "not_found", "message": str(exc)})
            elif isinstance(exc, (ValueError, TypeError, json.JSONDecodeError)):
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "invalid_request", "message": str(exc)},
                )
            else:
                self.log_error("request failed: %s", type(exc).__name__)
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error"})

    return FederationRequestHandler


def create_federation_server(
    service: FederatedMemoryService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_federation_handler(service))
