from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from continuum_memory.mcp_auth import IntrospectionTokenVerifier, require_write_scope


class _Response:
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self) -> dict:
        return self.payload


class _Client:
    payload: dict = {}

    def __init__(self, **_kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, *_args, **_kwargs):
        return _Response(self.payload)


class MCPAuthTests(unittest.IsolatedAsyncioTestCase):
    async def test_introspection_binds_subject_issuer_audience_and_scopes(self) -> None:
        _Client.payload = {
            "active": True,
            "sub": "alice",
            "client_id": "chatgpt",
            "aud": ["https://memory.example.com/mcp"],
            "scope": "memory:read memory:write",
            "exp": 2_000_000_000,
        }
        verifier = IntrospectionTokenVerifier(
            endpoint="https://auth.example.com/introspect",
            issuer_url="https://auth.example.com/",
            resource_url="https://memory.example.com/mcp",
        )
        with patch("httpx2.AsyncClient", _Client):
            token = await verifier.verify_token("opaque-token")
        self.assertIsNotNone(token)
        self.assertEqual("alice", token.subject)
        self.assertEqual("https://auth.example.com/", token.claims["iss"])
        self.assertIn("memory:write", token.scopes)

    async def test_introspection_rejects_wrong_audience_or_issuer(self) -> None:
        verifier = IntrospectionTokenVerifier(
            endpoint="https://auth.example.com/introspect",
            issuer_url="https://auth.example.com/",
            resource_url="https://memory.example.com/mcp",
        )
        for payload in (
            {"active": True, "sub": "alice", "aud": "https://other.example.com/mcp"},
            {
                "active": True,
                "sub": "alice",
                "aud": "https://memory.example.com/mcp",
                "iss": "https://evil.example.com/",
            },
        ):
            _Client.payload = payload
            with patch("httpx2.AsyncClient", _Client):
                self.assertIsNone(await verifier.verify_token("opaque-token"))

    def test_write_scope_fails_closed_without_an_authenticated_token(self) -> None:
        with patch(
            "mcp.server.auth.middleware.auth_context.get_access_token", return_value=None
        ):
            # Local stdio has no OAuth context and needs none.
            require_write_scope(authenticated=False)
            # Under an authenticated transport, a missing token means the request
            # never passed the auth middleware.
            with self.assertRaises(PermissionError):
                require_write_scope(authenticated=True)

    def test_write_scope_requires_the_write_scope(self) -> None:
        read_only = SimpleNamespace(subject="user-1", scopes=["memory:read"], claims={})
        writer = SimpleNamespace(
            subject="user-1", scopes=["memory:read", "memory:write"], claims={}
        )
        with patch(
            "mcp.server.auth.middleware.auth_context.get_access_token", return_value=read_only
        ):
            with self.assertRaises(PermissionError):
                require_write_scope(authenticated=True)
        with patch(
            "mcp.server.auth.middleware.auth_context.get_access_token", return_value=writer
        ):
            require_write_scope(authenticated=True)

    def test_introspection_endpoint_requires_https(self) -> None:
        with self.assertRaises(ValueError):
            IntrospectionTokenVerifier(
                endpoint="http://auth.example.com/introspect",
                issuer_url="https://auth.example.com/",
                resource_url="https://memory.example.com/mcp",
            )


if __name__ == "__main__":
    unittest.main()
