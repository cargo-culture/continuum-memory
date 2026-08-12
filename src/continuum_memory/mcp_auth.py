from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any


READ_SCOPE = "memory:read"
WRITE_SCOPE = "memory:write"


def local_namespace() -> str:
    value = os.environ.get("CONTINUUM_NAMESPACE", "local").strip()
    if not value:
        raise RuntimeError("CONTINUUM_NAMESPACE cannot be empty")
    return value


def authenticated_namespace() -> str:
    from mcp.server.auth.middleware.auth_context import get_access_token

    token = get_access_token()
    if token is None or not token.subject:
        raise PermissionError("an authenticated user subject is required")
    issuer = str((token.claims or {}).get("iss", ""))
    principal = f"{issuer}\0{token.subject}".encode("utf-8")
    return "oauth-" + hashlib.sha256(principal).hexdigest()[:40]


def require_write_scope() -> None:
    from mcp.server.auth.middleware.auth_context import get_access_token

    token = get_access_token()
    if token is not None and WRITE_SCOPE not in token.scopes:
        raise PermissionError(f"the {WRITE_SCOPE!r} OAuth scope is required")


@dataclass(slots=True)
class IntrospectionTokenVerifier:
    """OAuth 2.1 resource-server verifier using RFC 7662 introspection."""

    endpoint: str
    issuer_url: str
    resource_url: str
    client_id: str | None = None
    client_secret: str | None = None

    def __post_init__(self) -> None:
        safe_local = self.endpoint.startswith(("http://127.0.0.1", "http://localhost"))
        if not self.endpoint.startswith("https://") and not safe_local:
            raise ValueError("the token introspection endpoint must use HTTPS")
        if bool(self.client_id) != bool(self.client_secret):
            raise ValueError("introspection client id and secret must be configured together")

    async def verify_token(self, token: str):
        import httpx2
        from mcp.server.auth.provider import AccessToken

        auth = (
            (self.client_id, self.client_secret)
            if self.client_id is not None and self.client_secret is not None
            else None
        )
        try:
            async with httpx2.AsyncClient(
                timeout=httpx2.Timeout(10.0, connect=5.0),
                limits=httpx2.Limits(max_connections=20, max_keepalive_connections=10),
                verify=True,
            ) as client:
                response = await client.post(
                    self.endpoint,
                    data={"token": token, "token_type_hint": "access_token"},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    auth=auth,
                )
            if response.status_code != 200:
                return None
            data: dict[str, Any] = response.json()
        except Exception:
            return None
        if data.get("active") is not True or not data.get("sub"):
            return None
        token_issuer = data.get("iss")
        if token_issuer is not None and str(token_issuer) != self.issuer_url:
            return None
        data.setdefault("iss", self.issuer_url)
        audiences = data.get("aud")
        if isinstance(audiences, str):
            audiences = [audiences]
        if not isinstance(audiences, list) or self.resource_url not in audiences:
            return None
        raw_scopes = data.get("scope") or []
        scopes = raw_scopes.split() if isinstance(raw_scopes, str) else list(raw_scopes)
        return AccessToken(
            token=token,
            client_id=str(data.get("client_id", "unknown")),
            scopes=[str(scope) for scope in scopes],
            expires_at=int(data["exp"]) if data.get("exp") is not None else None,
            resource=self.resource_url,
            subject=str(data["sub"]),
            claims=data,
        )


def oauth_configuration_from_env():
    from mcp.server.auth.settings import AuthSettings

    issuer_url = os.environ.get("CONTINUUM_MCP_ISSUER_URL")
    resource_url = os.environ.get("CONTINUUM_MCP_RESOURCE_URL")
    introspection_url = os.environ.get("CONTINUUM_MCP_INTROSPECTION_URL")
    values = (issuer_url, resource_url, introspection_url)
    if not any(values):
        return None
    if not all(values):
        raise ValueError(
            "CONTINUUM_MCP_ISSUER_URL, CONTINUUM_MCP_RESOURCE_URL, and "
            "CONTINUUM_MCP_INTROSPECTION_URL must be configured together"
        )
    verifier = IntrospectionTokenVerifier(
        endpoint=str(introspection_url),
        issuer_url=str(issuer_url),
        resource_url=str(resource_url),
        client_id=os.environ.get("CONTINUUM_MCP_INTROSPECTION_CLIENT_ID"),
        client_secret=os.environ.get("CONTINUUM_MCP_INTROSPECTION_CLIENT_SECRET"),
    )
    auth = AuthSettings(
        issuer_url=str(issuer_url),
        resource_server_url=str(resource_url),
        required_scopes=[READ_SCOPE],
    )
    return auth, verifier
