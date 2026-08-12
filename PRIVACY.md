# Continuum Memory Privacy Notice

Effective date: August 12, 2026

Continuum Memory is an open-source memory service published by cargo-culture. This notice describes the reference ChatGPT plugin and any hosted service whose listing links to this document. A third party that self-hosts Continuum Memory is responsible for publishing its own privacy terms.

## Data the service processes

When a user links and uses the hosted plugin, the service may process:

- Memory content the user asks ChatGPT or Codex to store, plus summaries, tags, entities, confidence, timestamps, provenance URLs, and revision history.
- Search queries and the identifiers of selected memory records for retrieval quality and auditability.
- An opaque namespace derived one-way from the authenticated token issuer and subject. The reference server does not store OAuth access tokens.
- Minimal operational data such as timestamps, error categories, and request correlation identifiers. Production operators must not log access tokens or raw request bodies at the proxy layer.

Continuum Memory does not sell personal information or use stored memories for advertising. The reference service uses data only to store, retrieve, correct, secure, and operate the user's memory library.

## Storage, retention, and deletion

The service keeps active memories until the user supersedes them or asks that they be removed. Superseded and tombstoned records may remain in the audit history so corrections are explainable. A user may request permanent deletion of their hosted namespace through the support URL in the plugin listing. The operator will remove the namespace from primary storage and allow encrypted backups to expire under the operator's published backup schedule.

Local and self-hosted installations keep data only in the storage configured by their operator. Uninstalling a local plugin does not itself delete its data directory.

## Sharing and subprocessors

The reference hashing embedder runs locally. If the operator enables the optional OpenAI embedding provider, memory text is sent to the configured OpenAI API project to produce embeddings and is handled under that project's data terms. Hosting, identity, logging, and backup providers may process data only to operate the service and must be disclosed in the production listing or service-specific notice.

The service may disclose information when required by law, to protect users and the service, or as part of a business transfer with appropriate notice and safeguards.

## Security and choices

The hosted plugin requires OAuth 2.1, validates issuer, audience, expiration, and scopes, and isolates each user in an opaque namespace. Users should not store passwords, API keys, payment credentials, authentication codes, or other secrets as memories.

Users may inspect provenance, correct obsolete information through supersession, and request access, export, or permanent deletion through the support URL in the plugin listing. Requests may require identity verification.

## Contact

Questions and requests can be filed through the Continuum Memory repository's support channel: https://github.com/cargo-culture/continuum-memory/issues
