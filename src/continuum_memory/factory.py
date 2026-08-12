from __future__ import annotations

from pathlib import Path

from .config import Settings
from .embeddings import HashingEmbedder, OpenAIEmbedder
from .federation import FederatedMemoryService
from .service import MemoryService
from .store import SQLiteMemoryStore


def _create_embedder(
    settings: Settings,
    *,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
):
    provider = (embedding_provider or settings.embedding_provider).casefold()
    dimensions = embedding_dimensions or settings.embedding_dimensions
    if provider == "hashing":
        return HashingEmbedder(dimension=dimensions)
    if provider == "openai":
        return OpenAIEmbedder(
            model=embedding_model or settings.embedding_model,
            dimensions=dimensions,
        )
    raise ValueError("embedding provider must be 'hashing' or 'openai'")


def create_service(
    *,
    db_path: str | Path | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    strict_embeddings: bool = False,
) -> tuple[MemoryService, Settings]:
    settings = Settings.from_env()
    embedder = _create_embedder(
        settings,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    store = SQLiteMemoryStore(db_path or settings.db_path)
    return (
        MemoryService(store, embedder, strict_embeddings=strict_embeddings),
        settings,
    )


def create_federated_service(
    *,
    library_dir: str | Path | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    max_books: int | None = None,
    max_workers: int | None = None,
    strict_embeddings: bool = False,
) -> tuple[FederatedMemoryService, Settings]:
    settings = Settings.from_env()
    embedder = _create_embedder(
        settings,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
    )
    return (
        FederatedMemoryService(
            library_dir or settings.library_dir,
            embedder,
            max_books=max_books or settings.library_max_books,
            max_workers=max_workers or settings.library_workers,
            strict_embeddings=strict_embeddings,
        ),
        settings,
    )
