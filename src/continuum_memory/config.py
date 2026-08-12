from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return key, value


def load_env_file(path: str | Path, *, override: bool = False) -> bool:
    """Load a simple dotenv file without logging secret values."""
    env_path = Path(path)
    if not env_path.is_file():
        return False
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
    return True


def load_nearest_env(start: str | Path | None = None) -> Path | None:
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *list(current.parents)[:3]):
        for filename in (".env.local", ".env"):
            candidate = directory / filename
            if load_env_file(candidate):
                return candidate
    return None


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class Settings:
    db_path: Path
    library_dir: Path
    library_max_books: int
    library_workers: int
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    consolidation_model: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_nearest_env()
        dimensions_raw = os.environ.get("CONTINUUM_EMBEDDING_DIMENSIONS", "512")
        try:
            dimensions = max(64, int(dimensions_raw))
        except ValueError:
            dimensions = 512
        return cls(
            db_path=Path(os.environ.get("CONTINUUM_DB_PATH", "./data/continuum.db")),
            library_dir=Path(os.environ.get("CONTINUUM_LIBRARY_DIR", "./data/books")),
            library_max_books=_positive_env_int("CONTINUUM_LIBRARY_MAX_BOOKS", 3),
            library_workers=_positive_env_int("CONTINUUM_LIBRARY_WORKERS", 4),
            embedding_provider=os.environ.get("CONTINUUM_EMBEDDING_PROVIDER", "hashing").casefold(),
            embedding_model=os.environ.get("CONTINUUM_EMBEDDING_MODEL", "text-embedding-3-small"),
            embedding_dimensions=dimensions,
            consolidation_model=os.environ.get("CONTINUUM_CONSOLIDATION_MODEL", "gpt-5.6"),
        )
