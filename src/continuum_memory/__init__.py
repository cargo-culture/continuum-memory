"""Continuum Memory: persistent, provenance-aware memory for LLM systems."""

from .embeddings import HashingEmbedder, OpenAIEmbedder
from .federation import (
    BookDefinition,
    BookRoute,
    FederatedContextPacket,
    FederatedMemoryService,
    FederatedRecallHit,
)
from .models import (
    ContextPacket,
    MemoryInput,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    PolicyRecord,
    RecallHit,
)
from .service import MemoryService
from .store import SQLiteMemoryStore

__all__ = [
    "ContextPacket",
    "BookDefinition",
    "BookRoute",
    "FederatedContextPacket",
    "FederatedMemoryService",
    "FederatedRecallHit",
    "HashingEmbedder",
    "MemoryInput",
    "MemoryKind",
    "MemoryRecord",
    "MemoryService",
    "MemoryStatus",
    "OpenAIEmbedder",
    "PolicyRecord",
    "RecallHit",
    "SQLiteMemoryStore",
]

__version__ = "0.4.0"
