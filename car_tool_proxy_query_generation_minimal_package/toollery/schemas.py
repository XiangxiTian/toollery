from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """Metadata passed to retrievers and final tool selectors."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    category: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolSpec":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            parameters=dict(data.get("parameters", {})),
            category=data.get("category"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ManualEntry:
    """A self-verified natural-language intent mapped to one tool."""

    query: str
    tool_name: str
    source: str = "synthetic"
    verification_score: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManualEntry":
        return cls(
            query=str(data["query"]),
            tool_name=str(data["tool_name"]),
            source=str(data.get("source", "synthetic")),
            verification_score=float(data.get("verification_score", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalHit:
    query: str
    tool_name: str
    score: float


@dataclass(frozen=True)
class ToolCandidate:
    tool: ToolSpec
    score: float
    supporting_queries: list[RetrievalHit]


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
