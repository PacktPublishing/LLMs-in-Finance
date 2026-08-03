from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def ensure_aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    source: str
    available_at: datetime
    issuer: str = ""
    section: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")
        if not self.text.strip():
            raise ValueError("text cannot be empty")
        object.__setattr__(self, "available_at", ensure_aware(self.available_at, "available_at"))


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    material: bool = True


@dataclass(frozen=True)
class Evidence:
    """A retrieved passage.

    ``score`` is normalized within the returned result set, so the top hit always
    scores 1.0; it ranks, it does not measure confidence. ``raw_score`` keeps the
    unnormalized retriever score for anything that needs an absolute threshold.
    """

    evidence_id: str
    document_id: str
    passage: str
    available_at: datetime
    score: float
    source: str = ""
    raw_score: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "available_at", ensure_aware(self.available_at, "available_at"))
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("evidence score must lie in [0, 1]")


@dataclass(frozen=True)
class EvidenceBundle:
    claim: Claim
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class ToolCall:
    request_id: str
    actor: str
    tool: str
    arguments: Mapping[str, Any]
    requested_at: datetime
    side_effect: Literal["none", "reversible", "material"] = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_at", ensure_aware(self.requested_at, "requested_at"))


@dataclass(frozen=True)
class GateDecision:
    status: Literal["allow", "reject", "escalate"]
    reason: str
    sanitized_arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    event_type: str
    timestamp: datetime
    payload: Mapping[str, Any]
    parent_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_aware(self.timestamp, "timestamp"))


Side = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    side: Side
    quantity: int
    limit_price: float | None
    thesis: str
    evidence_ids: tuple[str, ...]
    confidence: float

    def validate(self) -> None:
        if not self.symbol or not self.symbol.replace(".", "").isalnum():
            raise ValueError("symbol must be a non-empty alphanumeric identifier")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be positive when supplied")
        if self.quantity < 0:
            raise ValueError("quantity must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        if self.side == "HOLD" and self.quantity != 0:
            raise ValueError("HOLD requires quantity = 0")
        if self.side != "HOLD" and self.quantity == 0:
            raise ValueError("BUY or SELL requires a positive quantity")
        if self.side != "HOLD" and not self.evidence_ids:
            raise ValueError("material trade intent requires evidence")
