"""Integrity findings: stable code, severity, plain-English explanation and exact location."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Severity = Literal["block", "warn", "info"]


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    title: str
    explanation: str
    location: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
