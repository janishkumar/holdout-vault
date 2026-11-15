"""Typed tool outputs. Each one is published as the tool's ``outputSchema``."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

VerdictStatus = Literal["scored", "invalid", "over_budget", "refused", "tamper"]


class FindingModel(BaseModel):
    code: str = Field(description="Stable finding code, e.g. HV-CAN-002")
    severity: Literal["block", "warn", "info"]
    title: str
    explanation: str
    location: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class ScoreVerdict(BaseModel):
    status: VerdictStatus = Field(description="'scored' means metrics were revealed and one reveal was spent")
    dataset: str
    message: str = Field(description="Plain-English summary of what happened")
    metrics: dict[str, float] | None = None
    n_scored: int | None = None
    budget: int | None = None
    remaining: int | None = Field(None, description="Reveals left after this call")
    mode: Literal["watch", "sealed"]
    stamp: dict[str, Any] | None = Field(
        None, description="Provenance: commit, dirty flag, data/scorer/lockfile hashes, seed, ledger entry"
    )
    findings: list[FindingModel] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    name: str
    task: str
    metrics: list[str]
    id_col: str
    n_test_rows: int
    export_dir: str
    scorer: str


class VaultStatus(BaseModel):
    project: str
    mode: Literal["watch", "sealed"]
    budget: int
    spent: int
    remaining: int
    ledger_ok: bool
    ledger_entries: int
    ledger_head: str | None
    datasets: list[DatasetInfo]
    note: str


class ApproveReveal(BaseModel):
    approve: bool = Field(description="Allow one more sealed score beyond the budget")
