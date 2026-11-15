"""Audit the claims in a workspace against a local or sealed vault."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from holdout_vault.checks.code import Context, code_findings
from holdout_vault.claims.audit import LABELS, AuditReport, audit_workspace
from holdout_vault.claims.records import vault_records, workspace_records

LEAKAGE_CODES = {"HV-COD-001", "HV-COD-002", "HV-COD-003", "HV-COD-005"}


def leakage_codes(vault: Any, repo: Path) -> set[str]:
    """Leakage warnings in the workspace code, for any registered dataset."""
    codes: set[str] = set()
    for d in vault.status().get("datasets", []):
        try:
            ctx = vault.check_context(d["name"])
        except Exception:
            continue
        ctx_ = Context(targets={ctx["target"]}, labelled_test=[ctx["test_path"]], vault_home=ctx["home"])
        codes |= {f.code for f in code_findings(repo, ctx_).findings} & LEAKAGE_CODES
    return codes


def run_audit(vault: Any, repo: str | Path, *, paths: list[str] | None = None, text: str | None = None) -> AuditReport:
    repo = Path(repo).resolve()
    return audit_workspace(
        repo,
        vault_records(vault.ledger_entries()),
        workspace_records(repo),
        paths=paths,
        text=text,
        leakage_codes=leakage_codes(vault, repo),
    )


def counts(report: AuditReport) -> dict[str, int]:
    return {label: sum(r.label == label for r in report.results) for label in LABELS}
