"""Run every integrity check for one dataset against the agent's workspace.

Works with a local :class:`~holdout_vault.core.vault.Vault` or a sealed
:class:`~holdout_vault.remote.RemoteVault`: code checks and table collection happen here, on the host,
and the canary comparison happens wherever the canary labels live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from holdout_vault.checks.artifacts import collect_tables
from holdout_vault.checks.code import Context, code_findings
from holdout_vault.core.findings import Finding


@dataclass
class CheckResult:
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    tables_scanned: int = 0
    unparsed: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "block"]


def workspace_checks(
    vault: Any,
    dataset: str,
    repo: str | Path,
    *,
    paths: list[str] | None = None,
    exclude: list[str | Path] = (),
) -> CheckResult:
    """Code checks (HV-COD) and canary artifacts (HV-CAN-001) for ``dataset`` over ``repo``."""
    ctx = vault.check_context(dataset)
    report = code_findings(
        repo,
        Context(targets={ctx["target"]}, labelled_test=[ctx["test_path"]], vault_home=ctx["home"]),
        paths=paths,
    )
    tables = collect_tables(repo, [ctx["id_col"], "id"], exclude=exclude)
    found = list(report.findings)
    if tables:
        found += vault.artifact_findings(dataset, tables)
    return CheckResult(found, report.files, len(tables), report.unparsed)


def full_checks(vault: Any, dataset: str, repo: str | Path, *, paths: list[str] | None = None) -> CheckResult:
    """Everything ``check_leakage`` reports: the split's row checks plus the workspace checks."""
    result = workspace_checks(vault, dataset, repo, paths=paths)
    result.findings = [*vault.row_findings(dataset), *result.findings]
    return result
