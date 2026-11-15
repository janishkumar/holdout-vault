"""MCP front door.

Exposes only what an agent may do: ask for aggregate scores and read status. Registration, budgets and
grants are human-only CLI commands. Going over budget asks the human through a mid-call confirmation
(a multi-round-trip request on protocol 2026-07-28, a classic elicitation on older clients).
"""

# No `from __future__ import annotations`: the SDK evaluates tool signatures at runtime, and the
# resolver referenced in the tools' annotations is a local closure.
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from mcp.server.elicitation import AcceptedElicitation, ElicitationResult
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.resolve import Elicit, Resolve

from holdout_vault.checks.run import CheckResult, full_checks, workspace_checks
from holdout_vault.core.errors import BudgetExhausted, Refused, TamperDetected, ValidationError, VaultError
from holdout_vault.core.findings import Finding
from holdout_vault.core.hashing import file_sha256
from holdout_vault.core.provenance import collect_stamp
from holdout_vault.core.registry import read_table
from holdout_vault.core.vault import Approval, Mode, Vault
from holdout_vault.mcp.models import ApproveReveal, FindingModel, LeakageReport, ScoreVerdict, VaultStatus
from holdout_vault.remote import RemoteVault
from holdout_vault.runner.launcher import RunnerConfig, RunnerError, run_model

WATCH_NOTE = (
    "Watch-only mode: the test file stays on this machine where the agent could read it, and models "
    "run in a plain subprocess. The vault detects tampering and label use; it does not prevent access."
)
SEALED_NOTE = (
    "Sealed mode: labels and scorer live in the vault container, and models run in an isolated "
    "runner container with no labels and no network."
)

INSTRUCTIONS = """\
holdout-vault holds the sealed test labels and the scorer for this project.
Train on train.csv. Then either write a predictions CSV (the id column, 'prediction', plus 'proba' or
'proba_<class>' for probability metrics) and call `score`, or save the model (joblib file, or a folder
with predict.py defining predict(features)) and call `evaluate`.
Each successful score spends one reveal from a limited budget and is recorded in a tamper-evident
ledger. Report only numbers returned by this server, and quote the ledger entry they came from.
Before each score the server checks the workspace code and data files for leakage; `check_leakage`
runs the same checks for free.
"""


def _findings(fs: list[Finding]) -> list[FindingModel]:
    return [FindingModel(**f.to_dict()) for f in fs]


def model_sha256(path: Path) -> str:
    if path.is_file():
        return file_sha256(path)
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(path)).encode() + b"\0" + file_sha256(f).encode())
    return h.hexdigest()


def build_server(
    project: str,
    home: str | Path | None = None,
    repo: str | Path | None = ".",
    mode: Mode | None = None,
    vault: Any = None,
    runner: RunnerConfig | None = None,
) -> MCPServer:
    if vault is None:
        vault = (RemoteVault.from_config(project, home) if mode != "watch" else None) or Vault(home, project, mode=mode)
    runner_cfg = runner or RunnerConfig(use_docker=vault.mode == "sealed")
    repo_dir = Path(repo).resolve() if repo is not None else None
    mcp = MCPServer("holdout-vault", instructions=INSTRUCTIONS)

    def resolve_path(p: str) -> Path:
        path = Path(p).expanduser()
        return path if path.is_absolute() or repo_dir is None else repo_dir / path

    def reveal_gate(dataset: str) -> ApproveReveal | Elicit[ApproveReveal] | None:
        try:
            state = vault.budget_state()
        except VaultError:
            return None  # surfaced by the tool itself
        if state.remaining > 0:
            return None
        return Elicit(
            f"The reveal budget for project '{project}' is used up ({state.spent} sealed scores revealed). "
            f"Allow one more score on '{dataset}'? Every extra reveal is logged in the ledger.",
            ApproveReveal,
        )

    def approval_from(gate, dataset: str) -> tuple[Approval | None, ScoreVerdict | None]:
        if isinstance(gate, AcceptedElicitation):
            if gate.data is None:
                return None, None
            if gate.data.approve:
                return Approval(by="human", via="mcp-elicitation"), None
            return None, _over_budget(dataset, vault, "The human declined another reveal.")
        return None, _over_budget(dataset, vault, f"The confirmation was {gate.action}; nothing was revealed.")

    def checks(dataset: str, exclude: list[Path]) -> CheckResult:
        if repo_dir is None:
            return CheckResult()
        try:
            return workspace_checks(vault, dataset, repo_dir, exclude=exclude)
        except ValidationError:
            raise
        except Exception as exc:  # a checker bug must not block scoring; say so instead
            note = Finding(
                code="HV-CHK-000",
                severity="info",
                title="Workspace checks could not run",
                explanation=f"The code and artifact checks failed with {exc.__class__.__name__}; nothing was checked.",
            )
            return CheckResult(findings=[note])

    def reveal(
        dataset: str,
        produce: Callable[[], pd.DataFrame],
        stamp: dict[str, Any],
        approval: Approval | None,
        kind: str,
        exclude: list[Path] = (),
    ) -> ScoreVerdict:
        try:
            checked = checks(dataset, list(exclude))
            if checked.blocking:
                vault.refuse(dataset, checked.blocking, source="front-door")
                raise Refused(checked.blocking)
            preds = produce()
            r = vault.score(dataset, preds, stamp=stamp, approval=approval, kind=kind, extra_findings=checked.findings)
        except (ValidationError, RunnerError) as exc:
            return _invalid(dataset, vault, str(exc))
        except BudgetExhausted as exc:
            return _over_budget(dataset, vault, str(exc))
        except TamperDetected as exc:
            return ScoreVerdict(
                status="tamper",
                dataset=dataset,
                mode=vault.mode,
                message="Integrity check failed, so nothing was scored. This has been logged.",
                findings=_findings(exc.findings),
            )
        except Refused as exc:
            return ScoreVerdict(
                status="refused",
                dataset=dataset,
                mode=vault.mode,
                message=("A blocking integrity finding stopped this score. It has been logged; no reveal was spent."),
                findings=_findings(exc.findings),
            )
        shown = ", ".join(f"{k}={v:.4f}" for k, v in r.metrics.items())
        warned = [f.code for f in r.findings if f.severity == "warn"]
        tail = f" Warnings: {', '.join(sorted(set(warned)))}." if warned else ""
        return ScoreVerdict(
            status="scored",
            dataset=dataset,
            mode=r.mode,
            message=(
                f"{shown} on {r.n_scored} sealed rows. Ledger entry #{r.entry_seq}. {r.remaining} reveals left." + tail
            ),
            metrics=r.metrics,
            n_scored=r.n_scored,
            budget=r.budget,
            remaining=r.remaining,
            stamp=r.stamp,
            findings=_findings(r.findings),
        )

    @mcp.tool(title="Score predictions against the sealed test labels")
    def score(
        dataset: str,
        predictions_path: str,
        gate: Annotated[ElicitationResult[ApproveReveal], Resolve(reveal_gate)],
        seed: int | None = None,
    ) -> ScoreVerdict:
        """Score a predictions file (CSV or Parquet) on the sealed test set.

        The file needs the dataset's id column and a 'prediction' column. Returns aggregate metrics
        only, never labels or per-row results. Spends one reveal when it succeeds.
        """
        approval, early = approval_from(gate, dataset)
        if early:
            return early
        path = resolve_path(predictions_path)

        def produce() -> pd.DataFrame:
            if not path.is_file():
                raise ValidationError(f"predictions file not found: {path}")
            try:
                return read_table(path)
            except Exception as exc:
                raise ValidationError(f"could not read predictions: {exc}") from exc

        stamp = {**collect_stamp(repo_dir, seed=seed), "predictions_path": str(path)}
        return reveal(dataset, produce, stamp, approval, "score", exclude=[path])

    @mcp.tool(title="Run a saved model on the sealed test set and score it")
    def evaluate(
        dataset: str,
        model_path: str,
        gate: Annotated[ElicitationResult[ApproveReveal], Resolve(reveal_gate)],
        seed: int | None = None,
    ) -> ScoreVerdict:
        """Run a saved model on the test features, then score its predictions on the sealed labels.

        model_path is a joblib/pickle estimator, or a folder with predict.py defining
        predict(features) -> predictions. In sealed mode the model runs in an isolated container with no
        network and no access to labels. Spends one reveal when it succeeds.
        """
        approval, early = approval_from(gate, dataset)
        if early:
            return early
        path = resolve_path(model_path)
        if not path.exists():
            return _invalid(dataset, vault, f"model not found: {path}")

        def produce() -> pd.DataFrame:
            info = vault.dataset_info(dataset)
            features = vault.test_features(dataset)
            return run_model(path, features, info["id_col"], runner_cfg)

        stamp = {
            **collect_stamp(repo_dir, seed=seed),
            "model_path": str(path),
            "model_sha256": model_sha256(path),
            "runner": "container" if runner_cfg.use_docker else "subprocess",
        }
        return reveal(dataset, produce, stamp, approval, "evaluate")

    @mcp.tool(title="Check the workspace for leakage without scoring")
    def check_leakage(dataset: str, paths: list[str] | None = None) -> LeakageReport:
        """Run the integrity checks without spending a reveal.

        Reports overlap between the sealed split and train (HV-ROW), leakage and tampering patterns in
        the workspace's Python files and notebooks (HV-COD), and data files carrying the hidden canary
        labels (HV-CAN-001). `paths` limits the code checks to some files or folders.
        """
        try:
            if repo_dir is None:
                found = CheckResult(findings=vault.row_findings(dataset))
            else:
                found = full_checks(vault, dataset, repo_dir, paths=paths)
        except ValidationError as exc:
            return LeakageReport(dataset=dataset, message=str(exc), blocking=0, warnings=0, files_scanned=0,
                                 tables_scanned=0)  # fmt: skip
        blocking = len(found.blocking)
        warnings = sum(f.severity == "warn" for f in found.findings)
        if not found.findings:
            msg = f"No problems found in {found.files_scanned} code file(s) and {found.tables_scanned} data file(s)."
        else:
            codes = ", ".join(sorted({f.code for f in found.findings}))
            msg = f"{blocking} blocking and {warnings} warning finding(s): {codes}. " + (
                "Scores are refused until the blocking ones are fixed." if blocking else "Scores are still allowed."
            )
        return LeakageReport(
            dataset=dataset,
            message=msg,
            blocking=blocking,
            warnings=warnings,
            files_scanned=found.files_scanned,
            tables_scanned=found.tables_scanned,
            unparsed=found.unparsed,
            findings=_findings(found.findings),
        )

    @mcp.tool(title="Vault status")
    def vault_status() -> VaultStatus:
        """Registered datasets, reveal budget, mode, and whether the ledger verifies."""
        s = vault.status()
        return VaultStatus(**s, note=WATCH_NOTE if vault.mode == "watch" else SEALED_NOTE)

    return mcp


def _invalid(dataset: str, vault: Any, msg: str) -> ScoreVerdict:
    try:
        remaining = vault.budget_state().remaining
    except VaultError:
        remaining = None
    return ScoreVerdict(
        status="invalid",
        dataset=dataset,
        mode=vault.mode,
        message=f"{msg.rstrip('.')}. No reveal was spent.",
        remaining=remaining,
    )


def _over_budget(dataset: str, vault: Any, msg: str) -> ScoreVerdict:
    st = vault.budget_state()
    return ScoreVerdict(
        status="over_budget",
        dataset=dataset,
        mode=vault.mode,
        message=f"{msg} Ask the human to run `holdout-vault grant` if more reveals are justified.",
        budget=st.budget + st.granted,
        remaining=st.remaining,
    )


def run(project: str, home: str | None, repo: str | None, http: bool = False, port: int = 7342) -> None:
    server = build_server(project, home=home, repo=repo)
    if http:
        server.run("streamable-http", host="127.0.0.1", port=port, stateless_http=True)
    else:
        server.run("stdio")
