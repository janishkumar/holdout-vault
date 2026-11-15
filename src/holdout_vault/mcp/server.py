"""MCP front door.

Exposes only what an agent may do: ask for aggregate scores and read status. Registration, budgets and
grants are human-only CLI commands. Going over budget asks the human through a mid-call confirmation
(a multi-round-trip request on protocol 2026-07-28, a classic elicitation on older clients).
"""

# No `from __future__ import annotations`: the SDK evaluates tool signatures at runtime, and the
# resolver referenced in `score`'s annotation is a local closure.
from pathlib import Path
from typing import Annotated

from mcp.server.elicitation import AcceptedElicitation, ElicitationResult
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.resolve import Elicit, Resolve

from holdout_vault.core.errors import BudgetExhausted, Refused, TamperDetected, ValidationError
from holdout_vault.core.findings import Finding
from holdout_vault.core.provenance import collect_stamp
from holdout_vault.core.registry import read_table
from holdout_vault.core.vault import Approval, Mode, Vault
from holdout_vault.mcp.models import ApproveReveal, FindingModel, ScoreVerdict, VaultStatus

WATCH_NOTE = (
    "Watch-only mode: the test file stays on this machine where the agent could read it. "
    "The vault detects tampering and label use; it does not prevent access."
)
SEALED_NOTE = "Sealed mode: labels and scorer live in the vault container, out of the agent's reach."

INSTRUCTIONS = """\
holdout-vault holds the sealed test labels and the scorer for this project.
Train on train.csv, predict on test_features.csv, write a CSV with the id column and a
'prediction' column (plus 'proba' or 'proba_<class>' for probability metrics), then call `score`.
Each successful score spends one reveal from a limited budget and is recorded in a tamper-evident
ledger. Report only numbers returned by this server, and quote the ledger entry they came from.
"""


def _findings(fs: list[Finding]) -> list[FindingModel]:
    return [FindingModel(**f.to_dict()) for f in fs]


def build_server(
    project: str, home: str | Path | None = None, repo: str | Path | None = ".", mode: Mode | None = None
) -> MCPServer:
    vault = Vault(home, project, mode=mode)
    repo_dir = Path(repo).resolve() if repo is not None else None
    mcp = MCPServer("holdout-vault", instructions=INSTRUCTIONS)

    def reveal_gate(dataset: str) -> ApproveReveal | Elicit[ApproveReveal] | None:
        state = vault.budget_state()
        if state.remaining > 0:
            return None
        return Elicit(
            f"The reveal budget for project '{project}' is used up ({state.spent} sealed scores revealed). "
            f"Allow one more score on '{dataset}'? Every extra reveal is logged in the ledger.",
            ApproveReveal,
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
        approval = None
        if isinstance(gate, AcceptedElicitation) and gate.data is not None:
            if not gate.data.approve:
                return _over_budget(dataset, vault, "The human declined another reveal.")
            approval = Approval(by="human", via="mcp-elicitation")
        elif not isinstance(gate, AcceptedElicitation):
            return _over_budget(dataset, vault, f"The confirmation was {gate.action}; no score was revealed.")

        path = Path(predictions_path)
        if not path.is_absolute() and repo_dir is not None:
            path = repo_dir / path
        try:
            preds = read_table(path)
        except FileNotFoundError:
            return _invalid(dataset, vault, f"predictions file not found: {path}")
        except Exception as exc:
            return _invalid(dataset, vault, f"could not read predictions: {exc}")

        stamp = collect_stamp(repo_dir, seed=seed)
        stamp["predictions_path"] = str(path)
        try:
            r = vault.score(dataset, preds, stamp=stamp, approval=approval)
        except ValidationError as exc:
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
                message=(
                    "A blocking integrity finding stopped this score. "
                    "It has been logged; no reveal was spent."
                ),
                findings=_findings(exc.findings),
            )
        shown = ", ".join(f"{k}={v:.4f}" for k, v in r.metrics.items())
        return ScoreVerdict(
            status="scored",
            dataset=dataset,
            mode=r.mode,
            message=(
                f"{shown} on {r.n_scored} sealed rows. "
                f"Ledger entry #{r.entry_seq}. {r.remaining} reveals left."
            ),
            metrics=r.metrics,
            n_scored=r.n_scored,
            budget=r.budget,
            remaining=r.remaining,
            stamp=r.stamp,
            findings=_findings(r.findings),
        )

    @mcp.tool(title="Vault status")
    def vault_status() -> VaultStatus:
        """Registered datasets, reveal budget, mode, and whether the ledger verifies."""
        s = vault.status()
        return VaultStatus(**s, note=WATCH_NOTE if vault.mode == "watch" else SEALED_NOTE)

    return mcp


def _invalid(dataset: str, vault: Vault, msg: str) -> ScoreVerdict:
    return ScoreVerdict(
        status="invalid",
        dataset=dataset,
        mode=vault.mode,
        message=f"{msg}. No reveal was spent.",
        remaining=vault.budget_state().remaining,
    )


def _over_budget(dataset: str, vault: Vault, msg: str) -> ScoreVerdict:
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
