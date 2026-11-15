"""Dashboard data and the MCP App view (``ui://holdout-vault/dashboard.html``).

The same data backs the App and a plain-text summary, because many clients (Claude Code among them)
do not render Apps.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from holdout_vault.claims.audit import metric_key, record_value
from holdout_vault.claims.extract import LOWER_IS_BETTER
from holdout_vault.claims.records import workspace_records

DASHBOARD_URI = "ui://holdout-vault/dashboard.html"


def score_gaps(entries: list[dict[str, Any]], repo: Path | None) -> list[dict[str, Any]]:
    """For each dataset's latest sealed score: the agent's own best number for the same metric, and the gap.

    A large positive gap (own number far above the sealed one) is what overfitting to a validation set,
    or leakage in the agent's own evaluation, looks like.
    """
    latest: dict[str, dict[str, Any]] = {}
    for e in entries:
        if e["kind"] in ("score", "evaluate"):
            latest[e["payload"]["dataset"]] = e
    own = workspace_records(repo) if repo is not None else []
    gaps = []
    for dataset, e in sorted(latest.items()):
        for metric, sealed in e["payload"].get("metrics", {}).items():
            key = metric_key(metric)
            mine = [r for r in own if metric_key(r.metric) == key]
            if not mine:
                continue
            pick = (min if key in LOWER_IS_BETTER else max)(mine, key=record_value)
            value = record_value(pick)
            better = sealed - value if key in LOWER_IS_BETTER else value - sealed
            gaps.append(
                {
                    "dataset": dataset,
                    "metric": key,
                    "sealed": round(float(sealed), 6),
                    "sealed_entry": e["seq"],
                    "own": round(value, 6),
                    "own_source": f"{pick.source} {pick.run or pick.ref}",
                    "optimism": round(better, 6),
                }
            )
    return gaps


def dashboard_data(vault: Any, repo: Path | None, findings: list[dict[str, Any]]) -> dict[str, Any]:
    status = vault.status()
    entries = vault.ledger_entries()
    scores = [
        {
            "seq": e["seq"],
            "kind": e["kind"],
            "dataset": e["payload"].get("dataset"),
            "metrics": e["payload"].get("metrics"),
            "commit": (e["payload"].get("stamp") or {}).get("git_commit", "")[:10] or None,
            "findings": e["payload"].get("findings") or [],
            "ts": e.get("ts"),
        }
        for e in entries
        if e["kind"] in ("score", "evaluate")
    ]
    refusals = [
        {
            "seq": e["seq"],
            "dataset": e["payload"].get("dataset"),
            "reason": e["payload"].get("reason")
            or ", ".join(f["code"] for f in e["payload"].get("findings", []) if isinstance(f, dict)),
        }
        for e in entries
        if e["kind"] in ("refused", "tamper")
    ]
    data = {
        "project": status["project"],
        "mode": status["mode"],
        "budget": status["budget"],
        "spent": status["spent"],
        "remaining": status["remaining"],
        "ledger_ok": status["ledger_ok"],
        "ledger_entries": status["ledger_entries"],
        "ledger_head": status["ledger_head"],
        "scores": scores[-50:],
        "refusals": refusals[-50:],
        "findings": findings,
        "gaps": score_gaps(entries, repo),
    }
    data["summary"] = summary_text(data)
    return data


def summary_text(d: dict[str, Any]) -> str:
    lines = [
        f"Project {d['project']} ({d['mode']} mode). "
        f"Reveals: {d['spent']} of {d['budget']} used, {d['remaining']} left.",
        f"Ledger {'verifies' if d['ledger_ok'] else 'is BROKEN'}: {d['ledger_entries']} entries, "
        f"head {str(d['ledger_head'])[:12]}.",
    ]
    if d["scores"]:
        lines.append("Sealed scores:")
        for s in d["scores"][-10:]:
            shown = ", ".join(f"{k}={v:.4f}" for k, v in (s["metrics"] or {}).items())
            warn = f"  warnings: {', '.join(s['findings'])}" if s["findings"] else ""
            lines.append(f"  #{s['seq']} {s['kind']} {s['dataset']}: {shown}{warn}")
    if d["refusals"]:
        lines.append(f"Refused or tampered submissions: {len(d['refusals'])} (latest #{d['refusals'][-1]['seq']}).")
    if d["gaps"]:
        lines.append("Own measurement vs sealed score:")
        for g in d["gaps"]:
            lines.append(
                f"  {g['dataset']} {g['metric']}: own {g['own']:.4f} ({g['own_source']}) vs sealed {g['sealed']:.4f} "
                f"(#{g['sealed_entry']}), optimism {g['optimism']:+.4f}"
            )
    blocking = [f for f in d["findings"] if f.get("severity") == "block"]
    if d["findings"]:
        lines.append(f"Workspace findings: {len(blocking)} blocking, {len(d['findings']) - len(blocking)} other.")
        for f in d["findings"][:10]:
            loc = f.get("location", {})
            where = f" ({loc.get('file')}:{loc.get('line')})" if loc.get("file") else ""
            lines.append(f"  {f['code']} [{f['severity']}] {f['title']}{where}")
    return "\n".join(lines)


DASHBOARD_HTML = (resources.files("holdout_vault.mcp") / "dashboard.html").read_text(encoding="utf-8")
