import json
import sys
from pathlib import Path

import pytest
from mcp import Client

from holdout_vault.claims.audit import audit_workspace, metric_key, summary
from holdout_vault.claims.extract import extract_text
from holdout_vault.claims.records import vault_records, workspace_records
from holdout_vault.cli import main
from holdout_vault.core.vault import Vault
from holdout_vault.mcp.server import build_server

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from claims_world import build_world


@pytest.fixture
def world(tmp_path):
    return build_world(tmp_path)


def audit(world, text):
    vault = Vault(world["home"], world["project"])
    return audit_workspace(
        world["ws"], vault_records(vault.ledger_entries()), workspace_records(world["ws"]), text=text
    ).results


def labels(world, text):
    return [(r.claim.metric, r.claim.shown, r.label) for r in audit(world, text)]


# extraction ------------------------------------------------------------------------------------
def test_extracts_prose_tables_and_percentages():
    text = """
The model reached an AUC of 0.9012 on the sealed test set (ledger entry #4).
Validation accuracy was 87.1% and log loss 0.312.
We trained for 0.5 hours with learning rate 0.05, 300 trees, in 2024.
```python
auc = 0.99
```
| Model | Accuracy | ROC AUC |
|---|---|---|
| baseline | 0.790 | - |
| lgbm (val) | **0.871** | 0.942 |
"""
    got = [(c.metric, c.value, c.scope, c.ref_seq) for c in extract_text(text, "r.md")]
    assert got == [
        ("roc_auc", 0.9012, "sealed", 4),
        ("accuracy", 0.871, "validation", None),
        ("log_loss", 0.312, "validation", None),
        ("accuracy", 0.79, "unspecified", None),
        ("accuracy", 0.871, "validation", None),
        ("roc_auc", 0.942, "validation", None),
    ]


def test_tolerance_follows_precision_shown():
    (a,) = extract_text("accuracy 0.84", "x")
    (b,) = extract_text("accuracy of 84.11%", "x")
    assert a.tolerance == pytest.approx(0.005) and b.value == pytest.approx(0.8411)
    assert b.tolerance == pytest.approx(0.00005)


def test_backward_form_and_inline_code_ignored():
    got = [(c.metric, c.value) for c in extract_text("We got 0.91 AUC, while `acc = 0.5` is code.", "x")]
    assert got == [("roc_auc", 0.91)]


def test_metric_key_normalises_recorded_names():
    assert metric_key("val_auc") == "roc_auc"
    assert metric_key("eval/accuracy_score") == "accuracy"
    assert metric_key("f1_score") == "f1"
    assert metric_key("val_log_loss") == "log_loss"


# records ---------------------------------------------------------------------------------------
def test_workspace_records_cover_every_source(world):
    recs = workspace_records(world["ws"])
    sources = {r.source for r in recs}
    assert sources == {"mlflow", "json", "wandb"}
    assert any(r.run == "lgbm-v3" and r.metric == "val_auc" and r.value == 0.942 for r in recs)
    assert any(r.run == "xgb-tuned" and r.metric == "val_f1" for r in recs)
    assert any(r.source == "json" and r.metric == "roc_auc" and r.meta["key"] == "cv.roc_auc" for r in recs)


def test_vault_records_skip_non_scores(world):
    recs = vault_records(Vault(world["home"], world["project"]).ledger_entries())
    assert {r.ref for r in recs} == {"2", "3", "4", "5", "7", "8", "10"}


# labels ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Sealed accuracy was 0.8203.", "traced"),
        ("Test AUC of 0.89 (entry #3).", "traced"),
        ("Fraud F1 is 0.671.", "traced"),
        ("Our model hit 84.11% accuracy.", "cherry-picked"),
        ("Best of 4 submissions: accuracy 0.8411.", "traced"),
        ("The evaluated model scored 0.8975 AUC.", "stale"),
        ("Entry #4 shows accuracy 0.85.", "mismatched"),
        ("Accuracy was 0.80 according to ledger entry #42.", "orphaned"),
        ("We reached an AUC of 0.95.", "orphaned"),
        ("Validation AUC was 0.942.", "untrusted-source"),
        ("Cross-validated AUC was 0.931.", "untrusted-source"),
        ("Validation RMSE 11.2.", "untrusted-source"),
        ("Run lgbm-v3 reached accuracy 0.88.", "mismatched"),
        ("Fraud log loss 0.3011.", "traced"),
    ],
)
def test_labels(world, text, expected):
    (result,) = audit(world, text)
    assert result.label == expected, result.explanation


def test_listing_other_scores_discloses(world):
    got = labels(world, "Our churn accuracy went from 0.8125 to 0.8203 and then 0.8411.")
    assert [g[2] for g in got] == ["traced", "traced", "traced"]


def test_summary_line(world):
    results = audit(world, "Accuracy 0.8203. AUC of 0.95. Validation AUC 0.942.")
    assert summary(results) == "3 metric claim(s), 1 traced, 2 unsupported (1 orphaned, 1 untrusted-source)."
    assert summary([]) == "No metric claims found."


def test_untrusted_number_flags_leakage_warnings(world):
    vault = Vault(world["home"], world["project"])
    (r,) = audit_workspace(
        world["ws"], vault_records(vault.ledger_entries()), workspace_records(world["ws"]),
        text="Validation AUC was 0.942.", leakage_codes=["HV-COD-001"],
    ).results  # fmt: skip
    assert r.flags == ["leakage_warnings"] and "HV-COD-001" in r.explanation


def test_documents_and_notebooks_in_workspace(world):
    ws = world["ws"]
    (ws / "REPORT.md").write_text("Final sealed accuracy: 0.8203 (entry #3).\n")
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {},
          "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["Our AUC reached 0.97."]},
                    {"cell_type": "code", "metadata": {}, "source": ["auc = 0.5"], "outputs": []}]}  # fmt: skip
    (ws / "analysis.ipynb").write_text(json.dumps(nb))
    vault = Vault(world["home"], world["project"])
    report = audit_workspace(ws, vault_records(vault.ledger_entries()), workspace_records(ws))
    got = {(r.claim.file, r.label) for r in report.results}
    assert got == {("REPORT.md", "traced"), ("analysis.ipynb", "orphaned")}
    nb_result = next(r for r in report.results if r.claim.file == "analysis.ipynb")
    assert nb_result.to_dict()["location"] == {"file": "analysis.ipynb", "line": 1, "cell": 0}
    assert report.records["vault"] == 15


async def test_audit_claims_tool(world):
    (world["ws"] / "REPORT.md").write_text("Sealed accuracy 0.8203, validation AUC 0.942, AUC 0.99.\n")
    server = build_server(world["project"], home=world["home"], repo=world["ws"], mode="watch")
    async with Client(server) as c:
        out = (await c.call_tool("audit_claims", {})).structured_content
        only = (await c.call_tool("audit_claims", {"text": "Accuracy 0.8411."})).structured_content
    assert out["summary"] == "3 metric claim(s), 1 traced, 2 unsupported (1 orphaned, 1 untrusted-source)."
    assert out["counts"] == {"traced": 1, "orphaned": 1, "untrusted-source": 1}
    assert "REPORT.md" in out["documents"]
    assert only["claims"][0]["label"] == "cherry-picked"


def test_audit_cli(world, capsys):
    (world["ws"] / "REPORT.md").write_text("Sealed accuracy 0.8203.\n")
    args = ["audit", "--project", world["project"], "--home", str(world["home"]), "--repo", str(world["ws"])]
    assert main(args) == 0
    assert "[traced] accuracy 0.8203" in capsys.readouterr().out
    assert main([*args, "--text", "AUC 0.99", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["claims"][0]["label"] == "orphaned"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("F1 was around 0.671 on fraud.", [("f1", "0.671")]),
        ("Log loss on the fraud set is 0.3011.", [("log_loss", "0.3011")]),
        ("Accuracy for entry #3: 0.8203", [("accuracy", "0.8203")]),
        ("Accuracy is 0.8411, up from 0.8203.", [("accuracy", "0.8411"), ("accuracy", "0.8203")]),
        ("AUC improved from 0.8911 to 0.9012.", [("roc_auc", "0.8911"), ("roc_auc", "0.9012")]),
        ("It reached 0.8411 accuracy and a ROC AUC of 0.9012.", [("accuracy", "0.8411"), ("roc_auc", "0.9012")]),
        ("The 0.90 AUC comes from the best of 4.", [("roc_auc", "0.90")]),
    ],
)
def test_phrasings(text, expected):
    assert [(c.metric, c.shown) for c in extract_text(text, "x")] == expected


def test_named_run_beats_vault_match(world):
    (r,) = audit(world, "On run lgbm-v3 the AUC was 0.9012.")
    assert r.label == "mismatched" and r.evidence["run"] == "lgbm-v3"


def test_transposed_table_and_cited_sequence(world):
    text = """
| Metric | lgbm-v3 | entry #4 |
|---|---|---|
| ROC-AUC | 0.9420 | 0.9012 |
| accuracy | 0.8710 | 0.85 |
| rows | 7,043 | 7,043 |

Accuracy moved from 0.8125 at entry #2 to 0.8411 at entry #4.
"""
    got = [(c.metric, c.shown, c.ref_seq) for c in extract_text(text, "x")]
    assert got == [
        ("roc_auc", "0.9420", None), ("roc_auc", "0.9012", 4),
        ("accuracy", "0.8710", None), ("accuracy", "0.85", 4),
        ("accuracy", "0.8125", 2), ("accuracy", "0.8411", 4),
    ]  # fmt: skip
    assert [r.label for r in audit(world, text)] == [
        "untrusted-source", "cherry-picked", "untrusted-source", "mismatched", "traced", "traced",
    ]  # fmt: skip
