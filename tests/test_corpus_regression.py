"""The three blind corpora as a regression bar: every case must keep its expected codes.

Their first-run scores (before any fix informed by them) are the held-out numbers; see
benchmark/results/code_checks_*_first_run.json and the README.
"""

import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH))

from code_checks import evaluate  # noqa: E402


@pytest.mark.parametrize("corpus", ["code_blind", "code_blind2", "code_blind3"])
def test_corpus_still_exact(corpus):
    res = evaluate(BENCH / "corpus" / corpus)
    assert res["mismatches"] == []


@pytest.mark.parametrize(("corpus", "min_recall"), [("claims_blind", 0.99), ("claims_blind2", 0.86)])
def test_claim_corpus_floor(corpus, min_recall):
    from claim_checks import evaluate as evaluate_claims

    res = evaluate_claims(BENCH / "corpus" / corpus)
    assert res["extraction_precision"] == 1.0
    assert res["extraction_recall"] >= min_recall
    assert res["label_accuracy_on_matched"] == 1.0
