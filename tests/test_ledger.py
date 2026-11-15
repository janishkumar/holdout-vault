import json

import pytest

from holdout_vault.core.ledger import GENESIS_HASH, Ledger


@pytest.fixture
def ledger(tmp_path):
    lg = Ledger(tmp_path / "ledger.jsonl")
    lg.append("register", {"dataset": "d1"})
    lg.append("score", {"dataset": "d1", "metrics": {"accuracy": 0.9}})
    lg.append("score", {"dataset": "d1", "metrics": {"accuracy": 0.91}})
    return lg


def _lines(lg):
    return lg.path.read_text().splitlines()


def test_append_chains_hashes(ledger):
    entries = ledger.entries()
    assert [e.seq for e in entries] == [0, 1, 2]
    assert entries[0].prev_hash == GENESIS_HASH
    assert entries[1].prev_hash == entries[0].hash
    assert ledger.head().hash == entries[2].hash


def test_verify_ok(ledger):
    result = ledger.verify()
    assert result.ok and result.count == 3 and result.bad_seq is None


def test_empty_ledger_verifies(tmp_path):
    lg = Ledger(tmp_path / "none.jsonl")
    assert lg.verify().ok and lg.head() is None


def test_edited_payload_detected(ledger):
    lines = _lines(ledger)
    obj = json.loads(lines[1])
    obj["payload"]["metrics"]["accuracy"] = 0.99
    lines[1] = json.dumps(obj)
    ledger.path.write_text("\n".join(lines) + "\n")
    result = ledger.verify()
    assert not result.ok and result.bad_seq == 1


def test_rehashed_edit_breaks_next_link(ledger):
    # An attacker who recomputes the edited entry's own hash still breaks the chain.
    from holdout_vault.core.ledger import entry_hash

    lines = _lines(ledger)
    obj = json.loads(lines[1])
    obj["payload"]["metrics"]["accuracy"] = 0.99
    obj["hash"] = entry_hash(obj)
    lines[1] = json.dumps(obj)
    ledger.path.write_text("\n".join(lines) + "\n")
    result = ledger.verify()
    assert not result.ok and result.bad_seq == 2


def test_deleted_entry_detected(ledger):
    lines = _lines(ledger)
    del lines[1]
    ledger.path.write_text("\n".join(lines) + "\n")
    result = ledger.verify()
    assert not result.ok and result.bad_seq == 1


def test_reordered_entries_detected(ledger):
    lines = _lines(ledger)
    lines[1], lines[2] = lines[2], lines[1]
    ledger.path.write_text("\n".join(lines) + "\n")
    assert not ledger.verify().ok


def test_truncated_line_detected(ledger):
    text = ledger.path.read_text()
    ledger.path.write_text(text[:-20])
    result = ledger.verify()
    assert not result.ok and result.bad_seq == 2
    assert "parse" in result.reason


def test_append_refuses_on_broken_chain(ledger):
    lines = _lines(ledger)
    del lines[0]
    ledger.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="ledger"):
        ledger.append("score", {})


def test_head_hash_survives_reopen(ledger):
    again = Ledger(ledger.path)
    assert again.head().hash == ledger.head().hash
    again.append("grant", {"n": 1})
    assert again.verify().ok
